"""Closed-loop sounding and MRT data transmission inside one 5 ms frame.

The legacy sounding collector and the predictor-driven MRT transmitter remain
available.  This entry point implements the lower-latency path described in
``D-MIMO_MRT_Modification_Plan.pdf``:

* slot 0: TX0 references, per-antenna FDM sounding, SSS/PSS;
* one or more muted processing slots;
* remaining slots: one common stream whose references and data are both MRT
  precoded from CSI measured in slot 0 of that same frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from multiprocessing import freeze_support, get_context
from pathlib import Path
from queue import Empty, Full
import secrets
import time
from typing import Sequence

import numpy as np

from config.cjt_config import (
    CJT_BANDWIDTH_MHZ,
    CJT_CLOCK_SOURCE,
    CJT_PROCESSING_GAP_SLOTS,
    CJT_TX_DEADLINE_GUARD_S,
    CJT_OTW_FORMAT,
    CJT_REFERENCE_SEQUENCE_SEED,
    CJT_RX_ANTENNA_PORT,
    CJT_RX_CHANNELS,
    CJT_RX_SUBDEVICE,
    CJT_RX_DEVICE_ARGS,
    CJT_RX_GAIN_DB,
    CJT_TIME_SOURCE,
    CJT_TX_ANTENNA_PORT,
    CJT_TX_CHANNELS,
    CJT_TX_SUBDEVICE,
    CJT_TX_DEVICE_ARGS,
    CJT_TX_GAIN_DB,
    CJT_START_DELAY_S,
)
from runtime.cjt_metrics import make_cjt_hard_decisions, save_cjt_constellation_plot
from runtime.cjt_precoding import apply_cjt_mrt_precoding, calculate_cjt_mrt_weights
from runtime.cjt_receiver import CJTReceiver
from USRP.cjt_frame import CJTFramePlan, build_cjt_frame_plan
from USRP.cjt_parameters import (
    make_cjt_reference_sequence,
    build_cjt_params,
)
from USRP.cjt_ofdm import ofdm_demodulate_slots
from USRP.cjt_signal_processing import (
    estimate_and_correct_ffo_slots,
    estimate_and_correct_cjt_rfo,
    process_cjt_sounding,
)


DEFAULT_OUTPUT_DIR = "cjt_results"


@dataclass
class CJTChannelFeedback:
    """First-slot estimate passed from RX processing to timed TX."""

    frame_id: int
    csi: np.ndarray
    first_sample_time: float | None
    measurement_usrp_time: float | None
    processing_duration_s: float
    sync_index: int
    ffo_hz: np.ndarray
    rfo_hz: np.ndarray


@dataclass
class CJTTransmissionRecord:
    """Per-frame scheduling result, including deadline and UHD diagnostics."""

    frame_id: int
    sounding_time_usrp_s: float
    data_time_usrp_s: float
    feedback_measurement_usrp_s: float | None
    feedback_processing_duration_s: float | None
    feedback_received_usrp_s: float | None
    deadline_margin_s: float | None
    csi: np.ndarray | None
    weights: np.ndarray | None
    data_transmitted: bool
    failure_reason: str | None
    async_events: tuple[str, ...]


class CJTTimedTransmitter:
    """Small UHD sender for independently timed sounding and data bursts."""

    def __init__(
        self,
        *,
        usrp,
        carrier_frequency,
        sampling_rate,
        gain,
        channels,
        otw_format,
        uhd_module,
    ):
        self.uhd = uhd_module
        self.usrp = usrp
        self.channels = tuple(channels)
        if not self.channels:
            raise ValueError("at least one TX channel must be selected")
        gains = (
            [gain] * len(self.channels)
            if np.isscalar(gain) and not isinstance(gain, bool)
            else list(gain)
        )
        if len(gains) != len(self.channels):
            raise ValueError("gain must be scalar or one value per TX channel")
        for index, channel in enumerate(self.channels):
            usrp.set_tx_rate(sampling_rate, channel)
            usrp.set_tx_freq(
                uhd_module.libpyuhd.types.tune_request(carrier_frequency),
                channel,
            )
            usrp.set_tx_gain(gains[index], channel)
        stream_args = uhd_module.usrp.StreamArgs("fc32", otw_format)
        stream_args.channels = list(self.channels)
        self.streamer = usrp.get_tx_stream(stream_args)

    def send_timed_burst(self, samples, transmission_time):
        samples = np.asarray(samples, dtype=np.complex64)
        if samples.ndim != 2 or samples.shape[0] != len(self.channels):
            raise ValueError(
                "samples must have shape (configured_channel, sample)"
            )
        if samples.shape[1] == 0:
            raise ValueError("samples must not be empty")
        metadata = self.uhd.types.TXMetadata()
        metadata.start_of_burst = True
        metadata.has_time_spec = True
        metadata.time_spec = self.uhd.types.TimeSpec(float(transmission_time))
        sent = 0
        stalled = 0
        while sent < samples.shape[1]:
            count = self.streamer.send(
                np.ascontiguousarray(samples[:, sent:]),
                metadata,
            )
            if count < 0 or count > samples.shape[1] - sent:
                raise RuntimeError("USRP TX streamer returned an invalid count")
            if count:
                sent += count
                stalled = 0
                metadata.start_of_burst = False
                metadata.has_time_spec = False
            else:
                stalled += 1
                if stalled >= 10:
                    raise TimeoutError(
                        "USRP transmit made no progress after 10 attempts"
                    )

        # A zero-length EOB cleanly separates the sounding and data bursts.
        end_metadata = self.uhd.types.TXMetadata()
        end_metadata.end_of_burst = True
        self.streamer.send(
            np.zeros((len(self.channels), 0), dtype=np.complex64),
            end_metadata,
        )

    def drain_async_events(self):
        receiver = getattr(self.streamer, "recv_async_msg", None)
        metadata_type = getattr(self.uhd.types, "TXAsyncMetadata", None)
        if receiver is None or metadata_type is None:
            return ()
        events = []
        while True:
            metadata = metadata_type()
            try:
                available = receiver(metadata, 0.0)
            except TypeError:
                break
            if not available:
                break
            code = str(getattr(metadata, "event_code", "unknown"))
            if "burst_ack" not in code.lower():
                events.append(code)
        return tuple(events)


def _put_latest(target_queue, value):
    try:
        target_queue.put_nowait(value)
        return True
    except Full:
        try:
            target_queue.get_nowait()
        except Empty:
            return False
    try:
        target_queue.put_nowait(value)
        return True
    except Full:
        return False


def process_cjt_sounding_captures(
    *,
    sounding_queue,
    feedback_queue,
    diagnostics_queue,
    stop_event,
    params,
    known_ref_seq,
    sync_template,
    num_frames,
):
    """Convert first-slot blocks to CSI without waiting for a full frame."""
    processed = 0
    try:
        while not stop_event.is_set() and processed < int(num_frames):
            try:
                block = sounding_queue.get(timeout=0.05)
            except Empty:
                continue
            started = time.perf_counter()
            estimate = process_cjt_sounding(
                params,
                known_ref_seq,
                block.samples,
                sync_template=sync_template,
            )
            csi_by_tx_rx = np.asarray(estimate["csi"], dtype=np.complex64)
            if csi_by_tx_rx.shape != (
                int(params["num_tx_ant"]),
                int(params["num_rx_ant"]),
            ):
                raise ValueError("first-slot CSI has an unexpected shape")
            if csi_by_tx_rx.shape[1] != 1:
                raise ValueError(
                    "same-frame MRT currently supports exactly one UE RX"
                )
            processing_duration = time.perf_counter() - started
            measurement_time = None
            if block.first_sample_time is not None:
                measurement_time = (
                    float(block.first_sample_time)
                    + (
                        float(estimate["sync_idx"])
                        + float(params["cjt_csi_timestamp_offset_samples"])
                    )
                    / float(params["sampling_rate"])
                )
            feedback = CJTChannelFeedback(
                frame_id=int(block.frame_id),
                csi=np.ascontiguousarray(csi_by_tx_rx[:, 0]),
                first_sample_time=block.first_sample_time,
                measurement_usrp_time=measurement_time,
                processing_duration_s=processing_duration,
                sync_index=int(estimate["sync_idx"]),
                ffo_hz=np.asarray(estimate["ffo_hz"], dtype=np.float32),
                rfo_hz=np.asarray(estimate["rfo_hz"], dtype=np.float32),
            )
            _put_latest(feedback_queue, feedback)
            _put_latest(diagnostics_queue, feedback)
            processed += 1
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise


def _wait_for_frame_feedback(
    feedback_queue,
    *,
    frame_id,
    usrp,
    deadline_usrp_s,
    stop_event,
):
    while not stop_event.is_set():
        remaining = float(deadline_usrp_s) - float(
            usrp.get_time_now().get_real_secs()
        )
        if remaining <= 0:
            return None
        try:
            feedback = feedback_queue.get(timeout=min(remaining, 0.01))
        except Empty:
            continue
        if int(feedback.frame_id) < int(frame_id):
            continue
        if int(feedback.frame_id) > int(frame_id):
            raise RuntimeError(
                "CSI feedback arrived out of order: "
                f"expected {frame_id}, got {feedback.frame_id}"
            )
        return feedback
    return None


def transmit_cjt_frames(
    *,
    stop_event,
    feedback_queue,
    tx_record_queue,
    start_time_queue,
    receiver_ready_event,
    sync_barrier,
    params,
    plan,
    num_frames,
    deadline_guard_s,
    transmission_delay,
):
    """Schedule sounding, consume matching CSI, then schedule MRT data."""
    try:
        import uhd
        from USRP.cjt_usrp import initialize_tx_usrp

        usrp = initialize_tx_usrp(
            device_args=CJT_TX_DEVICE_ARGS,
            tx_subdev_spec=CJT_TX_SUBDEVICE,
            tx_antenna=CJT_TX_ANTENNA_PORT,
            tx_channels=CJT_TX_CHANNELS,
            clock_source=CJT_CLOCK_SOURCE,
            time_source=CJT_TIME_SOURCE,
            sync_barrier=sync_barrier,
        )
        sender = CJTTimedTransmitter(
            usrp=usrp,
            carrier_frequency=params["carrier_frequency"],
            sampling_rate=params["sampling_rate"],
            gain=CJT_TX_GAIN_DB,
            channels=CJT_TX_CHANNELS,
            otw_format=CJT_OTW_FORMAT,
            uhd_module=uhd,
        )
        while not stop_event.is_set():
            if receiver_ready_event.wait(timeout=0.05):
                break
        else:
            return

        first_frame_time = (
            float(usrp.get_time_now().get_real_secs())
            + float(transmission_delay)
        )
        start_time_queue.put(first_frame_time)
        frame_period_s = float(params["frame_length"]) / float(
            params["sampling_rate"]
        )
        data_offset_s = float(params["cjt_data_start_sample"]) / float(
            params["sampling_rate"]
        )
        records = []

        for frame_id in range(int(num_frames)):
            if stop_event.is_set():
                break
            sounding_time = first_frame_time + frame_id * frame_period_s
            data_time = sounding_time + data_offset_s
            deadline = data_time - float(deadline_guard_s)
            csi = None
            weights = None
            feedback = None
            feedback_received_time = None
            margin = None
            transmitted = False
            reason = None

            if float(usrp.get_time_now().get_real_secs()) >= (
                sounding_time - float(deadline_guard_s)
            ):
                reason = "sounding_schedule_missed"
            else:
                sender.send_timed_burst(plan.sounding_waveform, sounding_time)
                feedback = _wait_for_frame_feedback(
                    feedback_queue,
                    frame_id=frame_id,
                    usrp=usrp,
                    deadline_usrp_s=deadline,
                    stop_event=stop_event,
                )
                feedback_received_time = float(
                    usrp.get_time_now().get_real_secs()
                )
                margin = deadline - feedback_received_time
                if feedback is None:
                    reason = "csi_feedback_deadline_missed"
                else:
                    csi = np.asarray(feedback.csi, dtype=np.complex64)
                    try:
                        weights = calculate_cjt_mrt_weights(
                            csi,
                            num_tx_antennas=int(params["num_tx_ant"]),
                        )
                        data_waveform = apply_cjt_mrt_precoding(
                            plan.common_data_waveforms[frame_id],
                            csi,
                            num_tx_antennas=int(params["num_tx_ant"]),
                        )
                    except ValueError as exc:
                        reason = f"invalid_csi: {exc}"
                    else:
                        now = float(usrp.get_time_now().get_real_secs())
                        margin = deadline - now
                        if now >= deadline:
                            reason = "precoding_completed_after_deadline"
                        else:
                            sender.send_timed_burst(data_waveform, data_time)
                            transmitted = True

            record = CJTTransmissionRecord(
                frame_id=frame_id,
                sounding_time_usrp_s=sounding_time,
                data_time_usrp_s=data_time,
                feedback_measurement_usrp_s=(
                    None if feedback is None else feedback.measurement_usrp_time
                ),
                feedback_processing_duration_s=(
                    None if feedback is None else feedback.processing_duration_s
                ),
                feedback_received_usrp_s=feedback_received_time,
                deadline_margin_s=margin,
                csi=csi,
                weights=weights,
                data_transmitted=transmitted,
                failure_reason=reason,
                async_events=sender.drain_async_events(),
            )
            records.append(record)

        # UHD reports underflow/time errors asynchronously. Give the final
        # scheduled burst time to complete, then preserve any last events in
        # the final record instead of silently losing them at process exit.
        if records:
            final_frame_end = first_frame_time + len(records) * frame_period_s
            remaining = final_frame_end - float(
                usrp.get_time_now().get_real_secs()
            )
            if remaining > 0:
                time.sleep(min(remaining + 0.001, 0.01))
            final_events = sender.drain_async_events()
            if final_events:
                records[-1].async_events += final_events
        for record in records:
            tx_record_queue.put(record)
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise


def receive_cjt_frames(
    *,
    sounding_queue,
    capture_queue,
    stop_event,
    start_time_queue,
    receiver_ready_event,
    sync_barrier,
    params,
    num_frames,
):
    """Keep UHD RX streaming while slot 0 is processed in another process."""
    try:
        import uhd
        from USRP.cjt_usrp import initialize_rx_usrp

        usrp = initialize_rx_usrp(
            device_args=CJT_RX_DEVICE_ARGS,
            rx_subdev_spec=CJT_RX_SUBDEVICE,
            rx_antenna=CJT_RX_ANTENNA_PORT,
            rx_channels=CJT_RX_CHANNELS,
            clock_source=CJT_CLOCK_SOURCE,
            time_source=CJT_TIME_SOURCE,
            sync_barrier=sync_barrier,
        )
        receiver = CJTReceiver(
            usrp=usrp,
            frames_queue=capture_queue,
            num_samples=int(params["frame_length"]),
            carrier_frequency=params["carrier_frequency"],
            sampling_rate=params["sampling_rate"],
            gain=CJT_RX_GAIN_DB,
            channels=CJT_RX_CHANNELS,
            start_time=None,
            otw_format=CJT_OTW_FORMAT,
            stop_event=stop_event,
            uhd_module=uhd,
        )
        receiver_ready_event.set()
        while not stop_event.is_set():
            try:
                receiver.start_time = start_time_queue.get(timeout=0.05)
                break
            except Empty:
                continue
        else:
            return
        receiver.run_cjt(
            sounding_queue=sounding_queue,
            capture_queue=capture_queue,
            num_frames=num_frames,
            sounding_num_samples=int(
                params["cjt_sounding_capture_num_samples"]
            ),
        )
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise
    finally:
        for process_queue in (sounding_queue, capture_queue):
            cancel_join = getattr(process_queue, "cancel_join_thread", None)
            if cancel_join is not None:
                cancel_join()


def recover_cjt_data(
    received_samples,
    params,
    known_ref_seq,
    plan: CJTFramePlan,
):
    """Correct offsets and equalize only the MRT data region."""
    samples = np.asarray(received_samples, dtype=np.complex64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    expected_shape = (
        int(params["num_rx_ant"]),
        int(params["frame_length"]),
    )
    if samples.shape != expected_shape:
        raise ValueError(
            f"received_samples has shape {samples.shape}, expected {expected_shape}"
        )
    corrected, ffo_hz = estimate_and_correct_ffo_slots(samples, params)
    corrected, rfo_hz, rfo_pair_hz = estimate_and_correct_cjt_rfo(
        corrected,
        params,
    )
    slots = ofdm_demodulate_slots(corrected, params)
    data_slots = slots[:, int(params["cjt_data_start_global_slot"]):]
    reference = np.asarray(known_ref_seq, dtype=np.complex64)
    effective_channel = data_slots[:, :, 0, :] / reference[None, None, :]
    slot_idx, symbol_idx, subcarrier_idx = plan.pdsch_indices
    raw = data_slots[:, slot_idx, symbol_idx, subcarrier_idx]
    pdsch_channel = effective_channel[:, slot_idx, subcarrier_idx]
    equalized = np.full_like(raw, np.nan + 1j * np.nan)
    np.divide(
        raw,
        pdsch_channel,
        out=equalized,
        where=np.abs(pdsch_channel) >= 1e-8,
    )
    return {
        "raw_symbols": raw,
        "equalized_symbols": equalized,
        "effective_channel": effective_channel,
        "ffo_hz": ffo_hz,
        "rfo_hz": rfo_hz,
        "rfo_pair_hz": rfo_pair_hz,
    }


def assemble_cjt_frame(
    params,
    plan: CJTFramePlan,
    *,
    frame_id,
    channel_coefficients,
):
    """Assemble a complete synthetic frame with a truly muted processing gap."""
    frame = np.zeros(
        (int(params["num_tx_ant"]), int(params["frame_length"])),
        dtype=np.complex64,
    )
    sounding_end = int(params["cjt_sounding_end_sample"])
    data_start = int(params["cjt_data_start_sample"])
    frame[:, :sounding_end] = plan.sounding_waveform
    frame[:, data_start:] = apply_cjt_mrt_precoding(
        plan.common_data_waveforms[int(frame_id)],
        channel_coefficients,
        num_tx_antennas=int(params["num_tx_ant"]),
    )
    return frame


def _link_metrics(transmitted, equalized, modulation_order):
    transmitted = np.asarray(transmitted, dtype=np.complex64).reshape(-1)
    equalized = np.asarray(equalized, dtype=np.complex64).reshape(-1)
    valid = np.isfinite(equalized.real) & np.isfinite(equalized.imag)
    if not np.any(valid):
        return None
    transmitted = transmitted[valid]
    equalized = equalized[valid]
    error_power = float(np.mean(np.abs(equalized - transmitted) ** 2))
    reference_power = float(np.mean(np.abs(transmitted) ** 2))
    evm = math.sqrt(error_power / max(reference_power, 1e-20))
    decisions = make_cjt_hard_decisions(equalized, int(modulation_order))
    return {
        "num_compared_symbols": int(transmitted.size),
        "evm_rms_percent": 100.0 * evm,
        "evm_snr_db": -20.0 * math.log10(max(evm, 1e-20)),
        "symbol_error_rate": float(np.mean(decisions != transmitted)),
    }


def run_cjt_dry_loopback(params, known_ref_seq, plan, channel_coefficients):
    """Exercise the complete split-frame DSP path without opening UHD."""
    channel = np.asarray(channel_coefficients, dtype=np.complex64).reshape(-1)
    if channel.shape != (int(params["num_tx_ant"]),):
        raise ValueError("synthetic channel must have one value per TX antenna")
    sounding_rx = np.sum(
        channel[:, None] * plan.sounding_waveform,
        axis=0,
        keepdims=True,
    )
    estimate = process_cjt_sounding(
        params,
        known_ref_seq,
        sounding_rx,
        sync_template=plan.sounding_sync_template,
    )
    measured_csi = np.asarray(estimate["csi"], dtype=np.complex64)[:, 0]
    frame = assemble_cjt_frame(
        params,
        plan,
        frame_id=0,
        channel_coefficients=measured_csi,
    )
    received = np.sum(channel[:, None] * frame, axis=0, keepdims=True)
    recovered = recover_cjt_data(
        received,
        params,
        known_ref_seq,
        plan,
    )
    metrics = _link_metrics(
        plan.pdsch_symbols[0],
        recovered["equalized_symbols"][0],
        params["modulation_order"],
    )
    return {
        "measured_csi": measured_csi,
        "weights": calculate_cjt_mrt_weights(measured_csi),
        "metrics": metrics,
        "recovered": recovered,
        "transmitted_frame": frame,
    }


def _stop_processes(processes, stop_event):
    stop_event.set()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


def _drain_queue(process_queue, output):
    while True:
        try:
            output.append(process_queue.get_nowait())
        except Empty:
            return


def _save_run(
    *,
    output_dir,
    params,
    known_ref_seq,
    plan,
    tx_records,
    feedbacks,
    captures,
    frame_seed,
    deadline_guard_s,
    max_plot_symbols,
):
    num_frames = int(plan.common_data_waveforms.shape[0])
    num_tx = int(params["num_tx_ant"])
    num_rx = int(params["num_rx_ant"])
    num_pdsch = int(plan.pdsch_symbols.shape[1])
    csi = np.full((num_frames, num_tx), np.nan + 1j * np.nan, np.complex64)
    weights = np.full_like(csi, np.nan + 1j * np.nan)
    equalized = np.full(
        (num_frames, num_rx, num_pdsch),
        np.nan + 1j * np.nan,
        dtype=np.complex64,
    )
    raw = np.full_like(equalized, np.nan + 1j * np.nan)
    processing_s = np.full(num_frames, np.nan, dtype=np.float64)
    measurement_time = np.full(num_frames, np.nan, dtype=np.float64)
    sync_indices = np.full(num_frames, -1, dtype=np.int64)
    deadline_margin_s = np.full(num_frames, np.nan, dtype=np.float64)
    data_transmitted = np.zeros(num_frames, dtype=bool)
    failure_reasons = np.full(num_frames, "", dtype="U128")
    async_events = np.full(num_frames, "", dtype="U256")
    ffo_hz = np.full((num_frames, num_rx), np.nan, dtype=np.float32)
    rfo_hz = np.full((num_frames, num_rx), np.nan, dtype=np.float32)

    feedback_by_id = {int(value.frame_id): value for value in feedbacks}
    record_by_id = {int(value.frame_id): value for value in tx_records}
    capture_by_id = {int(value.frame_id): value for value in captures}
    for frame_id, feedback in feedback_by_id.items():
        if 0 <= frame_id < num_frames:
            csi[frame_id] = feedback.csi
            processing_s[frame_id] = feedback.processing_duration_s
            sync_indices[frame_id] = feedback.sync_index
            if feedback.measurement_usrp_time is not None:
                measurement_time[frame_id] = feedback.measurement_usrp_time
    for frame_id, record in record_by_id.items():
        if not 0 <= frame_id < num_frames:
            continue
        data_transmitted[frame_id] = record.data_transmitted
        if record.csi is not None:
            csi[frame_id] = record.csi
        if record.weights is not None:
            weights[frame_id] = record.weights
        if record.deadline_margin_s is not None:
            deadline_margin_s[frame_id] = record.deadline_margin_s
        failure_reasons[frame_id] = record.failure_reason or ""
        async_events[frame_id] = ";".join(record.async_events)
        if not record.data_transmitted or frame_id not in capture_by_id:
            continue
        recovered = recover_cjt_data(
            capture_by_id[frame_id].samples,
            params,
            known_ref_seq,
            plan,
        )
        raw[frame_id] = recovered["raw_symbols"]
        equalized[frame_id] = recovered["equalized_symbols"]
        ffo_hz[frame_id] = recovered["ffo_hz"]
        rfo_hz[frame_id] = recovered["rfo_hz"]

    successful = data_transmitted & np.all(
        np.isfinite(equalized[:, 0].real),
        axis=1,
    )
    metrics = _link_metrics(
        plan.pdsch_symbols[successful],
        equalized[successful, 0],
        params["modulation_order"],
    )
    underflow_or_time_errors = sum(
        ("underflow" in event.lower()) or ("time" in event.lower())
        for event in async_events
        if event
    )
    summary = {
        "bandwidth_mhz": float(params["bandwidth_mhz"]),
        "num_frames": num_frames,
        "num_tx_antennas": num_tx,
        "processing_gap_slots": int(params["cjt_processing_gap_slots"]),
        "processing_gap_us": (
            float(params["cjt_processing_gap_samples"])
            / float(params["sampling_rate"])
            * 1e6
        ),
        "data_start_us": (
            float(params["cjt_data_start_sample"])
            / float(params["sampling_rate"])
            * 1e6
        ),
        "deadline_guard_us": float(deadline_guard_s) * 1e6,
        "num_data_transmitted": int(np.count_nonzero(data_transmitted)),
        "deadline_miss_rate": float(1.0 - np.mean(data_transmitted)),
        "num_tx_underflow_or_time_errors": int(underflow_or_time_errors),
        "processing_latency_us": {
            "median": (
                None
                if not np.any(np.isfinite(processing_s))
                else float(np.nanmedian(processing_s) * 1e6)
            ),
            "p95": (
                None
                if not np.any(np.isfinite(processing_s))
                else float(np.nanpercentile(processing_s, 95) * 1e6)
            ),
            "max": (
                None
                if not np.any(np.isfinite(processing_s))
                else float(np.nanmax(processing_s) * 1e6)
            ),
        },
        "link_metrics": metrics,
        "failure_reasons": failure_reasons.tolist(),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("cjt_%Y%m%d_%H%M%S")
    npz_path = output_dir / f"{run_name}.npz"
    json_path = output_dir / f"{run_name}.json"
    plot_path = output_dir / f"{run_name}_constellation.png"
    np.savez_compressed(
        npz_path,
        csi=csi,
        cjt_mrt_weights=weights,
        data_transmitted=data_transmitted,
        processing_duration_s=processing_s,
        csi_measurement_usrp_s=measurement_time,
        sounding_sync_indices=sync_indices,
        deadline_margin_s=deadline_margin_s,
        failure_reasons=failure_reasons,
        tx_async_events=async_events,
        tx_pdsch_symbols=plan.pdsch_symbols,
        rx_pdsch_symbols_raw=raw,
        rx_pdsch_symbols_equalized=equalized,
        ffo_hz=ffo_hz,
        rfo_hz=rfo_hz,
        known_reference_sequence=known_ref_seq,
        frame_seed=np.asarray(frame_seed, dtype=np.int64),
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if metrics is not None:
        save_cjt_constellation_plot(
            plan.pdsch_symbols[successful],
            equalized[successful, 0],
            plot_path,
            modulation_order=int(params["modulation_order"]),
            max_symbols=int(max_plot_symbols),
        )
    else:
        plot_path = None
    return {
        "npz_path": npz_path,
        "json_path": json_path,
        "plot_path": plot_path,
        "summary": summary,
    }


def run_cjt(
    *,
    num_frames=20,
    frame_seed=None,
    bandwidth_mhz=CJT_BANDWIDTH_MHZ,
    processing_gap_slots=CJT_PROCESSING_GAP_SLOTS,
    deadline_guard_s=CJT_TX_DEADLINE_GUARD_S,
    output_dir=DEFAULT_OUTPUT_DIR,
    max_plot_symbols=10000,
):
    """Run the three-process UE/RU experiment and save timing/link results."""
    if int(num_frames) <= 0:
        raise ValueError("num_frames must be greater than zero")
    if float(deadline_guard_s) < 0:
        raise ValueError("deadline_guard_s must be non-negative")
    resolved_seed = secrets.randbits(63) if frame_seed is None else int(frame_seed)
    if resolved_seed < 0:
        raise ValueError("frame_seed must be non-negative")
    params = build_cjt_params(
        bandwidth_mhz,
        processing_gap_slots=processing_gap_slots,
    )
    if len(CJT_TX_CHANNELS) != int(params["num_tx_ant"]):
        raise ValueError("CJT_TX_CHANNELS does not match params['num_tx_ant']")
    if len(CJT_RX_CHANNELS) != int(params["num_rx_ant"]):
        raise ValueError("CJT_RX_CHANNELS does not match params['num_rx_ant']")
    reference_seed = CJT_REFERENCE_SEQUENCE_SEED + int(10 * bandwidth_mhz)
    known_ref_seq = make_cjt_reference_sequence(
        params["N"],
        reference_seed,
    )
    plan = build_cjt_frame_plan(
        params,
        known_ref_seq,
        num_frames=int(num_frames),
        frame_seed=resolved_seed,
    )

    context = get_context("spawn")
    sounding_queue = context.Queue(maxsize=4)
    feedback_queue = context.Queue(maxsize=4)
    diagnostics_queue = context.Queue(maxsize=max(4, int(num_frames)))
    capture_queue = context.Queue(maxsize=max(4, int(num_frames)))
    tx_record_queue = context.Queue(maxsize=max(4, int(num_frames)))
    start_time_queue = context.Queue(maxsize=1)
    stop_event = context.Event()
    receiver_ready_event = context.Event()
    sync_barrier = context.Barrier(2)

    processor = context.Process(
        name="mrt-sounding-processor",
        target=process_cjt_sounding_captures,
        kwargs={
            "sounding_queue": sounding_queue,
            "feedback_queue": feedback_queue,
            "diagnostics_queue": diagnostics_queue,
            "stop_event": stop_event,
            "params": params,
            "known_ref_seq": known_ref_seq,
            "sync_template": plan.sounding_sync_template,
            "num_frames": num_frames,
        },
    )
    receiver = context.Process(
        name="mrt-continuous-receiver",
        target=receive_cjt_frames,
        kwargs={
            "sounding_queue": sounding_queue,
            "capture_queue": capture_queue,
            "stop_event": stop_event,
            "start_time_queue": start_time_queue,
            "receiver_ready_event": receiver_ready_event,
            "sync_barrier": sync_barrier,
            "params": params,
            "num_frames": num_frames,
        },
    )
    transmitter = context.Process(
        name="mrt-timed-transmitter",
        target=transmit_cjt_frames,
        kwargs={
            "stop_event": stop_event,
            "feedback_queue": feedback_queue,
            "tx_record_queue": tx_record_queue,
            "start_time_queue": start_time_queue,
            "receiver_ready_event": receiver_ready_event,
            "sync_barrier": sync_barrier,
            "params": params,
            "plan": plan,
            "num_frames": num_frames,
            "deadline_guard_s": deadline_guard_s,
            "transmission_delay": CJT_START_DELAY_S,
        },
    )
    processes = (processor, receiver, transmitter)
    started = []
    feedbacks = []
    captures = []
    tx_records = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        while any(process.is_alive() for process in processes):
            _drain_queue(diagnostics_queue, feedbacks)
            _drain_queue(capture_queue, captures)
            _drain_queue(tx_record_queue, tx_records)
            for process in processes:
                process.join(timeout=0.02)
            if not receiver.is_alive() and not transmitter.is_alive():
                stop_event.set()
            if any(
                process.exitcode not in (None, 0) for process in processes
            ):
                break
    except KeyboardInterrupt:
        print("Stopping same-frame MRT experiment...")
        raise
    finally:
        _stop_processes(started, stop_event)
        _drain_queue(diagnostics_queue, feedbacks)
        _drain_queue(capture_queue, captures)
        _drain_queue(tx_record_queue, tx_records)
        for process_queue in (
            sounding_queue,
            feedback_queue,
            diagnostics_queue,
            capture_queue,
            tx_record_queue,
            start_time_queue,
        ):
            process_queue.close()
            process_queue.join_thread()

    failures = [
        f"{process.name} (exit code {process.exitcode})"
        for process in processes
        if process.exitcode not in (0, None)
    ]
    if failures:
        raise RuntimeError("Worker process failed: " + ", ".join(failures))
    return _save_run(
        output_dir=output_dir,
        params=params,
        known_ref_seq=known_ref_seq,
        plan=plan,
        tx_records=tx_records,
        feedbacks=feedbacks,
        captures=captures,
        frame_seed=resolved_seed,
        deadline_guard_s=deadline_guard_s,
        max_plot_symbols=max_plot_symbols,
    )


def _parse_complex(value):
    try:
        return complex(value.strip().replace("i", "j"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid complex value {value!r}") from exc


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Measure first-slot CSI and transmit MRT-precoded data later in "
            "the same 5 ms USRP frame."
        )
    )
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--frame-seed", type=int, default=None)
    parser.add_argument("--bandwidth", type=float, default=CJT_BANDWIDTH_MHZ)
    parser.add_argument(
        "--processing-gap-slots",
        type=int,
        default=CJT_PROCESSING_GAP_SLOTS,
        help="Muted 0.5 ms slots between sounding and MRT data.",
    )
    parser.add_argument(
        "--deadline-guard-us",
        type=float,
        default=CJT_TX_DEADLINE_GUARD_S * 1e6,
        help="Minimum lead time required before scheduling the data burst.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--max-plot-symbols", type=int, default=10000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run an ideal synthetic loopback without opening USRPs.",
    )
    parser.add_argument(
        "--synthetic-channel",
        type=_parse_complex,
        nargs="+",
        default=None,
        help="Optional dry-run channel coefficient for each TX antenna.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = _parse_args(argv)
    if args.deadline_guard_us < 0:
        raise ValueError("--deadline-guard-us must be non-negative")
    if args.dry_run:
        seed = secrets.randbits(63) if args.frame_seed is None else args.frame_seed
        params = build_cjt_params(
            args.bandwidth,
            processing_gap_slots=args.processing_gap_slots,
        )
        known_ref_seq = make_cjt_reference_sequence(
            params["N"],
            CJT_REFERENCE_SEQUENCE_SEED + int(10 * args.bandwidth),
        )
        plan = build_cjt_frame_plan(
            params,
            known_ref_seq,
            num_frames=max(1, args.frames),
            frame_seed=seed,
        )
        if args.synthetic_channel is None:
            indices = np.arange(int(params["num_tx_ant"]), dtype=np.float32)
            channel = (1.0 - 0.1 * indices) * np.exp(0.35j * indices)
        else:
            channel = np.asarray(args.synthetic_channel, dtype=np.complex64)
        result = run_cjt_dry_loopback(params, known_ref_seq, plan, channel)
        print(f"Measured first-slot CSI: {result['measured_csi']}")
        print(f"MRT weights: {result['weights']}")
        print(f"Muted processing samples: {params['cjt_processing_gap_samples']}")
        print(f"Data starts at sample: {params['cjt_data_start_sample']}")
        print(f"Ideal loopback metrics: {result['metrics']}")
        return

    artifacts = run_cjt(
        num_frames=args.frames,
        frame_seed=args.frame_seed,
        bandwidth_mhz=args.bandwidth,
        processing_gap_slots=args.processing_gap_slots,
        deadline_guard_s=args.deadline_guard_us * 1e-6,
        output_dir=args.output_dir,
        max_plot_symbols=args.max_plot_symbols,
    )
    print("Same-frame MRT run complete")
    for key in ("npz_path", "json_path", "plot_path"):
        print(f"  {key}: {artifacts[key]}")
    print(json.dumps(artifacts["summary"], indent=2))


if __name__ == "__main__":
    freeze_support()
    main()
