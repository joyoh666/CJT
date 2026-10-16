"""Synchronization and offset correction for the CJT frame layout."""

import numpy as np
from scipy.signal import correlate

from USRP.cjt_channel_estimation import estimate_cjt_sounding_channel
from USRP.cjt_ofdm import ofdm_demodulate_slots


def estimate_and_correct_ffo_slots(slot_samples, params):
    """Estimate CP-based fractional frequency offset from complete slots."""
    samples = np.asarray(slot_samples, dtype=np.complex64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2:
        raise ValueError("slot_samples must have shape (RX, sample)")

    fft_size = int(params["FFT_SIZE"])
    first_cp = int(params["first_CP_length"])
    normal_cp = int(params["normal_CP_length"])
    num_symbols = int(params["num_symbols_per_slot"])
    slot_length = int(params["slot_length"])
    if samples.shape[1] == 0 or samples.shape[1] % slot_length:
        raise ValueError("slot_samples must contain complete slots")
    num_slots = samples.shape[1] // slot_length
    slots = samples.reshape(samples.shape[0], num_slots, slot_length)
    first_cps = slots[:, :, :first_cp]
    first_tails = slots[:, :, fft_size:fft_size + first_cp]
    remaining = slots[:, :, fft_size + first_cp:].reshape(
        samples.shape[0],
        num_slots * (num_symbols - 1),
        fft_size + normal_cp,
    )
    other_cps = remaining[:, :, :normal_cp]
    other_tails = remaining[:, :, -normal_cp:]
    correlation = np.sum(np.conj(first_cps) * first_tails, axis=(1, 2))
    correlation += np.sum(np.conj(other_cps) * other_tails, axis=(1, 2))
    sampling_rate = float(params["sampling_rate"])
    ffo_hz = (
        np.angle(correlation) * sampling_rate / (2.0 * np.pi * fft_size)
    ).astype(np.float32)
    sample_indices = np.arange(samples.shape[1], dtype=np.float64)[None, :]
    correction = np.exp(
        -1j * 2.0 * np.pi * ffo_hz[:, None] * sample_indices / sampling_rate
    ).astype(np.complex64)
    return samples * correction, ffo_hz


def estimate_and_correct_cjt_sounding_rfo(slot_samples, params):
    """Estimate residual CFO from sounding reference symbols 0 and 1."""
    samples = np.asarray(slot_samples, dtype=np.complex64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    slot_length = int(params["slot_length"])
    if samples.ndim != 2 or samples.shape[1] < slot_length:
        raise ValueError("slot_samples must contain the complete sounding slot")
    fft_size = int(params["FFT_SIZE"])
    first_cp = int(params["first_CP_length"])
    normal_cp = int(params["normal_CP_length"])
    first_symbol = samples[:, first_cp:first_cp + fft_size]
    second_start = first_cp + fft_size + normal_cp
    second_symbol = samples[:, second_start:second_start + fft_size]
    separation = fft_size + normal_cp
    correlation = np.sum(np.conj(first_symbol) * second_symbol, axis=-1)
    sampling_rate = float(params["sampling_rate"])
    rfo_hz = (
        np.angle(correlation) * sampling_rate / (2.0 * np.pi * separation)
    ).astype(np.float32)
    sample_indices = np.arange(samples.shape[1], dtype=np.float64)[None, :]
    correction = np.exp(
        -1j * 2.0 * np.pi * rfo_hz[:, None] * sample_indices / sampling_rate
    ).astype(np.complex64)
    return samples * correction, rfo_hz


def estimate_and_correct_cjt_rfo(frame_samples, params):
    """Correct RFO without correlating across the sounding/MRT boundary."""
    samples = np.asarray(frame_samples, dtype=np.complex64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    expected_length = int(params["frame_length"])
    if samples.ndim != 2 or samples.shape[1] != expected_length:
        raise ValueError(
            f"frame_samples must have shape (RX, {expected_length})"
        )

    fft_size = int(params["FFT_SIZE"])
    first_cp = int(params["first_CP_length"])
    normal_cp = int(params["normal_CP_length"])
    slot_length = int(params["slot_length"])
    sampling_rate = float(params["sampling_rate"])

    first_symbol = samples[:, first_cp:first_cp + fft_size]
    second_start = first_cp + fft_size + normal_cp
    second_symbol = samples[:, second_start:second_start + fft_size]
    correlations = [np.sum(np.conj(first_symbol) * second_symbol, axis=-1)]
    separations = [fft_size + normal_cp]

    data_start_slot = int(params["cjt_data_start_global_slot"])
    data_references = []
    for local_slot in range(int(params["cjt_data_num_slots"])):
        start = (data_start_slot + local_slot) * slot_length + first_cp
        data_references.append(samples[:, start:start + fft_size])
    for previous, current in zip(data_references[:-1], data_references[1:]):
        correlations.append(np.sum(np.conj(previous) * current, axis=-1))
        separations.append(slot_length)

    correlation_array = np.stack(correlations, axis=1)
    separation_array = np.asarray(separations, dtype=np.float64)[None, :]
    pair_hz = (
        np.angle(correlation_array)
        * sampling_rate
        / (2.0 * np.pi * separation_array)
    )
    weights = np.abs(correlation_array)
    weight_sum = np.sum(weights, axis=1)
    rfo_hz = np.divide(
        np.sum(weights * pair_hz, axis=1),
        weight_sum,
        out=np.zeros(samples.shape[0], dtype=np.float64),
        where=weight_sum > 0,
    ).astype(np.float32)
    sample_indices = np.arange(expected_length, dtype=np.float64)[None, :]
    correction = np.exp(
        -1j * 2.0 * np.pi * rfo_hz[:, None] * sample_indices / sampling_rate
    ).astype(np.complex64)
    return samples * correction, rfo_hz, pair_hz.astype(np.float32)


def synchronize_cjt_sounding(slot_capture, sync_template, params):
    """Locate sounding slot zero while searching only the configured margin."""
    capture = np.asarray(slot_capture, dtype=np.complex64)
    if capture.ndim == 1:
        capture = capture.reshape(1, -1)
    if capture.ndim != 2:
        raise ValueError("slot_capture must have shape (RX, sample)")
    template = np.asarray(sync_template, dtype=np.complex64).reshape(-1)
    if template.size == 0:
        raise ValueError("sync_template must not be empty")
    slot_length = int(params["cjt_sounding_num_samples"])
    if capture.shape[1] < slot_length:
        raise ValueError("slot_capture is shorter than the sounding slot")
    max_start = min(
        capture.shape[1] - slot_length,
        int(params["cjt_sync_margin_samples"]),
    )
    sync_symbol_start = (
        int(params["first_CP_length"])
        + int(params["FFT_SIZE"])
        + 4 * (int(params["normal_CP_length"]) + int(params["FFT_SIZE"]))
    )
    correlations = np.stack(
        [correlate(row, template, mode="valid", method="fft") for row in capture]
    )
    first_candidate = sync_symbol_start
    last_candidate = sync_symbol_start + max_start + 1
    if last_candidate > correlations.shape[1]:
        raise ValueError("capture does not contain the complete sync template")
    combined_power = np.sum(
        np.abs(correlations[:, first_candidate:last_candidate]) ** 2,
        axis=0,
    )
    sync_peak = first_candidate + int(np.argmax(combined_power))
    sync_index = sync_peak - sync_symbol_start
    peak_values = np.abs(correlations[:, sync_peak]).astype(np.float32)
    aligned = capture[:, sync_index:sync_index + slot_length]
    return np.ascontiguousarray(aligned), sync_index, peak_values


def process_cjt_sounding(
    params,
    known_reference,
    slot_samples,
    *,
    sync_template=None,
):
    """Run the latency-sensitive sync, FFO, RFO, and CSI pipeline."""
    samples = np.asarray(slot_samples, dtype=np.complex64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2 or samples.shape[0] != int(params["num_rx_ant"]):
        raise ValueError("slot_samples must have shape (configured_RX, sample)")

    if sync_template is None:
        expected_length = int(params["cjt_sounding_num_samples"])
        if samples.shape[1] != expected_length:
            raise ValueError(
                f"slot_samples has {samples.shape[1]} samples, "
                f"expected {expected_length} without a sync template"
            )
        sync_index = 0
        peak_values = np.full(samples.shape[0], np.nan, dtype=np.float32)
    else:
        samples, sync_index, peak_values = synchronize_cjt_sounding(
            samples, sync_template, params
        )
    ffo_corrected, ffo_hz = estimate_and_correct_ffo_slots(samples, params)
    rfo_corrected, rfo_hz = estimate_and_correct_cjt_sounding_rfo(
        ffo_corrected, params
    )
    grid = ofdm_demodulate_slots(rfo_corrected, params)
    estimate = estimate_cjt_sounding_channel(params, grid, known_reference)
    estimate.update(
        {
            "resource_grid": grid,
            "ffo_hz": ffo_hz,
            "rfo_hz": rfo_hz,
            "sync_idx": sync_index,
            "timing_correlation_peak": peak_values,
        }
    )
    return estimate
