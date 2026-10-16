"""Continuous UE receive path for same-frame CJT feedback and diagnostics."""

from dataclasses import dataclass
from queue import Full

import numpy as np


@dataclass
class CJTSoundingCapture:
    frame_id: int
    samples: np.ndarray
    first_sample_time: float | None


@dataclass
class CJTFrameCapture:
    frame_id: int
    samples: np.ndarray
    first_sample_time: float | None


class CJTReceiver:
    """Receive slot-zero early while keeping the UHD stream continuous."""

    def __init__(
        self,
        usrp,
        frames_queue,
        num_samples,
        carrier_frequency,
        sampling_rate,
        gain,
        channels,
        start_time,
        otw_format,
        stop_event,
        *,
        recv_timeout=0.2,
        max_consecutive_timeouts=10,
        uhd_module=None,
    ):
        if uhd_module is None:
            import uhd as uhd_module

        if int(num_samples) <= 0:
            raise ValueError("num_samples must be greater than zero")
        if not channels:
            raise ValueError("at least one RX channel must be selected")
        if float(recv_timeout) <= 0 or int(max_consecutive_timeouts) <= 0:
            raise ValueError("receive timeout limits must be positive")

        self.uhd = uhd_module
        self.usrp = usrp
        self.frames_queue = frames_queue
        self.num_samples = int(num_samples)
        self.carrier_frequency = float(carrier_frequency)
        self.sampling_rate = float(sampling_rate)
        self.channels = tuple(channels)
        self.start_time = start_time
        self.otw_format = otw_format
        self.stop_event = stop_event
        self.recv_timeout = float(recv_timeout)
        self.max_consecutive_timeouts = int(max_consecutive_timeouts)

        gains = (
            [gain] * len(self.channels)
            if np.isscalar(gain) and not isinstance(gain, bool)
            else list(gain)
        )
        if len(gains) != len(self.channels):
            raise ValueError("gain must be scalar or one value per RX channel")
        for index, channel in enumerate(self.channels):
            self.usrp.set_rx_rate(self.sampling_rate, channel)
            self.usrp.set_rx_freq(
                self.uhd.libpyuhd.types.tune_request(self.carrier_frequency),
                channel,
            )
            self.usrp.set_rx_gain(gains[index], channel)
            self.usrp.set_rx_dc_offset(False, channel)

        stream_args = self.uhd.usrp.StreamArgs("fc32", self.otw_format)
        stream_args.channels = list(self.channels)
        self.streamer = self.usrp.get_rx_stream(stream_args)
        self.max_samples_per_packet = int(self.streamer.get_max_num_samps())
        if self.max_samples_per_packet <= 0:
            raise RuntimeError("USRP RX streamer reported an invalid packet size")

    @staticmethod
    def _metadata_time(metadata):
        if not getattr(metadata, "has_time_spec", False):
            return None
        return float(metadata.time_spec.get_real_secs())

    def receive_samples(self, metadata, num_samples):
        """Drain exactly ``num_samples`` from UHD without breaking continuity."""
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero")
        packet_buffer = np.empty(
            (len(self.channels), self.max_samples_per_packet),
            dtype=np.complex64,
        )
        output = np.empty((len(self.channels), num_samples), np.complex64)
        received_total = 0
        consecutive_timeouts = 0
        first_sample_time = None
        while received_total < num_samples:
            if self.stop_event.is_set():
                return None
            remaining = num_samples - received_total
            packet_view = packet_buffer[:, :min(remaining, self.max_samples_per_packet)]
            count = self.streamer.recv(packet_view, metadata, self.recv_timeout)
            error_code = metadata.error_code
            if error_code not in (
                self.uhd.types.RXMetadataErrorCode.none,
                self.uhd.types.RXMetadataErrorCode.timeout,
            ):
                raise RuntimeError(f"USRP receive error: {error_code}")
            if count < 0 or count > packet_view.shape[1]:
                raise RuntimeError("USRP RX streamer returned an invalid count")
            if count == 0:
                consecutive_timeouts += 1
                if consecutive_timeouts >= self.max_consecutive_timeouts:
                    raise TimeoutError("USRP receive made no progress")
                continue
            if received_total == 0:
                first_sample_time = self._metadata_time(metadata)
            output[:, received_total:received_total + count] = packet_view[:, :count]
            received_total += count
            consecutive_timeouts = 0
        return output, first_sample_time

    def _start_stream(self):
        command = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.start_cont)
        command.stream_now = self.start_time is None
        if self.start_time is not None:
            command.time_spec = self.uhd.types.TimeSpec(self.start_time)
        self.streamer.issue_stream_cmd(command)

    def _stop_stream(self):
        command = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.stop_cont)
        self.streamer.issue_stream_cmd(command)

    def run_cjt_capture(
        self,
        *,
        sounding_queue,
        capture_queue,
        num_frames,
        sounding_num_samples,
    ):
        num_frames = int(num_frames)
        sounding_num_samples = int(sounding_num_samples)
        if num_frames <= 0:
            raise ValueError("num_frames must be greater than zero")
        if not 0 < sounding_num_samples < self.num_samples:
            raise ValueError("sounding_num_samples must lie inside one frame")

        metadata = self.uhd.types.RXMetadata()
        stream_started = False
        failure = None
        try:
            self._start_stream()
            stream_started = True
            for frame_id in range(num_frames):
                if self.stop_event.is_set():
                    break
                sounding_result = self.receive_samples(metadata, sounding_num_samples)
                if sounding_result is None:
                    break
                sounding_samples, first_sample_time = sounding_result
                try:
                    sounding_queue.put_nowait(
                        CJTSoundingCapture(
                            frame_id=frame_id,
                            samples=sounding_samples,
                            first_sample_time=first_sample_time,
                        )
                    )
                except Full:
                    pass

                remaining_result = self.receive_samples(
                    metadata, self.num_samples - sounding_num_samples
                )
                if remaining_result is None:
                    break
                remaining_samples, _ = remaining_result
                try:
                    capture_queue.put_nowait(
                        CJTFrameCapture(
                            frame_id=frame_id,
                            samples=np.concatenate(
                                (sounding_samples, remaining_samples), axis=1
                            ),
                            first_sample_time=first_sample_time,
                        )
                    )
                except Full:
                    pass
        except KeyboardInterrupt:
            self.stop_event.set()
        except Exception as exc:
            failure = exc
            self.stop_event.set()
        finally:
            if stream_started and self.streamer is not None:
                try:
                    self._stop_stream()
                except Exception as exc:
                    if failure is None:
                        failure = exc
            self.streamer = None
        if failure is not None:
            raise failure
