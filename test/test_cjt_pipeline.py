import types
import unittest

import numpy as np

from runtime.cjt_precoding import apply_cjt_mrt_precoding, calculate_cjt_mrt_weights
from runtime.cjt_runner import (
    CJTTimedTransmitter,
    assemble_cjt_frame,
    run_cjt_dry_loopback,
)
from USRP.cjt_frame import build_cjt_frame_plan
from USRP.cjt_parameters import (
    make_cjt_reference_sequence,
    build_cjt_params,
)
from USRP.cjt_ofdm import ofdm_demodulate_slots
from USRP.cjt_signal_processing import (
    estimate_and_correct_cjt_rfo,
    process_cjt_sounding,
)


class CJTPipelineDSPTests(unittest.TestCase):
    def setUp(self):
        self.params = build_cjt_params(1.4)
        self.reference = make_cjt_reference_sequence(
            self.params["N"],
            2040,
        )
        self.plan = build_cjt_frame_plan(
            self.params,
            self.reference,
            num_frames=2,
            frame_seed=77,
        )
        tx_indices = np.arange(
            int(self.params["num_tx_ant"]),
            dtype=np.float32,
        )
        self.channel = (
            (1.0 - 0.1 * tx_indices) * np.exp(0.27j * tx_indices)
        ).astype(np.complex64)

    def test_first_slot_layout_and_data_start(self):
        grid = self.plan.sounding_resource_grid[:, 0]
        sync_tx = int(self.params["sync_tx_idx"])
        for symbol_idx in (0, 1):
            np.testing.assert_allclose(grid[sync_tx, symbol_idx], self.reference)
            other_tx = np.arange(int(self.params["num_tx_ant"])) != sync_tx
            np.testing.assert_array_equal(grid[other_tx, symbol_idx], 0)

        for tx_idx, active_idx in enumerate(
            self.params["fdm_pilot_active_indices"]
        ):
            for symbol_idx in (2, 3, 4):
                self.assertEqual(np.count_nonzero(grid[tx_idx, symbol_idx]), 1)
                np.testing.assert_allclose(
                    grid[tx_idx, symbol_idx, active_idx],
                    self.reference[active_idx],
                )
        self.assertEqual(self.params["cjt_sounding_num_samples"], 960)
        self.assertEqual(self.params["cjt_processing_gap_samples"], 960)
        self.assertEqual(self.params["cjt_data_start_sample"], 1920)
        self.assertEqual(self.params["cjt_data_num_samples"], 7680)

    def test_first_slot_estimates_every_tx_without_a_full_frame(self):
        received = np.sum(
            self.channel[:, None] * self.plan.sounding_waveform,
            axis=0,
            keepdims=True,
        )
        estimate = process_cjt_sounding(
            self.params,
            self.reference,
            received,
        )
        expected = np.sqrt(self.params["waveform_power"]) * self.channel
        np.testing.assert_allclose(
            estimate["csi"][:, 0],
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(estimate["resource_grid"].shape[1], 1)

    def test_first_slot_sync_uses_only_the_processing_gap_margin(self):
        received = np.sum(
            self.channel[:, None] * self.plan.sounding_waveform,
            axis=0,
            keepdims=True,
        )
        timing_offset = 23
        margin = int(self.params["cjt_sync_margin_samples"])
        capture = np.concatenate(
            (
                np.zeros((1, timing_offset), dtype=np.complex64),
                received,
                np.zeros((1, margin - timing_offset), dtype=np.complex64),
            ),
            axis=1,
        )
        estimate = process_cjt_sounding(
            self.params,
            self.reference,
            capture,
            sync_template=self.plan.sounding_sync_template,
        )
        self.assertEqual(estimate["sync_idx"], timing_offset)
        expected = np.sqrt(self.params["waveform_power"]) * self.channel
        np.testing.assert_allclose(
            estimate["csi"][:, 0],
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_reference_and_pdsch_receive_the_same_mrt_weights(self):
        weights = calculate_cjt_mrt_weights(self.channel)
        precoded = apply_cjt_mrt_precoding(
            self.plan.common_data_waveforms[0],
            self.channel,
        )
        recovered = ofdm_demodulate_slots(
            precoded / np.sqrt(self.params["waveform_power"]),
            self.params,
        )
        expected = (
            weights[:, None, None, None]
            * self.plan.common_data_resource_grids[0][None, ...]
        )
        np.testing.assert_allclose(recovered, expected, rtol=1e-5, atol=1e-5)

    def test_complete_synthetic_link_has_a_muted_gap_and_zero_ser(self):
        result = run_cjt_dry_loopback(
            self.params,
            self.reference,
            self.plan,
            self.channel,
        )
        frame = result["transmitted_frame"]
        gap = frame[
            :,
            self.params["cjt_sounding_end_sample"]:
            self.params["cjt_data_start_sample"],
        ]
        np.testing.assert_array_equal(gap, 0)
        self.assertEqual(result["metrics"]["symbol_error_rate"], 0.0)
        self.assertLess(result["metrics"]["evm_rms_percent"], 1e-3)

    def test_rfo_does_not_compare_tx0_sounding_to_mrt_reference(self):
        frame = assemble_cjt_frame(
            self.params,
            self.plan,
            frame_id=0,
            channel_coefficients=self.channel,
        )
        received = np.sum(
            self.channel[:, None] * frame,
            axis=0,
            keepdims=True,
        )
        expected_rfo_hz = 123.0
        samples = np.arange(self.params["frame_length"], dtype=np.float64)
        received *= np.exp(
            1j
            * 2.0
            * np.pi
            * expected_rfo_hz
            * samples
            / self.params["sampling_rate"]
        )[None, :]
        corrected, estimated, pair_estimates = (
            estimate_and_correct_cjt_rfo(received, self.params)
        )
        np.testing.assert_allclose(estimated, expected_rfo_hz, atol=1e-3)
        np.testing.assert_allclose(pair_estimates, expected_rfo_hz, atol=1e-3)
        ideal = np.sum(
            self.channel[:, None] * frame,
            axis=0,
            keepdims=True,
        )
        np.testing.assert_allclose(corrected, ideal, rtol=2e-5, atol=2e-5)

    def test_assemble_rejects_no_layout_and_places_regions_exactly(self):
        frame = assemble_cjt_frame(
            self.params,
            self.plan,
            frame_id=0,
            channel_coefficients=self.channel,
        )
        np.testing.assert_array_equal(
            frame[:, : self.params["cjt_sounding_end_sample"]],
            self.plan.sounding_waveform,
        )
        self.assertEqual(frame.shape[-1], self.params["frame_length"])


class FakeTXMetadata:
    def __init__(self):
        self.has_time_spec = False
        self.time_spec = None
        self.start_of_burst = False
        self.end_of_burst = False


class FakeTimeSpec:
    def __init__(self, seconds):
        self.seconds = seconds


class FakeStreamArgs:
    def __init__(self, cpu_format, otw_format):
        self.cpu_format = cpu_format
        self.otw_format = otw_format
        self.channels = None


class FakeStreamer:
    def __init__(self):
        self.calls = []

    def send(self, samples, metadata):
        self.calls.append(
            {
                "samples": samples.copy(),
                "has_time_spec": metadata.has_time_spec,
                "time_spec": metadata.time_spec,
                "start_of_burst": metadata.start_of_burst,
                "end_of_burst": metadata.end_of_burst,
            }
        )
        return min(3, samples.shape[1])


class FakeUSRP:
    def __init__(self):
        self.streamer = FakeStreamer()

    def set_tx_rate(self, rate, channel):
        pass

    def set_tx_freq(self, frequency, channel):
        pass

    def set_tx_gain(self, gain, channel):
        pass

    def get_tx_stream(self, stream_args):
        return self.streamer


def fake_uhd():
    return types.SimpleNamespace(
        libpyuhd=types.SimpleNamespace(
            types=types.SimpleNamespace(tune_request=lambda value: value)
        ),
        usrp=types.SimpleNamespace(StreamArgs=FakeStreamArgs),
        types=types.SimpleNamespace(
            TXMetadata=FakeTXMetadata,
            TimeSpec=FakeTimeSpec,
        ),
    )


class CJTTimedTransmitterTests(unittest.TestCase):
    def test_partial_sends_keep_time_only_on_first_packet_and_close_burst(self):
        usrp = FakeUSRP()
        sender = CJTTimedTransmitter(
            usrp=usrp,
            carrier_frequency=2.2e9,
            sampling_rate=1.92e6,
            gain=30,
            channels=(0, 1),
            otw_format="sc16",
            uhd_module=fake_uhd(),
        )
        samples = np.ones((2, 8), dtype=np.complex64)
        sender.send_timed_burst(samples, 4.25)

        self.assertEqual(len(usrp.streamer.calls), 4)
        first, second, third, end = usrp.streamer.calls
        self.assertTrue(first["has_time_spec"])
        self.assertTrue(first["start_of_burst"])
        self.assertEqual(first["time_spec"].seconds, 4.25)
        for call in (second, third):
            self.assertFalse(call["has_time_spec"])
            self.assertFalse(call["start_of_burst"])
        self.assertTrue(end["end_of_burst"])
        self.assertEqual(end["samples"].shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
