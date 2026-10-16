"""CJT-only waveform parameters and deterministic reference sequences."""

import numpy as np

from config.cjt_config import (
    CJT_BANDWIDTH_OPTIONS,
    CJT_CARRIER_FREQUENCY_HZ,
    CJT_FDM_PILOT_CENTERED_BINS,
    CJT_FIRST_CP_TIME_S,
    CJT_MODULATION_ORDER,
    CJT_NORMAL_CP_TIME_S,
    CJT_NUM_RX_ANTENNAS,
    CJT_NUM_SLOTS_PER_SUBFRAME,
    CJT_NUM_SUBFRAMES_PER_FRAME,
    CJT_NUM_SYMBOLS_PER_FRAME,
    CJT_NUM_SYMBOLS_PER_SLOT,
    CJT_NUM_TX_ANTENNAS,
    CJT_NUM_VIRTUAL_PILOTS,
    CJT_PHASE_ALIGN_VIRTUAL_PILOTS,
    CJT_PILOT_SYMBOL_START,
    CJT_PROCESSING_GAP_SLOTS,
    CJT_PSS_LENGTH,
    CJT_RX_GAIN_DB,
    CJT_SOUNDING_REFERENCE_SYMBOLS,
    CJT_SOUNDING_SLOT_INDEX,
    CJT_SUBCARRIER_SPACING_HZ,
    CJT_SYNC_MARGIN_SAMPLES,
    CJT_SYNC_TX_INDEX,
    CJT_TX_GAIN_DB,
    CJT_WAVEFORM_POWER,
)


def make_cjt_zc_sequence(length, root=25):
    sample = np.arange(int(length))
    phase = -np.pi * int(root) * sample * (sample + 1) / int(length)
    return np.exp(1j * phase).astype(np.complex64)


def make_cjt_sss_sequence(length, seed):
    rng = np.random.default_rng(int(seed))
    bits = rng.integers(0, 2, size=int(length))
    return (2 * bits - 1 + 0.0j).astype(np.complex64)


def make_cjt_reference_sequence(num_subcarriers, seed=2026):
    rng = np.random.default_rng(int(seed))
    constellation = np.array(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64
    ) / np.sqrt(2)
    sequence = rng.choice(constellation, size=int(num_subcarriers))
    sequence /= np.sqrt(np.mean(np.abs(sequence) ** 2))
    return sequence.astype(np.complex64)


def centered_bin_to_active_index(centered_bin, num_subcarriers):
    """Map a centered, non-DC FFT bin to the active-subcarrier axis."""
    centered_bin = int(centered_bin)
    half = int(num_subcarriers) // 2
    if centered_bin == 0 or abs(centered_bin) > half:
        raise ValueError("centered pilot bin is outside the active allocation")
    return centered_bin + half if centered_bin < 0 else half + centered_bin - 1


def get_cjt_pilot_active_indices(num_subcarriers):
    if len(CJT_FDM_PILOT_CENTERED_BINS) != CJT_NUM_TX_ANTENNAS:
        raise ValueError("CJT requires one FDM pilot bin per TX antenna")
    if len(set(CJT_FDM_PILOT_CENTERED_BINS)) != CJT_NUM_TX_ANTENNAS:
        raise ValueError("CJT FDM pilot bins must be distinct")
    return tuple(
        centered_bin_to_active_index(value, num_subcarriers)
        for value in CJT_FDM_PILOT_CENTERED_BINS
    )


def phase_align_channel_estimates(channel_repetitions):
    """Align repeated channel estimates to the phase of repetition zero."""
    estimates = np.asarray(channel_repetitions)
    if estimates.ndim < 2 or estimates.shape[0] == 0:
        raise ValueError("channel_repetitions needs a non-empty repeat axis")
    reference = estimates[0]
    correlation = np.sum(
        np.conj(reference)[None, ...] * estimates,
        axis=-1,
    )
    common_phase = np.angle(correlation)
    aligned = estimates * np.exp(-1j * common_phase[..., None])
    return aligned.astype(np.complex64), common_phase.astype(np.float32)


def _symbol_useful_center_sample(
    slot_index,
    symbol_index,
    *,
    slot_length,
    fft_size,
    first_cp_length,
    normal_cp_length,
):
    slot_start = int(slot_index) * int(slot_length)
    if int(symbol_index) == 0:
        return slot_start + first_cp_length + fft_size / 2
    return (
        slot_start
        + first_cp_length
        + fft_size
        + (int(symbol_index) - 1) * (normal_cp_length + fft_size)
        + normal_cp_length
        + fft_size / 2
    )


def build_cjt_params(
    bandwidth_mhz,
    *,
    processing_gap_slots=CJT_PROCESSING_GAP_SLOTS,
):
    """Build the isolated CJT sounding/gap/data frame configuration."""
    if bandwidth_mhz not in CJT_BANDWIDTH_OPTIONS:
        raise ValueError(f"unsupported CJT bandwidth: {bandwidth_mhz}")
    processing_gap_slots = int(processing_gap_slots)
    if processing_gap_slots < 1:
        raise ValueError("processing_gap_slots must be at least one")

    num_subcarriers, fft_size = CJT_BANDWIDTH_OPTIONS[bandwidth_mhz]
    sampling_rate = fft_size * CJT_SUBCARRIER_SPACING_HZ
    normal_cp_length = round(CJT_NORMAL_CP_TIME_S * sampling_rate)
    first_cp_length = round(CJT_FIRST_CP_TIME_S * sampling_rate)
    slot_length = (
        fft_size * CJT_NUM_SYMBOLS_PER_SLOT
        + normal_cp_length * (CJT_NUM_SYMBOLS_PER_SLOT - 1)
        + first_cp_length
    )
    total_slots = CJT_NUM_SLOTS_PER_SUBFRAME * CJT_NUM_SUBFRAMES_PER_FRAME
    frame_length = total_slots * slot_length

    sounding_slot = int(CJT_SOUNDING_SLOT_INDEX)
    if sounding_slot != 0:
        raise ValueError("CJT sounding must occupy slot zero")
    reference_symbols = tuple(int(v) for v in CJT_SOUNDING_REFERENCE_SYMBOLS)
    pilot_symbols = tuple(
        range(
            int(CJT_PILOT_SYMBOL_START),
            int(CJT_PILOT_SYMBOL_START) + int(CJT_NUM_VIRTUAL_PILOTS),
        )
    )
    occupied_symbols = set(reference_symbols) | set(pilot_symbols) | {5, 6}
    if len(occupied_symbols) != len(reference_symbols) + len(pilot_symbols) + 2:
        raise ValueError("CJT sounding resources overlap")
    if min(occupied_symbols) < 0 or max(occupied_symbols) >= CJT_NUM_SYMBOLS_PER_SLOT:
        raise ValueError("CJT sounding resources do not fit inside one slot")

    data_start_slot = sounding_slot + 1 + processing_gap_slots
    if data_start_slot >= total_slots:
        raise ValueError("processing gap leaves no CJT data slots")
    sync_margin = int(CJT_SYNC_MARGIN_SAMPLES)
    if sync_margin < 0 or sync_margin >= processing_gap_slots * slot_length:
        raise ValueError("CJT sync margin must fit inside the processing gap")

    pilot_indices = get_cjt_pilot_active_indices(num_subcarriers)
    pilot_offsets = np.empty(
        (CJT_NUM_TX_ANTENNAS, CJT_NUM_VIRTUAL_PILOTS), dtype=np.float64
    )
    for tx_index in range(CJT_NUM_TX_ANTENNAS):
        for repetition, symbol_index in enumerate(pilot_symbols):
            pilot_offsets[tx_index, repetition] = _symbol_useful_center_sample(
                sounding_slot,
                symbol_index,
                slot_length=slot_length,
                fft_size=fft_size,
                first_cp_length=first_cp_length,
                normal_cp_length=normal_cp_length,
            )

    return {
        "bandwidth_mhz": float(bandwidth_mhz),
        "N": int(num_subcarriers),
        "FFT_SIZE": int(fft_size),
        "delta_f": float(CJT_SUBCARRIER_SPACING_HZ),
        "sampling_rate": float(sampling_rate),
        "T": 1.0 / float(CJT_SUBCARRIER_SPACING_HZ),
        "normal_CP_length": int(normal_cp_length),
        "first_CP_length": int(first_cp_length),
        "slot_length": int(slot_length),
        "frame_length": int(frame_length),
        "carrier_frequency": float(CJT_CARRIER_FREQUENCY_HZ),
        "tx_gain_db": float(CJT_TX_GAIN_DB),
        "rx_gain_db": float(CJT_RX_GAIN_DB),
        "modulation_order": int(CJT_MODULATION_ORDER),
        "waveform_power": float(CJT_WAVEFORM_POWER),
        "num_symbols_per_slot": int(CJT_NUM_SYMBOLS_PER_SLOT),
        "num_slot_per_subframe": int(CJT_NUM_SLOTS_PER_SUBFRAME),
        "num_subframe_per_frame": int(CJT_NUM_SUBFRAMES_PER_FRAME),
        "num_symbols_frame": int(CJT_NUM_SYMBOLS_PER_FRAME),
        "num_tx_ant": int(CJT_NUM_TX_ANTENNAS),
        "num_rx_ant": int(CJT_NUM_RX_ANTENNAS),
        "pss_length": int(CJT_PSS_LENGTH),
        "num_virtual_pilots": int(CJT_NUM_VIRTUAL_PILOTS),
        "fdm_pilot_centered_bins": tuple(CJT_FDM_PILOT_CENTERED_BINS),
        "fdm_pilot_active_indices": pilot_indices,
        "phase_align_virtual_pilots": bool(CJT_PHASE_ALIGN_VIRTUAL_PILOTS),
        "sync_tx_idx": int(CJT_SYNC_TX_INDEX),
        "cjt_sounding_global_slot": sounding_slot,
        "cjt_sounding_reference_symbols": reference_symbols,
        "cjt_pilot_symbols": pilot_symbols,
        "cjt_virtual_pilot_positions": tuple(
            (tx_index, symbol_index)
            for tx_index in range(CJT_NUM_TX_ANTENNAS)
            for symbol_index in pilot_symbols
        ),
        "cjt_processing_gap_slots": processing_gap_slots,
        "cjt_processing_gap_samples": processing_gap_slots * slot_length,
        "cjt_data_start_global_slot": data_start_slot,
        "cjt_data_start_sample": data_start_slot * slot_length,
        "cjt_data_num_slots": total_slots - data_start_slot,
        "cjt_data_num_samples": (total_slots - data_start_slot) * slot_length,
        "cjt_sounding_num_samples": slot_length,
        "cjt_sync_margin_samples": sync_margin,
        "cjt_sounding_capture_num_samples": slot_length + sync_margin,
        "cjt_sounding_end_sample": slot_length,
        "cjt_pilot_timestamp_offsets_samples_by_tx": pilot_offsets.tolist(),
        "cjt_csi_timestamp_offset_samples": float(np.mean(pilot_offsets)),
    }
