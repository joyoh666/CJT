"""Build the split sounding/gap/data waveform used by same-frame MRT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from USRP.cjt_parameters import make_cjt_zc_sequence, make_cjt_sss_sequence
from USRP.cjt_modulation import modulate
from USRP.cjt_ofdm import ofdm_modulate_slots


@dataclass(frozen=True)
class CJTFramePlan:
    """Static sounding and precomputed common data for a finite run."""

    sounding_resource_grid: np.ndarray
    sounding_waveform: np.ndarray
    sounding_sync_template: np.ndarray
    common_data_resource_grids: np.ndarray
    common_data_waveforms: np.ndarray
    pdsch_symbols: np.ndarray
    pdsch_indices: tuple[np.ndarray, np.ndarray, np.ndarray]


def _build_sounding_resource_grid(params, known_ref_seq):
    num_tx = int(params["num_tx_ant"])
    num_symbols = int(params["num_symbols_per_slot"])
    num_active = int(params["N"])
    sync_tx = int(params["sync_tx_idx"])
    reference = np.asarray(known_ref_seq, dtype=np.complex64)
    if reference.shape != (num_active,):
        raise ValueError("known_ref_seq has the wrong number of subcarriers")

    grid = np.zeros(
        (num_tx, 1, num_symbols, num_active),
        dtype=np.complex64,
    )
    for symbol_idx in params["cjt_sounding_reference_symbols"]:
        grid[sync_tx, 0, int(symbol_idx)] = reference

    active_indices = tuple(int(v) for v in params["fdm_pilot_active_indices"])
    for tx_idx, active_idx in enumerate(active_indices):
        for symbol_idx in params["cjt_pilot_symbols"]:
            grid[tx_idx, 0, int(symbol_idx), active_idx] = reference[active_idx]

    pss_length = int(params["pss_length"])
    sync_start = num_active - (pss_length + 1) - (
        (num_active - (pss_length + 1)) // 2
    )
    grid[
        sync_tx,
        0,
        5,
        sync_start:sync_start + pss_length,
    ] = make_cjt_sss_sequence(pss_length, 0)
    grid[
        sync_tx,
        0,
        6,
        sync_start:sync_start + pss_length,
    ] = make_cjt_zc_sequence(pss_length, 25)
    return grid


def _build_common_data_grids(params, known_ref_seq, num_frames, frame_seed):
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than zero")
    if frame_seed < 0:
        raise ValueError("frame_seed must be non-negative")

    num_slots = int(params["cjt_data_num_slots"])
    num_symbols = int(params["num_symbols_per_slot"])
    num_active = int(params["N"])
    reference = np.asarray(known_ref_seq, dtype=np.complex64)
    grids = np.empty(
        (num_frames, num_slots, num_symbols, num_active),
        dtype=np.complex64,
    )
    grids[:, :, 0, :] = reference[None, None, :]

    # Every non-reference RE in the data region carries the same user stream
    # before MRT maps it to the physical antennas.
    data_mask = np.ones((num_slots, num_symbols, num_active), dtype=bool)
    data_mask[:, 0, :] = False
    pdsch_indices = tuple(
        np.asarray(axis, dtype=np.int64) for axis in np.where(data_mask)
    )
    num_pdsch = int(pdsch_indices[0].size)
    bits_per_symbol = int(np.log2(int(params["modulation_order"])))
    rng = np.random.default_rng(frame_seed)
    for frame_idx in range(num_frames):
        bits = rng.integers(
            0,
            2,
            num_pdsch * bits_per_symbol,
            dtype=np.uint8,
        )
        grids[frame_idx][pdsch_indices] = modulate(
            bits,
            int(params["modulation_order"]),
        )
    pdsch_symbols = grids[(slice(None),) + pdsch_indices]
    return grids, pdsch_symbols, pdsch_indices


def build_cjt_frame_plan(
    params,
    known_ref_seq,
    *,
    num_frames,
    frame_seed,
):
    """Create sounding and common data waveforms without MRT coefficients."""
    sounding_grid = _build_sounding_resource_grid(params, known_ref_seq)
    sounding_waveform = ofdm_modulate_slots(params, sounding_grid)
    data_grids, pdsch_symbols, pdsch_indices = _build_common_data_grids(
        params,
        known_ref_seq,
        int(num_frames),
        int(frame_seed),
    )
    common_data_waveforms = ofdm_modulate_slots(params, data_grids)
    amplitude = np.sqrt(float(params["waveform_power"]))
    sounding_waveform = np.ascontiguousarray(
        sounding_waveform * amplitude,
        dtype=np.complex64,
    )
    sync_symbol_start = (
        int(params["first_CP_length"])
        + int(params["FFT_SIZE"])
        + 4
        * (int(params["normal_CP_length"]) + int(params["FFT_SIZE"]))
    )
    sync_tx = int(params["sync_tx_idx"])
    return CJTFramePlan(
        sounding_resource_grid=sounding_grid,
        sounding_waveform=sounding_waveform,
        sounding_sync_template=np.ascontiguousarray(
            sounding_waveform[sync_tx, sync_symbol_start:],
            dtype=np.complex64,
        ),
        common_data_resource_grids=data_grids,
        common_data_waveforms=np.ascontiguousarray(
            common_data_waveforms * amplitude,
            dtype=np.complex64,
        ),
        pdsch_symbols=np.asarray(pdsch_symbols, dtype=np.complex64),
        pdsch_indices=pdsch_indices,
    )
