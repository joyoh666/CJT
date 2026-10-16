"""Slot-oriented OFDM modulation and demodulation for CJT."""

import numpy as np


def ofdm_modulate_slots(params, resource_grids):
    """Modulate ``(stream, slot, symbol, subcarrier)`` resource grids."""
    grids = np.asarray(resource_grids, dtype=np.complex64)
    expected_tail = (int(params["num_symbols_per_slot"]), int(params["N"]))
    if grids.ndim != 4 or grids.shape[-2:] != expected_tail:
        raise ValueError(
            "resource_grids must have shape "
            "(stream, slot, num_symbols_per_slot, N)"
        )
    if grids.shape[0] == 0 or grids.shape[1] == 0:
        raise ValueError("resource_grids must contain a stream and a slot")

    fft_size = int(params["FFT_SIZE"])
    num_active = int(params["N"])
    first_cp = int(params["first_CP_length"])
    normal_cp = int(params["normal_CP_length"])
    fft_bins = np.zeros(grids.shape[:-1] + (fft_size,), dtype=np.complex64)
    fft_bins[..., 1:num_active // 2 + 1] = grids[..., num_active // 2:]
    fft_bins[..., -(num_active // 2):] = grids[..., :num_active // 2]
    time_symbols = np.fft.ifft(fft_bins, axis=-1, norm="ortho").astype(
        np.complex64
    )
    first_symbol = np.concatenate(
        (time_symbols[..., 0:1, -first_cp:], time_symbols[..., 0:1, :]),
        axis=-1,
    )
    remaining_symbols = np.concatenate(
        (time_symbols[..., 1:, -normal_cp:], time_symbols[..., 1:, :]),
        axis=-1,
    )
    slots = np.concatenate(
        (
            first_symbol.reshape(grids.shape[0], grids.shape[1], -1),
            remaining_symbols.reshape(grids.shape[0], grids.shape[1], -1),
        ),
        axis=-1,
    )
    return np.ascontiguousarray(slots.reshape(grids.shape[0], -1))


def ofdm_demodulate_slots(slot_samples, params):
    """Demodulate an integer number of aligned OFDM slots."""
    samples = np.asarray(slot_samples, dtype=np.complex64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2:
        raise ValueError("slot_samples must have shape (receiver, sample)")

    slot_length = int(params["slot_length"])
    if samples.shape[1] == 0 or samples.shape[1] % slot_length:
        raise ValueError("slot_samples must contain complete slots")
    num_slots = samples.shape[1] // slot_length
    num_symbols = int(params["num_symbols_per_slot"])
    fft_size = int(params["FFT_SIZE"])
    first_cp = int(params["first_CP_length"])
    normal_cp = int(params["normal_CP_length"])
    num_active = int(params["N"])

    slots = samples.reshape(samples.shape[0], num_slots, slot_length)
    normalized_slots = slots[..., first_cp - normal_cp:]
    symbols_with_cp = normalized_slots.reshape(
        samples.shape[0],
        num_slots,
        num_symbols,
        normal_cp + fft_size,
    )
    symbols = symbols_with_cp[..., normal_cp:]
    fft_bins = np.fft.fft(symbols, axis=-1, norm="ortho")
    grids = np.zeros(
        (samples.shape[0], num_slots, num_symbols, num_active),
        dtype=np.complex64,
    )
    grids[..., num_active // 2:] = fft_bins[..., 1:num_active // 2 + 1]
    grids[..., :num_active // 2] = fft_bins[..., -(num_active // 2):]
    return grids
