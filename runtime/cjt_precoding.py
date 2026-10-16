"""MRT precoding primitives used only by the CJT runtime."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from phy.cjt_precoder import CJTMRTPrecoder


def calculate_cjt_mrt_weights(
    channel_coefficients: Sequence[complex] | np.ndarray,
    *,
    num_tx_antennas: int | None = None,
) -> np.ndarray:
    """Return the unit-norm single-user MRT vector ``conj(h) / ||h||``."""
    channel = np.asarray(channel_coefficients, dtype=np.complex64)
    if num_tx_antennas is None:
        if channel.ndim != 1:
            raise ValueError("channel_coefficients must be one-dimensional")
        num_tx_antennas = int(channel.size)
    if channel.shape != (int(num_tx_antennas),):
        raise ValueError(
            "channel_coefficients must contain one value per TX antenna"
        )
    if not np.all(np.isfinite(channel)):
        raise ValueError("channel_coefficients must contain finite values")
    norm = float(np.linalg.norm(channel))
    if norm < 1e-12:
        raise ValueError("channel_coefficients must not be all zero")
    return np.asarray(CJTMRTPrecoder().precode(channel), dtype=np.complex64)


def apply_cjt_mrt_precoding(
    common_waveform: Sequence[complex] | np.ndarray,
    channel_coefficients: Sequence[complex] | np.ndarray,
    *,
    num_tx_antennas: int | None = None,
) -> np.ndarray:
    """Map one common waveform to all RU antennas through the MRT vector."""
    waveform = np.asarray(common_waveform, dtype=np.complex64)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("common_waveform must be a non-empty 1-D array")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("common_waveform must contain finite values")
    weights = calculate_cjt_mrt_weights(
        channel_coefficients,
        num_tx_antennas=num_tx_antennas,
    )
    return np.ascontiguousarray(weights[:, None] * waveform[None, :])
