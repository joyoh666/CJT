"""Constellation decisions and plots for CJT link verification."""

from pathlib import Path

import numpy as np

from USRP.cjt_modulation import modulate


def _constellation_points(modulation_order: int) -> np.ndarray:
    bits_per_symbol = int(np.log2(modulation_order))
    indices = np.arange(modulation_order, dtype=np.uint64)
    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.uint64)
    bits = ((indices[:, None] >> shifts) & 1).astype(np.uint8)
    return np.asarray(modulate(bits.reshape(-1), modulation_order))


def make_cjt_hard_decisions(symbols, modulation_order):
    values = np.asarray(symbols, dtype=np.complex64)
    points = _constellation_points(int(modulation_order))
    flat = values.reshape(-1)
    decisions = np.full(flat.shape, np.nan + 1j * np.nan, np.complex64)
    valid = np.isfinite(flat.real) & np.isfinite(flat.imag)
    distances = np.abs(flat[valid, None] - points[None, :])
    decisions[valid] = points[np.argmin(distances, axis=1)]
    return decisions.reshape(values.shape)


def save_cjt_constellation_plot(
    transmitted_symbols,
    equalized_symbols,
    output_path: Path,
    *,
    modulation_order: int,
    max_symbols: int = 10000,
):
    if max_symbols <= 0:
        raise ValueError("max_symbols must be greater than zero")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transmitted = np.asarray(transmitted_symbols).reshape(-1)
    equalized = np.asarray(equalized_symbols).reshape(-1)
    valid = np.isfinite(equalized.real) & np.isfinite(equalized.imag)
    transmitted = transmitted[valid]
    equalized = equalized[valid]
    if transmitted.size > max_symbols:
        selected = np.linspace(
            0, transmitted.size - 1, max_symbols, dtype=np.int64
        )
        transmitted = transmitted[selected]
        equalized = equalized[selected]
    ideal = _constellation_points(modulation_order)
    all_values = np.concatenate((ideal, transmitted, equalized))
    limit = max(1.0, float(np.max(np.abs(all_values)))) * 1.2
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].scatter(transmitted.real, transmitted.imag, s=8, alpha=0.35)
    axes[0].set_title("CJT transmitted symbols")
    axes[1].scatter(equalized.real, equalized.imag, s=8, alpha=0.35)
    axes[1].set_title("CJT equalized symbols")
    for axis in axes:
        axis.scatter(ideal.real, ideal.imag, marker="x", color="black")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal")
        axis.grid(True, alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
