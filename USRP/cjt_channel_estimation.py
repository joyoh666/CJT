"""First-slot FDM channel estimation for CJT feedback."""

import numpy as np

from USRP.cjt_parameters import phase_align_channel_estimates


def estimate_cjt_sounding_channel(params, resource_grid, known_reference):
    """Estimate every TX-to-RX coefficient from the CJT sounding slot."""
    grid = np.asarray(resource_grid, dtype=np.complex64)
    if grid.ndim == 4:
        if grid.shape[1] != 1:
            raise ValueError("resource_grid must contain exactly one slot")
        grid = grid[:, 0]
    expected_shape = (
        int(params["num_rx_ant"]),
        int(params["num_symbols_per_slot"]),
        int(params["N"]),
    )
    if grid.shape != expected_shape:
        raise ValueError(
            f"resource_grid has shape {grid.shape}, expected {expected_shape}"
        )

    reference = np.asarray(known_reference, dtype=np.complex64)
    if reference.shape != (int(params["N"]),):
        raise ValueError("known_reference has the wrong number of subcarriers")
    pilot_symbols = tuple(int(v) for v in params["cjt_pilot_symbols"])
    active_indices = np.asarray(
        params["fdm_pilot_active_indices"], dtype=np.int64
    )
    num_tx = int(params["num_tx_ant"])
    num_rx = int(params["num_rx_ant"])
    if active_indices.shape != (num_tx,):
        raise ValueError("CJT requires one active FDM pilot index per TX")

    repetitions = np.empty(
        (num_tx, len(pilot_symbols), num_rx), dtype=np.complex64
    )
    received_pilots = np.empty_like(repetitions)
    noise_indices = np.setdiff1d(
        np.arange(int(params["N"]), dtype=np.int64),
        active_indices,
        assume_unique=True,
    )
    noise_power = np.empty((len(pilot_symbols), num_rx), dtype=np.float32)
    for repetition, symbol_index in enumerate(pilot_symbols):
        received_symbol = grid[:, symbol_index, :]
        noise_power[repetition] = np.mean(
            np.abs(received_symbol[:, noise_indices]) ** 2,
            axis=-1,
        )
        for tx_index, active_index in enumerate(active_indices):
            value = received_symbol[:, active_index]
            received_pilots[tx_index, repetition] = value
            repetitions[tx_index, repetition] = value / reference[active_index]

    aligned = np.empty_like(repetitions)
    common_phases = np.empty(repetitions.shape, dtype=np.float32)
    for tx_index in range(num_tx):
        aligned_tx, phases_tx = phase_align_channel_estimates(
            repetitions[tx_index, :, :, None]
        )
        aligned[tx_index] = aligned_tx[..., 0]
        common_phases[tx_index] = phases_tx

    raw_csi = np.mean(repetitions, axis=1).astype(np.complex64)
    aligned_csi = np.mean(aligned, axis=1).astype(np.complex64)
    selected_csi = (
        aligned_csi
        if bool(params["phase_align_virtual_pilots"])
        else raw_csi
    )
    return {
        "csi": selected_csi,
        "csi_raw": raw_csi,
        "csi_aligned": aligned_csi,
        "csi_repetitions": repetitions,
        "csi_repetitions_aligned": aligned,
        "pilot_received": received_pilots,
        "pilot_noise_power": noise_power,
        "pilot_common_phases_rad": common_phases,
    }
