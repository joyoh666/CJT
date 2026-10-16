"""USRP initialization shared by the CJT transmit and receive workers."""

import time

import uhd


def configure_cjt_usrp_timing(
    usrp,
    *,
    clock_source=None,
    time_source=None,
    sync_barrier=None,
    sync_timeout=10.0,
):
    """Align all CJT RU/UE device times to the same external PPS edge."""
    if (clock_source is None) != (time_source is None):
        raise ValueError("clock_source and time_source must be provided together")
    if clock_source is None:
        usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
        return

    for motherboard in range(usrp.get_num_mboards()):
        usrp.set_clock_source(clock_source, motherboard)
        usrp.set_time_source(time_source, motherboard)
        if usrp.get_clock_source(motherboard) != clock_source:
            raise RuntimeError("USRP clock source configuration failed")
        if usrp.get_time_source(motherboard) != time_source:
            raise RuntimeError("USRP time source configuration failed")

    if sync_barrier is not None:
        sync_barrier.wait(timeout=sync_timeout)
        previous_pps = usrp.get_time_last_pps().get_real_secs()
        deadline = time.monotonic() + float(sync_timeout)
        while usrp.get_time_last_pps().get_real_secs() == previous_pps:
            if time.monotonic() >= deadline:
                raise TimeoutError("no external PPS edge was detected")
            time.sleep(0.01)
        sync_barrier.wait(timeout=sync_timeout)

    usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))
    time.sleep(1.1)


def initialize_tx_usrp(
    device_args,
    tx_subdev_spec,
    tx_antenna,
    tx_channels,
    *,
    clock_source=None,
    time_source=None,
    sync_barrier=None,
):
    """Create the TX-only distributed-RU MultiUSRP used by CJT."""
    usrp = uhd.usrp.MultiUSRP(device_args)
    configure_cjt_usrp_timing(
        usrp,
        clock_source=clock_source,
        time_source=time_source,
        sync_barrier=sync_barrier,
    )
    for motherboard in range(usrp.get_num_mboards()):
        usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(tx_subdev_spec), motherboard)
    channels = tuple(tx_channels)
    if not channels or min(channels) < 0:
        raise ValueError("CJT TX channels must be non-empty and non-negative")
    if max(channels) >= usrp.get_tx_num_channels():
        raise ValueError("configured CJT TX channel is not exposed by the USRP")
    for channel in channels:
        usrp.set_tx_antenna(tx_antenna, channel)
    print(f"CJT RU USRPs ready; TX channels: {channels}")
    return usrp


def initialize_rx_usrp(
    device_args,
    rx_subdev_spec,
    rx_antenna,
    rx_channels,
    *,
    clock_source=None,
    time_source=None,
    sync_barrier=None,
):
    """Create the RX-only UE MultiUSRP used by CJT."""
    usrp = uhd.usrp.MultiUSRP(device_args)
    configure_cjt_usrp_timing(
        usrp,
        clock_source=clock_source,
        time_source=time_source,
        sync_barrier=sync_barrier,
    )
    for motherboard in range(usrp.get_num_mboards()):
        usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(rx_subdev_spec), motherboard)
    channels = tuple(rx_channels)
    if not channels or min(channels) < 0:
        raise ValueError("CJT RX channels must be non-empty and non-negative")
    if max(channels) >= usrp.get_rx_num_channels():
        raise ValueError("configured CJT RX channel is not exposed by the USRP")
    for channel in channels:
        usrp.set_rx_antenna(rx_antenna, channel)
    print(f"CJT UE USRP ready; RX channels: {channels}")
    return usrp
