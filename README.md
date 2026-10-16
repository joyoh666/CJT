# CJT: same-frame sounding and MRT data transmission

This directory is an independent USRP transmission project. It measures CSI
in slot 0, leaves a muted processing interval, and schedules MRT-precoded
reference/data bursts in the remainder of the same 5 ms frame.

The channel-sounder project is not imported. CJT-specific settings live in
`config/cjt_config.py`, and CJT runtime/DSP names use `CJT` or `cjt_` prefixes.
The `models` directory is a separate copy because both projects consume the
same predictor and RU-scheduler model sources.

## Hardware-free verification

```bash
uv run python -m runtime.cjt_runner --dry-run --frames 2 --frame-seed 7
```

## USRP execution

```bash
uv run python -m runtime.cjt_runner \
  --frames 100 \
  --processing-gap-slots 1 \
  --deadline-guard-us 100 \
  --output-dir cjt_results
```

If a 0.5 ms processing gap produces deadline misses, increase
`--processing-gap-slots` to 2 and inspect the generated JSON latency summary.
