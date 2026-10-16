"""Convenience entry point for the standalone CJT transmitter."""

from multiprocessing import freeze_support

from runtime.cjt_runner import main


if __name__ == "__main__":
    freeze_support()
    main()
