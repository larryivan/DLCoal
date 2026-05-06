#!/usr/bin/env python3
"""Compatibility wrapper for the simulator package.

Prefer:
    python -m simulator
"""

from simulator.cli import main


if __name__ == "__main__":
    main()
