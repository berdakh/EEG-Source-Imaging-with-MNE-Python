"""PyInstaller entry point: a plain script so the package's relative imports work when frozen."""

import multiprocessing
import sys

from eegagent.desktop import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
