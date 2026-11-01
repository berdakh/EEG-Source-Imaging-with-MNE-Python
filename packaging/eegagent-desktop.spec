# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone EEG-Agent desktop bundle.

    pip install -e ".[desktop]" pyinstaller
    pyinstaller packaging/eegagent-desktop.spec

Produces dist/EEG-Agent/ (one-directory bundle). The launcher spawns itself with
--serve-api / --serve-ui for the two services, so one executable is enough.
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # repository root, whatever the cwd
block_cipher = None

datas, binaries, hiddenimports = [], [], []
for pkg in ("streamlit", "mne", "pooch"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
for pkg in ("eegagent", "eegsi", "eegmodels"):
    datas += collect_data_files(pkg, include_py_files=True)  # ui.py must exist on disk for Streamlit
    hiddenimports += collect_submodules(pkg)
for pkg in ("streamlit", "mne", "anthropic", "fastapi", "uvicorn", "pydantic", "pyyaml", "pooch", "huggingface_hub"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass
hiddenimports += collect_submodules("uvicorn") + collect_submodules("scipy.special") + [
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on", "anyio._backends._asyncio", "matplotlib.backends.backend_agg", "lazy_loader",
    "webview", "webbrowser",
]

a = Analysis(
    [os.path.join(SPECPATH, "entry.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "torch", "braindecode", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EEG-Agent",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no terminal window; child logs go to ~/.eegagent/logs
)
import shutil
import sys

coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="EEG-Agent")

# Linux: ship the menu-integration helper next to the executable.
if sys.platform.startswith("linux"):
    shutil.copy(os.path.join(SPECPATH, "linux", "install-desktop-entry.sh"), os.path.join(DISTPATH, "EEG-Agent"))
