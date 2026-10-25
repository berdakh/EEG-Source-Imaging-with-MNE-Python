"""Pull the Ollama models for this machine's hardware tier, as declared in the pack manifest."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def total_memory_gb() -> float:
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        return int(out) / 1024**3
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1024**3
    return 8.0


def main() -> int:
    pack = os.environ.get("LOA_PACK", "church")
    manifest = yaml.safe_load((REPO / "packs" / pack / "pack.yaml").read_text("utf-8"))
    mem = total_memory_gb()
    tier = next((t for t in manifest["hardware_tiers"] if mem >= t["min_memory_gb"]), None)
    if tier is None:
        print(
            f"Only {mem:.0f} GB memory; no tier fits. Smallest tier needs "
            f"{manifest['hardware_tiers'][-1]['min_memory_gb']} GB."
        )
        return 1
    print(
        f"Detected {mem:.0f} GB memory -> chat={tier['chat_model']} "
        f"embed={tier['embed_model']} whisper={tier['whisper_model']}"
    )
    for model in (tier["chat_model"], tier["embed_model"]):
        print(f"ollama pull {model}")
        subprocess.run(["ollama", "pull", model], check=True)
    print("\nAdd to .env:")
    print(f"LOA_CHAT_MODEL={tier['chat_model']}")
    print(f"LOA_EMBED_MODEL={tier['embed_model']}")
    print(f"LOA_WHISPER_MODEL={tier['whisper_model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
