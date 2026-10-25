"""Load the pack's synthetic samples into a running backend's library (for demos and dev)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    pack = os.environ.get("LOA_PACK", "church")
    samples = REPO / "packs" / pack / "samples"
    files = [
        p
        for p in samples.iterdir()
        if p.suffix in {".md", ".txt", ".pdf", ".docx"} and p.name != "README.md"
    ]
    with httpx.Client(base_url=base, timeout=60) as c:
        for p in files:
            r = c.post("/api/library", files={"file": (p.name, p.read_bytes())})
            print(f"{p.name}: {r.status_code} {r.json()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
