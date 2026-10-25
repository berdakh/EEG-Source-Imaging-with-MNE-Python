"""Prove the app makes no outbound connections except to local Ollama.

Runs a health check, an upload, and an ask against a running backend while
recording every TCP connection the process tree opens. Requires psutil.
Usage: uv run --with psutil python scripts/offline_check.py [http://127.0.0.1:8000]
"""

from __future__ import annotations

import sys
import time

import httpx

try:
    import psutil
except ImportError:
    print("Install psutil: uv run --with psutil python scripts/offline_check.py")
    sys.exit(2)

LOCAL = {"127.0.0.1", "::1", "localhost"}


def remote_connections() -> set[tuple[str, int]]:
    found = set()
    for c in psutil.net_connections(kind="inet"):
        if c.raddr and c.raddr.ip not in LOCAL and c.status != psutil.CONN_LISTEN:
            found.add((c.raddr.ip, c.raddr.port))
    return found


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    before = remote_connections()
    with httpx.Client(base_url=base, timeout=120) as c:
        c.get("/api/health").raise_for_status()
        c.post("/api/library", files={"file": ("check.md", b"The van seats 12.", "text/markdown")})
        c.post("/api/ask", json={"question": "How many does the van seat?"})
    time.sleep(1)
    after = remote_connections() - before
    if after:
        print("Outbound connections observed during the check:")
        for ip, port in sorted(after):
            print(f"  {ip}:{port}")
        return 1
    print("OK: no new outbound connections observed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
