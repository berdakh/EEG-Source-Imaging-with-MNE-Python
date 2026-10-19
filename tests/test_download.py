import hashlib
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pytest

from eegmodels import download
from eegmodels.download import DownloadError, is_downloaded, pull, sha256_of
from eegmodels.registry import parse_spec


@pytest.fixture
def http_root(tmp_path):
    """Serve tmp_path/src over localhost so pooch exercises its real HTTP path."""
    root = tmp_path / "src"
    root.mkdir()
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    srv = HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def url_spec(http_root, content=b"weights", bad_hash=False):
    root, base = http_root
    (root / "w.bin").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    if bad_hash:
        digest = "0" * 64
    return parse_spec(
        {
            "name": "urltoy",
            "status": "available",
            "source": {"type": "url", "files": [{"url": f"{base}/w.bin", "sha256": digest, "filename": "w.bin"}]},
            "license": {"spdx": "MIT", "research_only": True},
            "input": {"sfreq": 100},
            "adapter": {"class": "planned"},
        }
    )


def test_cache_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EEGMODELS_CACHE", str(tmp_path / "c"))
    assert download.cache_dir() == tmp_path / "c"
    assert download.cache_dir(tmp_path / "o") == tmp_path / "o"


def test_pull_url_verifies_and_caches(tmp_path, http_root):
    spec = url_spec(http_root)
    cache = tmp_path / "cache"
    msgs = []
    out = pull(spec, cache=cache, progress=False, log=msgs.append)
    assert (out / "w.bin").read_bytes() == b"weights"
    assert is_downloaded(spec, cache)
    assert any("Research use only" in m for m in msgs)
    # second call is a cache hit: no banner printed
    msgs.clear()
    pull(spec, cache=cache, progress=False, log=msgs.append)
    assert msgs == []


def test_pull_url_bad_checksum_fails(tmp_path, http_root):
    spec = url_spec(http_root, bad_hash=True)
    with pytest.raises(ValueError, match="hash"):
        pull(spec, cache=tmp_path / "cache", progress=False, log=lambda s: None)
    assert not is_downloaded(spec, tmp_path / "cache")


def test_pull_requires_license_acceptance(tmp_path, http_root):
    spec = url_spec(http_root)
    with pytest.raises(DownloadError, match="not accepted"):
        pull(spec, cache=tmp_path / "cache", accept_license=False, log=lambda s: None)


def test_pull_planned_refuses(tmp_path):
    spec = parse_spec(
        {"name": "p", "status": "planned", "source": {"type": "hf", "repo_id": "x/y"},
         "input": {"sfreq": 1}, "adapter": {"class": "planned"}}
    )
    with pytest.raises(DownloadError, match="planned"):
        pull(spec, cache=tmp_path)


def test_sha256_of(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_of(p) == hashlib.sha256(b"abc").hexdigest()
