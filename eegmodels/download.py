"""Download layer: resolve a registry entry's source to local files in the cache.

* Hugging Face sources use ``huggingface_hub.snapshot_download`` (resume, dedup,
  revision pinning, gated-repo auth via the user's HF token).
* Direct URL sources use ``pooch`` (resume-friendly, SHA256 verification).

Cache location, in order of precedence: the ``cache_dir`` argument, the
``EEGMODELS_CACHE`` environment variable, then ``~/.cache/eegmodels``.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path

from .registry import ModelSpec, get_spec


class DownloadError(RuntimeError):
    pass


def cache_dir(override: os.PathLike | str | None = None) -> Path:
    if override:
        root = Path(override)
    elif os.environ.get("EEGMODELS_CACHE"):
        root = Path(os.environ["EEGMODELS_CACHE"])
    else:
        root = Path.home() / ".cache" / "eegmodels"
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_dir(spec: ModelSpec, root: os.PathLike | str | None = None) -> Path:
    return cache_dir(root) / spec.name


def sha256_of(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def license_banner(spec: ModelSpec) -> str:
    lines = [f"[{spec.name}] license: {spec.license.spdx}"]
    if spec.license.research_only:
        lines.append("  Research use only. Do not use for diagnosis or commercial purposes.")
    if spec.license.notes:
        lines.append(f"  {spec.license.notes}")
    if spec.status == "gated":
        lines.append(
            "  Access is gated. Request access on the model page and log in with `huggingface-cli login`."
        )
    return "\n".join(lines)


def is_downloaded(spec: ModelSpec, root: os.PathLike | str | None = None) -> bool:
    d = model_dir(spec, root)
    if not d.is_dir():
        return False
    if spec.source.type == "url":
        return all((d / f["filename"]).exists() for f in spec.source.files)
    return any(d.iterdir())


def _pull_hf(spec: ModelSpec, dest: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError as e:  # pragma: no cover
        raise DownloadError(
            "huggingface_hub is required for Hugging Face sources: pip install huggingface_hub"
        ) from e
    try:
        snapshot_download(
            repo_id=spec.source.repo_id,
            revision=spec.source.revision,
            local_dir=dest,
            token=token,
        )
    except GatedRepoError as e:
        raise DownloadError(
            f"'{spec.source.repo_id}' is gated. Request access at https://huggingface.co/{spec.source.repo_id} "
            "and authenticate with `huggingface-cli login` or the HF_TOKEN environment variable."
        ) from e
    except RepositoryNotFoundError as e:
        raise DownloadError(
            f"Hugging Face repo '{spec.source.repo_id}' not found (revision {spec.source.revision})."
        ) from e
    return dest


def _pull_url(spec: ModelSpec, dest: Path, progress: bool) -> Path:
    try:
        import pooch
    except ImportError as e:  # pragma: no cover
        raise DownloadError("pooch is required for URL sources: pip install pooch") from e
    for f in spec.source.files:
        pooch.retrieve(
            url=f["url"],
            known_hash=f"sha256:{f['sha256']}",
            fname=f["filename"],
            path=dest,
            progressbar=progress,
        )
    return dest


def pull(
    name_or_spec: str | ModelSpec,
    *,
    cache: os.PathLike | str | None = None,
    force: bool = False,
    accept_license: bool = True,
    token: str | None = None,
    progress: bool = True,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
) -> Path:
    """Download a model's weights into the cache and return the model directory.

    ``accept_license=False`` prints the banner and raises instead of downloading, for
    interactive front-ends that want an explicit confirmation step.
    """
    spec = get_spec(name_or_spec) if isinstance(name_or_spec, str) else name_or_spec
    if spec.status == "planned":
        raise DownloadError(
            f"'{spec.name}' is in the registry as 'planned': its adapter is not implemented yet "
            "and no download source is wired up."
        )
    dest = model_dir(spec, cache)
    if is_downloaded(spec, cache) and not force:
        return dest

    log(license_banner(spec))
    if not accept_license:
        raise DownloadError(f"License for '{spec.name}' not accepted; nothing downloaded.")

    dest.mkdir(parents=True, exist_ok=True)
    if spec.source.type == "hf":
        return _pull_hf(spec, dest, token or os.environ.get("HF_TOKEN"))
    return _pull_url(spec, dest, progress)
