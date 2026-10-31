"""Command-line interface: ``eegmodels list | info | pull | embed``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .adapters.base import AdapterError
from .download import DownloadError, is_downloaded, license_banner, pull
from .registry import RegistryError, get_spec, load_registry


def cmd_list(args: argparse.Namespace) -> int:
    specs = load_registry()
    print(f"{'NAME':<22} {'STATUS':<10} {'LICENSE':<28} DESCRIPTION")
    for name in sorted(specs):
        s = specs[name]
        if args.status and s.status != args.status:
            continue
        mark = "*" if is_downloaded(s, args.cache) else " "
        print(mark + s.summary())
    print("\n* = weights already in cache", file=sys.stderr)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    s = get_spec(args.name)
    print(f"name:        {s.name}")
    print(f"family:      {s.family}")
    print(f"status:      {s.status}")
    print(f"description: {s.description}")
    print(f"source:      {s.source.type} {s.source.repo_id or ''} @ {s.source.revision}")
    for f in s.source.files:
        print(f"             {f['filename']}  {f['url']}")
    print(license_banner(s))
    i = s.input
    print(
        f"input:       {i.sfreq} Hz, units {i.units}, window {i.window_seconds}s, "
        f"channels {len(i.channels) if i.channels else (i.channel_vocab or 'any')}"
    )
    print(f"contract:    {'validated' if i.contract_verified else 'UNVALIDATED (inference disabled)'}")
    print(f"segment:     {i.segment_seconds}s (tokenizer segment, not full context)")
    if i.channel_vocab_size:
        print(f"vocabulary:  {i.channel_vocab_size} slots (not an input-channel limit)")
    if i.validation_notes:
        print(f"validation:  {i.validation_notes}")
    for step in i.preprocessing:
        print(f"requires:    {step}")
    print(f"adapter:     {s.adapter.cls} {s.adapter.kwargs}")
    for k, v in s.references.items():
        print(f"{k + ':':<13}{v}")
    print(f"downloaded:  {is_downloaded(s, args.cache)}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    rc = 0
    for name in args.names:
        try:
            path = pull(name, cache=args.cache, force=args.force, accept_license=not args.no_accept)
            print(f"{name}: {path}")
        except (DownloadError, RegistryError) as e:
            print(f"{name}: {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_embed(args: argparse.Namespace) -> int:
    from .api import load

    try:
        import mne
    except ImportError:
        print("`eegmodels embed` needs MNE to read recordings: pip install 'eegmodels[mne]'", file=sys.stderr)
        return 1
    raw = mne.io.read_raw(args.recording, preload=True, verbose="error")
    model = load(args.name, device=args.device, cache=args.cache)
    z = model.embed(raw, strict_channels=not args.loose_channels)
    np.save(args.out, z)
    print(f"wrote {args.out} with shape {z.shape}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eegmodels", description="Download and run open-weight EEG models.")
    p.add_argument("--cache", help="cache directory (default: $EEGMODELS_CACHE or ~/.cache/eegmodels)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list registry entries")
    sp.add_argument("--status", choices=["available", "gated", "planned"])
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("info", help="show one registry entry")
    sp.add_argument("name")
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("pull", help="download weights into the cache")
    sp.add_argument("names", nargs="+")
    sp.add_argument("--force", action="store_true", help="re-download even if cached")
    sp.add_argument("--no-accept", action="store_true", help="print the license banner and stop")
    sp.set_defaults(fn=cmd_pull)

    sp = sub.add_parser("embed", help="run a model on a recording and save embeddings as .npy")
    sp.add_argument("name")
    sp.add_argument("recording", help="any file mne.io.read_raw understands")
    sp.add_argument("-o", "--out", default="embeddings.npy")
    sp.add_argument("--device", default="cpu")
    sp.add_argument(
        "--loose-channels", action="store_true", help="skip channels the recording lacks instead of failing"
    )
    sp.set_defaults(fn=cmd_embed)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (RegistryError, DownloadError, AdapterError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
