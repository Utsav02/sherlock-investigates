#!/usr/bin/env python3
"""
Fetch an OSF project's files into v2/data/sources/<name>/ and write a manifest.

Stage A step 3 of `v2/experiment_design.md`. The sources layer is IMMUTABLE:
this script only ever writes files that do not already exist, and refuses to
overwrite. Everything downstream (canonical/, sft/) is derived, never edited
in place here.

The manifest records, per §8.2 of the design doc, what is needed to reproduce
and to prove nothing was silently altered:

  - the OSF node id and page URL
  - the "revision" of each file: OSF's own version identifier plus the
    server-reported date_modified (OSF has no single project-wide revision,
    so per-file version is the honest unit)
  - the download date (UTC)
  - the size and sha256 computed LOCALLY over the bytes on disk, alongside
    the sha256/md5 the OSF API reports, so a mismatch is visible

Stdlib only, so it runs under the repo venv's `python` without extra installs.

Usage:
    venv/bin/python v2/scripts/fetch_osf_source.py \
        --node jk7bw --name jones_bergen_2025 [--include-img] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OSF_API = "https://api.osf.io/v2"
# The house rule from CLAUDE.md: always send a descriptive User-Agent.
USER_AGENT = "sherlock-investigates-v2-research/0.1 (academic dataset audit)"

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_DIR = REPO_ROOT / "v2" / "data" / "sources"


def _request(url: str, retries: int = 4) -> bytes:
    """GET with a descriptive UA and exponential backoff on transient failures."""
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:  # pragma: no cover - network
            last_error = exc
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 3
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last_error})")


def _get_json(url: str) -> dict:
    return json.loads(_request(url).decode("utf-8"))


def list_node_files(node: str, prefix: str = "") -> list[dict]:
    """Walk an OSF node's osfstorage recursively, returning flat file records."""
    start = f"{OSF_API}/nodes/{node}/files/osfstorage/?page%5Bsize%5D=100"
    return _walk(start, prefix)


def _walk(url: str | None, prefix: str) -> list[dict]:
    out: list[dict] = []
    while url:
        payload = _get_json(url)
        for entry in payload["data"]:
            attrs = entry["attributes"]
            name = attrs["name"]
            if attrs["kind"] == "folder":
                child = entry["relationships"]["files"]["links"]["related"]["href"]
                out.extend(_walk(child, f"{prefix}{name}/"))
            else:
                hashes = (attrs.get("extra") or {}).get("hashes") or {}
                out.append(
                    {
                        "path": f"{prefix}{name}",
                        "osf_file_id": entry["id"],
                        "download_url": entry["links"]["download"],
                        "size_reported": attrs.get("size"),
                        "date_modified": attrs.get("date_modified"),
                        "osf_version": (attrs.get("extra") or {}).get("version"),
                        "md5_reported": hashes.get("md5"),
                        "sha256_reported": hashes.get("sha256"),
                    }
                )
        url = payload["links"].get("next")
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_files(files: list[dict], include_img: bool) -> tuple[list[dict], list[dict]]:
    """Split into (fetch, skip). img/ holds rendered figures, not data."""
    if include_img:
        return files, []
    fetch = [f for f in files if not f["path"].startswith("img/")]
    skip = [f for f in files if f["path"].startswith("img/")]
    return fetch, skip


def fetch(node: str, name: str, include_img: bool, dry_run: bool) -> Path:
    dest = SOURCES_DIR / name
    files = list_node_files(node)
    to_fetch, skipped = select_files(files, include_img)

    print(f"OSF node {node}: {len(files)} files; fetching {len(to_fetch)}, "
          f"skipping {len(skipped)}")
    if dry_run:
        for record in to_fetch:
            print(f"  would fetch {record['path']} ({record['size_reported']} B)")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for record in to_fetch:
        target = dest / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # Sources are immutable: never overwrite, just re-verify.
            print(f"  exists, verifying {record['path']}")
        else:
            print(f"  downloading {record['path']} ({record['size_reported']} B)")
            target.write_bytes(_request(record["download_url"]))
        local_sha = sha256_file(target)
        records.append(
            {
                **record,
                "size_local": target.stat().st_size,
                "sha256_local": local_sha,
                "sha256_matches_osf": (
                    None
                    if record["sha256_reported"] is None
                    else local_sha == record["sha256_reported"]
                ),
            }
        )

    manifest = {
        "source_name": name,
        "osf_node": node,
        "osf_page_url": f"https://osf.io/{node}/",
        "osf_api_url": f"{OSF_API}/nodes/{node}/",
        "download_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetched_by": "v2/scripts/fetch_osf_source.py",
        "user_agent": USER_AGENT,
        "note": (
            "OSF has no project-wide revision identifier. Per-file 'osf_version' "
            "and 'date_modified' are the revision unit. Files are immutable here: "
            "the fetcher never overwrites and downstream code is read-only."
        ),
        "skipped_paths": [f["path"] for f in skipped],
        "skipped_reason": (
            "" if include_img else
            "img/ holds rendered figures reproducible from the released .Rmd; "
            "excluded to keep the tracked corpus small. Re-fetch with --include-img."
        ),
        "file_count": len(records),
        "total_bytes": sum(r["size_local"] for r in records),
        "files": sorted(records, key=lambda r: r["path"]),
    }
    manifest_path = dest / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    mismatches = [r["path"] for r in records if r["sha256_matches_osf"] is False]
    if mismatches:
        print(f"!! sha256 MISMATCH on {len(mismatches)} files: {mismatches}")
        return dest
    print(f"OK: {len(records)} files, {manifest['total_bytes']} bytes, "
          f"all sha256 match OSF. Manifest: {manifest_path}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="jk7bw", help="OSF node id")
    parser.add_argument("--name", default="jones_bergen_2025",
                        help="directory name under v2/data/sources/")
    parser.add_argument("--include-img", action="store_true",
                        help="also fetch the img/ figure folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    fetch(args.node, args.name, args.include_img, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
