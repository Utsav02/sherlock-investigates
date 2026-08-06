#!/usr/bin/env python3
"""Push a trained adapter to the HuggingFace Hub.

WHY THIS EXISTS: Kaggle deletes /kaggle/working when the session ends, and
RunPod deletes the pod's disk when it is torn down. HuggingFace is the
authoritative copy of every adapter (CLAUDE.md, Data & secrets). If you start a
run and go to sleep, the upload has to happen unattended or the run is lost.

The token is read from the environment or a Kaggle Secret. It is NEVER accepted
as a command-line argument, because argv ends up in shell history and in the
notebook output that Kaggle saves publicly.

    # Kaggle: Add-ons -> Secrets -> name it HF_TOKEN
    python scripts/training/upload_adapter.py \
        --adapter outputs/kaggle_t4_validation_seed42/final_adapter \
        --repo-id Utsav02/sherlock-r1distill-7b-validation

Uploads the adapter plus the config and training log alongside it, so the
artifact carries the exact settings that produced it. An adapter without its
config is not reproducible.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def get_token() -> str:
    """Environment first, then Kaggle Secrets. Never from argv."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    sys.exit(
        "ERROR: no HF token found.\n"
        "  Kaggle : Add-ons -> Secrets -> add HF_TOKEN, then re-run.\n"
        "  Local  : export HF_TOKEN=hf_...\n"
        "  Create one at https://huggingface.co/settings/tokens "
        "with WRITE permission."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--repo-id", required=True,
                    help="e.g. Utsav02/sherlock-r1distill-7b-validation")
    ap.add_argument("--config", default=None,
                    help="config YAML to bundle with the adapter (recommended)")
    ap.add_argument("--log", default=None,
                    help="training log to bundle (recommended)")
    ap.add_argument("--public", action="store_true",
                    help="Default is PRIVATE, matching the repo's stated IP "
                         "posture (CLAUDE.md: the corpus and design are the "
                         "experiment's IP). Opt in explicitly to publish.")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    if not adapter.exists():
        sys.exit(f"ERROR: no adapter at {adapter}")

    from huggingface_hub import HfApi

    token = get_token()
    api = HfApi()

    # Bundle config + log INTO the adapter folder so they travel with it.
    # An adapter whose hyperparameters are unknown cannot be reproduced or
    # compared against another run.
    for src, name in ((args.config, "training_config.yaml"),
                      (args.log, "training_log.txt")):
        if not src:
            continue
        p = Path(src) if Path(src).is_absolute() else ROOT / src
        if p.exists():
            shutil.copy(p, adapter / name)
            print(f"  bundled {name}")
        else:
            print(f"  WARNING: {p} not found, not bundled")

    size_mb = sum(f.stat().st_size for f in adapter.rglob("*") if f.is_file()) / 1e6
    print(f"\n  adapter : {adapter}")
    print(f"  size    : {size_mb:.0f} MB")
    print(f"  repo    : {args.repo_id}  ({'PUBLIC' if args.public else 'private'})")

    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=not args.public, exist_ok=True, token=token)
    api.upload_folder(folder_path=str(adapter), repo_id=args.repo_id,
                      repo_type="model", token=token)

    print(f"\n  DONE -> https://huggingface.co/{args.repo_id}")
    print("  Safe to close the session; the adapter is off-machine.")


if __name__ == "__main__":
    main()
