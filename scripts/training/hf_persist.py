"""Off-machine persistence helpers — HF Hub uploads that never crash the caller.

WHY THIS EXISTS: three runs have now been lost to ephemeral disks. 2026-08-06:
21 checkpoints died with a Kaggle session; the same day save_total_limit=3
discarded the dose curve a paid run had already produced. 2026-08-07: the
re-run's 12 checkpoints, final adapter, and results JSON were wiped when the
owner ended the interactive session — /kaggle/working does not survive session
end. An artifact that exists only on the machine producing it is not an
artifact; it is a bet that nothing goes wrong before someone remembers to copy
it. This module makes persistence part of the run path instead of a manual
step afterwards.

Design constraints, in priority order:
1. NEVER raise into the caller. A failed upload of checkpoint-40 must not kill
   the training run that is about to produce checkpoint-45 — losing one
   checkpoint to a flaky uplink is recoverable; losing the run is not.
2. Retry with backoff. Kaggle's uplink drops transiently; most failures heal
   within a minute.
3. Loud on failure. A silent skip recreates the exact failure class this
   exists to end.
4. Token from env or Kaggle Secrets only — never argv (shell history, and
   Kaggle saves notebook output publicly).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# 4 attempts total: immediate, then 10s / 30s / 90s. ~2 min worst case per
# artifact — bounded, so a dead network delays a run by minutes, not hours.
RETRY_DELAYS = (10, 30, 90)

# Repos already created this process — create_repo(exist_ok=True) is cheap but
# there is no reason to call it once per checkpoint.
_ensured: set[tuple[str, str]] = set()


def find_hf_token() -> str | None:
    """Environment first, then Kaggle Secrets. Returns None instead of raising
    so callers can degrade to a loud warning — training must proceed either way."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def upload_path(
    local_path: str | Path,
    repo_id: str,
    token: str,
    *,
    path_in_repo: str | None = None,
    repo_type: str = "model",
    private: bool = True,
    label: str | None = None,
) -> bool:
    """Upload a file or folder to the Hub. Returns True on success.

    Never raises — every failure path prints loudly and returns False, because
    the callers are mid-run (training between checkpoints, eval between rows)
    and must keep producing the artifacts even when persistence is failing.
    """
    local_path = Path(local_path)
    label = label or local_path.name
    if not local_path.exists():
        print(f"  [persist] NOTHING at {local_path} — cannot upload {label}", flush=True)
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(f"  [persist] huggingface_hub NOT INSTALLED — {label} exists only "
              "on this machine. pip install huggingface_hub.", flush=True)
        return False

    api = HfApi()
    attempts = 1 + len(RETRY_DELAYS)
    for attempt in range(attempts):
        if attempt:
            delay = RETRY_DELAYS[attempt - 1]
            print(f"  [persist] retrying {label} in {delay}s "
                  f"(attempt {attempt + 1}/{attempts})", flush=True)
            time.sleep(delay)
        try:
            if (repo_id, repo_type) not in _ensured:
                api.create_repo(repo_id=repo_id, repo_type=repo_type,
                                private=private, exist_ok=True, token=token)
                _ensured.add((repo_id, repo_type))
            t0 = time.time()
            if local_path.is_dir():
                api.upload_folder(folder_path=str(local_path), repo_id=repo_id,
                                  repo_type=repo_type, path_in_repo=path_in_repo,
                                  token=token)
            else:
                api.upload_file(path_or_fileobj=str(local_path), repo_id=repo_id,
                                repo_type=repo_type,
                                path_in_repo=path_in_repo or local_path.name,
                                token=token)
            print(f"  [persist] {label} -> {repo_id}"
                  f"{'/' + path_in_repo if path_in_repo else ''} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            return True
        except Exception as e:  # noqa: BLE001 — constraint 1: never kill the run
            print(f"  [persist] upload FAILED for {label} "
                  f"({type(e).__name__}: {e})", flush=True)

    print("  " + "!" * 68, flush=True)
    print(f"  [persist] GIVING UP on {label} after {attempts} attempts.", flush=True)
    print(f"  [persist] It survives ONLY on this machine — re-upload it before "
          "the session ends.", flush=True)
    print("  " + "!" * 68, flush=True)
    return False
