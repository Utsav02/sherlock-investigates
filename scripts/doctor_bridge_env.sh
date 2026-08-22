#!/usr/bin/env bash
set -u

REPO=$(git rev-parse --show-toplevel)
PY="$REPO/.bridge-venv/bin/python"
LOCK="$REPO/requirements-bridge.lock"
MODEL_DIR="$REPO/v2/.cache/huggingface"
failed=0

check() {
  label=$1
  shift
  if "$@" >/dev/null 2>&1; then
    echo "PASS $label"
  else
    echo "FAIL $label"
    failed=1
  fi
}

check "bridge interpreter exists" test -x "$PY"
check "dependency lock exists" test -f "$LOCK"
if [ -x "$PY" ]; then
  check "Python is 3.12" "$PY" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'
  check "torch/transformers/safetensors import" "$PY" -c 'import torch, transformers, safetensors'
  check "installed versions match lock" "$PY" -c 'import importlib.metadata as m, pathlib, re, sys; lock=pathlib.Path(sys.argv[1]).read_text().splitlines(); expected={}; [expected.__setitem__(*re.split(r"==", x, maxsplit=1)) for x in lock if x and not x.startswith(("#", "-")) and "==" in x]; actual={m.metadata(k)["Name"].lower():m.version(k) for k in expected}; raise SystemExit(any(actual[k.lower()] != v for k,v in expected.items()))' "$LOCK"
  check "CPU tensor inference executes" "$PY" -c 'import torch; x=torch.tensor([1.0]); assert (x+x).item() == 2.0'
fi
check "canonical manifest materialized" test -r "$REPO/v2/data/canonical/main_study_v1/manifest.json"
check "at least 2 GiB free" sh -c 'test "$(df -Pk "$1" | awk "NR==2 {print \$4}")" -gt 2097152' sh "$REPO"
check "model cache parent writable" sh -c 'mkdir -p "$1" && test -w "$1"' sh "$MODEL_DIR"
check "git worktree resolves" git -C "$REPO" rev-parse HEAD

exit "$failed"
