#!/usr/bin/env bash
set -eu

REPO=$(git rev-parse --show-toplevel)
SYSTEM_PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
ENV_DIR="$REPO/.bridge-venv"
LOCK="$REPO/requirements-bridge.lock"

if [ ! -x "$SYSTEM_PY" ]; then
  echo "missing required Python 3.12 interpreter: $SYSTEM_PY" >&2
  exit 1
fi
if [ ! -f "$LOCK" ]; then
  echo "missing $LOCK; dependency resolution must be reviewed and committed first" >&2
  exit 1
fi

if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$SYSTEM_PY" -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --requirement "$LOCK"
echo "bridge environment ready at $ENV_DIR"
echo "NOTE: pip installation and first model fetch require network access"
