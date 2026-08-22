#!/usr/bin/env python3
"""Small durable-output helpers for long or interruptible experiments."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

__all__ = ["Run", "append_jsonl", "atomic_write_json"]


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_jsonl(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class _StageHandle:
    def __init__(self, run: "Run", name: str) -> None:
        self._run, self._name = run, name
        self.count = int(run._state["progress"].get(name, {}).get("count", 0))
        self.skipped = False

    def tick(self, n: int = 1, **extra: Any) -> None:
        self.count += n
        self._run._state["progress"][self._name] = {
            "count": self.count,
            "at": time.time(),
            **extra,
        }
        self._run._flush()

    def note(self, **kv: Any) -> None:
        self._run._state["progress"].setdefault(self._name, {}).update(kv)
        self._run._flush()


class Run:
    """Atomic stage state. Output IDs, not this counter, own exact resume."""

    def __init__(self, state_path: str | Path, resume: bool = False,
                 config: dict | None = None) -> None:
        self.state_path = Path(state_path)
        self._state: dict[str, Any] = {
            "started": time.time(), "completed": [], "failed": None,
            "progress": {}, "config": config or {}, "argv": sys.argv,
        }
        if resume and self.state_path.exists():
            try:
                prior = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"corrupt state file {self.state_path}: {exc}")
            if config and prior.get("config") != config:
                raise SystemExit("resume state does not match the requested config")
            self._state.update(prior)
            self._state["resumed_at"] = time.time()
            self._state["failed"] = None
        self._flush()

    def _flush(self) -> None:
        atomic_write_json(self.state_path, self._state)

    @contextmanager
    def stage(self, name: str) -> Iterator[_StageHandle]:
        handle = _StageHandle(self, name)
        if name in self._state["completed"]:
            handle.skipped = True
            yield handle
            return
        started = time.time()
        try:
            yield handle
        except BaseException as exc:
            self._state["failed"] = {
                "stage": name, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8), "at": time.time(),
            }
            self._flush()
            raise
        self._state["completed"].append(name)
        self._state["progress"].setdefault(name, {})["elapsed"] = time.time() - started
        self._flush()
