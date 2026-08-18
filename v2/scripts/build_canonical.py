#!/usr/bin/env python3
"""
Normalise the cleared 5-minute (main) study into the canonical layer.

`v2/experiment_design.md` §9 keeps three layers apart:

    v2/data/sources/     immutable downloads + licence metadata   (never edited)
    v2/data/canonical/   normalized conversations + provenance    (written here)
    v2/data/sft/         derived, versioned training examples

Scope is deliberately narrow, and the narrowness is a permission decision, not a
convenience one:

  * Only `jones_bergen_2025/data` — the March 2025 five-minute study — is
    normalised. Its Gate 0 is CONDITIONAL: evaluation and local development are
    approved (registry §12). The `15_mins/` study inside the same OSF node has
    NO resolved Gate 0 (registry §14 item 3: no paper, no preregistration,
    consent unconfirmed) and is therefore not read by this script at all.
  * `tt_profile.other` is never materialised. `canonical_policy.check_columns`
    is called with the columns this loader intends to KEEP for every source
    table it touches, so the exclusion fails the pipeline closed rather than
    relying on anyone remembering it.

Two records are written per game:

  * a **dialogue** row per conversation (2,280) — one witness, its messages, and
    its ground-truth `is_human`. This is the unit a passive detector scores.
  * a **game** row per game (1,140) — the A/B pairing, the human interrogator's
    verdict, and the frozen split/component assignment. This is the unit Track
    A's frozen contrast P1 is measured on, and the unit that clusters.

Provenance per §9: every row carries `source_dataset`, `source_revision`,
`source_conversation_id`, `transformation_version`, `target_origin` and
`review_status`. `source_revision` is a 12-hex digest over the OSF file ids and
`date_modified` stamps of exactly the files read; `manifest.json` beside the
data expands it back to the per-file list, so a row traces to bytes with a known
hash without carrying a kilobyte of provenance on every line.

`target_origin` is **null** on every canonical row, on purpose. That field
describes where an SFT *target* came from; a canonical conversation has no
target. Design §7.1's rule — "fields an arm does not produce must be null, not
filled with a plausible untrained number" — is the same principle, so the field
is present and empty rather than given an invented value.

Stdlib only.

Usage:
    venv/bin/python v2/scripts/build_canonical.py
    venv/bin/python v2/scripts/build_canonical.py --check   # verify, write nothing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_splits  # noqa: E402
import canonical_policy  # noqa: E402
from inspect_three_party import SOURCE_DIR, is_blank, norm_id, read_csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "v2" / "data" / "canonical" / "main_study_v1"
MANIFEST = SOURCE_DIR / "MANIFEST.json"

csv.field_size_limit(10_000_000)

CANONICAL_VERSION = "canonical_v1"
SOURCE_DATASET = "jones_bergen_2025/data"
STUDY = "main_5min_2025_03"

# Files this loader reads. The revision digest covers exactly these.
SOURCE_FILES = (
    "data/tt_game.csv",
    "data/tt_conversation.csv",
    "data/tt_witness.csv",
    "data/tt_interrogator.csv",
    "data/tt_message_anonymized.csv",
    "data/tt_verdict.csv",
    "data/tt_transcripts.csv",
    "data/tt_profile.csv",
    "data/tt_aimodel.csv",
)

# Columns this loader intends to KEEP, per table. Passed to the policy check.
KEPT_COLUMNS = {
    "tt_game": ["id", "interrogator_id", "human_witness_id", "ai_witness_id"],
    "tt_conversation": ["id", "game_id", "witness_id", "label"],
    "tt_witness": ["id", "witness_type", "user_id", "ai_model_id"],
    "tt_interrogator": ["id", "user_id"],
    "tt_message_anonymized": [
        "id", "conversation_id", "sender_role", "timestamp", "content", "is_changed",
    ],
    "tt_verdict": ["id", "game_id", "verdict", "confidence", "is_correct"],
    "tt_transcripts": ["game_id", "conversation_label", "is_human", "witness", "transcript"],
    # NOTE: `other` is deliberately absent. canonical_policy fails closed on it.
    "tt_profile": ["user_id", "source", "expt_aware"],
    "tt_aimodel": ["id", "name", "family", "model_name"],
}

# Anonymisation placeholders inserted by the authors' GPT-4o redaction pass.
PLACEHOLDER_RE = re.compile(r"<[A-Z_]{3,}>|\[[A-Z_]{3,}\]")
WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_v2_track_a.py)
# ---------------------------------------------------------------------------

def collapse_ws(text: str) -> str:
    """Collapse newlines/runs of whitespace. Harness formatting, not writing."""
    return WS_RE.sub(" ", text or "").strip()


def strip_placeholders(text: str) -> str:
    """Remove `<NAME>` / `[LOCATION]`-style redaction tokens.

    These are collection artefacts: they were written by the authors'
    anonymisation model, not by the witness. Their PRESENCE is still a signal a
    detector can use ("this speaker named a person"), which is exactly why the
    template-normalisation sensitivity analysis strips them.
    """
    return collapse_ws(PLACEHOLDER_RE.sub(" ", text or ""))


def truthy(value: str | None) -> bool:
    return (value or "").strip().upper() == "TRUE"


def revision_digest(manifest_files: list[dict], paths: tuple[str, ...]) -> tuple[str, dict]:
    """12-hex digest over (path, osf_file_id, date_modified, sha256) of `paths`."""
    by_path = {f["path"]: f for f in manifest_files}
    missing = [p for p in paths if p not in by_path]
    if missing:
        raise KeyError(f"MANIFEST.json does not list {missing}")
    detail = {
        p: {
            "osf_file_id": by_path[p]["osf_file_id"],
            "date_modified": by_path[p]["date_modified"],
            "sha256": by_path[p]["sha256_local"],
        }
        for p in paths
    }
    blob = json.dumps(detail, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12], detail


def digest_rows(rows: list[dict]) -> str:
    blob = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _check_policy(tables: dict[str, list[str]]) -> None:
    for table, columns in tables.items():
        canonical_policy.check_columns(table, columns)


def build() -> dict:
    _check_policy(KEPT_COLUMNS)

    base = SOURCE_DIR / "data"
    _, games = read_csv(base / "tt_game.csv")
    _, conversations = read_csv(base / "tt_conversation.csv")
    _, witnesses = read_csv(base / "tt_witness.csv")
    _, interrogators = read_csv(base / "tt_interrogator.csv")
    _, messages = read_csv(base / "tt_message_anonymized.csv")
    _, verdicts = read_csv(base / "tt_verdict.csv")
    _, transcripts = read_csv(base / "tt_transcripts.csv")
    _, profiles = read_csv(base / "tt_profile.csv")
    _, aimodels = read_csv(base / "tt_aimodel.csv")

    notes: list[str] = []

    seat_user_i = {norm_id(r["id"]): norm_id(r["user_id"]) for r in interrogators}
    witness_row = {norm_id(r["id"]): r for r in witnesses}
    model_row = {norm_id(r["id"]): r for r in aimodels}
    profile_source = {
        norm_id(r["user_id"]): (r.get("source") or "").strip() for r in profiles
    }
    profile_aware = {norm_id(r["user_id"]): truthy(r.get("expt_aware")) for r in profiles}

    # --- messages, grouped by conversation, ordered by (timestamp, id) --------
    by_conversation: dict[str, list[dict]] = defaultdict(list)
    for row in messages:
        by_conversation[norm_id(row["conversation_id"])].append(row)
    for rows in by_conversation.values():
        rows.sort(key=lambda r: (r["timestamp"], int(norm_id(r["id"]) or 0)))

    # --- the witness-system label + is_human, from the transcript table -------
    # tt_transcripts has 2,282 rows for 2,280 conversations: game 2197 is
    # duplicated on both labels. Deduplicate on (game_id, label), keeping the
    # first occurrence, and record it.
    transcript_of: dict[tuple[str, str], dict] = {}
    transcript_dupes: list[str] = []
    for row in transcripts:
        key = (norm_id(row["game_id"]), row["conversation_label"])
        if key in transcript_of:
            transcript_dupes.append(f"{key[0]}/{key[1]}")
            continue
        transcript_of[key] = row
    if transcript_dupes:
        notes.append(
            "tt_transcripts duplicates dropped (kept first occurrence in file "
            f"order): {sorted(set(transcript_dupes))}"
        )

    # --- verdicts, deduplicated deterministically -----------------------------
    # Game 2197 carries two verdict rows agreeing on verdict/reason and
    # disagreeing on confidence (100 vs 45). Keep the LOWEST verdict id, which is
    # a rule that does not depend on file order, and record which was kept.
    verdict_rows: dict[str, list[dict]] = defaultdict(list)
    for row in verdicts:
        verdict_rows[norm_id(row["game_id"])].append(row)
    verdict_of: dict[str, dict] = {}
    for gid, rows in verdict_rows.items():
        rows.sort(key=lambda r: int(norm_id(r["id"]) or 0))
        verdict_of[gid] = rows[0]
        if len(rows) > 1:
            notes.append(
                f"game {gid}: {len(rows)} verdict rows; kept verdict id "
                f"{norm_id(rows[0]['id'])} (confidence {rows[0]['confidence']}), "
                f"dropped {[norm_id(r['id']) for r in rows[1:]]} "
                f"(confidence {[r['confidence'] for r in rows[1:]]}). "
                "Rule: lowest verdict id."
            )

    # --- frozen split assignment ---------------------------------------------
    split_payload = build_splits.build()
    game_split = split_payload["game_split"]
    component_of_user = split_payload["component_of_user"]
    itb = build_splits.itb_game_ids()

    # --- conversations by game ------------------------------------------------
    conv_by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in conversations:
        conv_by_game[norm_id(row["game_id"])][row["label"]] = row

    manifest_files = json.loads(MANIFEST.read_text())["files"]
    revision, revision_detail = revision_digest(manifest_files, SOURCE_FILES)

    dialogue_rows: list[dict] = []
    game_rows: list[dict] = []

    for game in sorted(games, key=lambda r: int(norm_id(r["id"]))):
        gid = norm_id(game["id"])
        interrogator_user = seat_user_i.get(norm_id(game["interrogator_id"]), "")
        split = game_split.get(gid)
        component = component_of_user.get(interrogator_user)

        sides = conv_by_game.get(gid, {})
        if set(sides) != {"A", "B"}:
            raise AssertionError(f"game {gid} has conversation labels {sorted(sides)}")

        per_side: dict[str, dict] = {}
        for label in ("A", "B"):
            conv = sides[label]
            cid = norm_id(conv["id"])
            wit = witness_row.get(norm_id(conv["witness_id"]), {})
            is_human = wit.get("witness_type", "").strip().upper() == "H"
            transcript = transcript_of.get((gid, label), {})
            system = (transcript.get("witness") or "").strip()
            model = model_row.get(norm_id(wit.get("ai_model_id", "")), {})

            turns = []
            for row in by_conversation.get(cid, []):
                turns.append({
                    "role": row["sender_role"].strip(),
                    "timestamp": row["timestamp"],
                    "is_changed": truthy(row["is_changed"]),
                    "content": collapse_ws(row["content"]),
                })
            witness_turns = [t for t in turns if t["role"] == "W"]

            row_out = {
                # --- design §9 provenance block ---------------------------------
                "example_id": f"jb2025main-{gid}-{label}",
                "source_dataset": SOURCE_DATASET,
                "source_revision": revision,
                "source_conversation_id": cid,
                "transformation_version": CANONICAL_VERSION,
                "target_origin": None,          # canonical rows carry no SFT target
                "review_status": "unreviewed",
                # --- content -----------------------------------------------------
                "study": STUDY,
                "game_id": gid,
                "conversation_label": label,
                "is_human": is_human,
                "witness_system": system,
                "witness_model_family": (model.get("family") or "").strip() or None,
                "witness_model_name": (model.get("model_name") or "").strip() or None,
                "witness_user_id": norm_id(wit.get("user_id", "")) if is_human else None,
                "interrogator_user_id": interrogator_user,
                "split": split,
                "component": component,
                "n_messages": len(turns),
                "n_witness_messages": len(witness_turns),
                "n_witness_chars": sum(len(t["content"]) for t in witness_turns),
                "witness_messages_changed": sum(1 for t in witness_turns if t["is_changed"]),
                "witness_messages_with_placeholder": sum(
                    1 for t in witness_turns if PLACEHOLDER_RE.search(t["content"])
                ),
                "turns": turns,
            }
            per_side[label] = row_out
            dialogue_rows.append(row_out)

        verdict = verdict_of.get(gid, {})
        human_label = "A" if per_side["A"]["is_human"] else "B"
        ai_label = "B" if human_label == "A" else "A"
        game_rows.append({
            "example_id": f"jb2025main-{gid}",
            "source_dataset": SOURCE_DATASET,
            "source_revision": revision,
            "source_conversation_id": gid,
            "transformation_version": CANONICAL_VERSION,
            "target_origin": None,
            "review_status": "unreviewed",
            "study": STUDY,
            "game_id": gid,
            "split": split,
            "component": component,
            "interrogator_user_id": interrogator_user,
            "interrogator_recruitment_source": profile_source.get(interrogator_user) or None,
            "interrogator_expt_aware": profile_aware.get(interrogator_user),
            "human_conversation_label": human_label,
            "ai_conversation_label": ai_label,
            "human_witness_user_id": per_side[human_label]["witness_user_id"],
            "witness_system": per_side[ai_label]["witness_system"],
            "witness_model_family": per_side[ai_label]["witness_model_family"],
            "verdict": (verdict.get("verdict") or "").strip() or None,
            "verdict_confidence_signed": (verdict.get("confidence") or "").strip() or None,
            "verdict_is_correct": truthy(verdict.get("is_correct")) if verdict else None,
            "verdict_row_id": norm_id(verdict.get("id", "")) or None,
            "in_itb_557": gid in itb,
            "n_messages_total": per_side["A"]["n_messages"] + per_side["B"]["n_messages"],
            "empty_side": any(per_side[s]["n_messages"] == 0 for s in ("A", "B")),
        })

    summary = {
        "dialogues": len(dialogue_rows),
        "games": len(game_rows),
        "human_dialogues": sum(1 for d in dialogue_rows if d["is_human"]),
        "ai_dialogues": sum(1 for d in dialogue_rows if not d["is_human"]),
        "games_by_split": dict(sorted(Counter(g["split"] for g in game_rows).items())),
        "games_with_empty_side": sum(1 for g in game_rows if g["empty_side"]),
        "dialogues_with_zero_messages": sum(1 for d in dialogue_rows if d["n_messages"] == 0),
        "human_position_A": sum(1 for g in game_rows if g["human_conversation_label"] == "A"),
        "witness_systems": dict(sorted(Counter(g["witness_system"] for g in game_rows).items())),
        "itb_557_games": sum(1 for g in game_rows if g["in_itb_557"]),
        "interrogator_expt_aware_games": sum(1 for g in game_rows if g["interrogator_expt_aware"]),
        "placeholder_dialogues": {
            "human": sum(1 for d in dialogue_rows
                         if d["is_human"] and d["witness_messages_with_placeholder"]),
            "ai": sum(1 for d in dialogue_rows
                      if not d["is_human"] and d["witness_messages_with_placeholder"]),
        },
        "changed_dialogues": {
            "human": sum(1 for d in dialogue_rows
                         if d["is_human"] and d["witness_messages_changed"]),
            "ai": sum(1 for d in dialogue_rows
                      if not d["is_human"] and d["witness_messages_changed"]),
        },
    }

    return {
        "manifest": {
            "canonical_version": CANONICAL_VERSION,
            "source_dataset": SOURCE_DATASET,
            "study": STUDY,
            "gate_0": "CONDITIONAL - evaluation and local development approved "
                      "(registry jones_bergen_2025 §12); training and republication excluded",
            "excluded_by_policy": {
                f"{t}.{c}": reason
                for (t, c), reason in canonical_policy.EXCLUDED_COLUMNS.items()
            },
            "kept_columns": KEPT_COLUMNS,
            "source_revision": revision,
            "source_revision_detail": revision_detail,
            "split_version": split_payload["split_version"],
            "split_sha256": split_payload["sha256"],
            "dedup_notes": notes,
            "summary": summary,
            "dialogues_sha256": digest_rows(dialogue_rows),
            "games_sha256": digest_rows(game_rows),
        },
        "dialogues": dialogue_rows,
        "games": game_rows,
    }


def write(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("dialogues", "games"):
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in payload[name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(payload["manifest"], indent=2) + "\n"
    )


def load() -> tuple[list[dict], list[dict], dict]:
    """Read the canonical layer back. Used by the Track A scripts."""
    manifest = json.loads((OUT_DIR / "manifest.json").read_text())
    def _rows(name: str) -> list[dict]:
        with (OUT_DIR / f"{name}.jsonl").open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return _rows("dialogues"), _rows("games"), manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="rebuild and compare digests against what is on disk")
    args = parser.parse_args(argv)

    payload = build()
    m = payload["manifest"]

    if args.check:
        path = OUT_DIR / "manifest.json"
        if not path.exists():
            print(f"FAIL: {path} does not exist", file=sys.stderr)
            return 1
        on_disk = json.loads(path.read_text())
        ok = all(on_disk.get(k) == m[k] for k in
                 ("dialogues_sha256", "games_sha256", "source_revision", "split_sha256"))
        print(f"on disk  {on_disk.get('dialogues_sha256', '')[:16]}…")
        print(f"rebuilt  {m['dialogues_sha256'][:16]}…")
        print("MATCH" if ok else "MISMATCH")
        return 0 if ok else 1

    write(payload)
    s = m["summary"]
    print(f"canonical_version : {CANONICAL_VERSION}")
    print(f"source_revision   : {m['source_revision']}  ({len(SOURCE_FILES)} files)")
    print(f"split             : {m['split_version']} {m['split_sha256'][:16]}…")
    print(f"dialogues         : {s['dialogues']} ({s['human_dialogues']} human / "
          f"{s['ai_dialogues']} AI); sha256 {m['dialogues_sha256'][:16]}…")
    print(f"games             : {s['games']} {s['games_by_split']}; "
          f"sha256 {m['games_sha256'][:16]}…")
    print(f"human in slot A   : {s['human_position_A']} of {s['games']}")
    print(f"empty sides       : {s['dialogues_with_zero_messages']} dialogues, "
          f"{s['games_with_empty_side']} games")
    print(f"placeholders      : human {s['placeholder_dialogues']['human']} / "
          f"AI {s['placeholder_dialogues']['ai']} dialogues")
    print(f"anonymiser-changed: human {s['changed_dialogues']['human']} / "
          f"AI {s['changed_dialogues']['ai']} dialogues")
    for note in m["dedup_notes"]:
        print(f"  note: {note}")
    print(f"\nwrote {OUT_DIR}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
