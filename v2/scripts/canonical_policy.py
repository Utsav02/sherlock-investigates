#!/usr/bin/env python3
"""
Single source of truth for what may cross from `v2/data/sources/` into
`v2/data/canonical/`.

This module exists so that a data-handling decision recorded in a registry is
also enforced in code. The Stage A review of the Jones & Bergen release found
participant re-identifiers in a free-text column (registry §8, §8.1); the
decision was to exclude that column permanently. A decision written only in
Markdown is a decision that the next loader silently forgets, so the exclusion
lives here and `tests/test_v2_policy.py` asserts it.

The canonical loader (Stage B, not yet written) must call `check_columns()` on
every source table it reads. Failing closed is deliberate: a new source column
that nobody has reviewed should stop the pipeline, not flow into a derived
artefact.

Stdlib only.
"""

from __future__ import annotations

# (table, column) pairs that must never enter a derived artefact.
# Each entry names the registry section that justifies it.
EXCLUDED_COLUMNS: dict[tuple[str, str], str] = {
    ("tt_profile", "other"): (
        "registry jones_bergen_2025 §8.1 — free-text column holding a "
        "24-char Prolific worker id (main study) and a Prolific relay e-mail "
        "whose local part is that same id (15-minute study). Excluded "
        "unconditionally; exclusion is not contingent on redaction because the "
        "PII screens are regex flags with unknown recall."
    ),
}

# Columns that are pseudonymous but re-identifying in combination. Permitted
# inside local canonical artefacts, never in anything published (registry §8).
RESTRICTED_COLUMNS: dict[tuple[str, str], str] = {
    ("tt_profile", "gender"): "demographic; local analysis only",
    ("tt_profile", "year_of_birth"): "demographic; local analysis only",
    ("tt_profile", "education"): "demographic; local analysis only",
    ("tt_profile", "chatbot_interaction_frequency"): "demographic; local analysis only",
    ("tt_profile", "familiarity_with_GPT"): "demographic; local analysis only",
}


class ExcludedColumnError(RuntimeError):
    """Raised when a loader tries to carry an excluded column forward."""


def is_excluded(table: str, column: str) -> bool:
    return (table, column) in EXCLUDED_COLUMNS


def is_restricted(table: str, column: str) -> bool:
    return (table, column) in RESTRICTED_COLUMNS


def check_columns(table: str, columns: list[str]) -> None:
    """Fail closed if any excluded column is about to be carried forward.

    Call this with the columns a loader intends to KEEP, not with every column
    present in the source file.
    """
    offenders = [c for c in columns if is_excluded(table, c)]
    if offenders:
        reasons = "; ".join(EXCLUDED_COLUMNS[(table, c)] for c in offenders)
        raise ExcludedColumnError(
            f"{table}: refusing to carry excluded column(s) {offenders}. {reasons}"
        )


def allowed_columns(table: str, columns: list[str]) -> list[str]:
    """The subset of `columns` a canonical loader may keep."""
    return [c for c in columns if not is_excluded(table, c)]


if __name__ == "__main__":
    print("Excluded (never enter a derived artefact):")
    for (table, column), reason in sorted(EXCLUDED_COLUMNS.items()):
        print(f"  {table}.{column}\n    {reason}")
    print("\nRestricted (local only, never published):")
    for (table, column), reason in sorted(RESTRICTED_COLUMNS.items()):
        print(f"  {table}.{column} — {reason}")
