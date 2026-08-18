# `v2/data/canonical/` — normalized conversations + provenance

Layer 2 of the three-layer rule in `v2/experiment_design.md` §9:

```text
v2/data/sources/       immutable downloads + licence metadata   (never edited)
v2/data/canonical/     normalized conversations + provenance    (this directory)
v2/data/sft/           derived, versioned training examples
```

**Empty by design as of 2026-08-17.** Stage A only inspected the sources; the
canonical schema is written in Stage B, after the split policy is frozen.

Rules for anything written here:

- Every record keeps `source_dataset`, `source_revision` (the OSF file id +
  `date_modified` from the source `MANIFEST.json`), and
  `source_conversation_id`, so a row can always be traced back to bytes with a
  known hash.
- Normalisation must apply `inspect_three_party.norm_id` (or equivalent): the
  15-minute release writes some foreign keys as floats, and a naive join
  silently drops every human-witness link.
- Game 2197 of the main study has duplicate transcript and verdict rows that
  disagree on confidence. Deduplicate deterministically and record which row was
  kept.
- `tt_profile.other` is excluded pending PII review (registry §8, item 1).
- Contents inherit the source's redistribution limits and are gitignored until a
  registry record approves publication.
