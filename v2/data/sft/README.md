# `v2/data/sft/` — derived, versioned training examples

Layer 3 of `v2/experiment_design.md` §9. **Empty by design as of 2026-08-17,
and it stays empty for the three-party corpus.**

Two rules decide what may ever appear here:

1. **v2.1's only required SFT set is D0** — synthetic identity tasks with known
   response distributions (§11.1). D1–D5 are deferred; D5 (new human data) is
   not authorized by the design document.
2. **Nothing derived from the Jones & Bergen three-party corpus may enter an SFT
   set in v2.1.** Its registry record approves evaluation and local development
   only, and excludes fine-tuning any generative adapter on it — the source has
   no declared licence, and an author of the data has published a caution about
   training on these transcripts (registry §10–§12).

Every example written here must carry the §9 provenance block:

```json
{
  "example_id": "d0-<scenario-family>-<episode>-<turn>",
  "source_dataset": "...",
  "source_revision": "...",
  "source_conversation_id": "...",
  "transformation_version": "...",
  "target_origin": "human_trace|teacher_proposal|empirical_search|synthetic",
  "review_status": "unreviewed|verified|rejected"
}
```

Contents are gitignored (except this file) until a registry record approves
publication.
