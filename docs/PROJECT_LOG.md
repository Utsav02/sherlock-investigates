# Project log

This file captures cross-cutting decisions and environmental gotchas that
are not specific to a single file. For code-specific notes, see inline
comments in the relevant scripts.

## Decisions

### 2026-05-17: TOC-overwrite strategy for Gutenberg splitting
Ebook #1661 has 24 heading matches for 12 stories because the table of
contents uses the same Roman-numeral format as the story bodies. The
splitter iterates forward and lets later entries overwrite earlier ones
in the dict, exploiting the invariant that story bodies always come
after TOC entries. See download_adventures.py for the implementation.

### 2026-05-17: Training story selection
Picked "A Scandal in Bohemia" and "The Red-Headed League" from Adventures
collection. Reasoning: both have heavy Holmes-explaining-deduction
content. Combined with "A Study in Scarlet" novel, gives ~61K training
words. Held-out: "The Speckled Band."

### 2026-05-18: qwen2.5:7b chosen over phi4-mini for local classification
In the 20-chunk pilot benchmark phi4-mini labeled 16/20 chunks "minor" regardless
of content (including pure action and short dialogue with zero deductive material).
qwen2.5:7b made a meaningful none/minor/central distinction: 11 "none", 8 "minor",
1 "central" on the same sample, and correctly identified the one passage in the
sample where Holmes reasons through possibilities as "central". The phi4-mini
pattern means its "minor" label has near-zero precision for anything deductive,
making it unusable for downstream passage selection. qwen2.5:7b is about 1.5× slower
per call (~5–6 s vs ~3–4 s), adding roughly 30 minutes to the cold classification pass.

### 2026-05-18: Paragraph-level chunking with 10-word floor
The chunker splits on one or more consecutive blank lines and drops chunks
shorter than 10 words. The 10-word floor retains single-sentence deductions
("You mean the retired sergeant of Marines") while excluding section headings
("I.", "II.") and chapter numbers. Result: 957 chunks from 61K training words
(avg 62 words/chunk). A higher floor (e.g. 20 words) would have cleaner chunks
but would drop a meaningful fraction of Holmes's single-line deductions, which
are precisely what the classification is looking for.

### 2026-05-18: paragraph_index is raw (pre-filter) index into the source file
The JSONL chunk records store the paragraph's original position in the source file
(counting all paragraphs including filtered-out short ones), not the position among
accepted chunks. This is a stable reference back to the source and makes it possible
to locate a chunk in the original text by counting blank-line-separated blocks.
The chunk_id field is the monotonic accepted-chunk counter across all stories.

## Gotchas

### 2026-05-17: Gutenberg TOC duplicates
Gutenberg ebooks include a table of contents that uses the same heading
format as the story bodies. Naive splitters will produce twice as many
sections as expected. Confirmed on #1661; expect on other multi-story
collections. Solution implemented in download_adventures.py.

### 2026-05-18: tqdm not installed despite being in requirements.txt
The venv was created in session 2 but requirements.txt was not fully installed
into it at the time. On the first attempt to run classify_chunks.py the tqdm
import failed. Fix: `source venv/bin/activate && pip install -r requirements.txt`.
For future sessions, if any standard dependency is missing, run that command
before assuming there is a code problem.

### 2026-05-18: venv activation required for all scripts
The system Python3 is Homebrew Python 3.13 with no project packages installed.
All scripts must be run from within the venv (`source venv/bin/activate`). The
chunk_stories.py script worked on the first run only because it has no third-party
imports; benchmark_models.py and classify_chunks.py failed immediately without
the venv active.