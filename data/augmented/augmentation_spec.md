# Augmentation Framings Specification

**Generated:** 2026-06-07  
**Source:** `data/processed/chunks_labeled.jsonl` (64 central + 164 minor)  
**Purpose:** Template definitions and worked examples for the 3–5× augmentation pipeline

---

## The Five Framings

### Framing 1: Raw Narrative (VERBATIM)

**Description:** The original passage exactly as extracted from the source text, with no modification.

**Template:**

```
{{CHUNK}}
```

**When to apply:** Central and minor chunks. This framing is always included — it is the baseline pass. No generation call needed; simply emit the chunk as-is.

**Notes:** Including the verbatim source ensures the model sees the original Victorian prose register alongside the augmented framings, preventing stylistic drift toward modern instruction prose.

---

### Framing 2: Observation–Deduction Q&A (QA)

**Description:** A natural-language question that describes the observable facts of the scene, followed by Holmes's deduction as the answer. The question is posed as if a puzzled observer is asking; the answer is Holmes's reasoning chain.

**Template:**

```
Below is a passage from a Victorian detective story, followed by a question and answer based on it.

PASSAGE:
{{CHUNK}}

Q: Given the observations described above, what does the detective conclude, and how does he arrive at that conclusion?

A: [Holmes's reasoning chain, drawn from the passage, written in first person as Holmes speaking]
```

**When to apply:** Central chunks only. The Q&A framing requires a rich enough deductive chain to produce a meaningful answer; minor chunks often lack sufficient reasoning content to make the answer non-trivial.

---

### Framing 3: Watson Retrospective Summary (WATSON)

**Description:** A third-person Watson narrating the scene after the fact, emphasizing what he observed of Holmes's method and what he understood (or failed to understand) at the time.

**Template:**

```
Below is a passage from a Victorian detective story, followed by Watson's retrospective account of the same scene.

PASSAGE:
{{CHUNK}}

WATSON'S ACCOUNT:
[Watson summarizes the passage in past tense, emphasizing what Holmes noticed, what Watson himself initially missed, and what the experience taught him about Holmes's method. 3–5 sentences.]
```

**When to apply:** Central and minor chunks. Watson's retrospective can work even for minor chunks by focusing on atmosphere and Holmes's manner rather than a full reasoning chain. For minor chunks, the Watson summary naturally stays shorter.

---

### Framing 4: Structured Observation–Inference Chain (CHAIN)

**Description:** A structured prompt that explicitly lists the observable facts as premises and then presents Holmes's inference as numbered steps, making the reasoning structure maximally explicit.

**Template:**

```
Below is a passage from a Victorian detective story. Read it carefully, then complete the structured analysis.

PASSAGE:
{{CHUNK}}

OBSERVATIONS (what can be directly seen or measured):
1. [first observable fact from the passage]
2. [second observable fact]
3. [additional facts as needed]

INFERENCES (what Holmes concludes from each observation):
1. [inference from observation 1]
2. [inference from observation 2]
3. [combined or final conclusion]

CONCLUSION: [Holmes's final deduction in one sentence]
```

**When to apply:** Central chunks only. This framing imposes explicit logical structure that only works when the source passage contains enough reasoning steps to populate at least two observation–inference pairs.

---

### Framing 5: Reverse Construction (REVERSE)

**Description:** Holmes's conclusion is given first, and the model (or generation script) must reconstruct the observational evidence that led there. The original passage serves as the ground truth for the generated observations.

**Template:**

```
Below is a conclusion reached by a Victorian detective. Before reading further, consider what observations might have led him to this conclusion. Then read the passage that follows to see how he actually arrived at it.

CONCLUSION:
[Holmes's final deduction from the passage, extracted and stated baldly]

What observations would you expect a skilled detective to have made in order to reach this conclusion?

ORIGINAL PASSAGE:
{{CHUNK}}
```

**When to apply:** Central chunks only. Reverse construction requires a clean, quotable conclusion — which most minor chunks do not provide. This framing is high-value for training because it explicitly links conclusion back to evidence, reinforcing the bidirectionality of deductive reasoning.

---

## Summary Table

| Framing | Name | Apply to | Generation needed? |
|---|---|---|---|
| 1 | VERBATIM | Central + minor | No |
| 2 | QA | Central only | Yes |
| 3 | WATSON | Central + minor | Yes |
| 4 | CHAIN | Central only | Yes |
| 5 | REVERSE | Central only | Yes |

Central chunks receive all 5 framings; minor chunks receive framings 1 (verbatim) and 3 (Watson). This gives an effective multiplier of 5× for central and 2× for minor. With the 3× oversample for central chunks already in the training mix, the effective training signal from central chunks is 5 × 3 = 15× the raw token count — meaningfully more than the design doc's 3–5× target and more than sufficient to push past the 240K threshold identified in the audit report.

---

## Worked Examples

Three central chunks are shown below with all five framings applied.

---

### Example A — chunk_id=645 | *A Study in Scarlet* | 200 words

**Source passage:**

> "This was the first point gained. I then walked slowly down the garden path, which happened to be composed of a clay soil, peculiarly suitable for taking impressions. No doubt it appeared to you to be a mere trampled line of slush, but to my trained eyes every mark upon its surface had a meaning. There is no branch of detective science which is so important and so much neglected as the art of tracing footsteps. Happily, I have always laid great stress upon it, and much practice has made it second nature to me. I saw the heavy footmarks of the constables, but I saw also the track of the two men who had first passed through the garden. It was easy to tell that they had been before the others, because in places their marks had been entirely obliterated by the others coming upon the top of them. In this way my second link was formed, which told me that the nocturnal visitors were two in number, one remarkable for his height (as I calculated from the length of his stride), and the other fashionably dressed, to judge from the small and elegant impression left by his boots."

**Framing 1 — VERBATIM:**

*(The source passage is emitted as-is. No generation needed.)*

---

**Framing 2 — QA:**

> Q: Given the observations described above, what does the detective conclude, and how does he arrive at that conclusion?
>
> A: I began with the garden path itself — clay soil, excellent for impressions, though to an untrained eye it would appear nothing more than churned mud. The first matter was chronology: whose marks were laid down first? The answer was plain enough. Where the constables' heavy boots had trodden directly upon earlier prints and obliterated them, those earlier prints must have preceded the constables. From this I was able to isolate the original visitors' tracks. Two men, not one: the stride patterns were distinct and consistent. The first man was tall — the length of his stride told me that without hesitation, for stride scales reliably with height. The second man was fashionably dressed: his boots left a small, narrow, elegant impression, the kind worn by a man who attends to his appearance and can afford to. These were not assumptions; they were readings of the evidence, the same as reading a text.

---

**Framing 3 — WATSON:**

> Holmes led me down the garden path at a pace I found almost insultingly slow, though I later understood why. Where I had seen only trampled mud — the inevitable confusion left by the constables' boots — he had seen a complete record. He explained afterward that footprint analysis is the most undervalued of the detective sciences, and that long practice had made it as natural to him as reading a printed page. He identified two sets of tracks predating the police, distinguishing them from the later impressions by the simple fact that the constables had obliterated portions of them. From the first set he calculated a man of considerable height; from the second, a man of fashion — the sole's shape was narrow, carefully made, the mark of someone who spent money on his boots. I had walked the same path and seen nothing.

---

**Framing 4 — CHAIN:**

> OBSERVATIONS:
> 1. The garden path is composed of clay soil, which retains clear impressions.
> 2. The constables' heavy footmarks appear on top of an earlier set of prints, partially obliterating them.
> 3. A second set of earlier prints shows a long stride length.
> 4. A third set of earlier prints shows a small, narrow, elegantly shaped sole impression.
>
> INFERENCES:
> 1. Any prints beneath the constables' prints must have been laid down before the police arrived — therefore they belong to the original nocturnal visitors.
> 2. Stride length scales reliably with height; a long stride indicates a tall man.
> 3. A small, narrow, elegant boot impression indicates a fashionably dressed man of refined taste.
>
> CONCLUSION: Two men visited the scene before the police: one tall, one fashionably dressed and of smaller stature.

---

**Framing 5 — REVERSE:**

> CONCLUSION: Two men visited the scene before the police — one tall, one fashionably dressed with narrow, elegant boots.
>
> What observations would you expect a skilled detective to have made in order to reach this conclusion?
>
> ORIGINAL PASSAGE: *(source passage follows)*

---

### Example B — chunk_id=173 | *A Study in Scarlet* | 129 words

**Source passage:**

> "Come along, Doctor," he said; "we shall go and look him up. I'll tell you one thing which may help you in the case," he continued, turning to the two detectives. "There has been murder done, and the murderer was a man. He was more than six feet high, was in the prime of life, had small feet for his height, wore coarse, square-toed boots and smoked a Trichinopoly cigar. He came here with his victim in a four-wheeled cab, which was drawn by a horse with three old shoes and one new one on his off fore leg. In all probability the murderer had a florid face, and the finger-nails of his right hand were remarkably long. These are only a few indications, but they may assist you."

**Framing 1 — VERBATIM:**

*(Emitted as-is.)*

---

**Framing 2 — QA:**

> Q: Given the observations described above, what does the detective conclude, and how does he arrive at that conclusion?
>
> A: The indications were there for anyone to read, though I admit they require some practice to see. The man was tall — the stride and the reach of certain marks I observed made that plain. His feet were small for his height, which is not uncommon in a man of his build. The boots were coarse and square-toed — working-man's boots, not those of a gentleman, and the tread marks confirmed it. As for the Trichinopoly cigar, the ash is quite distinctive; I have made a small study of tobacco and its residues. The cab was four-wheeled, as you can tell from the turning radius of the wheel marks, and one of the horse's shoes was newer than the others — a slightly different depth of impression in the soft ground. The florid complexion and long nails I infer from other details I need not explain just now. These are not guesses. They are readings.

---

**Framing 3 — WATSON:**

> Holmes turned from the crime scene with an air of easy confidence that I confess astonished me. He had spent perhaps twenty minutes in that room, and now he was rattling off a description of the murderer — height, footwear, tobacco habit, the condition of the horse that drew his cab — as though he were reading from a dossier compiled over months. Lestrade and Gregson exchanged their characteristic looks of half-admiration, half-scepticism. Holmes smiled at their reaction. He had seen things in that room that I had walked past entirely, and the description he offered was not speculation but inference — each element traceable back to a mark, a stain, an ash, an impression.

---

**Framing 4 — CHAIN:**

> OBSERVATIONS:
> 1. Marks in the room indicate a tall man with a long stride; foot impressions show small feet relative to height.
> 2. Boot impressions are coarse and square-toed.
> 3. Cigar ash near the body is of a distinctive pale grey colour and coarse texture.
> 4. Wheel marks outside indicate a four-wheeled cab; one hoof impression is shallower and sharper-edged than the other three.
> 5. Additional physical evidence (not detailed) suggests a ruddy complexion and long right-hand fingernails.
>
> INFERENCES:
> 1. Stride length and foot impression size establish height and build.
> 2. Boot type indicates social class and likely occupation.
> 3. Ash composition identifies the cigar brand — Trichinopoly — confirming a specific demographic.
> 4. Wheel geometry identifies cab type; uneven hoof depth identifies a freshly shod shoe on one leg.
> 5. Remaining indicators suggest physical type.
>
> CONCLUSION: The murderer was a tall man over six feet, in the prime of life, small-footed, wearing coarse square-toed boots, who smoked Trichinopoly cigars, arrived in a four-wheeled cab with a horse recently shod on one leg, and had a florid face with unusually long right-hand fingernails.

---

**Framing 5 — REVERSE:**

> CONCLUSION: The murderer was a tall man, over six feet, with small feet for his height, wearing coarse square-toed boots, who smoked Trichinopoly cigars and arrived by four-wheeled cab drawn by a horse freshly shod on one leg.
>
> What observations would you expect a skilled detective to have made in order to reach this conclusion?
>
> ORIGINAL PASSAGE: *(source passage follows)*

---

### Example C — chunk_id=364 | *A Study in Scarlet* | 86 words

**Source passage:**

> Sherlock Holmes drew a long breath, and wiped the perspiration from his forehead. "I should have more faith," he said; "I ought to know by this time that when a fact appears to be opposed to a long train of deductions, it invariably proves to be capable of bearing some other interpretation. Of the two pills in that box one was of the most deadly poison, and the other was entirely harmless. I ought to have known that before ever I saw the box at all."

**Framing 1 — VERBATIM:**

*(Emitted as-is.)*

---

**Framing 2 — QA:**

> Q: Given the observations described above, what does the detective conclude, and how does he arrive at that conclusion?
>
> A: I had allowed myself a moment of doubt — which I consider a professional failing, not a virtue. When the evidence in the box appeared to contradict what my reasoning told me must be true, I felt the pull to revise my conclusions. But I have learned — or should have learned by now — that a contradiction almost never means the deduction was wrong. It means the interpretation of one fact has been too narrow. The box contained two pills: one poisonous, one harmless. That was the answer to the apparent contradiction, and I ought to have anticipated it from the first principles of the case. A murderer who wishes to leave no witnesses to the act may well carry insurance.

---

**Framing 3 — WATSON:**

> I had rarely seen Holmes show anything resembling doubt, and when he wiped the perspiration from his brow I understood that we had reached a critical juncture. He recovered himself almost immediately, with a remark that I found memorable: that when a fact appears to contradict a long chain of reasoning, the fault lies with the interpretation of the fact, not with the chain. He had deduced that the box contained two pills — one lethal, one inert — without having seen the box at all. He said he ought to have known it earlier, which I took to mean that the structure of the crime had always implied a failsafe, and he had been slow to read it.

---

**Framing 4 — CHAIN:**

> OBSERVATIONS:
> 1. A box containing two pills is present; one pill has been confirmed as a deadly poison.
> 2. Holmes had previously reached a conclusion through deduction that appeared to be contradicted by observable facts.
>
> INFERENCES:
> 1. If a deduction is sound and a fact appears to contradict it, the fact must admit of an alternative interpretation — the deductive chain is not invalidated by surface-level contradiction.
> 2. A murderer who relies on a victim ingesting a pill would rationally hedge against uncertainty by making one pill lethal and one harmless, gambling on the victim's choice.
>
> CONCLUSION: The box contains one poisonous and one harmless pill; the apparent contradiction between the physical evidence and Holmes's prior deductions resolves when the pill-pair design is understood as an intentional feature of the crime.

---

**Framing 5 — REVERSE:**

> CONCLUSION: Of the two pills in the box, one was a deadly poison and the other entirely harmless — and Holmes realizes he should have deduced this before ever examining the box.
>
> What observations or prior reasoning steps would you expect a skilled detective to have assembled in order to arrive at this conclusion without seeing the box?
>
> ORIGINAL PASSAGE: *(source passage follows)*

---

## Risk Note on Framings

**Framing 5 (REVERSE)** carries the highest quality risk. The framing asks a generation script to reason backward from a conclusion to its evidence — a more demanding task than summarizing or structuring a forward chain. For short passages like chunk_id=364 (86 words), the model generating the reverse reconstruction has very little source material to work with, and the resulting "expected observations" may be vague or circular. The fallback is to restrict REVERSE to central chunks with a word count above 100, which retains 45 of the 64 central chunks and ensures the source passage contains enough content to support a non-trivial reconstruction.

**Framing 3 (WATSON)** carries moderate risk on very short central chunks (under 20 words). Watson's retrospective requires some narrative fabric to work with; on a 10-word deduction like "You mean the retired sergeant of Marines," the summary will inevitably feel thin. The fallback is to apply a 30-word minimum for WATSON generation, treating sub-threshold chunks as VERBATIM-only.
