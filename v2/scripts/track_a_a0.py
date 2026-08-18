#!/usr/bin/env python3
"""
Track A, arm A0: majority/random plus lexical and statistical passive detectors,
evaluated ONLY on the frozen contrasts P1 and P2.

What this script is, stated precisely so the result is not over-read
--------------------------------------------------------------------
`v2/results/stage_a/precision_and_contrasts.md` §5 freezes two primary contrasts:

    P1  A2 vs A0, paired accuracy, leave-one-component-out cross-fitting over
        train+dev (911 games, 11 folds). MDD 5.25pp at ICC 0.10.
    P2  Out-of-fold calibration of the same estimator: Brier score and
        reliability curve, with clustered 95% intervals.

**A2 does not exist yet.** A2 is a classifier head on a temporally clean frozen
representation model, and choosing that checkpoint is still an open design
decision (§19). So P1 *as frozen* is NOT evaluated here and this script does not
claim to have evaluated it. What runs here is:

  * the P1 **procedure** — the same 11 folds, the same pairing, the same
    clustered intervals — with A0's trained detectors in the estimator slot and
    A0's trivial baseline in the reference slot. That establishes the number A2
    will have to beat, and it exercises the machinery end to end before any GPU
    is spent.
  * P2 **directly**, because calibration of an out-of-fold estimator is
    well-defined for any estimator that emits a probability, including A0's.

Arm A1 (Inverse Turing Bench reproduction) is deliberately NOT implemented. Its
557 games are 48.9% of the corpus and biased long (mean 90.5 tokens per released
dialogue against 65.0 corpus-wide), and the head-to-head A2-vs-A1 contrast was
demoted to interval-reporting. It needs its own session with that bias stated up
front.

The frozen test split is never read. It is Gate 5, one shot.

Units, and why there are several
--------------------------------
Two different graph facts govern this evaluation and are easy to conflate:

  * **Splitting/folding** happens on the connected component of the participant
    co-occurrence graph. That controls *leakage*: two people who shared a game
    cannot land in different folds.
  * **Inference** clusters on the participant. That controls *dependence*: one
    user's ~4 games are correlated, but two users in the same component who
    never shared a game are not thereby correlated.

Games are clustered by interrogator AND by human witness — crossed, not nested.
A single-level bootstrap on either role alone understates that. This script
therefore reports intervals under every unit, so the reader can see the size of
the choice rather than take it on trust:

    game            each game independent (ANTI-conservative; the naive default)
    interrogator    cluster bootstrap on the interrogator
    human_witness   cluster bootstrap on the human witness
    participant     the wider of the two above (conservative stand-in for a
                    proper two-way crossed bootstrap, which is NOT implemented)
    component       cluster bootstrap on the 11 co-occurrence components
                    (valid but only 11 clusters, one holding ~a third of games)

The headline interval is `participant`. `game` is reported to make the
understatement visible, not to be used.

Stdlib only (no numpy/sklearn in this venv), so the logistic regressions are
implemented here: ridge-IRLS for the low-dimensional dense feature sets and
full-batch gradient descent with momentum for sparse TF-IDF.

Usage:
    venv/bin/python v2/scripts/track_a_a0.py
    venv/bin/python v2/scripts/track_a_a0.py --bootstrap 200   # quick pass
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_canonical  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "v2" / "results" / "track_a"

# Text variants. The delta between them IS the template-normalisation
# sensitivity analysis; see `TEXT_VARIANTS` docstrings in `dialogue_text`.
VARIANTS = ("raw", "nostub", "nostub_nochanged")

BOOTSTRAP_DEFAULT = 1000
SEED = 20260817

# Mosteller-Wallace-style closed-class list: words whose rate is a stylometric
# signature rather than a topic signal. Fixed in advance, never tuned on scores.
FUNCTION_WORDS = """
a about all also an and any are as at be been but by can could did do does
down even for from had has have he her here him his how i if in into is it
its just like me more most my no not now of on one only or other our out over
so some than that the their them then there these they this those through to
too under up very was we well were what when where which while who will with
would you your am been being had having i'm it's don't didn't can't that's
yeah ok okay hi hey lol haha yes nope yep gonna wanna kinda sorta
""".split()

TOKEN_RE = re.compile(r"[a-z']+")
PUNCT_CLASSES = {
    "period": ".", "comma": ",", "question": "?", "exclaim": "!",
    "apostrophe": "'", "quote": '"', "dash": "-", "semicolon": ";",
    "colon": ":", "slash": "/", "paren": "(",
}


# ---------------------------------------------------------------------------
# ablation conditions
# ---------------------------------------------------------------------------

# Feature names that are derived from HOW MUCH text there is rather than from
# what the text says. Under `drop_length` these are removed from every dense
# family.
#
# Why each one is here (the list is deliberately aggressive — a feature stays
# only if it is a rate that is invariant to how long the conversation ran):
#   n_messages/total_*/mean_*/sd_*/max_*  quantity of text, directly
#   frac_long / frac_short                per-message length distribution
#   empty                                 message count == 0
#   fw_type_token                         type-token ratio falls with length by
#                                         construction, so it is a length proxy
#                                         wearing a lexical-diversity costume
LENGTH_DERIVED = frozenset({
    "n_messages", "total_chars", "total_words", "mean_chars", "mean_words",
    "max_chars", "sd_chars", "frac_long", "frac_short", "empty",
    "fw_type_token",
})


# Fixed budget for the capped control. 20 whitespace tokens sits just under the
# human median witness length (24 words) and well under the AI median (31), so
# most dialogues on both sides actually reach it and are truncated to the SAME
# length. Chosen from the measured medians, not tuned against any score.
TOKEN_CAP = 20


class Condition:
    """One ablation cell: which speaker turns are read, and whether length counts.

    `sides`:
        "witness"  only the witness's own turns (what RQ1 asks about)
        "both"     witness + interrogator, i.e. what a loader reading the
                   released `tt_transcripts.transcript` column would featurize,
                   since that column interleaves both sides
    """

    def __init__(self, name: str, sides: str, drop_length: bool, note: str = "",
                 token_cap: int | None = None):
        self.name = name
        self.sides = sides
        self.drop_length = drop_length
        self.token_cap = token_cap
        self.note = note

    def as_dict(self) -> dict:
        return {"name": self.name, "sides": self.sides,
                "drop_length": self.drop_length, "token_cap": self.token_cap,
                "note": self.note}


CONDITIONS = [
    Condition("A0-full", "witness", False,
              "as run in a0_baselines_20260818_010835.json"),
    Condition("A0-witness", "witness", False,
              "witness turns only, I: lines stripped — identical to A0-full by "
              "construction, run separately to prove that rather than assert it"),
    Condition("A0-wit-nolen", "witness", True,
              "witness turns only AND every length-derived feature removed"),
    Condition("A0-bothsides", "both", False,
              "DIAGNOSTIC, not requested: what including interrogator turns "
              "would have scored, i.e. the artefact the transcript column invites"),
    Condition("A0-wit-nolen-capped", "witness", True,
              "CONTROL for the residual length channel that `drop_length` cannot "
              "reach: TF-IDF has no explicit length features, so dropping them is "
              "a no-op for it, yet a longer document still activates MORE terms "
              "after L2 normalisation. Truncating every witness side to a fixed "
              "token budget equalises that channel directly.",
              token_cap=TOKEN_CAP),
]




# ---------------------------------------------------------------------------
# text variants (pure)
# ---------------------------------------------------------------------------

def dialogue_text(dialogue: dict, variant: str, sides: str = "witness",
                  token_cap: int | None = None) -> str:
    """The witness side of one conversation, under one normalisation variant.

    raw
        Witness messages as stored (whitespace already collapsed by the
        canonical loader, which is harness formatting, not writing style).
    nostub
        Anonymisation placeholders (`<NAME>`, `[LOCATION]`, ...) removed. These
        tokens were written by the authors' GPT-4o redaction pass, not by the
        witness. Their presence encodes "this speaker named a person or place",
        which a detector can exploit without learning anything about writing.
    nostub_nochanged
        Additionally drops every message the anonymiser rewrote
        (`is_changed = TRUE`). Those messages contain text partly authored by
        GPT-4o inside BOTH classes, so they contaminate the human class with
        machine prose and the AI class with a second machine's prose.

    Only the WITNESS side is used. The interrogator is the same person in both
    conversations of a game, and the question the arm asks is about the writer
    whose identity is in doubt.
    """
    keep = {"W"} if sides == "witness" else {"W", "I"}
    turns = [t for t in dialogue["turns"] if t["role"] in keep]
    if variant == "nostub_nochanged":
        turns = [t for t in turns if not t["is_changed"]]
    parts = []
    for turn in turns:
        text = turn["content"]
        if variant in ("nostub", "nostub_nochanged"):
            text = build_canonical.strip_placeholders(text)
        if text:
            parts.append(text)
    if token_cap is not None:
        capped, used = [], 0
        for part in parts:
            words = part.split()
            if used + len(words) >= token_cap:
                capped.append(" ".join(words[: token_cap - used]))
                break
            capped.append(part)
            used += len(words)
        parts = [c for c in capped if c]
    return "\n".join(parts)


def dialogue_messages(dialogue: dict, variant: str,
                      sides: str = "witness",
                      token_cap: int | None = None) -> list[str]:
    text = dialogue_text(dialogue, variant, sides, token_cap)
    return [m for m in text.split("\n") if m]


# ---------------------------------------------------------------------------
# features (pure)
# ---------------------------------------------------------------------------

def length_features(messages: list[str]) -> dict[str, float]:
    chars = [len(m) for m in messages]
    words = [len(m.split()) for m in messages]
    total_c, total_w, n = sum(chars), sum(words), len(messages)
    mean_c = total_c / n if n else 0.0
    return {
        "n_messages": float(n),
        "total_chars": float(total_c),
        "total_words": float(total_w),
        "mean_chars": mean_c,
        "mean_words": (total_w / n) if n else 0.0,
        "max_chars": float(max(chars)) if chars else 0.0,
        "sd_chars": (
            math.sqrt(sum((c - mean_c) ** 2 for c in chars) / n) if n else 0.0
        ),
        "frac_long": (sum(1 for c in chars if c > 100) / n) if n else 0.0,
        "frac_short": (sum(1 for c in chars if c <= 15) / n) if n else 0.0,
        "empty": 1.0 if n == 0 else 0.0,
    }


def punctuation_features(messages: list[str]) -> dict[str, float]:
    text = " ".join(messages)
    n_chars = max(len(text), 1)
    n_msg = max(len(messages), 1)
    letters = [c for c in text if c.isalpha()]
    feats = {
        f"punct_{name}": 1000.0 * text.count(ch) / n_chars
        for name, ch in PUNCT_CLASSES.items()
    }
    feats.update({
        "upper_rate": (sum(1 for c in letters if c.isupper()) / len(letters))
                      if letters else 0.0,
        "digit_rate": 1000.0 * sum(1 for c in text if c.isdigit()) / n_chars,
        "space_rate": 1000.0 * text.count(" ") / n_chars,
        "ellipsis_rate": 1000.0 * text.count("..") / n_chars,
        "frac_start_lower": sum(
            1 for m in messages if m[:1].islower()) / n_msg,
        "frac_no_terminal": sum(
            1 for m in messages if m and m[-1] not in ".?!") / n_msg,
        "frac_has_question": sum(1 for m in messages if "?" in m) / n_msg,
        "empty": 1.0 if not messages else 0.0,
    })
    return feats


def function_word_features(messages: list[str]) -> dict[str, float]:
    tokens = TOKEN_RE.findall(" ".join(messages).lower())
    n = max(len(tokens), 1)
    counts = Counter(tokens)
    feats = {f"fw_{w}": 1000.0 * counts.get(w, 0) / n for w in FUNCTION_WORDS}
    feats["fw_type_token"] = len(counts) / n
    feats["empty"] = 1.0 if not tokens else 0.0
    return feats


def tfidf_terms(messages: list[str]) -> list[str]:
    """Word unigrams + bigrams, lowercased."""
    tokens = TOKEN_RE.findall(" ".join(messages).lower())
    terms = list(tokens)
    terms += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return terms


# ---------------------------------------------------------------------------
# logistic regression (pure, unit-tested)
# ---------------------------------------------------------------------------

def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Small dense systems only."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[col][col] += 1e-9
            pivot = col
        aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = 1.0 / aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] * inv
            if factor:
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def fit_dense_logistic(X: list[list[float]], y: list[int], l2: float = 1.0,
                       iters: int = 25, tol: float = 1e-8) -> list[float]:
    """Ridge-penalised logistic regression by IRLS. Returns [bias, *weights].

    IRLS is used rather than gradient descent because these feature sets are
    low-dimensional (<= ~90), so one Newton step costs a small dense solve and
    convergence takes a handful of iterations with no learning rate to tune.
    """
    n_features = len(X[0]) if X else 0
    dim = n_features + 1
    beta = [0.0] * dim
    rows = [[1.0] + row for row in X]
    for _ in range(iters):
        hessian = [[0.0] * dim for _ in range(dim)]
        gradient = [0.0] * dim
        for row, label in zip(rows, y):
            p = sigmoid(sum(b * v for b, v in zip(beta, row)))
            w = max(p * (1 - p), 1e-6)
            resid = label - p
            for i in range(dim):
                vi = row[i]
                if vi == 0.0:
                    continue
                gradient[i] += resid * vi
                wv = w * vi
                hrow = hessian[i]
                for j in range(dim):
                    hrow[j] += wv * row[j]
        for i in range(1, dim):            # ridge; bias unpenalised
            hessian[i][i] += l2
            gradient[i] -= l2 * beta[i]
        step = solve(hessian, gradient)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < tol:
            break
    return beta


def predict_dense(beta: list[float], row: list[float]) -> float:
    return sigmoid(beta[0] + sum(b * v for b, v in zip(beta[1:], row)))


def fit_sparse_logistic(rows: list[list[tuple[int, float]]], y: list[int],
                        dim: int, l2: float = 1.0, iters: int = 250,
                        lr: float = 2.0, momentum: float = 0.9,
                        tol: float = 1e-7) -> tuple[list[float], float, dict]:
    """L2 logistic regression on sparse rows by full-batch GD with momentum.

    Rows are L2-normalised TF-IDF vectors, so the gradient is well scaled and a
    fixed step works. Returns (weights, bias, diagnostics). Diagnostics are
    returned rather than printed so the write-up can show that every fold
    actually converged instead of asserting it.
    """
    w = [0.0] * dim
    velocity = [0.0] * dim
    bias = 0.0
    bias_velocity = 0.0
    n = max(len(rows), 1)
    prev_obj = float("inf")
    obj = float("inf")
    used = 0
    for used in range(1, iters + 1):
        grad: dict[int, float] = defaultdict(float)
        grad_bias = 0.0
        loss = 0.0
        for row, label in zip(rows, y):
            z = bias + sum(w[i] * v for i, v in row)
            p = sigmoid(z)
            loss += -(math.log(p + 1e-12) if label else math.log(1 - p + 1e-12))
            resid = label - p
            grad_bias += resid
            for i, v in row:
                grad[i] += resid * v
        obj = loss / n + 0.5 * l2 * sum(x * x for x in w) / n
        bias_velocity = momentum * bias_velocity + grad_bias / n
        bias += lr * bias_velocity
        for i, g in grad.items():
            velocity[i] = momentum * velocity[i] + (g / n - l2 * w[i] / n)
            w[i] += lr * velocity[i]
        # decay-free ridge on untouched coordinates
        if l2:
            for i in range(dim):
                if i not in grad and w[i]:
                    velocity[i] = momentum * velocity[i] - l2 * w[i] / n
                    w[i] += lr * velocity[i]
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj
    return w, bias, {"iterations": used, "final_objective": round(obj, 6)}


# ---------------------------------------------------------------------------
# metrics (pure, unit-tested)
# ---------------------------------------------------------------------------

def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC with tie handling (Mann-Whitney U)."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier(probs: list[float], labels: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / max(len(probs), 1)


def log_loss(probs: list[float], labels: list[int]) -> float:
    total = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, 1e-12), 1 - 1e-12)
        total += -(math.log(p) if y else math.log(1 - p))
    return total / max(len(probs), 1)


def reliability(probs: list[float], labels: list[int], bins: int = 10) -> list[dict]:
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, y in zip(probs, labels):
        idx = min(int(p * bins), bins - 1)
        buckets[idx].append((p, y))
    out = []
    for idx in range(bins):
        items = buckets.get(idx, [])
        if not items:
            out.append({"bin": idx, "lo": idx / bins, "hi": (idx + 1) / bins,
                        "n": 0, "mean_p": None, "observed": None})
            continue
        out.append({
            "bin": idx, "lo": idx / bins, "hi": (idx + 1) / bins,
            "n": len(items),
            "mean_p": round(sum(p for p, _ in items) / len(items), 4),
            "observed": round(sum(y for _, y in items) / len(items), 4),
        })
    return out


def ece(probs: list[float], labels: list[int], bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    rows = reliability(probs, labels, bins)
    n = len(probs)
    return sum(
        (r["n"] / n) * abs(r["mean_p"] - r["observed"])
        for r in rows if r["n"]
    ) if n else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs.

    Ignores clustering, so it is anti-conservative here and is reported only
    alongside the clustered bootstrap interval, which is the actual inference.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

class Detector:
    """Fit on the training dialogues of a fold, predict p(AI) for held-out ones."""

    name = "base"
    trained = True
    # Set by `make_detectors(condition)`.
    sides = "witness"
    drop_length = False
    token_cap = None
    # Conversation label to choose when the two sides tie. None = split the
    # credit. Only the majority baseline sets this; a text detector must not
    # break ties on slot position, which is not a property of the writing.
    tie_break: str | None = None

    def fit(self, train: list[dict], variant: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict(self, dialogues: list[dict], variant: str) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}


class ConstantDetector(Detector):
    """`random`: p(AI) = 0.5 for every dialogue.

    At the game level both sides tie, so the pairwise choice scores 0.5 per
    game — exactly chance, by construction rather than by sampling.
    """

    name = "random"
    trained = False

    def fit(self, train, variant):
        self.p = 0.5

    def predict(self, dialogues, variant):
        return [self.p] * len(dialogues)


class MajorityDetector(Detector):
    """`majority`: the trivial learned baseline.

    Dialogue level: constant p(AI) = the training fold's AI rate (0.5 by
    construction here, since every game contributes one of each).

    Game level: always answer the training fold's majority human POSITION. That
    is the honest majority rule for a forced-choice A/B task and it is not 50% —
    the human sat in slot A in 577 of 1,140 games — so it is a slightly stronger
    baseline than chance and is the one P1 pairs against.

    The position rule is applied as an explicit TIE-BREAK, not by nudging the
    emitted probability. Nudging would push the slot-position prior into the
    dialogue-level AUROC (measured: 0.5148 instead of 0.5000), which would
    report a property of the seating arrangement as if it were a text signal.
    """

    name = "majority"

    def fit(self, train, variant):
        n_ai = sum(1 for d in train if not d["is_human"])
        self.p = n_ai / len(train) if train else 0.5
        positions = Counter(
            d["conversation_label"] for d in train if d["is_human"]
        )
        self.tie_break = positions.most_common(1)[0][0] if positions else "A"

    def predict(self, dialogues, variant):
        return [self.p] * len(dialogues)

    def diagnostics(self):
        return {"majority_human_position": self.tie_break}


class DenseDetector(Detector):
    """Standardised dense features + ridge-IRLS logistic regression."""

    def __init__(self, name: str, extractor, l2: float = 1.0):
        self.name = name
        self.extractor = extractor
        self.l2 = l2
        self.keys: list[str] = []
        self.mean: list[float] = []
        self.sd: list[float] = []
        self.beta: list[float] = []

    def _rows(self, dialogues, variant):
        feats = [self.extractor(dialogue_messages(d, variant, self.sides, self.token_cap))
                 for d in dialogues]
        return [[f[k] for k in self.keys] for f in feats]

    def fit(self, train, variant):
        feats = [self.extractor(dialogue_messages(d, variant, self.sides, self.token_cap))
                 for d in train]
        self.keys = sorted(feats[0])
        if self.drop_length:
            self.keys = [k for k in self.keys if k not in LENGTH_DERIVED]
        raw = [[f[k] for k in self.keys] for f in feats]
        n = len(raw)
        self.mean = [sum(r[i] for r in raw) / n for i in range(len(self.keys))]
        self.sd = []
        for i in range(len(self.keys)):
            var = sum((r[i] - self.mean[i]) ** 2 for r in raw) / n
            self.sd.append(math.sqrt(var) if var > 1e-12 else 1.0)
        X = [[(v - m) / s for v, m, s in zip(r, self.mean, self.sd)] for r in raw]
        y = [0 if d["is_human"] else 1 for d in train]
        self.beta = fit_dense_logistic(X, y, l2=self.l2)

    def predict(self, dialogues, variant):
        rows = self._rows(dialogues, variant)
        return [
            predict_dense(
                self.beta,
                [(v - m) / s for v, m, s in zip(r, self.mean, self.sd)],
            )
            for r in rows
        ]

    def diagnostics(self):
        return {"n_features": len(self.keys),
                "features_removed_as_length": self.drop_length,
                "degenerate_no_features": len(self.keys) == 0}


class TfidfDetector(Detector):
    """TF-IDF (word 1+2-grams) + L2 logistic regression, fit per fold."""

    name = "tfidf_lr"

    def __init__(self, min_df: int = 3, max_features: int = 6000, l2: float = 1.0):
        self.min_df = min_df
        self.max_features = max_features
        self.l2 = l2
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.w: list[float] = []
        self.bias = 0.0
        self.diag: dict = {}

    def _vector(self, terms: list[str]) -> list[tuple[int, float]]:
        counts = Counter(t for t in terms if t in self.vocab)
        if not counts:
            return []
        vec = [
            (self.vocab[t], (1.0 + math.log(c)) * self.idf[self.vocab[t]])
            for t, c in counts.items()
        ]
        norm = math.sqrt(sum(v * v for _, v in vec)) or 1.0
        return [(i, v / norm) for i, v in vec]

    def fit(self, train, variant):
        docs = [tfidf_terms(dialogue_messages(d, variant, self.sides, self.token_cap))
                for d in train]
        df = Counter()
        for terms in docs:
            df.update(set(terms))
        kept = [t for t, c in df.items() if c >= self.min_df]
        kept.sort(key=lambda t: (-df[t], t))
        kept = kept[: self.max_features]
        self.vocab = {t: i for i, t in enumerate(sorted(kept))}
        n_docs = len(docs)
        self.idf = [0.0] * len(self.vocab)
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n_docs) / (1 + df[t])) + 1.0
        rows = [self._vector(terms) for terms in docs]
        y = [0 if d["is_human"] else 1 for d in train]
        self.w, self.bias, self.diag = fit_sparse_logistic(
            rows, y, dim=len(self.vocab), l2=self.l2
        )
        self.diag["vocab"] = len(self.vocab)

    def predict(self, dialogues, variant):
        out = []
        for d in dialogues:
            vec = self._vector(
                tfidf_terms(dialogue_messages(d, variant, self.sides, self.token_cap)))
            out.append(sigmoid(self.bias + sum(self.w[i] * v for i, v in vec)))
        return out

    def diagnostics(self):
        return self.diag


def make_detectors(condition: Condition | None = None) -> list[Detector]:
    condition = condition or CONDITIONS[0]
    detectors = [
        ConstantDetector(),
        MajorityDetector(),
        DenseDetector("length", length_features),
        DenseDetector("punctuation", punctuation_features),
        DenseDetector("function_words", function_word_features, l2=5.0),
        TfidfDetector(),
    ]
    for detector in detectors:
        detector.sides = condition.sides
        detector.drop_length = condition.drop_length
        detector.token_cap = condition.token_cap
    return detectors


# ---------------------------------------------------------------------------
# cross-fitting
# ---------------------------------------------------------------------------

def cross_fit(dialogues: list[dict], games: list[dict], variant: str,
              condition: Condition | None = None) -> dict:
    """Leave-one-component-out over train+dev. Returns p(AI) per dialogue."""
    by_game = defaultdict(dict)
    for d in dialogues:
        by_game[d["game_id"]][d["conversation_label"]] = d

    eval_games = [g for g in games if g["split"] in ("train", "dev")]
    components = sorted({g["component"] for g in eval_games})

    predictions: dict[str, dict[str, list[float]]] = defaultdict(dict)
    diagnostics: dict[str, list[dict]] = defaultdict(list)
    tie_breaks: dict[str, str | None] = {}

    for fold, component in enumerate(components):
        train_games = [g for g in eval_games if g["component"] != component]
        held_games = [g for g in eval_games if g["component"] == component]
        train_dialogues = [
            by_game[g["game_id"]][lbl] for g in train_games for lbl in ("A", "B")
        ]
        held_dialogues = [
            by_game[g["game_id"]][lbl] for g in held_games for lbl in ("A", "B")
        ]
        for detector in make_detectors(condition):
            start = time.time()
            detector.fit(train_dialogues, variant)
            probs = detector.predict(held_dialogues, variant)
            for d, p in zip(held_dialogues, probs):
                predictions[detector.name][d["example_id"]] = p
            tie_breaks[detector.name] = detector.tie_break
            diag = dict(detector.diagnostics())
            diag.update({
                "fold": fold, "component": component,
                "n_train": len(train_dialogues), "n_held": len(held_dialogues),
                "seconds": round(time.time() - start, 2),
            })
            diagnostics[detector.name].append(diag)

    return {
        "predictions": {k: dict(v) for k, v in predictions.items()},
        "tie_breaks": tie_breaks,
        "diagnostics": {k: v for k, v in diagnostics.items()},
        "folds": len(components),
        "components": components,
        "eval_games": [g["game_id"] for g in eval_games],
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_games(prob_of: dict[str, float], games: list[dict],
                by_game: dict, tie_break: str | None = None) -> dict:
    """Per-game pairwise outcome plus per-dialogue probabilities.

    `tie_break` is the conversation label to answer when the two sides carry
    identical probabilities. Only the majority baseline supplies one; every text
    detector leaves ties at half credit. Tied games keep `q = 0.5`, so a
    constant predictor scores Brier 0.25 — the no-information reference P2 is
    read against.
    """
    game_correct: list[float] = []
    game_strict: list[int] = []
    game_prob_a_human: list[float] = []
    game_label_a_human: list[int] = []
    ties = 0
    dialogue_probs: list[float] = []
    dialogue_labels: list[int] = []

    for g in games:
        a = by_game[g["game_id"]]["A"]
        b = by_game[g["game_id"]]["B"]
        pa, pb = prob_of[a["example_id"]], prob_of[b["example_id"]]
        human_a = 1.0 - pa
        human_b = 1.0 - pb
        truth_a = 1 if g["human_conversation_label"] == "A" else 0
        denom = human_a + human_b
        q = human_a / denom if denom > 1e-12 else 0.5
        game_prob_a_human.append(q)
        game_label_a_human.append(truth_a)
        if abs(human_a - human_b) < 1e-12:
            ties += 1
            if tie_break is not None:
                hit = 1.0 if (tie_break == "A") == bool(truth_a) else 0.0
                game_correct.append(hit)
                game_strict.append(int(hit))
            else:
                game_correct.append(0.5)
                game_strict.append(0)
        else:
            picked_a = human_a > human_b
            hit = 1.0 if picked_a == bool(truth_a) else 0.0
            game_correct.append(hit)
            game_strict.append(int(hit))
        for d, p in ((a, pa), (b, pb)):
            dialogue_probs.append(p)
            dialogue_labels.append(0 if d["is_human"] else 1)

    return {
        "game_correct": game_correct,
        "game_strict": game_strict,
        "game_prob_a_human": game_prob_a_human,
        "game_label_a_human": game_label_a_human,
        "ties": ties,
        "dialogue_probs": dialogue_probs,
        "dialogue_labels": dialogue_labels,
    }


def metrics_from(scored: dict, index: list[int]) -> dict:
    """All headline metrics restricted to a (possibly resampled) game index."""
    gc = [scored["game_correct"][i] for i in index]
    gq = [scored["game_prob_a_human"][i] for i in index]
    gy = [scored["game_label_a_human"][i] for i in index]
    dp, dl = [], []
    for i in index:
        dp.extend(scored["dialogue_probs"][2 * i: 2 * i + 2])
        dl.extend(scored["dialogue_labels"][2 * i: 2 * i + 2])
    return {
        "game_accuracy": sum(gc) / len(gc) if gc else float("nan"),
        "game_brier": brier(gq, gy),
        "dialogue_auroc": auroc(dp, dl),
        "dialogue_brier": brier(dp, dl),
        "dialogue_log_loss": log_loss(dp, dl),
    }


METRIC_KEYS = ("game_accuracy", "game_brier", "dialogue_auroc",
               "dialogue_brier", "dialogue_log_loss")


# ---------------------------------------------------------------------------
# clustered bootstrap
# ---------------------------------------------------------------------------

def cluster_units(games: list[dict]) -> dict[str, dict[str, list[int]]]:
    """unit name -> cluster id -> game positions."""
    units: dict[str, dict[str, list[int]]] = {
        "game": defaultdict(list),
        "interrogator": defaultdict(list),
        "human_witness": defaultdict(list),
        "component": defaultdict(list),
    }
    for i, g in enumerate(games):
        units["game"][g["game_id"]].append(i)
        units["interrogator"][g["interrogator_user_id"] or f"?g{i}"].append(i)
        units["human_witness"][g["human_witness_user_id"] or f"?g{i}"].append(i)
        units["component"][str(g["component"])].append(i)
    return {k: dict(v) for k, v in units.items()}


def bootstrap_intervals(scored_by_detector: dict[str, dict],
                        games: list[dict], n_boot: int, seed: int) -> dict:
    """Percentile CIs for every metric, every detector, under every unit.

    One resample is shared across detectors and metrics, so paired differences
    (P1) are computed on the same replicate as the levels they come from.
    """
    units = cluster_units(games)
    names = sorted(scored_by_detector)
    baseline = "majority"
    out: dict[str, dict] = {}

    for unit_name, clusters in units.items():
        keys = sorted(clusters)
        rng = random.Random(seed)
        draws: dict[str, dict[str, list[float]]] = {
            n: {k: [] for k in METRIC_KEYS} for n in names
        }
        diffs: dict[str, list[float]] = {n: [] for n in names}
        for _ in range(n_boot):
            index: list[int] = []
            for _ in range(len(keys)):
                index.extend(clusters[keys[rng.randrange(len(keys))]])
            if not index:
                continue
            base_acc = None
            per_detector = {}
            for name in names:
                m = metrics_from(scored_by_detector[name], index)
                per_detector[name] = m
                if name == baseline:
                    base_acc = m["game_accuracy"]
            for name in names:
                for key in METRIC_KEYS:
                    draws[name][key].append(per_detector[name][key])
                if base_acc is not None:
                    diffs[name].append(per_detector[name]["game_accuracy"] - base_acc)

        def pct(values: list[float], q: float) -> float:
            clean = sorted(v for v in values if not math.isnan(v))
            if not clean:
                return float("nan")
            pos = min(len(clean) - 1, max(0, int(round(q * (len(clean) - 1)))))
            return clean[pos]

        out[unit_name] = {
            "n_clusters": len(keys),
            "detectors": {
                name: {
                    **{
                        key: {"lo": round(pct(draws[name][key], 0.025), 4),
                              "hi": round(pct(draws[name][key], 0.975), 4)}
                        for key in METRIC_KEYS
                    },
                    "game_accuracy_diff_vs_majority": {
                        "lo": round(pct(diffs[name], 0.025), 4),
                        "hi": round(pct(diffs[name], 0.975), 4),
                    },
                }
                for name in names
            },
        }
    return out


def widen(a: dict, b: dict) -> dict:
    """The wider of two intervals — the conservative crossed-cluster stand-in."""
    return a if (a["hi"] - a["lo"]) >= (b["hi"] - b["lo"]) else b


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_variant(dialogues: list[dict], games: list[dict], variant: str,
                n_boot: int, condition: Condition | None = None) -> dict:
    condition = condition or CONDITIONS[0]
    fit = cross_fit(dialogues, games, variant, condition)
    by_game = defaultdict(dict)
    for d in dialogues:
        by_game[d["game_id"]][d["conversation_label"]] = d
    eval_games = [g for g in games if g["split"] in ("train", "dev")]

    scored = {
        name: score_games(probs, eval_games, by_game, fit["tie_breaks"].get(name))
        for name, probs in fit["predictions"].items()
    }
    index = list(range(len(eval_games)))
    point = {name: metrics_from(s, index) for name, s in scored.items()}

    # --- P1: paired accuracy against the majority baseline -------------------
    p1 = {}
    base = scored["majority"]
    for name, s in scored.items():
        b = sum(1 for x, y in zip(s["game_strict"], base["game_strict"])
                if x == 1 and y == 0)
        c = sum(1 for x, y in zip(s["game_strict"], base["game_strict"])
                if x == 0 and y == 1)
        p1[name] = {
            "game_accuracy": round(point[name]["game_accuracy"], 4),
            "baseline_game_accuracy": round(point["majority"]["game_accuracy"], 4),
            "difference_pp": round(
                100 * (point[name]["game_accuracy"] - point["majority"]["game_accuracy"]), 2),
            "discordant_b": b, "discordant_c": c,
            "discordance_rate": round((b + c) / len(eval_games), 4),
            "mcnemar_exact_p_unclustered": mcnemar_exact(b, c),
            "ties": scored[name]["ties"],
        }

    # --- P2: out-of-fold calibration ----------------------------------------
    p2 = {
        name: {
            "dialogue_brier": round(point[name]["dialogue_brier"], 4),
            "dialogue_log_loss": round(point[name]["dialogue_log_loss"], 4),
            "dialogue_ece": round(ece(s["dialogue_probs"], s["dialogue_labels"]), 4),
            "game_brier": round(point[name]["game_brier"], 4),
            "reliability_dialogue": reliability(s["dialogue_probs"], s["dialogue_labels"]),
        }
        for name, s in scored.items()
    }

    intervals = bootstrap_intervals(scored, eval_games, n_boot, SEED)
    # participant = wider of interrogator / human_witness, per metric
    participant = {"n_clusters": None, "detectors": {}}
    for name in sorted(scored):
        row = {}
        for key in list(METRIC_KEYS) + ["game_accuracy_diff_vs_majority"]:
            row[key] = widen(
                intervals["interrogator"]["detectors"][name][key],
                intervals["human_witness"]["detectors"][name][key],
            )
        participant["detectors"][name] = row
    participant["n_clusters"] = (
        f"{intervals['interrogator']['n_clusters']} interrogators / "
        f"{intervals['human_witness']['n_clusters']} human witnesses (max-of-marginals)"
    )
    intervals["participant"] = participant

    # --- descriptive: accuracy by witness system (NOT a frozen contrast) ------
    by_system: dict[str, dict[str, float]] = {}
    for name, s in scored.items():
        rows = defaultdict(list)
        for g, hit in zip(eval_games, s["game_correct"]):
            rows[g["witness_system"]].append(hit)
        by_system[name] = {
            sysname: round(sum(v) / len(v), 4) for sysname, v in sorted(rows.items())
        }
    system_counts = dict(sorted(Counter(g["witness_system"] for g in eval_games).items()))

    # --- descriptive: accuracy on subsets that remove known easy cases -------
    # No tests, no intervals, no gate. These exist because a single pooled
    # accuracy hides which games are carrying it.
    subsets = {
        "all": lambda g: True,
        "excl_eliza": lambda g: g["witness_system"] != "eliza",
        "excl_empty_side": lambda g: not g["empty_side"],
        "excl_eliza_and_empty": lambda g: g["witness_system"] != "eliza" and not g["empty_side"],
        "persona_only": lambda g: g["witness_system"].endswith("_quinn"),
        "minimal_only": lambda g: g["witness_system"].endswith("_minimal"),
    }
    by_subset: dict[str, dict[str, float]] = {}
    subset_counts = {}
    for label, keep in subsets.items():
        idx = [i for i, g in enumerate(eval_games) if keep(g)]
        subset_counts[label] = len(idx)
        for name, s in scored.items():
            by_subset.setdefault(name, {})[label] = round(
                sum(s["game_correct"][i] for i in idx) / len(idx), 4
            ) if idx else float("nan")

    # --- descriptive: what the TF-IDF model actually keys on ------------------
    # Fit once on all of train+dev (NOT cross-fitted; this is an inspection of
    # the learned weights, not a score). Vocabulary items only — single tokens
    # and bigrams are lexical statistics, not participant utterances, so this
    # stays inside registry §5's bar on republishing transcript text.
    train_all = [by_game[g["game_id"]][lbl] for g in eval_games for lbl in ("A", "B")]
    inspector = TfidfDetector()
    inspector.sides = condition.sides
    inspector.drop_length = condition.drop_length
    inspector.token_cap = condition.token_cap
    inspector.fit(train_all, variant)
    inverse = {i: t for t, i in inspector.vocab.items()}
    order = sorted(range(len(inspector.w)), key=lambda i: -inspector.w[i])
    top_terms = {
        "toward_ai": [[inverse[i], round(inspector.w[i], 3)] for i in order[:30]],
        "toward_human": [[inverse[i], round(inspector.w[i], 3)] for i in order[-30:]],
        "note": "fit on all of train+dev for inspection only; not used for any score",
    }

    # --- descriptive: harness asymmetries the detectors can exploit ----------
    empty = Counter()
    msg_counts = defaultdict(list)
    for g in eval_games:
        for lbl in ("A", "B"):
            d = by_game[g["game_id"]][lbl]
            key = "human" if d["is_human"] else "ai"
            msg_counts[key].append(d["n_witness_messages"])
            if d["n_witness_messages"] == 0:
                empty[key] += 1
    harness = {
        "empty_witness_side_dialogues": dict(empty),
        "mean_witness_messages": {
            k: round(sum(v) / len(v), 3) for k, v in msg_counts.items()
        },
        "note": "AI witnesses emitted more messages and fell silent less often. "
                "Per the Stage A inspection §10 item 6, AI turns were released "
                "with an artificial per-character delay, so message count is "
                "partly a property of the harness rather than of the writer.",
    }

    surviving = {}
    for fam, extractor in (("length", length_features),
                           ("punctuation", punctuation_features),
                           ("function_words", function_word_features)):
        keys = sorted(extractor(["probe text here"]))
        kept = [k for k in keys if not (condition.drop_length and k in LENGTH_DERIVED)]
        surviving[fam] = {"total": len(keys), "kept": len(kept),
                          "dropped": sorted(set(keys) - set(kept))}

    return {
        "variant": variant,
        "condition": condition.as_dict(),
        "surviving_features": surviving,
        "folds": fit["folds"],
        "n_eval_games": len(eval_games),
        "point_estimates": {n: {k: round(v, 4) for k, v in m.items()}
                            for n, m in point.items()},
        "P1_paired_vs_majority": p1,
        "P2_calibration": p2,
        "intervals": intervals,
        "descriptive_accuracy_by_witness_system": by_system,
        "witness_system_game_counts": system_counts,
        "descriptive_accuracy_by_subset": by_subset,
        "subset_game_counts": subset_counts,
        "descriptive_top_tfidf_terms": top_terms,
        "descriptive_harness_asymmetry": harness,
        "fit_diagnostics": fit["diagnostics"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DEFAULT)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="ablation cells to run; default is the single "
                             "as-run cell A0-full")
    parser.add_argument("--ablation", action="store_true",
                        help="run the full three-way ablation (plus the "
                             "both-sides diagnostic) on the raw variant only")
    parser.add_argument("--tag", default="", help="suffix for the output filename")
    args = parser.parse_args(argv)

    if args.ablation:
        conditions = list(CONDITIONS)
        args.variants = ["raw"]
    elif args.conditions:
        by_name = {c.name: c for c in CONDITIONS}
        conditions = [by_name[n] for n in args.conditions]
    else:
        conditions = [CONDITIONS[0]]

    dialogues, games, manifest = build_canonical.load()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_by": "v2/scripts/track_a_a0.py",
        "generated_at_utc": stamp,
        "arm": "A0",
        "not_implemented": {
            "A1": "Inverse Turing Bench reproduction. Its 557 games are 48.9% of "
                  "the corpus and biased long (90.5 vs 65.0 mean tokens); the "
                  "A2-vs-A1 contrast was demoted to interval reporting. Needs its "
                  "own session with the coverage bias stated.",
            "A2": "Classifier head on a temporally clean frozen representation "
                  "model. The frozen checkpoint is still an open design decision "
                  "(§19), so contrast P1 AS FROZEN is not evaluated here.",
        },
        "test_split": "UNTOUCHED (Gate 5, one shot)",
        "canonical": {
            "source_revision": manifest["source_revision"],
            "split_sha256": manifest["split_sha256"],
            "dialogues_sha256": manifest["dialogues_sha256"],
            "games_sha256": manifest["games_sha256"],
        },
        "bootstrap": {"replicates": args.bootstrap, "seed": SEED,
                      "method": "percentile, cluster resampled with replacement"},
        "conditions": {},
    }

    for condition in conditions:
        payload["conditions"][condition.name] = {"variants": {}}
        for variant in args.variants:
            started = time.time()
            print(f"[{condition.name} / {variant}] cross-fitting ...", flush=True)
            result = run_variant(dialogues, games, variant, args.bootstrap, condition)
            result["seconds"] = round(time.time() - started, 1)
            payload["conditions"][condition.name]["variants"][variant] = result
            for name, m in result["point_estimates"].items():
                print(f"  {name:16s} game_acc={m['game_accuracy']:.4f} "
                      f"auroc={m['dialogue_auroc']:.4f} brier={m['dialogue_brier']:.4f}",
                      flush=True)
            print(f"  ({result['seconds']}s)", flush=True)

    # --- normalisation deltas, within each condition -------------------------
    for cname, cblock in payload["conditions"].items():
        variants = cblock["variants"]
        if "raw" not in variants or len(variants) < 2:
            continue
        deltas = {}
        for variant in variants:
            if variant == "raw":
                continue
            deltas[f"raw_minus_{variant}"] = {
                name: {
                    key: round(variants["raw"]["point_estimates"][name][key]
                               - variants[variant]["point_estimates"][name][key], 4)
                    for key in METRIC_KEYS
                }
                for name in variants["raw"]["point_estimates"]
            }
        cblock["normalisation_deltas"] = deltas

    # --- the ablation table, if more than one condition ran ------------------
    if len(payload["conditions"]) > 1:
        table = {}
        for cname, cblock in payload["conditions"].items():
            pe = cblock["variants"]["raw"]["point_estimates"]
            ci = cblock["variants"]["raw"]["intervals"]["participant"]["detectors"]
            table[cname] = {
                name: {
                    "game_accuracy": pe[name]["game_accuracy"],
                    "ci_participant": [ci[name]["game_accuracy"]["lo"],
                                       ci[name]["game_accuracy"]["hi"]],
                    "dialogue_auroc": pe[name]["dialogue_auroc"],
                }
                for name in pe
            }
        payload["ablation_table"] = table

    tag = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"a0_baselines_{stamp}{tag}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
