#!/usr/bin/env python3
"""Bag-of-words baseline: is the detector modelling stance, or memorising words?

WHY THIS EXISTS. Cross-project finding from decision-traceability, 2026-08-07:
their per-layer probes scored 95-98% with healthy control-task selectivity
(+0.42 to +0.48), and were still worthless. A bag-of-words logistic regression
on the RAW QUERY STRING — no model at all — matched or beat them. The concept
was lexically marked and a layer-1 residual probe is barely past the embedding.

**Control-task selectivity did not catch this.** Hewitt & Liang controls use
random labels with matched statistics and no linguistic content, so a lexical
cue clears the control easily and the gap still looks healthy. Selectivity is
necessary, not sufficient.

So: before trusting ANY detector number on think-block sentences, fit a
bag-of-words classifier on the same hand-labelled data and report its
precision/recall alongside. If they are close, the detector is lexical —
whatever its architecture.

    python scripts/eval/lexical_baseline.py

Pure numpy, no sklearn, no GPU. Runs in seconds on a laptop.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conversation"))

POSITIVE = "conclusion"
_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], min_df: int = 2) -> dict[str, int]:
    df = Counter()
    for t in texts:
        df.update(set(tokenize(t)))
    # min_df=2 drops hapax terms, which otherwise let the model memorise single
    # sentences and inflate CV scores.
    # Filter BEFORE enumerating — indexing the unfiltered sequence produces
    # indices larger than the resulting vocab and overruns the feature matrix.
    kept = [w for w, c in sorted(df.items()) if c >= min_df]
    return {w: i for i, w in enumerate(kept)}


def featurize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    X = np.zeros((len(texts), len(vocab) + 1), dtype=np.float32)
    X[:, -1] = 1.0                       # bias
    for i, t in enumerate(texts):
        for w in set(tokenize(t)):
            j = vocab.get(w)
            if j is not None:
                X[i, j] = 1.0            # binary presence, not counts
    return X


def fit_logreg(X, y, l2=0.5, steps=2000, lr=1.0) -> np.ndarray:
    """Full-batch logistic regression with L2 and CLASS WEIGHTING.

    Class weighting is not optional here. The positive rate is ~10%, and an
    unweighted fit simply predicts the majority class for every row — which
    scores 0.000 precision and looks like "BoW is worse than the regex" when
    it actually means "the baseline never ran". A degenerate baseline is worse
    than no baseline, because it licenses a false conclusion.
    """
    w = np.zeros(X.shape[1], dtype=np.float32)
    n = len(y)
    pos = max(float(y.sum()), 1.0)
    sw = np.where(y > 0.5, n / (2 * pos), n / (2 * max(n - pos, 1.0))).astype(np.float32)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (sw * (p - y)) / n + l2 * w / n
        grad[-1] -= l2 * w[-1] / n       # do not regularise the bias
        w -= lr * grad
    return w


def auc(scores: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    """Rank-based AUC with a Hanley-McNeil standard error.

    The SE is not decoration. At ~23 positives the 95% CI is roughly +/-0.12,
    so a point estimate of 0.73 is consistent with anything from 0.61 to 0.85 —
    and 0.85 would materially weaken any claim that lexical methods cannot
    reach a given bar. Reporting the point estimate alone invites exactly the
    over-reading that a 4-positive AUROC produced in decision-traceability.
    """
    pos, neg = scores[gold > 0.5], scores[gold <= 0.5]
    n_p, n_n = len(pos), len(neg)
    if not n_p or not n_n:
        return float("nan"), float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    a = float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_p - 1) * (q1 - a * a)
           + (n_n - 1) * (q2 - a * a)) / (n_p * n_n)
    return a, float(np.sqrt(max(var, 0.0)))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default="data/probes/think_stance_labels_v1.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--annotator-glob",
                    default="results/analysis/agent2_labels_*.jsonl",
                    help="per-annotator label files. Used to compute the "
                         "ceiling: a low BoW score is equally consistent with "
                         "'needs semantics' and 'labels are noisy', and only "
                         "inter-annotator agreement distinguishes them.")
    ap.add_argument("--show-top", type=int, default=15,
                    help="most predictive words. If these are the same terms "
                         "the regex matches on, both are the same detector.")
    args = ap.parse_args()

    import conv_logging

    path = ROOT / args.labels
    if not path.exists():
        sys.exit(f"ERROR: no labels at {path}")
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("label") != "skip"]
    texts = [r["text"] for r in rows]
    y = np.array([1.0 if r["label"] == POSITIVE else 0.0 for r in rows], dtype=np.float32)

    n_pos = int(y.sum())
    print(f"\n  {len(rows)} labelled sentences, {n_pos} positive "
          f"({n_pos/len(rows):.1%})")
    if n_pos < 15:
        print("  WARNING: too few positives for a stable CV estimate.")

    # --- cross-validated bag of words ---------------------------------------
    # CV, not fit-and-score: the regex was never fitted to this data, so a
    # BoW model evaluated on its own training rows would be an unfair
    # comparison in the other direction.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    folds = np.array_split(order, args.folds)

    oof = np.zeros(len(rows), dtype=np.float64)   # out-of-fold scores
    for k in range(args.folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        vocab = build_vocab([texts[i] for i in tr])       # vocab from TRAIN only
        Xtr, Xte = featurize([texts[i] for i in tr], vocab), featurize(
            [texts[i] for i in te], vocab)
        w = fit_logreg(Xtr, y[tr])
        oof[te] = 1.0 / (1.0 + np.exp(-(Xte @ w)))

    gold_b = y > 0.5
    bow_auc, bow_se = auc(oof, y)

    # --- the regex detector on the same rows --------------------------------
    dtp = dfp = dfn = 0
    for r in rows:
        pred = bool(conv_logging._think_block_suspicious(r["text"]))
        gold = r["label"] == POSITIVE
        dtp += pred and gold
        dfp += pred and not gold
        dfn += gold and not pred
    d_p, d_r, d_f = prf(dtp, dfp, dfn)

    # Compare at MATCHED RECALL. Comparing at a fixed 0.5 threshold is
    # meaningless across a rule and a probabilistic model — the fair question is
    # "at the recall the regex achieves, which has better precision?"
    thresholds = np.unique(oof)[::-1]
    bow_p = bow_r = bow_f = 0.0
    for t in thresholds:
        pred = oof >= t
        tp_, fp_, fn_ = (int((pred & gold_b).sum()), int((pred & ~gold_b).sum()),
                         int((~pred & gold_b).sum()))
        p_, r_, f_ = prf(tp_, fp_, fn_)
        if r_ >= d_r:                  # first threshold reaching the regex's recall
            bow_p, bow_r, bow_f = p_, r_, f_
            break
    # NESTED threshold selection. Sweeping the threshold over the same scores
    # you then report is selection on the evaluation set — the identical error
    # as picking a probe layer by argmax over TEST accuracy. It biases the
    # ceiling upward, which for a claim of the form "no lexical method exceeds
    # X" is the direction that makes the claim look weaker than it is, but it
    # is still an untrustworthy number. Threshold is chosen on the TRAIN folds
    # and applied to the held-out fold.
    ntp = nfp = nfn = 0
    for k in range(args.folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        best_t, best_tf = 0.5, -1.0
        for t in np.unique(oof[tr]):
            pr = oof[tr] >= t
            g = gold_b[tr]
            f_ = prf(int((pr & g).sum()), int((pr & ~g).sum()),
                     int((~pr & g).sum()))[2]
            if f_ > best_tf:
                best_tf, best_t = f_, t
        pr, g = oof[te] >= best_t, gold_b[te]
        ntp += int((pr & g).sum()); nfp += int((pr & ~g).sum())
        nfn += int((~pr & g).sum())
    nested_p, nested_r, best_f = prf(ntp, nfp, nfn)

    print(f"\n{'='*64}")
    print(f"  {'detector':<30}{'prec':>8}{'recall':>8}{'F1':>8}")
    print("  " + "-" * 62)
    print(f"  {'regex (t_think_07)':<30}{d_p:>8.3f}{d_r:>8.3f}{d_f:>8.3f}")
    print(f"  {'BoW @ matched recall (CV)':<30}{bow_p:>8.3f}{bow_r:>8.3f}{bow_f:>8.3f}")
    print(f"  {'BoW @ nested-selected thr':<30}{nested_p:>8.3f}{nested_r:>8.3f}{best_f:>8.3f}")
    lo, hi = bow_auc - 1.96 * bow_se, bow_auc + 1.96 * bow_se
    print(f"\n  BoW AUC : {bow_auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   "
          f"(0.5 = no lexical signal)")
    print(f"  n positives = {n_pos} — the CI is wide BECAUSE of that, and the")
    print(f"  upper bound is what any 'lexical methods cannot reach X' claim")
    print(f"  must be argued against, not the point estimate.")

    # --- most predictive words ----------------------------------------------
    vocab = build_vocab(texts)
    w = fit_logreg(featurize(texts, vocab), y)
    inv = {i: t for t, i in vocab.items()}
    top = np.argsort(w[:-1])[::-1][:args.show_top]
    print(f"\n  most predictive words: "
          + ", ".join(f"{inv[int(i)]}" for i in top))

    # --- the ceiling: is 0.8 even reachable on these labels? ----------------
    import glob as _glob
    afiles = sorted(_glob.glob(str(ROOT / args.annotator_glob)))
    if len(afiles) >= 2:
        ann = {}
        for f in afiles:
            nm = Path(f).stem.split("_")[-1]
            ann[nm] = {json.loads(l)["id"]: json.loads(l)["label"]
                       for l in open(f, encoding="utf-8") if l.strip()}
        common = sorted(set.intersection(*[set(d) for d in ann.values()]))
        nms = sorted(ann)
        f1s = []
        for held in nms:
            others = [n for n in nms if n != held]
            tp_ = fp_ = fn_ = 0
            for i in common:
                votes = [ann[o][i] for o in others]
                g = votes.count(POSITIVE) > len(others) / 2
                pr = ann[held][i] == POSITIVE
                tp_ += pr and g; fp_ += pr and not g; fn_ += g and not pr
            f1s.append(prf(tp_, fp_, fn_)[2])
        ceiling = float(np.mean(f1s))
        print(f"  {'ANNOTATOR ceiling (LOAO)':<30}{'':>8}{'':>8}{ceiling:>8.3f}")
        print(f"\n  Ceiling from {len(nms)} annotators over {len(common)} sentences.")
        print(f"  These are LLM annotators sharing a rubric and priors, so this")
        print(f"  is an upper bound on agreement, not on human performance.")
        if ceiling < 0.85:
            print(f"  WARNING: at {ceiling:.2f} the labels themselves cap any")
            print(f"  detector near there. A 0.8 gate would be unreachable by")
            print(f"  ANY method — the gate is wrong, not the detector.")

    print(f"\n{'='*64}")
    # THREE readings, not two. The original two-branch version reported
    # "task is lexically separable" whenever BoW matched the detector — which
    # is right when both score HIGH and badly wrong when both score LOW.
    # Sherlock and decision-traceability landed in different branches of this
    # from the same diagnostic, which is the whole reason it is worth running.
    ceil = locals().get("ceiling")
    close = abs(bow_f - d_f) < 0.10
    if ceil is not None and ceil < 0.85:
        print("  LABELS ARE THE LIMIT. Annotators agree at "
              f"{ceil:.2f}, so no detector can reliably exceed that.")
        print("  Fix the label definition or the rubric before touching the")
        print("  detector; the gate is wrong, not the method.")
    elif bow_f >= 0.7 and close:
        print("  TASK IS LEXICAL. BoW matches the detector at a HIGH score, so")
        print("  the concept is carried by word presence and a good score is")
        print("  no evidence of understanding. Hold out the concept's")
        print("  linguistic carrier, not just its topic, and re-measure.")
        print("  (This is the decision-traceability failure mode: probes at")
        print("   95-98% that a bag of words matched.)")
    elif close:
        print(f"  TASK IS NOT LEXICAL. BoW ({bow_f:.3f}) and the detector "
              f"({d_f:.3f}) are")
        print(f"  both far below the annotator ceiling ({ceil:.3f} if shown).")
        print("  Word presence does not carry this concept, so no n-gram,")
        print("  TF-IDF or regex method will close the gap — including a")
        print("  'stance classifier' built on lexical features.")
        print("  A replacement must be SEMANTIC: sentence embeddings plus a")
        print("  classifier, or an LLM judge per sentence.")
    else:
        print(f"  The detector ({d_f:.3f}) beats BoW ({bow_f:.3f}), so it uses")
        print("  something beyond word presence. Necessary, not sufficient —")
        print("  keep reporting this baseline beside it.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
