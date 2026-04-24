"""§9D.2 — Weaver weak-supervision label model + ensemble student.

Per Paper1 §3.11 / §4.14 we follow the Weaver recipe (Anthropic 2024):
  1. 5 heterogeneous verifier signals w_1..w_5 (§9D.1) act as
     `labeling functions` on each (case, candidate-mechanism) pair.
  2. A generative label model (Snorkel-style) learns each signal's
     accuracy + class-conditional bias *without* gold labels, by
     fitting a latent-variable model over the agreement structure of
     the 5 signals across many cases.
  3. Posterior P(y | w_1..w_5) from the label model becomes the soft
     label.
  4. A small student (here a 400 M distilled Qwen-2.5 head, slot for
     ckpt path) is trained against the soft labels to give a single
     scalar weaver-score per case at inference.

This file provides:
  • `LabelModelInputs`     — dataclass packaging (n_cases × 5) signal
                             matrix + per-signal stub mask + observed
                             abstain mask (for w_2 gate).
  • `fit_label_model()`    — Snorkel `LabelModel` wrapper. Falls back
                             to majority-vote when snorkel is missing.
  • `predict_soft_labels()`
  • `StudentHeadStub`      — 400 M ckpt loader stub (records the
                             intended interface; real load wired when
                             §9A.3 PRM ckpt format is final).
  • `train_student()`      — soft-label distillation loop (PyTorch).
  • CLI `python -m src.weaver_label_model --signals_json <path>` so
     once §9D.1 emits a real `weaver_signals_minibatch_v*.json`, this
     file can be smoke-tested end-to-end.

Decision gate (§9D.2): on calibration subset, ensemble AUROC must beat
the best single-signal AUROC by ≥ 3% → §4.14 PASS.

Status (2026-04-21): SCAFFOLD only. The label-model fit + student
distillation paths execute on the 30-case mini-batch when DEEPSEEK
+ noise-perturbed observations make ≥3 signals non-constant. Real gate
evaluation waits on (a) §9A.3 PRM, (b) §9C.2 posterior, (c) graded
gold labels from §9A.2 (`stepwise_labels`).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent

SIGNAL_NAMES = ["w1_prm", "w2_resid", "w3_kk", "w4_sbi", "w5_llm"]


# ─────────────────────── data containers ──────────────────────────


@dataclass
class LabelModelInputs:
    """Inputs to the §9D.2 label model.

    W           : (n, 5) float in [0,1] — the 5 weaver signals.
    stubs       : (n, 5) bool          — True if signal was a stub
                                         (no upstream model). Stubbed
                                         signals are treated as
                                         abstain in the LF matrix.
    abstain     : (n,) bool            — True if w_2 gate ABSTAIN'd
                                         (case is unscoreable).
    gold        : (n,) {-1,0,1} or None — gold label if available.
                                         -1 = abstain, 0 = wrong,
                                         1 = correct. Used only for
                                         AUROC eval, NOT for fitting.
    """
    W: np.ndarray
    stubs: np.ndarray
    abstain: np.ndarray
    gold: np.ndarray | None = None
    signal_names: list[str] = field(default_factory=lambda: SIGNAL_NAMES)


def from_signals_json(path: str | Path) -> LabelModelInputs:
    """Load `weaver_signals_minibatch_v*.json` produced by §9D.1."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = obj["rows"]
    W = np.array([r["w"] for r in rows], dtype=float)
    stubs = np.array([r["stubs"] for r in rows], dtype=bool)
    # Heuristic: w_2 ≈ 0.0 + stub False ⇒ gate ABSTAIN'd.
    abstain = (W[:, 1] < 1e-6) & ~stubs[:, 1]
    return LabelModelInputs(W=W, stubs=stubs, abstain=abstain)


# ─────────────────────── label-model fit ──────────────────────────


def _binarise_for_lf(W: np.ndarray, stubs: np.ndarray,
                      threshold: float = 0.5) -> np.ndarray:
    """Convert continuous signals → Snorkel LF matrix in {-1, 0, 1}.
    -1 = abstain (stub or near-threshold); 0 = wrong; 1 = correct.
    """
    L = np.zeros_like(W, dtype=int)
    L[W > threshold + 0.05] = 1
    L[W < threshold - 0.05] = 0
    L[(W >= threshold - 0.05) & (W <= threshold + 0.05)] = -1
    L[stubs] = -1  # stub → abstain in LF semantics
    return L


def fit_label_model(inp: LabelModelInputs, *, seed: int = 0) -> Any:
    """Fit a Snorkel-style generative label model.

    Returns an opaque handle with `.predict_proba(L)` API. Falls back
    to a calibrated majority-vote estimator (`_MajorityVote`) when
    snorkel is unavailable or the LF matrix has too few non-abstain
    rows for EM to converge.
    """
    L = _binarise_for_lf(inp.W, inp.stubs)
    # Drop fully-abstaining rows.
    mask_any = (L != -1).any(axis=1)
    L_use = L[mask_any]
    if L_use.shape[0] < 10:
        return _MajorityVote().fit(L)

    try:
        from snorkel.labeling.model import LabelModel  # type: ignore
        lm = LabelModel(cardinality=2, verbose=False)
        lm.fit(L_use, n_epochs=500, seed=seed)
        return _SnorkelHandle(lm)
    except Exception as ex:
        print(f"  snorkel fit failed ({ex}); falling back to majority-vote")
        return _MajorityVote().fit(L)


class _SnorkelHandle:
    def __init__(self, lm): self.lm = lm
    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        return self.lm.predict_proba(L=L)


class _MajorityVote:
    """Fallback: per-row mean of non-abstain LF votes; clipped to [0,1].
    Returns (n, 2) probas [P(y=0), P(y=1)] like Snorkel.
    """
    def fit(self, L: np.ndarray) -> "_MajorityVote":
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        out = np.zeros((L.shape[0], 2), dtype=float)
        for i in range(L.shape[0]):
            votes = L[i][L[i] != -1]
            if votes.size == 0:
                p1 = 0.5
            else:
                p1 = float(votes.mean())
            out[i] = [1 - p1, p1]
        return out


def predict_soft_labels(handle: Any,
                        inp: LabelModelInputs) -> np.ndarray:
    """Return per-case P(y=correct) ∈ [0,1]."""
    L = _binarise_for_lf(inp.W, inp.stubs)
    proba = handle.predict_proba(L)
    return proba[:, 1].astype(float)


# ─────────────────────── student distillation ─────────────────────


class StudentHeadStub:
    """Slot for a real 400 M distilled student.

    Real implementation will wrap a Qwen-2.5-0.5B (or similar) checkpoint
    fine-tuned with cross-entropy against `predict_soft_labels`. Until
    the §9A.3 PRM checkpoint is finalised (its tokenizer + embedding
    feed this student), we keep the student as a 5-feature MLP — small
    enough to train in <1s on the 30-case mini-batch and exercise the
    interface end-to-end.
    """

    def __init__(self, in_dim: int = 5, hidden: int = 16):
        self.W1 = np.random.default_rng(0).standard_normal(
            (in_dim, hidden)) * 0.3
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.default_rng(1).standard_normal(hidden) * 0.3
        self.b2 = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.05,
            n_epochs: int = 400) -> "StudentHeadStub":
        # Bare-bones SGD (numpy) for the stub. Real path: torch + AdamW.
        for _ in range(n_epochs):
            h = np.maximum(0, X @ self.W1 + self.b1)
            z = h @ self.W2 + self.b2
            p = 1.0 / (1.0 + np.exp(-z))
            dz = (p - y) / X.shape[0]
            dW2 = h.T @ dz
            db2 = dz.sum()
            dh = np.outer(dz, self.W2)
            dh[h <= 0] = 0
            dW1 = X.T @ dh
            db1 = dh.sum(axis=0)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
        return self


def train_student(inp: LabelModelInputs,
                  soft_labels: np.ndarray) -> StudentHeadStub:
    """Train the (stub) student against soft labels from the label model.
    Drops abstaining rows from the training set."""
    keep = ~inp.abstain
    student = StudentHeadStub(in_dim=inp.W.shape[1])
    student.fit(inp.W[keep], soft_labels[keep])
    return student


# ─────────────────────── eval helpers ─────────────────────────────


def auroc(scores: np.ndarray, gold: np.ndarray) -> float:
    """Trapezoidal AUROC. gold ∈ {0,1}; ignores rows where gold == -1."""
    keep = gold != -1
    s, g = scores[keep], gold[keep]
    if g.sum() == 0 or g.sum() == len(g):
        return float("nan")
    order = np.argsort(-s)
    g = g[order]
    tp = np.cumsum(g)
    fp = np.cumsum(1 - g)
    tpr = tp / max(g.sum(), 1)
    fpr = fp / max((1 - g).sum(), 1)
    return float(np.trapezoid(tpr, fpr))


def gate_decision(per_signal_auroc: dict[str, float],
                  ensemble_auroc: float,
                  margin: float = 0.03) -> tuple[bool, str]:
    """§9D.2 gate: ensemble AUROC ≥ best-single + margin → PASS."""
    valid = {k: v for k, v in per_signal_auroc.items()
             if not (v is None or math.isnan(v))}
    if not valid or math.isnan(ensemble_auroc):
        return False, "no valid AUROCs (insufficient gold labels?)"
    best = max(valid.values())
    delta = ensemble_auroc - best
    ok = delta >= margin
    return ok, (f"ensemble={ensemble_auroc:.3f} vs best-single="
                f"{best:.3f} (Δ={delta:+.3f}, threshold ≥{margin})")


# ─────────────────────── CLI smoke-run ────────────────────────────


def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals_json", default=str(
        HERE.parent / "results" / "weaver_signals_minibatch_v1.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("§9D.2 weak-supervision label model + student smoke")
    inp = from_signals_json(args.signals_json)
    n = inp.W.shape[0]
    n_stub_cols = int((inp.stubs.all(axis=0)).sum())
    print(f"  loaded n={n} cases, {n_stub_cols}/5 fully-stubbed signals, "
          f"{int(inp.abstain.sum())} abstain")

    handle = fit_label_model(inp, seed=args.seed)
    soft = predict_soft_labels(handle, inp)
    print(f"  soft labels: mean={soft.mean():.3f} "
          f"std={soft.std():.3f} "
          f"min={soft.min():.3f} max={soft.max():.3f}")

    student = train_student(inp, soft)
    pred = student.forward(inp.W)
    # Synthetic gold from §9D.1 mini-batch ordering: i%3==0 correct (1),
    # i%3==1 wrong_mech (0), i%3==2 cond_mismatch (-1 abstain in gate).
    gold = np.array([[1, 0, -1][i % 3] for i in range(n)], dtype=int)

    per_sig = {SIGNAL_NAMES[k]: auroc(inp.W[:, k], gold) for k in range(5)}
    ens = auroc(pred, gold)
    print("  per-signal AUROC:")
    for k, v in per_sig.items():
        print(f"    {k:9s} {v:.3f}" if not math.isnan(v) else
              f"    {k:9s}   nan (constant or no positives)")
    print(f"  ensemble AUROC: {ens:.3f}")

    ok, msg = gate_decision(per_sig, ens)
    print(f"§9D.2 gate: {'PASS' if ok else 'FAIL/PROVISIONAL'}  ({msg})")

    out_path = HERE.parent / "results" / "weaver_label_model_smoke_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n": int(n),
        "per_signal_auroc": {k: (None if math.isnan(v) else v)
                              for k, v in per_sig.items()},
        "ensemble_auroc": (None if math.isnan(ens) else float(ens)),
        "soft_labels": soft.tolist(),
        "gate_pass": ok,
        "gate_msg": msg,
    }, indent=2))
    print(f"wrote {out_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
