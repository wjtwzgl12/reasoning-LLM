"""§9E.1 pilot — minimal C0 baseline runner on a 5-scenario subset.

Goal: validate the end-to-end pipeline (benchmark load → LLM call → metric
computation → JSONL output) before scaling to the full 5 configs × 60
scenarios × 3 seeds run.

C0 definition (Paper1_骨架.md §9E.1): plain LLM call, no Arm-2/3, no
reasoning enhancement. Uses DeepSeek-V3 (cost-effective Qwen2.5-7B
substitute for pilot; full §9E.1 will use vLLM-hosted Qwen2.5-7B).

Metric (pilot v0): key-element recall — fraction of `ground_truth.key_elements`
strings that appear in the prediction (case-insensitive substring match).
This is a cheap proxy; the full §9E.1 will use LLM-as-judge + paired
bootstrap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCHMARK = ROOT / "data" / "benchmark" / "echem_reason_benchmark.jsonl"
OUT_DIR = ROOT / "results"


def load_benchmark(path: Path = BENCHMARK) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def pick_pilot_subset(all_cases: list[dict], n: int = 5, seed: int = 0
                      ) -> list[dict]:
    """Pick 5 cases spanning the level distribution (L1/L2/L3 mix)."""
    import random
    rng = random.Random(seed)
    by_level = {}
    for c in all_cases:
        by_level.setdefault(c.get("level", 0), []).append(c)
    picks = []
    for lvl in sorted(by_level):
        rng.shuffle(by_level[lvl])
    # Round-robin across levels until we have n.
    levels = sorted(by_level)
    i = 0
    while len(picks) < n and any(by_level[l] for l in levels):
        lvl = levels[i % len(levels)]
        if by_level[lvl]:
            picks.append(by_level[lvl].pop())
        i += 1
    return picks[:n]


def c0_predict(case: dict) -> dict:
    """C0 = plain LLM call with zero-shot QA prompt. No CoT, no verifier."""
    from .sbi_prior_emit import call_llm  # uses DeepSeek-V3 when key present
    msgs = [
        {"role": "system",
         "content": "You are an electrochemist. Answer the question concisely "
                    "in 2–4 sentences. Use Chinese."},
        {"role": "user", "content": case["question_text"]},
    ]
    t0 = time.time()
    try:
        raw = call_llm(msgs)
        return {"pred": raw, "latency_s": time.time() - t0, "error": None}
    except Exception as ex:
        return {"pred": "", "latency_s": time.time() - t0, "error": str(ex)[:150]}


C1_SYSTEM = (
    "你是资深电化学/锂电池阻抗谱专家。请按以下四阶段框架逐步推理后给出最终结论：\n"
    "  S1 数据质量 — 检查题目给出的频率范围、截距、KK 合理性，判定数据是否可用\n"
    "  S2 特征提取 — 从题干数值(欧姆阻抗R_ohm / 中频弧 / 低频斜率等)中提取关键特征\n"
    "  S3 机理识别 — 将特征映射到候选退化机理(healthy / LAM_negative / "
    "diffusion_degradation / combined_degradation)与严重程度(mild/moderate/severe)\n"
    "  S4 差分诊断 — 对主机理给出物理依据,并排除其他候选\n"
    "最后用一句话给出最终诊断。全部用中文。"
)


def c1_predict(case: dict) -> dict:
    """C1 = CoT with 4-stage reasoning scaffold (no verifier, no ensemble)."""
    from .sbi_prior_emit import call_llm
    msgs = [
        {"role": "system", "content": C1_SYSTEM},
        {"role": "user",
         "content": (case["question_text"]
                     + "\n\n请严格按 S1→S2→S3→S4 四步逐步分析，再给出最终诊断。")},
    ]
    t0 = time.time()
    try:
        raw = call_llm(msgs)
        return {"pred": raw, "latency_s": time.time() - t0, "error": None}
    except Exception as ex:
        return {"pred": "", "latency_s": time.time() - t0, "error": str(ex)[:150]}


C3_CRITIC_SYSTEM = (
    "你是阻抗谱诊断的审阅专家。下面给出题目和一个候选诊断回答，"
    "请对这条回答的可靠性从以下三个维度打分 (0..1)：\n"
    "  consistency  候选回答的四阶段(S1 数据质量→S2 特征→S3 机理→S4 差分)"
    "是否内部一致、无矛盾\n"
    "  grounding    候选回答引用的数值/特征是否全部出自题干,无新造数据\n"
    "  mechanism_fit 候选最终机理与题干给出的 EIS 特征(欧姆阻抗、弧、低频斜率)"
    "物理上是否相容\n"
    "只返回 json 对象: {\"consistency\": ..., \"grounding\": ..., "
    "\"mechanism_fit\": ..., \"reason\": <≤40 字>}."
)


def _c3_score_candidate(question: str, candidate: str) -> float:
    """Return aggregated critic score (∈[0,1]) for a candidate. Falls back
    to 0.5 on any API/parse error to avoid single-critic crash poisoning
    the whole BoN selection."""
    from .sbi_prior_emit import call_llm
    msgs = [
        {"role": "system", "content": C3_CRITIC_SYSTEM},
        {"role": "user",
         "content": f"题目:\n{question}\n\n候选回答:\n{candidate}\n\n请返回 json。"},
    ]
    try:
        raw = call_llm(msgs, temperature=0.0)
        obj = json.loads(raw)
        keys = ("consistency", "grounding", "mechanism_fit")
        vals = [max(0.0, min(1.0, float(obj.get(k, 0.5)))) for k in keys]
        return sum(vals) / 3.0
    except Exception:
        return 0.5


def c3_predict(case: dict, n_candidates: int = 4) -> dict:
    """C3 = Weaver-BoN. N=4 CoT candidates @ T=0.8, LLM-critic picks best.

    Candidate generation (4 calls) and critic scoring (4 calls) are each
    fanned out via ThreadPoolExecutor so the two rounds go from ~8×T
    sequential to ~2×T wall-clock (T ≈ single DeepSeek call latency).
    """
    from concurrent.futures import ThreadPoolExecutor
    from .sbi_prior_emit import call_llm
    t0 = time.time()

    def _gen_one(_):
        msgs = [
            {"role": "system", "content": C1_SYSTEM},
            {"role": "user",
             "content": (case["question_text"]
                         + "\n\n请严格按 S1→S2→S3→S4 四步逐步分析，再给出最终诊断。")},
        ]
        try:
            return (call_llm(msgs, temperature=0.8), None)
        except Exception as ex:
            return ("", str(ex)[:80])

    with ThreadPoolExecutor(max_workers=n_candidates) as ex:
        results = list(ex.map(_gen_one, range(n_candidates)))
    candidates = [r[0] for r in results]
    cand_errors = [r[1] for r in results if r[1]]

    valid = [(i, c) for i, c in enumerate(candidates) if c]
    if not valid:
        return {"pred": "", "latency_s": time.time() - t0,
                "error": f"all {n_candidates} candidates failed: "
                         f"{cand_errors[:2]}"}

    # Score each valid candidate IN PARALLEL.
    with ThreadPoolExecutor(max_workers=len(valid)) as ex:
        scores = list(ex.map(
            lambda c: _c3_score_candidate(case["question_text"], c),
            [c for _, c in valid]))
    best_idx_in_valid = max(range(len(valid)), key=lambda i: scores[i])
    best_cand_idx, best_cand = valid[best_idx_in_valid]
    return {
        "pred": best_cand,
        "latency_s": time.time() - t0,
        "error": None,
        "bon_meta": {
            "n_candidates": n_candidates,
            "n_valid": len(valid),
            "scores": scores,
            "picked_idx": best_cand_idx,
            "picked_score": scores[best_idx_in_valid],
            "score_spread": (max(scores) - min(scores)) if len(scores) > 1 else 0.0,
        },
    }


# ─────────── C3-div = contrarian-branch critic-BoN (§9E.1 v2) ──────
#
# Motivation: §9E.1 main-run (n=180) showed C3-critic-only Δagg ≈ 0 vs C0,
# with Δcorrectness = -0.105 (sig). Per-candidate inspection showed 4
# T=0.8 candidates of the same prompt converge on near-identical final
# mechanism → critic has no real choice to make. C3-div forces
# diversity by pinning each branch to a different mechanism hypothesis,
# then lets the critic pick the one best grounded in the observation.

C3DIV_BRANCHES = [
    ("LAM_negative",
     "假设本电池主要退化机理是 **负极活性材料损失 (LAM_negative)**。"
     "请在此假设下走完 S1→S4，并在 S4 给出该机理与题干特征是否一致的判定。"),
    ("diffusion_degradation",
     "假设本电池主要退化机理是 **扩散退化 (diffusion_degradation)** —— "
     "固态扩散系数下降导致低频 Warburg 尾加重。"
     "请在此假设下走完 S1→S4，并在 S4 给出该机理与题干特征是否一致的判定。"),
    ("combined_degradation",
     "假设本电池同时存在 **LAM + 扩散退化 (combined_degradation)**。"
     "请在此假设下走完 S1→S4，并在 S4 给出该机理与题干特征是否一致的判定。"),
    ("healthy",
     "假设本电池**基本健康 (healthy)**，所见特征属于正常运行而非退化。"
     "请在此假设下走完 S1→S4，并在 S4 给出该机理与题干特征是否一致的判定。"),
]


def c3_div_predict(case: dict, n_candidates: int = 4) -> dict:
    """C3-div = contrarian-branch critic-BoN.

    N candidates are generated with distinct mechanism assumptions
    (one per branch in `C3DIV_BRANCHES`). The critic then picks the
    branch whose S1→S4 best fits the observation. n_candidates > 4
    will cycle through branches with re-sampled temperature; n < 4
    truncates the branch list.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .sbi_prior_emit import call_llm
    t0 = time.time()

    branches = (C3DIV_BRANCHES * ((n_candidates + 3) // 4))[:n_candidates]

    def _gen_branch(branch_idx_and_spec):
        idx, (label, assumption) = branch_idx_and_spec
        msgs = [
            {"role": "system", "content": C1_SYSTEM},
            {"role": "user",
             "content": (case["question_text"]
                         + "\n\n" + assumption
                         + "\n\n请严格按 S1→S2→S3→S4 四步逐步分析，"
                         "在 S4 末尾用一句话给出最终诊断（可以是"
                         "\"该假设成立\"或\"该假设不成立，真实机理为 X\"）。")},
        ]
        try:
            return (idx, label, call_llm(msgs, temperature=0.8), None)
        except Exception as ex:
            return (idx, label, "", str(ex)[:80])

    with ThreadPoolExecutor(max_workers=n_candidates) as ex:
        results = list(ex.map(_gen_branch, enumerate(branches)))
    # results sorted by original idx for deterministic order.
    results.sort(key=lambda r: r[0])
    candidates = [r[2] for r in results]
    labels = [r[1] for r in results]
    cand_errors = [r[3] for r in results if r[3]]

    valid = [(i, c) for i, c in enumerate(candidates) if c]
    if not valid:
        return {"pred": "", "latency_s": time.time() - t0,
                "error": f"all {n_candidates} branch candidates failed: "
                         f"{cand_errors[:2]}"}

    # Critic scoring fans out in parallel (same critic prompt as C3).
    with ThreadPoolExecutor(max_workers=len(valid)) as ex:
        scores = list(ex.map(
            lambda c: _c3_score_candidate(case["question_text"], c),
            [c for _, c in valid]))

    best_idx_in_valid = max(range(len(valid)), key=lambda i: scores[i])
    best_cand_idx, best_cand = valid[best_idx_in_valid]
    return {
        "pred": best_cand,
        "latency_s": time.time() - t0,
        "error": None,
        "bon_meta": {
            "mode": "contrarian_branches",
            "n_candidates": n_candidates,
            "n_valid": len(valid),
            "branch_labels": labels,
            "scores": scores,
            "picked_idx": best_cand_idx,
            "picked_label": labels[best_cand_idx],
            "picked_score": scores[best_idx_in_valid],
            "score_spread": (max(scores) - min(scores)) if len(scores) > 1
                             else 0.0,
        },
    }


# ─────────────────────── C3-full = 5-signal Weaver-BoN ─────────────


_W4_SINGLETON: object | None = None


def _get_w4():
    """Load §9C.2 SBI density estimator once per process. None if ckpt
    missing → w_4 will fall back to stub 0.5 inside extract_all."""
    global _W4_SINGLETON
    if _W4_SINGLETON is not None:
        return _W4_SINGLETON if _W4_SINGLETON != "failed" else None
    try:
        from .sbi_w4_scorer import Wfour
        _W4_SINGLETON = Wfour()
    except Exception as ex:
        print(f"  [c3_full] w_4 unavailable ({ex!s}[:80]); falling back to stub")
        _W4_SINGLETON = "failed"
        return None
    return _W4_SINGLETON


_PRM_SINGLETON = None


def _get_prm():
    """Load §9A.3 EIS-PRM once per process. None if ckpt missing → w_1
    falls back to neutral 0.5 stub in extract_w1_prm.

    v3 fix (§9E.1, 2026-04-22): c3_full_predict previously called
    `_score_5signal` without any prm_model, so w_1 was a constant 0.5
    stub for every candidate, contributing zero discrimination. Now we
    try to load the PRM; env PVGAP_DISABLE_PRM=1 opts out (for CI /
    dep-less Colab runs)."""
    global _PRM_SINGLETON
    if _PRM_SINGLETON is not None:
        return _PRM_SINGLETON if _PRM_SINGLETON != "failed" else None
    import os
    if os.environ.get("PVGAP_DISABLE_PRM", "").lower() in ("1", "true", "yes"):
        _PRM_SINGLETON = "failed"
        return None
    try:
        from .prm_scorer import PRMScorer
        _PRM_SINGLETON = PRMScorer()
    except Exception as ex:
        print(f"  [c3_full] PRM unavailable ({str(ex)[:80]}); w_1 will stub to 0.5")
        _PRM_SINGLETON = "failed"
        return None
    return _PRM_SINGLETON


_PRM_SINGLETON = None


def _get_prm():
    """Load §9A.3 EIS-PRM once per process. None if ckpt missing → w_1
    falls back to neutral 0.5 stub in extract_w1_prm.

    v3 fix (§9E.1, 2026-04-22): c3_full_predict previously called
    `_score_5signal` without any prm_model, so w_1 was a constant 0.5
    stub for every candidate, contributing zero discrimination. Now we
    try to load the PRM; env PVGAP_DISABLE_PRM=1 opts out (for CI /
    dep-less Colab runs)."""
    global _PRM_SINGLETON
    if _PRM_SINGLETON is not None:
        return _PRM_SINGLETON if _PRM_SINGLETON != "failed" else None
    import os
    if os.environ.get("PVGAP_DISABLE_PRM", "").lower() in ("1", "true", "yes"):
        _PRM_SINGLETON = "failed"
        return None
    try:
        from .prm_scorer import PRMScorer
        _PRM_SINGLETON = PRMScorer()
    except Exception as ex:
        print(f"  [c3_full] PRM unavailable ({str(ex)[:80]}); w_1 will stub to 0.5")
        _PRM_SINGLETON = "failed"
        return None
    return _PRM_SINGLETON


def _score_5signal(case_for_signals: dict, w4_obj, prm_model=None) -> dict:
    """Return {score: float, meta: {...}} aggregating w_1..w_5.

    Aggregation: mean of non-stub signals. A stub (w_i.meta.stub=True)
    contributes nothing and is not counted in the denominator — this
    avoids the neutral-0.5 from dominating when a signal is simply
    unavailable. If all 5 signals are stubs, returns score=0.5 (fall
    through to neutral; BoN argmax becomes arbitrary).
    """
    from .weaver_signals import extract_all
    out = extract_all(case_for_signals, prm_model=prm_model)
    # Patch w_4 with real scorer if available.
    if w4_obj is not None:
        try:
            out["w4"] = w4_obj.score(case_for_signals)
        except Exception as ex:
            out["w4"] = {"w4": 0.5, "meta": {"stub": True,
                                              "reason": f"w4_fail:{ex!s}[:60]"}}
    vals, stubs = [], []
    for k in ("w1", "w2", "w3", "w4", "w5"):
        vals.append(float(out[k][k]))
        stubs.append(bool(out[k]["meta"].get("stub", False)))
    nonstub = [v for v, s in zip(vals, stubs) if not s]
    score = (sum(nonstub) / len(nonstub)) if nonstub else 0.5
    return {"score": score, "w_raw": vals, "stubs": stubs}


def c3_full_predict(case: dict, n_candidates: int = 4) -> dict:
    """C3-full = Weaver-BoN with 5 heterogeneous signals.

    **Status (§9E.1 post-mortem v3, 2026-04)**: archived / 留档. Two
    n=5 pilots (v2 shared-params, v3 per-candidate-params) both failed
    the paired-Δ gate vs critic-only C3. Root causes identified:
      (1) rule-based diagnosis-text → Chen2020 override parser
          over-labels "combined severe" (CoT differential-diagnosis
          sections mention alternative mechanisms that trigger
          keyword matches);
      (2) generator diversity is too low at T=0.8 single-prompt —
          all 4 candidates converge on the same final mechanism, so
          even a perfect extractor would yield identical per-candidate
          params for w_2/w_4.
    Leaving the code path intact so §9E.2 (future work) can swap in
    a structured-JSON LLM extractor + contrarian-branch generator.

    For scenarios with physical_params (L2/L3/L4-with-base): use
    extract_all over (observed_Z, committed_params, prediction_text).
    For L1 (no physical_params) fall back to the critic-only C3 path —
    we still return a well-formed prediction but with bon_meta.mode
    marked `critic_only_fallback`.
    """
    from .sbi_prior_emit import call_llm
    from .scenario_bridge import bridge

    weaver_case = bridge(case)
    if weaver_case is None or "_bridge_error" in (weaver_case or {}):
        # Fall back to critic-only BoN.
        out = c3_predict(case, n_candidates=n_candidates)
        if out.get("bon_meta"):
            out["bon_meta"]["mode"] = "critic_only_fallback"
            out["bon_meta"]["bridge_note"] = (
                weaver_case.get("_bridge_error")
                if isinstance(weaver_case, dict) else "no_physical_params")
        return out

    t0 = time.time()
    # 1) Generate N candidates at T=0.8.
    candidates: list[str] = []
    for _ in range(n_candidates):
        msgs = [
            {"role": "system", "content": C1_SYSTEM},
            {"role": "user",
             "content": case["question_text"]
                        + "\n\n请严格按 S1→S2→S3→S4 四步逐步分析，再给出最终诊断。"},
        ]
        try:
            candidates.append(call_llm(msgs, temperature=0.8))
        except Exception:
            candidates.append("")
    valid = [(i, c) for i, c in enumerate(candidates) if c]
    if not valid:
        return {"pred": "", "latency_s": time.time() - t0,
                "error": "all candidates failed"}

    # 2) 5-signal score per candidate.
    #    Per v3 post-mortem (§9E.1, 2026-04-22): the v2 regex mapper in
    #    scenario_bridge collapsed most candidates to an identical
    #    "combined-severe" bucket, so w_2/w_4 stopped discriminating. v3
    #    replaces it with an LLM structured-JSON extractor per candidate
    #    (candidate_param_extractor.extract_overrides), with the regex mapper
    #    kept as a validated fallback. Candidate-level extraction runs in
    #    parallel (one LLM call per candidate) — ~4 extra calls per qid.
    from concurrent.futures import ThreadPoolExecutor
    from .scenario_bridge import perturb_params_from_diagnosis
    from .candidate_param_extractor import extract_overrides
    base_committed = weaver_case["committed_mechanism_params"]
    w4 = _get_w4()
    prm = _get_prm()

    def _extract_one(cand_text: str):
        return extract_overrides(
            cand_text, base_committed,
            fallback=perturb_params_from_diagnosis,
        )

    with ThreadPoolExecutor(max_workers=min(4, len(valid))) as pool:
        extracted = list(pool.map(_extract_one, [c for _, c in valid]))

    per_cand = []
    per_cand_diag: list[dict] = []
    for (_, cand), (cand_params, diag_meta) in zip(valid, extracted):
        cfs = dict(weaver_case)
        cfs["prediction_text"] = cand
        cfs["committed_mechanism_params"] = cand_params
        cfs["param_overrides"] = cand_params
        per_cand_diag.append(diag_meta)
        per_cand.append(_score_5signal(cfs, w4, prm_model=prm))

    # 3) Argmax by aggregated score.
    scores = [p["score"] for p in per_cand]
    best_i = max(range(len(valid)), key=lambda i: scores[i])
    picked_idx, picked_text = valid[best_i]
    return {
        "pred": picked_text,
        "latency_s": time.time() - t0,
        "error": None,
        "bon_meta": {
            "mode": "5signal",
            "n_candidates": n_candidates,
            "n_valid": len(valid),
            "scores": scores,
            "picked_idx": picked_idx,
            "picked_score": scores[best_i],
            "per_candidate_w_raw": [p["w_raw"] for p in per_cand],
            "per_candidate_stubs": [p["stubs"] for p in per_cand],
            "per_candidate_diagnosis": per_cand_diag,
        },
    }


CONFIGS = {"C0": c0_predict, "C1": c1_predict, "C3": c3_predict,
           "C3div": c3_div_predict, "C3full": c3_full_predict}


def keyelem_recall(pred: str, key_elements: list[str]) -> float:
    if not key_elements:
        return 0.0
    p = (pred or "").lower()
    hits = sum(1 for k in key_elements if str(k).lower() in p)
    return hits / len(key_elements)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge", action="store_true",
                    help="Also run LLM-as-judge 4-axis scoring (§9E.1)")
    ap.add_argument("--config", default="C0", choices=sorted(CONFIGS))
    args = ap.parse_args()
    if args.out is None:
        args.out = str(OUT_DIR / f"9E1_pilot_{args.config}_v1.jsonl")
    predict_fn = CONFIGS[args.config]

    all_cases = load_benchmark()
    subset = pick_pilot_subset(all_cases, n=args.n, seed=args.seed)
    print(f"§9E.1 pilot {args.config}  |  {len(subset)}/{len(all_cases)} "
          f"scenarios, seed={args.seed}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, c in enumerate(subset):
            p = predict_fn(c)
            recall = keyelem_recall(p["pred"],
                                    c["ground_truth"].get("key_elements", []))
            row = {
                "qid": c["qid"],
                "level": c.get("level"),
                "question": c["question_text"][:100],
                "pred_head": (p["pred"] or "")[:200],
                "key_elements": c["ground_truth"].get("key_elements", []),
                "key_recall": recall,
                "latency_s": p["latency_s"],
                "error": p["error"],
            }
            if args.judge and p["pred"] and p["error"] is None:
                from .llm_judge import judge_case
                j = judge_case(c["question_text"], p["pred"],
                               c["ground_truth"], c.get("level", 0))
                row["judge"] = j
            if "bon_meta" in p:
                row["bon_meta"] = p["bon_meta"]
            results.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            err_tag = f" ERR={p['error'][:40]}" if p["error"] else ""
            j_tag = ""
            if args.judge and "judge" in row and row["judge"]["error"] is None:
                j = row["judge"]
                j_tag = (f"  judge(c/g/m/comp/agg)={j['correctness']:.2f}/"
                         f"{j['grounding']:.2f}/{j['mechanism']:.2f}/"
                         f"{j['completeness']:.2f}/{j['aggregate']:.2f}")
            print(f"  {i+1}/{len(subset)}  {c['qid']}  L{c.get('level')}  "
                  f"recall={recall:.2f}  t={p['latency_s']:.1f}s{err_tag}{j_tag}")

    ok = [r for r in results if r["error"] is None and r["pred_head"]]
    if not ok:
        print("FAIL: no successful predictions — check API key/network")
        return 1
    mean_recall = sum(r["key_recall"] for r in ok) / len(ok)
    mean_lat = sum(r["latency_s"] for r in ok) / len(ok)
    print(f"\npilot C0 mean key-element recall = {mean_recall:.3f}  "
          f"(n_ok={len(ok)}/{len(results)})")
    print(f"pilot C0 mean latency = {mean_lat:.1f}s")
    if args.judge:
        j_ok = [r for r in ok if "judge" in r and r["judge"]["error"] is None]
        if j_ok:
            for axis in ("correctness", "grounding", "mechanism",
                         "completeness", "aggregate"):
                m = sum(r["judge"][axis] for r in j_ok) / len(j_ok)
                print(f"pilot C0 mean judge.{axis} = {m:.3f}")
            print(f"  (n_judge_ok={len(j_ok)}/{len(ok)})")
    print(f"wrote {args.out}")
    # Gate: any non-zero recall on any non-L1 case → pipeline works; we gate
    # later with full run.
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
