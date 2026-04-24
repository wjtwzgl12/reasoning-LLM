"""§9E.1 LLM-as-judge scorer.

Replaces the cheap key-element substring recall with a grounded 4-axis
evaluation against the benchmark's structured `ground_truth`. A strong
LLM (DeepSeek-V3 by default) receives the question, the prediction, and
the ground_truth dict, then emits JSON with four [0,1] scores:

  1. correctness  — does the diagnosis/claim match the ground_truth?
  2. grounding    — does the prediction cite the observable EIS features
                    actually present in the scenario (not hallucinated)?
  3. mechanism    — is the attributed degradation mechanism physically
                    consistent with the observations?
  4. completeness — does it cover the required reasoning stages for the
                    level (L1: concept; L2/L3: data→feature→mechanism;
                    L4: NoOp invariance)?

Aggregate = mean of the four axes. The judge prompt is deterministic
(temp=0.0), always JSON-mode, retry-on-parse-fail once.

Gate (§9E.1): C1–C4 must each beat C0 on `aggregate` by ≥0.05 on at
least one level, paired-bootstrap p<0.05 (n=60 × 3 seeds). Evaluated by
`run_9e_full.py` — this module only provides the per-case scoring.
"""
from __future__ import annotations

import json
from typing import Any


JUDGE_SYSTEM = (
    "你是资深电化学/锂电池阻抗谱评审专家。你将收到一道题、一条"
    "候选回答、以及该题的结构化标准答案(ground_truth)。请严格按四个维度"
    "在[0,1]区间打分，并返回 JSON。不要输出除 JSON 以外的任何文字。"
)

JUDGE_INSTRUCTIONS = (
    "评分维度定义：\n"
    "  correctness  : 候选回答的主要结论(诊断/机理/数值)是否与 ground_truth 匹配\n"
    "                 (1.0=完全一致；0.5=部分匹配；0.0=错误或无关)\n"
    "  grounding    : 候选回答提到的EIS特征是否源自题干的可观测量,无幻觉\n"
    "  mechanism    : 给出的退化/电化学机理是否物理合理,与观察一致\n"
    "  completeness : 是否覆盖该题级别要求的推理步骤\n"
    "                 (L1 概念；L2/L3 数据质量→特征→机理→差分；L4 NoOp 不变性)\n"
    "\n"
    "返回 JSON 对象,键恰为：correctness, grounding, mechanism, "
    "completeness, reason。前四键为 0..1 浮点;reason 为 ≤60 字中文简述"
    "评分依据。请返回 json 对象。"
)


def judge_case(question: str,
               prediction: str,
               ground_truth: dict,
               level: int,
               llm_call=None) -> dict:
    """Score a single (prediction, ground_truth) pair. Returns
    {'correctness','grounding','mechanism','completeness','aggregate',
     'reason', 'error'}."""
    if llm_call is None:
        from .sbi_prior_emit import call_llm
        llm_call = call_llm

    gt_repr = json.dumps(ground_truth, ensure_ascii=False, indent=2)
    user = (
        f"题目级别: L{level}\n\n"
        f"题目:\n{question}\n\n"
        f"候选回答:\n{prediction}\n\n"
        f"ground_truth (json):\n{gt_repr}\n\n"
        f"{JUDGE_INSTRUCTIONS}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user},
    ]

    raw = ""
    last_err: str | None = None
    for attempt in range(2):  # one retry
        try:
            raw = llm_call(messages, temperature=0.0)
            obj = json.loads(raw)
            # Clamp + coerce.
            def _f(k):
                v = float(obj.get(k, 0.0))
                return max(0.0, min(1.0, v))
            scores = {k: _f(k) for k in
                      ("correctness", "grounding", "mechanism", "completeness")}
            agg = sum(scores.values()) / 4.0
            return {**scores, "aggregate": agg,
                    "reason": str(obj.get("reason", ""))[:120],
                    "error": None}
        except Exception as ex:
            last_err = f"{type(ex).__name__}: {str(ex)[:120]}"
            # On retry, nudge the judge with the previous bad output head.
            messages = messages + [
                {"role": "assistant", "content": raw[:200]},
                {"role": "user",
                 "content": "上条输出无法解析为 JSON。请严格只返回 json 对象。"},
            ]
    return {
        "correctness": 0.0, "grounding": 0.0, "mechanism": 0.0,
        "completeness": 0.0, "aggregate": 0.0,
        "reason": "", "error": last_err,
    }


# ─────────────────────── self-test CLI ─────────────────────────────────


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="L2_000")
    ap.add_argument("--pred_file", default=None,
                    help="Optional: path to a .txt with the prediction. "
                         "Default uses a canned good answer.")
    args = ap.parse_args()

    from pathlib import Path
    bench = (Path(__file__).resolve().parent.parent
             / "data" / "benchmark" / "echem_reason_benchmark.jsonl")
    case = None
    for l in open(bench, encoding="utf-8"):
        r = json.loads(l)
        if r["qid"] == args.qid:
            case = r; break
    if case is None:
        raise SystemExit(f"qid {args.qid} not in {bench}")

    if args.pred_file:
        pred = open(args.pred_file, encoding="utf-8").read()
    else:
        # Canned "good" answer for smoke test.
        gt = case["ground_truth"]
        pred = (f"根据EIS数据分析，诊断结果为: {gt.get('diagnosis','')}。"
                f"退化类型为 {gt.get('degradation_type','')}，"
                f"严重程度 {gt.get('severity','')}。欧姆阻抗正常范围，"
                f"中频半圆对应电荷转移过程，低频斜率指示扩散行为。")

    print(f"scoring qid={args.qid} level=L{case['level']}")
    out = judge_case(case["question_text"], pred,
                     case["ground_truth"], case["level"])
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
