"""
Convert E1 responses_*.jsonl (from `run_e1.py --step collect`) into the CoT JSONL
format consumed by `stepwise_label_bootstrap.py`.

Input schema (E1 LLMResponse.to_dict()):
    {
        "model_name": "deepseek-r1",
        "question_id": "L3_001",
        "response_text": "<final answer>",
        "reasoning_content": "<think>...</think> content or None",
        ...
    }

Output schema (stepwise_label_bootstrap input):
    {
        "qid": "L3_001",
        "model": "deepseek-r1",
        "cot_text": "<reasoning_content joined with response_text as final step>",
        "gt_mechanism": "<if present in benchmark>",
        "gt_severity": "<if present in benchmark>",
    }

Design
------
- For reasoning models (deepseek-r1): concatenate `reasoning_content` (the
  <think> trace) with `response_text` (the final answer) separated by a
  clear "Step N: Final diagnosis:" marker so the step splitter in
  `stepwise_label_bootstrap` can find step boundaries.
- Drop rows with error or empty reasoning_content.
- Optional: filter by level (e.g., only L3-L4 diagnosis questions).

Usage
-----
    python -m pvgap_experiment.scripts.responses_to_cot \\
        --responses pvgap_experiment/results/e1/responses/responses_deepseek-r1.jsonl \\
        --benchmark pvgap_experiment/data/benchmark/echem_reason_benchmark.jsonl \\
        --out pvgap_experiment/results/stepwise_labels/cot_deepseek-r1_N128.jsonl \\
        [--min-level 2]
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys


def load_benchmark_gt(path: str) -> dict[str, dict]:
    """Return {qid: {level, gt_mechanism?, gt_severity?, question_text}}."""
    gt: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid = d.get("qid") or d.get("id")
            if not qid:
                continue
            entry = {
                "level": d.get("level"),
                "question_text": d.get("question_text", ""),
            }
            # Ground-truth mechanism/severity can live either in the row's
            # top-level scenario dict or in `ground_truth`.
            gt_blob = d.get("ground_truth", {}) or {}
            entry["gt_mechanism"] = (
                d.get("gt_mechanism")
                or gt_blob.get("mechanism")
                or gt_blob.get("gt_mechanism")
            )
            entry["gt_severity"] = (
                d.get("gt_severity")
                or gt_blob.get("severity")
                or gt_blob.get("gt_severity")
            )
            gt[qid] = entry
    return gt


def join_reasoning_and_answer(reasoning: str | None, answer: str) -> str:
    """Combine <think> trace and final answer into a single CoT with explicit
    step markers where possible."""
    parts: list[str] = []
    if reasoning and reasoning.strip():
        # If reasoning already has explicit step markers, keep them; otherwise
        # prepend a generic marker to help the splitter.
        r = reasoning.strip()
        if not re.search(r"(?im)^\s*(?:Step|步骤)\s*\d+[:：.]", r):
            # add a leading "Step 1:" to the whole think block if unmarked
            r = "Step 1: " + r
        parts.append(r)
    if answer and answer.strip():
        # Mark final answer as its own step so the splitter can pick it out
        # even if the model used no step markers internally.
        parts.append("Step N: Final diagnosis: " + answer.strip())
    return "\n\n".join(parts)


def convert(responses_path: str, benchmark_path: str, out_path: str,
            min_level: int | None) -> None:
    gt = load_benchmark_gt(benchmark_path)
    print(f"benchmark: loaded {len(gt)} qids with ground-truth")

    n_in = 0
    n_out = 0
    n_err = 0
    n_empty = 0
    n_filtered = 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(responses_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            if d.get("error"):
                n_err += 1
                continue
            qid = d.get("question_id")
            if not qid:
                n_err += 1
                continue
            g = gt.get(qid, {})
            lvl = g.get("level")
            if min_level is not None and (lvl is None or lvl < min_level):
                n_filtered += 1
                continue
            cot = join_reasoning_and_answer(d.get("reasoning_content"),
                                            d.get("response_text", ""))
            if not cot.strip():
                n_empty += 1
                continue
            row = {
                "qid": qid,
                "model": d.get("model_name", "unknown"),
                "cot_text": cot,
            }
            if g.get("gt_mechanism"):
                row["gt_mechanism"] = g["gt_mechanism"]
            if g.get("gt_severity"):
                row["gt_severity"] = g["gt_severity"]
            # carry level for downstream analysis
            if lvl is not None:
                row["level"] = lvl
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"in={n_in}  out={n_out}  err={n_err}  empty={n_empty}  "
          f"filtered_by_level={n_filtered}")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 responses → stepwise-bootstrap CoT JSONL.")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-level", type=int, default=None,
                    help="keep only rows whose benchmark level >= this")
    args = ap.parse_args()
    convert(args.responses, args.benchmark, args.out, args.min_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
