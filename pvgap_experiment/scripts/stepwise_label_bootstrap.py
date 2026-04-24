"""
9A.2 stepwise label bootstrap — GPT-4o + DS-V3 disagreement filter.
=========================================================================

Purpose
-------
Take a CoT JSONL (one row per diagnosis trace), split each trace into
stepwise segments, ask GPT-4o and DS-V3 to judge each segment's quality
against the EChemRules corpus, and emit a stepwise-labelled JSONL plus a
disagreement-rate report. §9A.2 decision gate: DS-V3/GPT-4o disagreement
rate < 15% → proceed to 9A.3.

Input schema (CoT JSONL, one per line)
--------------------------------------
{
  "qid": "L3_000",
  "model": "deepseek-r1",
  "cot_text": "<full reasoning trace, incl. <think> tags if present>",
  "gt_mechanism": "LAM_negative",   # optional; if present, final-step label can cross-check
  "gt_severity": "moderate"         # optional
}

Output JSONL (one row per (qid, step_idx))
------------------------------------------
{
  "qid": ..., "step_idx": 0, "step_text": "...",
  "gpt4o":  {"label": "good"|"bad"|"neutral", "rule_cites": [...], "reason": "..."},
  "dsv3":   {...},
  "agree":  true|false,
  "consensus_label": "good"|"bad"|"neutral"|"disagree"
}

Usage
-----
    python -m pvgap_experiment.scripts.stepwise_label_bootstrap \\
        --cot cot_inputs.jsonl --out labels.jsonl --dry-run

    python -m pvgap_experiment.scripts.stepwise_label_bootstrap \\
        --cot cot_inputs.jsonl --out labels.jsonl \\
        --n-samples 50 --max-steps-per-trace 12

Design contract
---------------
- This script is a *pilot* tool. It MUST NOT be used to train a PRM before
  the disagreement rate is inspected. It writes disagreement stats to
  `<out>.summary.json` for §9A.2 decision-gate review.
- Rule corpus (50 seed + 15 gate) is loaded and the rule IDs are passed to
  both judges as grounding. Judges must cite rule IDs or return empty list.
- Temperature 0.2 for both judges; 1 sample per step per judge (judgement,
  not generation — no benefit to repeats here).
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from typing import Any


# ───────────────────── input split ──────────────────────────────────

_STEP_SPLIT_PATTERNS = [
    re.compile(r"\n\s*(?:Step|步骤)\s*\d+[:：.\-]\s*"),
    re.compile(r"\n\s*\d+\.\s+"),
    re.compile(r"\n\s*\n"),   # last-resort: double newline
]


def split_cot_into_steps(cot: str, max_steps: int = 15) -> list[str]:
    """Segment a CoT trace into steps. Tries explicit step markers first,
    falls back to paragraph breaks. Always returns ≥ 1 chunk."""
    # strip <think>...</think> wrappers but keep the content
    cot = re.sub(r"</?think>", "", cot)
    cot = cot.strip()
    for pat in _STEP_SPLIT_PATTERNS:
        parts = pat.split(cot)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) >= 2:
            return parts[:max_steps]
    # unsplittable: return as single step
    return [cot][:max_steps]


# ───────────────────── prompt ───────────────────────────────────────

JUDGE_PROMPT = """You are an electrochemistry-diagnosis step judge. Apply the
three labels with MUTUALLY EXCLUSIVE criteria: every step belongs to exactly
one category. Do not collapse to "neutral" on uncertainty; pick by the
checklist.

Candidate grounding rules (retrieved by ontology overlap with the step —
these are the specific rules you should check for match/contradiction):
{rules_block}

Diagnostic step to evaluate:
---
{step_text}
---

## DECISION CHECKLIST (apply in order — first match wins)

STEP 1. Does the step make a DIAGNOSTIC CLAIM?
  A claim names an EIS feature (semicircle, arc, Warburg tail, DRT peak,
  intercept, slope, CPE phase, inductive loop, tilt, ...) AND asserts a
  physical cause or mechanism (SEI, charge transfer, diffusion regime,
  contact loss, aging, ...).
  - NO: label = "neutral". (Pure observation, procedural "let me look at X",
    summary restatement, tool-call wrappers, or meta-commentary with no
    feature→mechanism assertion.) STOP HERE. Return neutral.
  - YES: go to STEP 2.

STEP 2. Does the claim CONTRADICT any rule in the grounding corpus, OR
  name a feature/mechanism that the corpus would mark hallucinated?
  - YES: label = "bad". Cite the rule(s) the claim contradicts in
    `rule_cites`. Return.
  - NO: go to STEP 3.

STEP 3. Does the step's claim MATCH at least one rule in the grounding
  corpus (the feature→mechanism pair is stated by a retrieved rule)?
  - NO (no retrieved rule pairs this feature with this mechanism):
      label = "neutral". Empty `rule_cites`. Return.
  - YES → go to STEP 3b to grade the strength.

STEP 3b. STRENGTH GRADING for a matched claim:
  - The step names an ALTERNATIVE mechanism it is ruling out, or cites a
    concrete DISCRIMINATOR (numerical range, `applies_to` condition, or
    counterexample) that narrows the match → label = "good_strong".
  - The step only states the feature→mechanism pair matching a rule,
    without explicit alternative-ruling-out or discriminator → label =
    "good_weak". (Still counts as a rule-grounded step, just not
    discriminative.)
  Populate `rule_cites` with the matching rule IDs either way.

## NOTES ON OVERLAP (READ BEFORE JUDGING)

- "good_strong" requires EXPLICIT alternative exclusion or discriminator —
  phrases like "as opposed to X", "not Y because …", "at low SoC (per
  applies_to)", or a numerical threshold.
- "good_weak" covers the common case where the CoT correctly identifies
  a feature→mechanism pair backed by a rule but does not argue against
  alternatives. This is still useful training signal (correct but
  non-discriminative).
- Procedural narration ("I will apply K-K test", "looking at the plot",
  "as shown above") = "neutral" regardless of rule proximity.
- Do NOT infer unstated grounding. If the step does not NAME a rule-like
  feature+mechanism pairing, neither "good_weak" nor "good_strong"
  applies — it is "neutral".
- "bad" requires an actual contradiction with a corpus rule or a
  hallucinated feature. Low quality without contradiction is "neutral".

Emit ONE JSON object, nothing else:
{{
  "label":      "good_strong" | "good_weak" | "bad" | "neutral",
  "rule_cites": ["<rule_id>", ...],
  "reason":     "<one sentence naming which checklist step decided the label>"
}}
"""


def _format_rule(r: dict) -> str:
    """Render one rule as a compact bullet for the judge prompt."""
    obs = (r.get("observation") or "").strip()
    mech = r.get("mechanism") or []
    alt = r.get("alt_mechanisms") or []
    disc = r.get("discriminators") or []
    appl = r.get("applies_to") or []
    parts = [f"- [{r['rule_id']}] observation: {obs}",
             f"    mechanism: {', '.join(mech) if mech else '(none)'}"]
    if alt:
        parts.append(f"    alt_mechanisms: {', '.join(alt)}")
    if disc:
        parts.append(f"    discriminators: {', '.join(disc)}")
    if appl:
        parts.append(f"    applies_to: {', '.join(appl)}")
    return "\n".join(parts)


def build_judge_messages(step_text: str, rules: list[dict]) -> list[dict]:
    """Build messages with full rule content (not just IDs) for grounding.

    Expects `rules` to already be retrieved/ranked — top-K most relevant
    for this step. Each rule appears as a multi-line bullet with
    observation, mechanism, alt_mechanisms, discriminators, applies_to,
    so the judge can actually check STEP-2 contradiction and STEP-3 match."""
    if rules:
        rules_block = "\n".join(_format_rule(r) for r in rules)
    else:
        rules_block = "(no candidate rules retrieved for this step)"
    return [
        {"role": "system", "content": "Output only a single JSON object."},
        {"role": "user", "content": JUDGE_PROMPT.format(
            step_text=step_text, rules_block=rules_block)},
    ]


# ───────────────────── rule corpus ──────────────────────────────────

def load_all_rules() -> list[dict]:
    """Load full rule objects (not just IDs) from the seed + calibration corpus."""
    here = os.path.dirname(os.path.abspath(__file__))
    rules_dir = os.path.abspath(os.path.join(here, "..", "data", "echem_rules"))
    rules: list[dict] = []
    for fname in ("echem_rules_seed.jsonl", "echem_rules_condition_calibration.jsonl"):
        p = os.path.join(rules_dir, fname)
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rules.append(json.loads(line))
    return rules


def load_all_rule_ids() -> list[str]:
    """Legacy helper — kept for back-compat; new code should use load_all_rules."""
    return [r["rule_id"] for r in load_all_rules()]


def retrieve_top_k(step_text: str, rules: list[dict], k: int = 8) -> list[dict]:
    """Rank rules by ontology-token overlap with the step; return top-K.

    Falls back to a substring/keyword heuristic if the ontology helper is
    unavailable (e.g. during unit tests without the full repo).
    """
    try:
        from pvgap_experiment.scripts.echem_ontology import ontology_tokens
    except Exception:
        ontology_tokens = None

    step_text_l = (step_text or "").lower()
    scored: list[tuple[int, int, dict]] = []  # (score, -len_obs, rule)
    step_toks = ontology_tokens(step_text) if ontology_tokens else set()
    for r in rules:
        rule_text = " ".join([
            r.get("observation", ""),
            " ".join(r.get("mechanism", [])),
            " ".join(r.get("alt_mechanisms", [])),
            " ".join(r.get("discriminators", [])),
        ])
        score = 0
        if ontology_tokens is not None:
            rule_toks = ontology_tokens(rule_text)
            score = len(step_toks & rule_toks)
        if score == 0:
            # fallback: substring overlap on observation head
            obs_head = (r.get("observation", "") or "")[:40].lower()
            if obs_head and obs_head in step_text_l:
                score = 1
        if score > 0:
            scored.append((score, -len(r.get("observation", "")), r))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [r for _, _, r in scored[:k]]


# ───────────────────── LLM wrapper ──────────────────────────────────

def _call_llm(messages: list[dict], model: str) -> str:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    try:
        from openai import OpenAI
        from config import MODELS, get_api_key
    except Exception as e:
        raise SystemExit(f"API deps not available: {e!r}") from e
    m = MODELS.get(model)
    if m is None:
        raise SystemExit(f"unknown model {model}; see config.py MODELS")
    client = OpenAI(api_key=get_api_key(m["provider"]), base_url=m.get("base_url"))
    resp = client.chat.completions.create(
        model=m["model_id"], messages=messages, temperature=0.2, max_tokens=300,
    )
    return resp.choices[0].message.content or ""


def _parse_judge_output(raw: str) -> dict | None:
    # find first {...} block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("label") not in {"good", "good_strong", "good_weak", "bad", "neutral"}:
        return None
    if not isinstance(obj.get("rule_cites", []), list):
        obj["rule_cites"] = []
    return obj


# ───────────────────── driver ──────────────────────────────────────

def run(cot_path: str, out_path: str,
        judge_a: str, judge_b: str,
        n_samples: int, max_steps: int, dry_run: bool,
        resume: bool = False) -> None:
    all_rules = load_all_rules()
    print(f"rule corpus: {len(all_rules)} rules loaded (full content)")

    rows = []
    with open(cot_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if n_samples and n_samples < len(rows):
        rows = rows[:n_samples]
    print(f"CoT traces: {len(rows)}")

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # --resume: scan existing out_path, skip (qid, step_idx) pairs already done.
    done_pairs: set[tuple[str, int]] = set()
    agreements = 0
    total = 0
    label_joint: dict[tuple[str, str], int] = {}
    if resume and os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for l in f:
                try:
                    o = json.loads(l)
                except Exception:
                    continue
                done_pairs.add((o["qid"], o["step_idx"]))
                total += 1
                agreements += int(o.get("agree", False))
                la = o.get(judge_a, {}).get("label")
                lb = o.get(judge_b, {}).get("label")
                if la and lb:
                    label_joint[(la, lb)] = label_joint.get((la, lb), 0) + 1
        print(f"resume: {len(done_pairs)} (qid,step) pairs already done, appending")

    fh_mode = "a" if (resume and done_pairs) else "w"
    fh = None if dry_run else open(out_path, fh_mode, encoding="utf-8")
    try:
        for row in rows:
            qid = row.get("qid", "?")
            cot = row.get("cot_text", "")
            if not cot:
                continue
            steps = split_cot_into_steps(cot, max_steps=max_steps)
            for i, step in enumerate(steps):
                if (qid, i) in done_pairs:
                    continue
                total += 1
                if dry_run:
                    print(f"  {qid} step {i}: {step[:60]!r}... (dry-run skip LLM)")
                    continue
                top_rules = retrieve_top_k(step, all_rules, k=8)
                ra = _parse_judge_output(_call_llm(build_judge_messages(step, top_rules), judge_a)) or \
                     {"label": "neutral", "rule_cites": [], "reason": "parse-fail"}
                rb = _parse_judge_output(_call_llm(build_judge_messages(step, top_rules), judge_b)) or \
                     {"label": "neutral", "rule_cites": [], "reason": "parse-fail"}
                # Collapse good_strong/good_weak → "good" for family-level agreement.
                def _fam(l):
                    return "good" if l in ("good", "good_strong", "good_weak") else l
                fam_a, fam_b = _fam(ra["label"]), _fam(rb["label"])
                agree = fam_a == fam_b
                agreements += int(agree)
                key = (ra["label"], rb["label"])
                label_joint[key] = label_joint.get(key, 0) + 1
                if agree:
                    # When both are "good" family but differ on strong/weak,
                    # take the weaker (more conservative) consensus.
                    if fam_a == "good":
                        priority = ["good_strong", "good_weak", "good"]
                        consensus = (ra["label"] if priority.index(ra["label"]) > priority.index(rb["label"])
                                     else rb["label"])
                    else:
                        consensus = ra["label"]
                else:
                    consensus = "disagree"
                fh.write(json.dumps({
                    "qid": qid, "step_idx": i, "step_text": step,
                    judge_a: ra, judge_b: rb,
                    "agree": agree, "consensus_label": consensus,
                }, ensure_ascii=False) + "\n")
    finally:
        if fh is not None:
            fh.close()

    if dry_run:
        print(f"DRY-RUN: would judge {total} steps × 2 judges = {total*2} API calls")
        return

    disagreement_rate = 1 - (agreements / total) if total else 0.0
    summary = {
        "cot_path": cot_path,
        "out_path": out_path,
        "n_steps": total,
        "n_agree": agreements,
        "disagreement_rate": disagreement_rate,
        "joint_labels": {f"{a}__{b}": n for (a, b), n in label_joint.items()},
        "decision_gate_9A2": ("PASS" if disagreement_rate < 0.15
                              else "FAIL — exceeds 15% threshold; add independent judge"),
    }
    with open(out_path + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="Stepwise CoT label bootstrap with 2-judge disagreement filter.")
    ap.add_argument("--cot", required=True, help="CoT JSONL input")
    ap.add_argument("--out", default="pvgap_experiment/results/stepwise_labels/pilot.jsonl",
                    help="stepwise-labelled JSONL output")
    ap.add_argument("--judge-a", default="gpt-4o", help="first judge model key (config.MODELS)")
    ap.add_argument("--judge-b", default="deepseek-v3", help="second judge model key")
    ap.add_argument("--n-samples", type=int, default=0, help="0 = use all CoT rows")
    ap.add_argument("--max-steps-per-trace", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="append to existing --out, skip already-labelled (qid,step_idx) pairs")
    args = ap.parse_args()
    run(args.cot, args.out, args.judge_a, args.judge_b,
        args.n_samples, args.max_steps_per_trace, args.dry_run,
        resume=args.resume)


if __name__ == "__main__":
    main()
