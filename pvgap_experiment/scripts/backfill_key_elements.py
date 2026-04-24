"""Backfill `ground_truth.key_elements` for L2/L3/L4 scenarios.

L1 already has curated lists. L2–L4 have structured `ground_truth`
(diagnosis/degradation_type/severity/...) but no free-text key-phrase
targets. We derive a Chinese keyword list from `degradation_type` and
`severity` so the cheap substring-recall proxy still has something to
compare against. LLM-as-judge (next step) will use the full dict.

Writes to data/benchmark/echem_reason_benchmark.jsonl in place. The
original file is copied to echem_reason_benchmark.bak.jsonl first.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "data" / "benchmark" / "echem_reason_benchmark.jsonl"

# degradation_type → list of Chinese + English substrings a correct
# prediction is expected to surface. Kept short (3–5 items) so recall
# scales from 0 to 1 in reasonable steps.
DEG_KE: dict[str, list[str]] = {
    "healthy": ["健康", "无明显退化", "正常"],
    "LAM_negative": ["负极", "活性材料损失", "LAM"],
    "diffusion_degradation": ["扩散", "颗粒", "固态扩散"],
    "combined_degradation": ["复合退化", "LAM", "扩散"],
}

SEV_KE: dict[str, list[str]] = {
    "mild":     ["轻度"],
    "moderate": ["中度"],
    "severe":   ["严重"],
    "none":     [],
}


def derive(gt: dict, level: int) -> list[str]:
    dt = gt.get("degradation_type")
    sev = gt.get("severity")
    ke: list[str] = []
    if dt in DEG_KE:
        ke.extend(DEG_KE[dt])
    if sev in SEV_KE:
        ke.extend(SEV_KE[sev])
    # L3 adds multi-stage reasoning targets.
    if level == 3:
        stages = gt.get("stages", {})
        if stages.get("S1_data_quality", {}).get("kk_pass") is False:
            ke.append("KK")
        ke.append("电荷转移")   # S2 is EIS feature extraction
    # L4 is NoOp adversarial. Parse diagnosis string (e.g. "LAM_negative
    # (moderate)") to harvest underlying-mechanism keywords, then add
    # the stability cue.
    if level == 4:
        diag = str(gt.get("diagnosis", ""))
        for tag, kws in DEG_KE.items():
            if tag in diag:
                ke.extend(kws); break
        for sev_tag, kws in SEV_KE.items():
            if sev_tag in diag.lower():
                ke.extend(kws); break
        ke.append("不受影响")
    # De-dup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for k in ke:
        if k not in seen:
            seen.add(k); out.append(k)
    return out


def main():
    assert BENCH.is_file(), f"{BENCH} not found"
    bak = BENCH.with_suffix(".bak.jsonl")
    if not bak.exists():
        shutil.copyfile(BENCH, bak)
        print(f"backup → {bak.name}")

    rows = [json.loads(l) for l in open(BENCH, encoding="utf-8")]
    n_filled = 0
    for r in rows:
        lvl = r.get("level", 0)
        gt = r.get("ground_truth", {})
        if lvl >= 2 and not gt.get("key_elements"):
            ke = derive(gt, lvl)
            if ke:
                gt["key_elements"] = ke
                n_filled += 1

    with open(BENCH, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"filled key_elements on {n_filled}/{len(rows)} rows")
    # Per-level summary.
    from collections import Counter
    lvl_filled = Counter()
    for r in rows:
        if r.get("ground_truth", {}).get("key_elements"):
            lvl_filled[r.get("level")] += 1
    print(f"coverage by level: {dict(lvl_filled)}")


if __name__ == "__main__":
    main()
