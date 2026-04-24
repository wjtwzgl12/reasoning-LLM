"""§9E.1 main-run summariser — paired deltas + bootstrap CIs.

Reads `results/9E_main/{config}_seed{seed}.jsonl` for every
(config, seed) present, computes:

  * per-cell means across 5 judge axes
  * paired Δ (C1−C0, C3−C0, C3−C1) at the (qid, seed) level
  * pooled-across-seeds Δ mean + bootstrap 95% CI (n_boot=2000)
  * per-level breakdown (L1/L2/L3/L4)

Writes `results/9E_main/_summary_v1.json` and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN_DIR = ROOT / "results" / "9E_main"

AXES = ("correctness", "grounding", "mechanism", "completeness", "aggregate")


def _load_cells(main_dir: Path) -> dict[tuple[str, int], list[dict]]:
    """Return {(config, seed): [rows]}."""
    out: dict[tuple[str, int], list[dict]] = {}
    for p in sorted(main_dir.glob("*_seed*.jsonl")):
        stem = p.stem  # e.g. "C3_seed1"
        try:
            cfg, seedtag = stem.rsplit("_seed", 1)
            seed = int(seedtag)
        except ValueError:
            continue
        rows = []
        for line in open(p, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        out[(cfg, seed)] = rows
    return out


def _judge_ok_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if r.get("error") is None and r.get("judge")
            and r["judge"].get("error") is None]


def _cell_means(rows: list[dict]) -> dict:
    ok = _judge_ok_rows(rows)
    d = {"n_total": len(rows), "n_judge_ok": len(ok)}
    if ok:
        for ax in AXES:
            d[ax] = mean(r["judge"][ax] for r in ok)
        d["key_recall"] = mean(r["key_recall"] for r in ok)
        d["latency_s"] = mean(r["latency_s"] for r in ok)
    return d


def _paired_deltas(rows_A: list[dict], rows_B: list[dict]) -> dict:
    """Δ = B − A, paired on qid. Returns {axis: [deltas across qids]}."""
    idx_A = {r["qid"]: r for r in _judge_ok_rows(rows_A)}
    idx_B = {r["qid"]: r for r in _judge_ok_rows(rows_B)}
    common = sorted(set(idx_A) & set(idx_B))
    if not common:
        return {ax: [] for ax in AXES}
    out = {ax: [idx_B[q]["judge"][ax] - idx_A[q]["judge"][ax]
                for q in common] for ax in AXES}
    out["_qids"] = common
    return out


def _bootstrap_ci(xs: list[float], n_boot: int = 2000, alpha: float = 0.05,
                  rng: random.Random | None = None) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a non-parametric resample."""
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng or random.Random(0)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (sum(xs) / n, lo, hi)


def summarise(main_dir: Path = MAIN_DIR) -> dict:
    cells = _load_cells(main_dir)
    configs = sorted({c for c, _ in cells})
    seeds = sorted({s for _, s in cells})
    print(f"found configs={configs}  seeds={seeds}  "
          f"cells={len(cells)}")

    per_cell = {f"{c}_seed{s}": _cell_means(cells[(c, s)])
                for (c, s) in cells}

    # Pool rows per config across seeds, keyed by (seed, qid).
    per_config: dict[str, list[dict]] = defaultdict(list)
    for (c, s), rows in cells.items():
        for r in rows:
            rr = dict(r); rr["_pair_key"] = (s, r["qid"])
            per_config[c].append(rr)

    # Paired Δ (B − A) pooled across seeds: join on (seed, qid).
    pairs = [("C1", "C0"), ("C3", "C0"), ("C3", "C1"),
             ("C3div", "C0"), ("C3div", "C1"), ("C3div", "C3"),
             ("C3full", "C0"), ("C3full", "C1"), ("C3full", "C3"),
             ("C3full", "C3div")]
    delta_summary = {}
    rng = random.Random(42)
    for B, A in pairs:
        if A not in per_config or B not in per_config:
            continue
        idx_A = {r["_pair_key"]: r for r in per_config[A]
                 if r.get("error") is None and r.get("judge")
                 and r["judge"].get("error") is None}
        idx_B = {r["_pair_key"]: r for r in per_config[B]
                 if r.get("error") is None and r.get("judge")
                 and r["judge"].get("error") is None}
        common = sorted(set(idx_A) & set(idx_B))
        entry = {"n_paired": len(common), "axes": {}}
        for ax in AXES:
            deltas = [idx_B[k]["judge"][ax] - idx_A[k]["judge"][ax]
                      for k in common]
            m, lo, hi = _bootstrap_ci(deltas, rng=rng)
            entry["axes"][ax] = {"mean": m, "ci95_lo": lo, "ci95_hi": hi,
                                  "n_pos": sum(1 for x in deltas if x > 0),
                                  "n_neg": sum(1 for x in deltas if x < 0)}
        # Per-level breakdown on aggregate axis.
        by_lvl: dict[int, list[float]] = defaultdict(list)
        for k in common:
            lvl = idx_A[k].get("level", 0)
            by_lvl[lvl].append(
                idx_B[k]["judge"]["aggregate"]
                - idx_A[k]["judge"]["aggregate"])
        entry["by_level_aggregate"] = {
            f"L{lvl}": {"n": len(xs), "mean_delta": mean(xs) if xs else float("nan")}
            for lvl, xs in sorted(by_lvl.items())}
        delta_summary[f"{B}_vs_{A}"] = entry

    out = {"per_cell": per_cell, "paired_deltas": delta_summary,
           "configs": configs, "seeds": seeds}

    # Pretty print markdown.
    print("\n### Per-cell means (judge.aggregate)")
    print("| config | " + " | ".join(f"seed{s}" for s in seeds)
          + " | mean |")
    print("|---|" + "---|" * (len(seeds) + 1))
    for c in configs:
        row = [c]
        vals = []
        for s in seeds:
            v = per_cell.get(f"{c}_seed{s}", {}).get("aggregate", float("nan"))
            vals.append(v)
            row.append(f"{v:.3f}")
        row.append(f"{mean([v for v in vals if v==v]):.3f}"
                   if any(v == v for v in vals) else "—")
        print("| " + " | ".join(row) + " |")

    print("\n### Paired Δ (bootstrap 95% CI, n_boot=2000)")
    print("| pair | axis | mean | 95% CI | n_paired | +/− |")
    print("|---|---|---|---|---|---|")
    for pair_name, e in delta_summary.items():
        for ax in AXES:
            a = e["axes"][ax]
            print(f"| {pair_name} | {ax} | {a['mean']:+.3f} | "
                  f"[{a['ci95_lo']:+.3f}, {a['ci95_hi']:+.3f}] | "
                  f"{e['n_paired']} | {a['n_pos']}/{a['n_neg']} |")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", default=str(MAIN_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main_dir = Path(args.main_dir)
    result = summarise(main_dir)
    out_path = Path(args.out) if args.out else main_dir / "_summary_v1.json"
    json.dump(result, open(out_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
