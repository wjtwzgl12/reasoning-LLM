"""§9E.1 main run — C0 / C1 / C3(critic-only) × seeds {0,1,2} × n=60.

Drives `run_9e_pilot.main()` via the same prediction functions, but:
  - loops (config, seed) pairs
  - writes per-(config, seed) JSONL with resumable qid-skip
  - deterministic scenario subset of size `--n` via the same seeded
    `pick_pilot_subset` (so seed controls both subset *and* generation
    randomness)
  - per-row LLM-judge (4-axis) when --judge

C3full is **not** scheduled by default — it's archived pending §9E.2
generator/extractor upgrade (see c3_full_predict docstring). Pass
`--include-c3full` to opt in.

Output layout:
  results/9E_main/{config}_seed{seed}.jsonl
  results/9E_main/_progress.json          ← resume pointer

Expected wall time (Colab, DeepSeek-V3):
  C0 × 60 × 3 seeds ≈ 90 min
  C1 × 60 × 3 seeds ≈ 180 min
  C3 × 60 × 3 seeds ≈ 540 min  (9 h)
  total ≈ 13.5 h, split across Colab sessions as (config, seed) cells.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .run_9e_pilot import (CONFIGS, load_benchmark, pick_pilot_subset,
                            keyelem_recall)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "results" / "9E_main"


def _load_done_qids(out_path: Path) -> set[str]:
    """Resume: read any already-written rows and return their qids."""
    if not out_path.is_file():
        return set()
    done = set()
    for line in open(out_path, encoding="utf-8"):
        try:
            done.add(json.loads(line)["qid"])
        except Exception:
            continue
    return done


def _log_row(row: dict, n_done: int, n_total: int, config: str,
             seed: int) -> None:
    j_tag = ""
    if "judge" in row and row["judge"].get("error") is None:
        j = row["judge"]
        j_tag = f"  agg={j['aggregate']:.2f} (c={j['correctness']:.2f})"
    err_tag = (f" ERR={row['error'][:40]}" if row.get("error") else "")
    elapsed = row.get("_t_total_s", row.get("latency_s", 0.0))
    print(f"  [{config} s{seed}] {n_done}/{n_total}  "
          f"{row['qid']:16s} L{row.get('level')}  "
          f"t={elapsed:.0f}s{err_tag}{j_tag}", flush=True)


def _process_one(c: dict, config: str, seed: int, predict_fn, judge: bool
                 ) -> dict:
    """Run predict + (optional) judge for one scenario. Pure function — no
    file I/O, safe to call from worker threads."""
    t0 = time.time()
    p = predict_fn(c)
    recall = keyelem_recall(p["pred"],
                            c["ground_truth"].get("key_elements", []))
    row = {
        "qid": c["qid"],
        "level": c.get("level"),
        "config": config,
        "seed": seed,
        "question": c["question_text"][:100],
        "pred_head": (p["pred"] or "")[:200],
        "pred_full": p["pred"] or "",
        "key_elements": c["ground_truth"].get("key_elements", []),
        "key_recall": recall,
        "latency_s": p["latency_s"],
        "error": p["error"],
        "_t_total_s": time.time() - t0,
    }
    if judge and p["pred"] and p["error"] is None:
        from .llm_judge import judge_case
        row["judge"] = judge_case(c["question_text"], p["pred"],
                                  c["ground_truth"], c.get("level", 0))
    if "bon_meta" in p:
        row["bon_meta"] = p["bon_meta"]
    return row


def run_cell(config: str, seed: int, n: int, judge: bool,
             out_dir: Path = OUT_DIR, concurrency: int = 1) -> dict:
    """Run a single (config, seed) cell. Returns a summary dict.

    concurrency: number of qids processed in parallel via ThreadPoolExecutor.
    Inside each qid, C3 already parallelises its 4 candidates + 4 critics.
    Effective API concurrency ≈ concurrency × 4 for C3, × 1 for C0/C1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config}_seed{seed}.jsonl"
    predict_fn = CONFIGS[config]

    all_cases = load_benchmark()
    subset = pick_pilot_subset(all_cases, n=n, seed=seed)
    done = _load_done_qids(out_path)
    todo = [c for c in subset if c["qid"] not in done]
    print(f"\n=== {config} seed={seed}  n={n}  resume {len(done)}/{len(subset)} "
          f"done; {len(todo)} remaining  concurrency={concurrency} "
          f"→ {out_path.name} ===", flush=True)

    mode = "a" if done else "w"
    t_cell0 = time.time()
    write_lock = threading.Lock()
    fh = open(out_path, mode, encoding="utf-8")
    completed = 0
    total_for_cell = len(todo)
    try:
        if concurrency <= 1:
            # Sequential path — unchanged behaviour.
            for c in todo:
                row = _process_one(c, config, seed, predict_fn, judge)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                completed += 1
                _log_row(row, len(done) + completed, len(subset), config, seed)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(_process_one, c, config, seed,
                                        predict_fn, judge): c for c in todo}
                for fut in as_completed(futures):
                    row = fut.result()
                    with write_lock:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                        completed += 1
                        _log_row(row, len(done) + completed, len(subset),
                                 config, seed)
    finally:
        fh.close()

    # Summary of cell.
    _ = total_for_cell  # silence unused
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    ok = [r for r in rows if r["error"] is None and r["pred_head"]]
    if not ok:
        return {"config": config, "seed": seed, "n_ok": 0,
                "out": str(out_path)}
    mean_recall = sum(r["key_recall"] for r in ok) / len(ok)
    mean_lat = sum(r["latency_s"] for r in ok) / len(ok)
    summary = {"config": config, "seed": seed, "n": n,
               "n_ok": len(ok), "n_total": len(rows),
               "mean_key_recall": mean_recall,
               "mean_latency_s": mean_lat,
               "wall_time_s": time.time() - t_cell0,
               "out": str(out_path)}
    j_ok = [r for r in ok if "judge" in r and r["judge"].get("error") is None]
    if j_ok:
        for ax in ("correctness", "grounding", "mechanism",
                   "completeness", "aggregate"):
            summary[f"judge_{ax}"] = sum(r["judge"][ax] for r in j_ok) / len(j_ok)
        summary["n_judge_ok"] = len(j_ok)
    print(f"  done  n_ok={summary['n_ok']}/{summary['n_total']}  "
          f"agg={summary.get('judge_aggregate', float('nan')):.3f}  "
          f"wall={summary['wall_time_s']/60:.1f}min")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60,
                    help="scenarios per (config, seed) cell")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated seeds")
    ap.add_argument("--configs", default="C0,C1,C3",
                    help="comma-separated configs; C3full archived by default")
    ap.add_argument("--include-c3full", action="store_true",
                    help="also run archived C3full (for ablation)")
    ap.add_argument("--judge", action="store_true", default=True)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--concurrency", type=int, default=8,
                    help="qid-level thread pool size; C3 further fans out "
                         "4× internally (candidates + critics)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    configs = [c for c in args.configs.split(",") if c.strip()]
    if args.include_c3full and "C3full" not in configs:
        configs.append("C3full")

    summaries = []
    for cfg in configs:
        if cfg not in CONFIGS:
            print(f"  skip unknown config={cfg}")
            continue
        for sd in seeds:
            s = run_cell(cfg, sd, args.n, args.judge, out_dir=out_dir,
                         concurrency=args.concurrency)
            summaries.append(s)

    # Write top-level summary.
    summ_path = out_dir / "_summaries.json"
    json.dump(summaries, open(summ_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {summ_path} ({len(summaries)} cells)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
