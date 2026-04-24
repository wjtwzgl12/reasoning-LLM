"""Build colab/9E1_c3full_pilot.ipynb.

Runs C3 (critic-only, already validated) + C3full (5-signal Weaver-BoN)
on the same 5 scenarios, seed=0, with --judge. Loads C0/C1 prior results
from the previously-downloaded pilot if uploaded; otherwise reruns them.

Gate: C3full must NOT regress vs C3 on aggregate (Δ ≥ -0.02), AND
      must recover at least 1 case where C3 picked wrong (targeted at
      L4_noop_006 which failed in the critic-only pilot).
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "9E1_c3full_pilot.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


CELLS = [
    md("# §9E.1 pilot — C3-full (5-signal Weaver-BoN)\n\n"
       "Compares **C3-full** against the previous critic-only C3 on the same "
       "5 pilot scenarios (seed=0). C3-full uses the full §3.11 weaver "
       "signal stack on each of N=4 CoT candidates:\n"
       "  - w_1 PRM (stubbed; §9A.3 ckpt not bundled — remains neutral)\n"
       "  - w_2 PyBaMM forward residual (gated by §9B.0 CC-008)\n"
       "  - w_3 lin-Kramers–Kronig soft score via `pyimpspec`\n"
       "  - w_4 SBI posterior match (§9C.2 v4 ckpt loaded)\n"
       "  - w_5 LLM critic (DeepSeek, 3-axis JSON)\n\n"
       "Aggregation: mean of non-stub signals; argmax over candidates.\n\n"
       "**Prior pilot (critic-only C3)**: agg=0.808 (vs C0 0.685, Δ=+0.123)."),
    md("## 0. Install heavy deps + **RESTART RUNTIME**\n\n"
       "≈ 5 min. SBI pulls torch + pyro; pybamm pulls casadi; pyimpspec "
       "is light. `pybamm` typically pulls a newer numpy which conflicts "
       "with Colab's pre-loaded numpy (ImportError `_center`).\n\n"
       "**The next cell auto-restarts the runtime after install.** Colab "
       "will show \"Your session crashed\" — that's expected. Just re-open "
       "and run from Cell 1 (upload bundle) onwards; deps are cached on "
       "the VM so the install is instant."),
    code(
        "import subprocess, sys, os\n"
        "# Install one-by-one so a failing package is identifiable. Keep\n"
        "# stderr visible (no -q) so Colab's Python-3.12 wheel gaps surface.\n"
        "PKGS = ['openai', 'pybamm', 'pybammeis', 'sbi', 'pyimpspec']\n"
        "failed = []\n"
        "for p in PKGS:\n"
        "    print(f'\\n--- pip install {p} ---', flush=True)\n"
        "    r = subprocess.run([sys.executable, '-m', 'pip', 'install', p],\n"
        "                       capture_output=True, text=True)\n"
        "    if r.returncode != 0:\n"
        "        print(f'  FAIL rc={r.returncode}')\n"
        "        print(r.stdout[-2000:]); print('STDERR:', r.stderr[-2000:])\n"
        "        failed.append(p)\n"
        "    else:\n"
        "        # Print last non-empty line (typically \"Successfully installed …\").\n"
        "        last = [l for l in r.stdout.splitlines() if l.strip()]\n"
        "        print(f'  ok: {last[-1] if last else \"(no output)\"}')\n"
        "if failed:\n"
        "    raise SystemExit(f'failed installs: {failed}')\n"
        "print('\\nall installs OK; restarting runtime …')\n"
        "os.kill(os.getpid(), 9)"
    ),
    md("## 1. Upload bundle\n\nUpload `pvgap_9E1_c3full.zip` (~196 KB)."),
    code(
        "import os, shutil\n"
        "from google.colab import files\n"
        "if os.path.isdir('/content/pvgap_experiment'):\n"
        "    shutil.rmtree('/content/pvgap_experiment')\n"
        "up = files.upload()                 # pvgap_9E1_c3full.zip\n"
        "zipname = next(iter(up))\n"
        "!unzip -q {zipname} -d /content/\n"
        "os.chdir('/content/pvgap_experiment')\n"
        "!ls -la src/ data/benchmark/ results/sbi_prior_emit/"
    ),
    md("## 2. API key"),
    code(
        "import os\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    os.environ['DEEPSEEK_API_KEY'] = userdata.get('DEEPSEEK_API_KEY')\n"
        "except Exception:\n"
        "    os.environ['DEEPSEEK_API_KEY'] = 'PASTE_KEY_HERE'\n"
        "assert os.environ.get('DEEPSEEK_API_KEY','').startswith('sk-')"
    ),
    md("## 3. Sanity: scenario bridge + SBI ckpt loadable"),
    code(
        "import json\n"
        "from src.scenario_bridge import bridge\n"
        "rows = [json.loads(l) for l in open('data/benchmark/echem_reason_benchmark.jsonl',encoding='utf-8')]\n"
        "l2 = next(r for r in rows if r.get('level')==2 and r['ground_truth'].get('physical_params'))\n"
        "c = bridge(l2)\n"
        "if c is None or '_bridge_error' in (c or {}):\n"
        "    raise SystemExit(f'bridge failed for {l2[\"qid\"]}: {c}')\n"
        "print(f'bridged qid={l2[\"qid\"]}: keys={list(c)}')\n"
        "print(f'  Z.shape={c[\"observed_Z\"].shape}, T={c[\"observed_temperature_K\"]}K')\n"
        "from src.sbi_w4_scorer import Wfour\n"
        "import time; t0=time.time(); w4 = Wfour(); print(f'W4 loaded in {time.time()-t0:.1f}s')"
    ),
    md("## 4. Run C3 (critic-only) + C3full (5-signal) on n=5\n\n"
       "Expected ~25–35 min total (C3 ≈ 13 min, C3full ≈ 18 min with w_4 "
       "sampling overhead)."),
    code(
        "import subprocess, sys, os\n"
        "os.makedirs('results', exist_ok=True)\n"
        "for cfg in ('C3','C3full'):\n"
        "    out = f'results/9E1_pilot_{cfg}_v2.jsonl'\n"
        "    print(f'\\n=== {cfg} → {out} ===', flush=True)\n"
        "    r = subprocess.run(\n"
        "        [sys.executable, '-m', 'src.run_9e_pilot',\n"
        "         '--config', cfg, '--n', '5', '--seed', '0',\n"
        "         '--judge', '--out', out],\n"
        "        capture_output=True, text=True)\n"
        "    print(r.stdout[-4000:])\n"
        "    if r.returncode != 0:\n"
        "        print('STDERR:', r.stderr[-4000:])\n"
        "        raise SystemExit(f'{cfg} failed rc={r.returncode}')"
    ),
    md("## 5. Paired deltas vs critic-only C3"),
    code(
        "import json, statistics\n"
        "def load(p):\n"
        "    return {json.loads(l)['qid']: json.loads(l) for l in open(p,encoding='utf-8')}\n"
        "c3  = load('results/9E1_pilot_C3_v2.jsonl')\n"
        "c3f = load('results/9E1_pilot_C3full_v2.jsonl')\n"
        "qids = sorted(set(c3) & set(c3f))\n"
        "axes = ('correctness','grounding','mechanism','completeness','aggregate')\n"
        "summary = {'n_paired': len(qids), 'per_config': {}, 'paired_deltas_vs_C3': {}}\n"
        "for tag,d in (('C3',c3),('C3full',c3f)):\n"
        "    summary['per_config'][tag] = {ax: sum(d[q]['judge'][ax] for q in qids)/len(qids) for ax in axes}\n"
        "for tag in ('C3','C3full'):\n"
        "    print(f'  {tag}:  ' + '  '.join(f'{ax}={summary[\"per_config\"][tag][ax]:.3f}' for ax in axes))\n"
        "print('\\n--- paired Δ C3full vs C3 ---')\n"
        "for ax in axes:\n"
        "    ds = [c3f[q]['judge'][ax] - c3[q]['judge'][ax] for q in qids]\n"
        "    mu = statistics.mean(ds); sd = statistics.stdev(ds) if len(ds)>1 else 0\n"
        "    pos = sum(1 for x in ds if x>0); neg = sum(1 for x in ds if x<0)\n"
        "    summary['paired_deltas_vs_C3'][ax] = {'mean':mu,'sd':sd,'n_pos':pos,'n_neg':neg,'deltas':ds}\n"
        "    print(f'  {ax:12s}  Δ={mu:+.3f} (sd={sd:.3f}, +{pos}/-{neg})')\n"
        "print('\\n--- per-scenario correctness C3 vs C3full ---')\n"
        "for q in qids:\n"
        "    print(f'  {q:16s} L{c3[q][\"level\"]}  C3={c3[q][\"judge\"][\"correctness\"]:.2f}  '\n"
        "          f'C3full={c3f[q][\"judge\"][\"correctness\"]:.2f}  '\n"
        "          f'Δ={c3f[q][\"judge\"][\"correctness\"]-c3[q][\"judge\"][\"correctness\"]:+.2f}')\n"
        "print('\\n--- C3full per-candidate diagnosis (5signal cases only) ---')\n"
        "for q in qids:\n"
        "    bm = c3f[q].get('bon_meta', {})\n"
        "    if bm.get('mode') != '5signal': continue\n"
        "    diag = bm.get('per_candidate_diagnosis', [])\n"
        "    w_raw = bm.get('per_candidate_w_raw', [])\n"
        "    print(f'  {q}:')\n"
        "    for i,(d,w) in enumerate(zip(diag, w_raw)):\n"
        "        print(f'    cand{i}: mech={d.get(\"mech_detected\")} sev={d.get(\"severity_detected\")}  '\n"
        "              f'w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f},{w[4]:.2f}]')\n"
        "# Gate: not worse on aggregate (Δ≥-0.02) AND recover ≥1 case.\n"
        "agg = summary['paired_deltas_vs_C3']['aggregate']\n"
        "corr = summary['paired_deltas_vs_C3']['correctness']\n"
        "no_regress = agg['mean'] >= -0.02\n"
        "recovered = corr['n_pos'] >= 1\n"
        "passed = no_regress and recovered\n"
        "summary['gate_passed'] = passed\n"
        "summary['gate_reason'] = f'no_regress={no_regress} recovered={recovered}'\n"
        "print(f'\\n§9E.1 C3-full gate: ' + ('PASS' if passed else 'FAIL')\n"
        "      + f'  (agg Δ={agg[\"mean\"]:+.3f}, correctness recovered={corr[\"n_pos\"]})')\n"
        "json.dump(summary, open('results/9E1_c3full_pilot_summary_v2.json','w',encoding='utf-8'),\n"
        "          indent=2, ensure_ascii=False)"
    ),
    md("## 6. Download artefacts"),
    code(
        "from google.colab import files\n"
        "for f in ('results/9E1_pilot_C3_v2.jsonl',\n"
        "          'results/9E1_pilot_C3full_v2.jsonl',\n"
        "          'results/9E1_c3full_pilot_summary_v2.json'):\n"
        "    files.download(f)"
    ),
]


NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info":{"name":"python"}, "colab":{"provenance":[]}},
      "nbformat": 4, "nbformat_minor": 5}


def main():
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        json.dump(NB, fh, indent=1, ensure_ascii=False)
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
