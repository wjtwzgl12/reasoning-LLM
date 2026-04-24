"""Build colab/9E1_pilot.ipynb — §9E.1 pilot runner for Colab.

Notebook flow:
  0. Install deps (openai only; judge uses same client)
  1. Upload pvgap_9E1_pilot.zip  → /content/pvgap_experiment
  2. Set DEEPSEEK_API_KEY from Colab secrets (or literal paste)
  3. Run C0, C1, C3 on n=5 scenarios × seed=0, with --judge
  4. Compute paired deltas per axis; write summary JSON
  5. Download results/*.jsonl + summary

Build with:
    python colab/_build_9E1_pilot_notebook.py
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "9E1_pilot.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


CELLS = [
    md("# §9E.1 pilot — C0 vs C1 vs C3 (Weaver-BoN)\n\n"
       "Paired pilot on 5 scenarios (seed=0) with LLM-as-judge 4-axis "
       "scoring. Goal: validate that **C3 (BoN + critic) uniformly beats "
       "C0 (plain) and C1 (CoT)** before committing to the full "
       "5 configs × 60 scenarios × 3 seeds run (~6h).\n\n"
       "Expected C0 baseline (local): aggregate=0.672, correctness=0.450.\n"
       "Expected C1 (local): aggregate=0.690 (ns), correctness=-0.130."),
    md("## 0. Install\n\nColab pre-installs numpy, scipy. We only need `openai`."),
    code("!pip install -q openai"),
    md("## 1. Upload bundle\n\nUpload `pvgap_9E1_pilot.zip` when prompted."),
    code(
        "import os, shutil\n"
        "from google.colab import files\n"
        "if os.path.isdir('/content/pvgap_experiment'):\n"
        "    shutil.rmtree('/content/pvgap_experiment')\n"
        "up = files.upload()                 # expect pvgap_9E1_pilot.zip\n"
        "zipname = next(iter(up))\n"
        "!unzip -q {zipname} -d /content/\n"
        "os.chdir('/content/pvgap_experiment')\n"
        "!ls -la src/ data/benchmark/"
    ),
    md("## 2. API key\n\nSet from Colab Secrets (gear icon → Secrets → "
       "`DEEPSEEK_API_KEY`), or paste below."),
    code(
        "import os\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    os.environ['DEEPSEEK_API_KEY'] = userdata.get('DEEPSEEK_API_KEY')\n"
        "except Exception:\n"
        "    os.environ['DEEPSEEK_API_KEY'] = 'PASTE_KEY_HERE'\n"
        "assert os.environ.get('DEEPSEEK_API_KEY','').startswith('sk-'), (\n"
        "    'set DEEPSEEK_API_KEY')"
    ),
    md("## 3. Run C0 / C1 / C3 on 5 scenarios\n\n"
       "Each config writes to `results/9E1_pilot_{Cx}_v1.jsonl`. C3 does "
       "N=4 candidates per scenario + critic scoring → 5× LLM calls per "
       "scenario. Total expected runtime ~15–25 min."),
    code(
        "import subprocess, sys, os\n"
        "os.makedirs('results', exist_ok=True)\n"
        "for cfg in ('C0','C1','C3'):\n"
        "    out = f'results/9E1_pilot_{cfg}_v1.jsonl'\n"
        "    print(f'\\n=== running {cfg} → {out} ===', flush=True)\n"
        "    r = subprocess.run(\n"
        "        [sys.executable, '-m', 'src.run_9e_pilot',\n"
        "         '--config', cfg, '--n', '5', '--seed', '0',\n"
        "         '--judge', '--out', out],\n"
        "        capture_output=True, text=True)\n"
        "    print(r.stdout)\n"
        "    if r.returncode != 0:\n"
        "        print('STDERR:', r.stderr[-2000:])\n"
        "        raise SystemExit(f'{cfg} failed rc={r.returncode}')"
    ),
    md("## 4. Paired deltas + gate"),
    code(
        "import json, statistics, math\n"
        "def load(p):\n"
        "    return {json.loads(l)['qid']: json.loads(l) for l in open(p,encoding='utf-8')}\n"
        "c0 = load('results/9E1_pilot_C0_v1.jsonl')\n"
        "c1 = load('results/9E1_pilot_C1_v1.jsonl')\n"
        "c3 = load('results/9E1_pilot_C3_v1.jsonl')\n"
        "qids = sorted(set(c0) & set(c1) & set(c3))\n"
        "axes = ('correctness','grounding','mechanism','completeness','aggregate')\n"
        "summary = {'n_paired': len(qids), 'axes': list(axes),\n"
        "           'per_config': {}, 'paired_deltas_vs_C0': {}}\n"
        "for tag, d in (('C0',c0),('C1',c1),('C3',c3)):\n"
        "    summary['per_config'][tag] = {\n"
        "        ax: sum(d[q]['judge'][ax] for q in qids)/len(qids)\n"
        "        for ax in axes}\n"
        "print(f'--- config means (n_paired={len(qids)}) ---')\n"
        "for tag in ('C0','C1','C3'):\n"
        "    print(f'  {tag}:  ' + '  '.join(\n"
        "        f'{ax}={summary[\"per_config\"][tag][ax]:.3f}' for ax in axes))\n"
        "print('\\n--- paired Δ vs C0 ---')\n"
        "for tag, d in (('C1',c1),('C3',c3)):\n"
        "    summary['paired_deltas_vs_C0'][tag] = {}\n"
        "    for ax in axes:\n"
        "        deltas = [d[q]['judge'][ax] - c0[q]['judge'][ax] for q in qids]\n"
        "        mu = statistics.mean(deltas)\n"
        "        sd = statistics.stdev(deltas) if len(deltas)>1 else 0.0\n"
        "        pos = sum(1 for x in deltas if x>0); neg = sum(1 for x in deltas if x<0)\n"
        "        summary['paired_deltas_vs_C0'][tag][ax] = {\n"
        "            'mean': mu, 'sd': sd, 'n_pos': pos, 'n_neg': neg}\n"
        "        print(f'  {tag}.{ax:12s}  Δ={mu:+.3f} (sd={sd:.3f}, +{pos}/-{neg})')\n"
        "# Gate: C3 must beat C0 on aggregate by ≥+0.05 with +k/-k ratio k≥4/5.\n"
        "c3_agg = summary['paired_deltas_vs_C0']['C3']['aggregate']\n"
        "passed = c3_agg['mean'] >= 0.05 and c3_agg['n_pos'] >= 4\n"
        "summary['gate_passed'] = passed\n"
        "verdict = 'PASS' if passed else 'FAIL'\n"
        "mu = c3_agg['mean']; npos = c3_agg['n_pos']; nneg = c3_agg['n_neg']\n"
        "print(f'\\n§9E.1 pilot gate (C3 vs C0 aggregate): {verdict}  '\n"
        "      f'Δ={mu:+.3f} (+{npos}/-{nneg})')\n"
        "with open('results/9E1_pilot_summary_v1.json','w',encoding='utf-8') as fh:\n"
        "    json.dump(summary, fh, indent=2, ensure_ascii=False)"
    ),
    md("## 5. Download artefacts"),
    code(
        "from google.colab import files\n"
        "for f in ('results/9E1_pilot_C0_v1.jsonl',\n"
        "          'results/9E1_pilot_C1_v1.jsonl',\n"
        "          'results/9E1_pilot_C3_v1.jsonl',\n"
        "          'results/9E1_pilot_summary_v1.json'):\n"
        "    files.download(f)"
    ),
]


NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        json.dump(NB, fh, indent=1, ensure_ascii=False)
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
