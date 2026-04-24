"""Build colab/9E_main.ipynb.

Drives the §9E.1 main run (C0/C1/C3 × seeds 0,1,2 × n=60) on Colab.

Design:
  - Cells 0..2: install (just openai), upload bundle, API key
  - Cell 3: optional upload of prior results/9E_main/*.jsonl for resume
  - Cell 4: USER-EDITABLE — pick which (config, seed) cells to run
  - Cell 5: kick off run_9e_main with that selection
  - Cell 6: summarise (works even on partial data)
  - Cell 7: zip & download results
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "9E_main.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


CELLS = [
    md("# §9E.1 main run — C0 / C1 / C3(critic-only) × seeds {0,1,2} × n=60\n\n"
       "Pilot v1 (n=5) showed **C3 critic-only** Δagg=+0.123 (+4/−1) vs C0. "
       "C3-full archived pending generator/extractor upgrade (§9E.2 future). "
       "This notebook drives the full paper-table run.\n\n"
       "**Wall time**: C0 ~30 min/seed, C1 ~60 min/seed, C3 ~180 min/seed. "
       "Total ≈ 13.5 h. Split across sessions via Cell 3 (upload prior "
       "results) + Cell 4 (select remaining cells)."),
    md("## 0. Install deps\n\n"
       "Only `openai` needed — no pybamm/sbi/pyimpspec for C0/C1/C3."),
    code(
        "import subprocess, sys\n"
        "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'openai'],\n"
        "                   capture_output=True, text=True)\n"
        "print(r.stdout.splitlines()[-1] if r.stdout.strip() else '(no output)')\n"
        "if r.returncode != 0: print('STDERR:', r.stderr[-1000:])"
    ),
    md("## 1. Upload bundle\n\nUpload `pvgap_9E_main.zip` (~45 KB)."),
    code(
        "import os, shutil\n"
        "from google.colab import files\n"
        "if os.path.isdir('/content/pvgap_experiment'):\n"
        "    shutil.rmtree('/content/pvgap_experiment')\n"
        "up = files.upload()\n"
        "zipname = next(iter(up))\n"
        "!unzip -q {zipname} -d /content/\n"
        "os.chdir('/content/pvgap_experiment')\n"
        "!ls src/ data/benchmark/"
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
    md("## 3. (Optional) Upload prior results for resume\n\n"
       "If you've already run some `(config, seed)` cells in a previous "
       "Colab session, upload their `*.jsonl` files here. They'll be placed "
       "in `results/9E_main/` and the runner will skip already-completed "
       "qids. Skip this cell on the first session."),
    code(
        "import os\n"
        "from google.colab import files\n"
        "os.makedirs('results/9E_main', exist_ok=True)\n"
        "try:\n"
        "    up = files.upload()\n"
        "    for name, data in up.items():\n"
        "        with open(f'results/9E_main/{name}', 'wb') as f:\n"
        "            f.write(data)\n"
        "    print('uploaded:', sorted(up.keys()))\n"
        "except Exception as ex:\n"
        "    print(f'(skipped: {ex})')\n"
        "!ls -la results/9E_main/ 2>/dev/null || echo '(empty)'"
    ),
    md("## 4. Select which cells to run\n\n"
       "Edit `CONFIGS` / `SEEDS` / `CONCURRENCY` below. With "
       "`CONCURRENCY=8` qid-level + C3's 4× internal candidate/critic "
       "fan-out, effective API concurrency is ~8 for C0/C1, ~32 for C3 "
       "(well within DeepSeek's per-key limits).\n\n"
       "Expected wall-time at `CONCURRENCY=8`:\n"
       "  - **C0** × 3 seeds ≈ 10 min (was 90 min sequential)\n"
       "  - **C1** × 3 seeds ≈ 20 min (was 180 min)\n"
       "  - **C3** × 3 seeds ≈ 60 min (was 540 min)\n"
       "  - full run one session ≈ 90 min total. If you hit rate-limit "
       "errors (HTTP 429), drop to `CONCURRENCY=4`."),
    code(
        "# EDIT ME\n"
        "# - To run the main table fresh:  ['C0','C1','C3']\n"
        "# - To add C3div on top of existing C0/C1/C3 (recommended now):\n"
        "#       CONFIGS = ['C3div']   (reuse prior C0/C1/C3 via Cell 3)\n"
        "CONFIGS     = ['C3div']\n"
        "SEEDS       = [0, 1, 2]\n"
        "N           = 60\n"
        "CONCURRENCY = 8                    # qid-level threads\n"
        "print(f'will run: {[(c,s) for c in CONFIGS for s in SEEDS]}  '\n"
        "      f'n={N}  concurrency={CONCURRENCY}')"
    ),
    md("## 5. Run selected cells\n\n"
       "Resumable: if a (config, seed) jsonl already exists from Cell 3 "
       "upload, completed qids are skipped. Each cell writes its jsonl "
       "incrementally (flush-on-row), so a mid-run crash still keeps "
       "the partial data."),
    code(
        "import subprocess, sys\n"
        "args = [sys.executable, '-m', 'src.run_9e_main',\n"
        "        '--n', str(N),\n"
        "        '--seeds', ','.join(str(s) for s in SEEDS),\n"
        "        '--configs', ','.join(CONFIGS),\n"
        "        '--concurrency', str(CONCURRENCY),\n"
        "        '--judge']\n"
        "print('>>>', ' '.join(args))\n"
        "# stream stdout live\n"
        "p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
        "                     text=True, bufsize=1)\n"
        "for line in p.stdout:\n"
        "    print(line, end='')\n"
        "p.wait()\n"
        "if p.returncode != 0:\n"
        "    raise SystemExit(f'run_9e_main failed rc={p.returncode}')"
    ),
    md("## 6. Summarise (paired Δ + bootstrap CI)\n\n"
       "Runs on whatever's present in `results/9E_main/`. Safe to call "
       "with partial data — useful as a progress check mid-run."),
    code(
        "import subprocess, sys\n"
        "r = subprocess.run([sys.executable, '-m', 'src.summarise_9e_main'],\n"
        "                   capture_output=True, text=True)\n"
        "print(r.stdout)\n"
        "if r.returncode != 0: print('STDERR:', r.stderr[-2000:])"
    ),
    md("## 7. Download all results"),
    code(
        "import shutil\n"
        "from google.colab import files\n"
        "shutil.make_archive('9E_main_results', 'zip', 'results/9E_main')\n"
        "files.download('9E_main_results.zip')"
    ),
]


NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                   "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}


def main():
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        json.dump(NB, fh, indent=1, ensure_ascii=False)
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
