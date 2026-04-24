"""Pack the §9E.1 main-run Colab bundle (C0/C1/C3 only, no heavy deps).

Contents:
  src/run_9e_pilot.py       — prediction functions (C0/C1/C3)
  src/run_9e_main.py        — (config × seed) driver, resumable
  src/summarise_9e_main.py  — paired-Δ bootstrap summariser
  src/sbi_prior_emit.py     — call_llm (DeepSeek/OpenAI wrapper)
  src/llm_judge.py          — LLM-as-judge 4-axis scorer
  src/__init__.py           — package marker
  data/benchmark/echem_reason_benchmark.jsonl

Excluded: everything C3full needs (pybamm/sbi/pyimpspec/ckpt/bridge).
Those stay in the C3-full archive bundle.

Writes pvgap_9E_main.zip in colab/ (~45 KB).
"""
from __future__ import annotations
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "pvgap_9E_main.zip"

FILES = [
    "src/__init__.py",
    "src/run_9e_pilot.py",
    "src/run_9e_main.py",
    "src/summarise_9e_main.py",
    "src/sbi_prior_emit.py",
    "src/llm_judge.py",
    "data/benchmark/echem_reason_benchmark.jsonl",
]


def main():
    missing = [f for f in FILES if not (ROOT / f).is_file()]
    if missing:
        raise SystemExit("missing files:\n  " + "\n  ".join(missing))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(ROOT / f, arcname=f"pvgap_experiment/{f}")
    print(f"wrote {OUT.name}  ({OUT.stat().st_size/1024:.1f} KB, "
          f"{len(FILES)} files)")


if __name__ == "__main__":
    main()
