"""Pack a minimal bundle for the §9E.1 pilot Colab run.

Contents:
  src/__init__.py
  src/run_9e_pilot.py
  src/llm_judge.py
  src/sbi_prior_emit.py    (only for call_llm; rest is dead code on Colab
                            but dropping it would require splitting the
                            module — not worth it for a 60 KB file)
  data/benchmark/echem_reason_benchmark.jsonl

Writes pvgap_9E1_pilot.zip in colab/.
"""
from __future__ import annotations
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "pvgap_9E1_pilot.zip"

FILES = [
    "src/__init__.py",
    "src/run_9e_pilot.py",
    "src/llm_judge.py",
    "src/sbi_prior_emit.py",
    "data/benchmark/echem_reason_benchmark.jsonl",
]


def main():
    missing = [f for f in FILES if not (ROOT / f).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(ROOT / f, arcname=f"pvgap_experiment/{f}")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.name}  ({size_kb:.1f} KB, {len(FILES)} files)")


if __name__ == "__main__":
    main()
