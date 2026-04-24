"""One-shot merger for round-3 tutorial-mode cards → seed.

Hand-curated reject list based on manual review of 62 staged cards.
Generates sequential ER-111.. IDs and appends to
echem_rules_seed.jsonl (after backup).
"""
import json
import os
import re

BASE = r"D:/Obsidian/data/PHD/Self-Planning Prompting+reasoning LLM做电化学可解释性/pvgap_experiment/data/echem_rules"
SEED = f"{BASE}/echem_rules_seed.jsonl"

# Manual reject list (IDs within staging file) based on reviewer notes:
#   meta-only, definitional tautology, observation/mechanism reversed, or too generic.
REJECTS = {
    "lazanas_tut":   {1, 2, 4, 10, 11},
    "wang_mfc_tut":  {1, 2, 3, 4, 5, 9},
    "morali_lib_tut": {1, 2, 3, 4, 6, 16},
}

STAGING_FILES = [
    ("lazanas_tut",   f"{BASE}/staging/lazanas_tut.jsonl"),
    ("wang_mfc_tut",  f"{BASE}/staging/wang_mfc_tut.jsonl"),
    ("morali_lib_tut", f"{BASE}/staging/morali_lib_tut.jsonl"),
]


def load_seed_max_id(path):
    max_n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            m = re.match(r"ER-(\d+)", o.get("rule_id", ""))
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n


def main():
    max_id = load_seed_max_id(SEED)
    print(f"current seed max ER-id: {max_id}")

    bak = f"{SEED}.bak.er{max_id:03d}_r3tut"
    if not os.path.exists(bak):
        with open(SEED, encoding="utf-8") as src, open(bak, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        print(f"backup written: {bak}")

    new_rows = []
    for src_key, path in STAGING_FILES:
        reject_set = REJECTS.get(src_key, set())
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i in reject_set:
                    continue
                o = json.loads(line)
                # promote rule_id, mark provenance
                max_id += 1
                o["rule_id"] = f"ER-{max_id:03d}"
                o["_source"] = f"literature_mined_round3_tutorial_{src_key}"
                new_rows.append(o)
        print(f"  {src_key}: staged={i}, rejected={len(reject_set)}, "
              f"merged={i-len(reject_set)}")

    with open(SEED, "a", encoding="utf-8") as f:
        for o in new_rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"\nappended {len(new_rows)} cards. new seed max = ER-{max_id:03d}")


if __name__ == "__main__":
    main()
