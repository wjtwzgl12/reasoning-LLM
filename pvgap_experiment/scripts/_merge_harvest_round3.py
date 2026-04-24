"""Merge hand-reviewed OA-harvest round-3 cards → seed.

Reviewer verdicts (12 staged → 10 kept):
  SCOPE NOTE: corpus is *electrochemistry*, not just EIS. CV / GCD /
  Tafel / LSV features are legitimate diagnostic observables too.
  Earlier "off-method" rejects for CV peaks are reversed here.

  supercap_40243:     0/1 — vague "high performance"
  jpowsour_2014:      3/3 — all legit PEM-FC EIS
  ncomms4529:         1/1 — impedance buildup ↔ surface reconstruction
  s41598_2017 (bio):  2/3 — #1 meaningless; #2 T-dep conductivity (EIS);
                            #3 CV three-oxidation-peaks → AP state shuttle (KEEP — CV is in scope)
  s41598_2024:        1/1 — Rs(CRct)(QR)(CR) equivalent circuit
  s11671_2016:        1/1 — recombination ↔ interfacial modification
  briac_2013:         2/2 — classic corrosion-inhibitor EIS
"""
import json, os, re

BASE = r"D:/Obsidian/data/PHD/Self-Planning Prompting+reasoning LLM做电化学可解释性/pvgap_experiment/data/echem_rules"
SEED = f"{BASE}/echem_rules_seed.jsonl"
STAGE = f"{BASE}/staging/oa_harvest"

KEEP = {
    "10_1007_s40243_018_0136_6.jsonl":      set(),
    "10_1016_j_jpowsour_2014_12_045.jsonl": {1, 2, 3},
    "10_1038_ncomms4529.jsonl":             {1},
    "10_1038_s41598_017_17486_9.jsonl":     {2, 3},
    "10_1038_s41598_024_79758_5.jsonl":     {1},
    "10_1186_s11671_016_1540_4.jsonl":      {1},
    "10_33263_briac115_1301913030.jsonl":   {1, 2},
}


def max_id():
    n = 0
    with open(SEED, encoding="utf-8") as f:
        for l in f:
            m = re.match(r"ER-(\d+)", json.loads(l).get("rule_id", ""))
            if m: n = max(n, int(m.group(1)))
    return n


def main():
    cur = max_id()
    print(f"seed max ER-{cur:03d}")
    bak = f"{SEED}.bak.er{cur:03d}_r3harv"
    if not os.path.exists(bak):
        with open(SEED, encoding="utf-8") as s, open(bak, "w", encoding="utf-8") as d:
            d.write(s.read())
        print(f"backup: {bak}")
    added = []
    for fname, keep_ids in KEEP.items():
        if not keep_ids: continue
        with open(f"{STAGE}/{fname}", encoding="utf-8") as f:
            for i, l in enumerate(f, 1):
                if i not in keep_ids: continue
                o = json.loads(l)
                cur += 1
                o["rule_id"] = f"ER-{cur:03d}"
                doi_key = fname.replace(".jsonl", "").replace("_", "/", 1)
                o["_source"] = f"literature_mined_round3_oa_harvest:{doi_key}"
                added.append(o)
    with open(SEED, "a", encoding="utf-8") as f:
        for o in added:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"added {len(added)}, new max ER-{cur:03d}")


if __name__ == "__main__":
    main()
