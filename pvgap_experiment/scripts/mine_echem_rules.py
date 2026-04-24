"""
9A.1 rule-mining scaffold — from source text to EChemRules JSONL.
=========================================================================

Reads a plain-text source file (a textbook chapter, a paper section, a DRT
review), segments it, and asks an LLM to extract diagnostic rule cards
conforming to the EChemRules schema. Output is a *staging* JSONL that a
human must review before merging into echem_rules_seed.jsonl.

Usage
-----
    # dry-run (no API calls, prints what would be sent)
    python -m pvgap_experiment.scripts.mine_echem_rules \\
        --source refs/orazem_tribollet_ch13.txt --dry-run

    # real extraction (requires API key in env via config.py)
    python -m pvgap_experiment.scripts.mine_echem_rules \\
        --source refs/orazem_tribollet_ch13.txt \\
        --out   data/echem_rules/staging/orazem_ch13.jsonl \\
        --judge gpt-4o --max-segments 20

Contract
--------
The script does *not* auto-merge into `echem_rules_seed.jsonl`. Every
extracted card lands in `staging/` and must pass a human review step
(acceptance criterion from §9A.1 decision gate: 100-segment manual
spot-check precision ≥ 90%). This is enforced by design: the staging
path is the only write target.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from typing import Iterable


# ───────────────────── source segmentation ──────────────────────────

def segment_source(text: str, max_chars: int = 1500) -> list[str]:
    """Split source text into paragraph-level chunks capped at max_chars.

    Prefers paragraph breaks, then sentence breaks, then hard cuts.
    """
    paras = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        # overlong paragraph: split on sentences
        sents = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sents:
            if len(buf) + len(s) + 1 <= max_chars:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)
    return chunks


# ───────────────────── extraction prompt ────────────────────────────

EXTRACTION_PROMPT = """You extract diagnostic rule cards from an electrochemistry source passage.

Source passage:
---
{passage}
---

Task: emit 0 to 3 JSON objects (one per line, no trailing commas, no prose)
following this exact schema:

{{
  "rule_id":         "STAGED-{source_key}-XXXX",
  "observation":     "<single observable EIS feature or pattern, concrete>",
  "mechanism":       ["<mechanism_1>", ...],
  "alt_mechanisms":  ["<alt_1>", ...],
  "discriminators":  ["<how to tell them apart>", ...],
  "confidence":      "primary" | "secondary" | "weak",
  "level":           "feature" | "fit" | "drt" | "meta" | "gate",
  "sources":         ["<short_citation_key>"],
  "units":           {{"<key>": [<low>, <high>] or <value>}},
  "applies_to":      ["<system>", ...],
  "counterexamples": ["<when the rule fails>", ...],
  "evidence_quote":  "<verbatim substring from the passage above, 10-240 chars, supports the observation AND the mechanism>"
}}

Strict constraints:
- If the passage does not contain a concrete diagnostic claim, emit NOTHING.
- `evidence_quote` MUST be a verbatim substring copied from the passage above. Do NOT paraphrase. Do NOT use ellipses. The exact bytes must appear in the passage. If you cannot find such a substring, emit NOTHING — do not guess from background knowledge.
- `observation` and `mechanism` must both be supported by `evidence_quote`. If the passage discusses only the tool, method, or workflow (not a feature→mechanism claim), emit NOTHING.
- If confidence is "primary", `counterexamples` MUST be non-empty and specific.
- `observation` must be an EIS feature (not a degradation cause itself).
- `mechanism` is what the observation points TO.
- Source key for `sources` field: "{source_key}".
- No preamble, no trailing commentary — only the JSON lines (or nothing).
"""


TUTORIAL_EXTRACTION_PROMPT = """You extract diagnostic rule cards from a TUTORIAL/REVIEW electrochemistry passage.

Source passage (tutorial/review — teaches GENERIC feature→mechanism mappings, not a
specific measurement on a specific cell):
---
{passage}
---

Task: emit 0 to 3 JSON objects (one per line, no trailing commas, no prose)
using the EChemRules schema (same fields as the standard extractor).

CRITICAL — this is a tutorial-aware variant. Tutorial text commonly teaches
"feature X in an impedance plot corresponds to mechanism Y" in GENERIC form
(no specific cell/experiment). You MUST accept such generic teaching claims
when they map an observable EIS feature to a mechanism, even if no numerical
result or specific system is mentioned. Examples of acceptable tutorial
claims:
  - "The high-frequency semicircle corresponds to charge transfer at the
    electrode/electrolyte interface."  → feature=HF semicircle, mech=charge transfer
  - "A 45° line at low frequency is characteristic of semi-infinite Warburg
    diffusion."  → feature=45° line, mech=Warburg diffusion
  - "An inductive loop at low frequency typically indicates adsorbed
    intermediates or corrosion-relaxation processes."  → feature=LF inductive loop

Rules specific to tutorial mode:
  * Set `confidence` to "secondary" for standard textbook mappings and "weak"
    for purely definitional statements (e.g. "Rct is the charge transfer
    resistance"). Do NOT emit "primary" for tutorial claims — tutorials
    almost never include proper counterexamples.
  * `applies_to` should be ["generic"] unless the passage explicitly names a
    chemistry/system (e.g. "lithium-ion batteries", "coatings").
  * Set `counterexamples` to a SPECIFIC alternative interpretation when the
    passage names one (e.g. "could also reflect a porous electrode if the
    semicircle is depressed"); otherwise leave [] and DO NOT claim primary.

Unchanged constraints (still strict):
  * `evidence_quote` is a VERBATIM substring of the passage above, 10-240
    chars, no ellipses, no paraphrase. If you cannot find one that supports
    BOTH the observation and the mechanism, emit NOTHING for that claim.
  * `observation` must be an observable EIS feature (semicircle, arc, slope,
    peak, DRT peak, phase angle, intercept, inductive loop, …), NOT the
    mechanism itself.
  * If the passage is pure motivation or describes the paper's structure
    (methods overview, "this review covers …"), emit NOTHING.

JSON schema:
{{
  "rule_id":         "STAGED-{source_key}-XXXX",
  "observation":     "<observable EIS feature>",
  "mechanism":       ["<mechanism_1>", ...],
  "alt_mechanisms":  ["<alt_1>", ...],
  "discriminators":  ["<how to tell them apart>", ...],
  "confidence":      "secondary" | "weak",
  "level":           "feature" | "fit" | "drt" | "meta" | "gate",
  "sources":         ["{source_key}"],
  "units":           {{}},
  "applies_to":      ["<system or 'generic'>"],
  "counterexamples": [],
  "evidence_quote":  "<verbatim substring from the passage above>"
}}

No preamble, no trailing commentary — only JSON lines (or nothing).
"""


def build_extraction_messages(passage: str, source_key: str,
                              mode: str = "standard") -> list[dict]:
    template = EXTRACTION_PROMPT if mode != "tutorial" else TUTORIAL_EXTRACTION_PROMPT
    sys_msg = ("You are an electrochemistry librarian. Output only valid JSON lines."
               if mode != "tutorial" else
               "You are an electrochemistry tutor. Extract generic feature→mechanism "
               "teaching claims as JSON rule cards. Output only valid JSON lines.")
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": template.format(passage=passage, source_key=source_key)},
    ]


# ───────────────────── LLM wrapper (lazy import) ────────────────────

def _call_llm(messages: list[dict], model: str) -> str:
    """Call OpenAI-compatible API via existing api_client.py / config.py.

    Deferred import so --dry-run works without API deps.
    """
    # Reuse project's config + SDK wiring
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    try:
        from openai import OpenAI
        from config import MODELS, get_api_key
    except Exception as e:
        raise SystemExit(f"API deps not available: {e!r}") from e
    m = MODELS.get(model)
    if m is None:
        raise SystemExit(f"unknown model {model}; see config.py MODELS")
    client = OpenAI(api_key=get_api_key(m["provider"]), base_url=m.get("base_url"))
    resp = client.chat.completions.create(
        model=m["model_id"], messages=messages,
        temperature=0.2, max_tokens=1200,
    )
    return resp.choices[0].message.content or ""


# ───────────────────── output validation ────────────────────────────

REQUIRED_FIELDS = [
    "rule_id", "observation", "mechanism", "alt_mechanisms",
    "discriminators", "confidence", "level", "sources",
    "units", "applies_to", "counterexamples", "evidence_quote",
]
VALID_CONF = {"primary", "secondary", "weak"}
VALID_LEVEL = {"feature", "fit", "drt", "meta", "gate"}


def _normalize_for_quote_match(s: str) -> str:
    # Case-fold, collapse whitespace, strip curly quotes; keep other punctuation
    # so the quote check is "did the LLM copy text that actually appears?".
    s = s.lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def validate_card(obj: dict, passage: str = "") -> tuple[bool, str]:
    for k in REQUIRED_FIELDS:
        if k not in obj:
            return False, f"missing field: {k}"
    if obj["confidence"] not in VALID_CONF:
        return False, f"bad confidence: {obj['confidence']}"
    if obj["level"] not in VALID_LEVEL:
        return False, f"bad level: {obj['level']}"
    if obj["confidence"] == "primary" and not obj.get("counterexamples"):
        return False, "primary rule without counterexamples (honesty constraint)"
    # Verbatim-quote grounding check.
    q = obj.get("evidence_quote", "")
    if not isinstance(q, str) or not (10 <= len(q) <= 240):
        return False, f"evidence_quote length out of range: {len(q) if isinstance(q, str) else 'not-str'}"
    if passage:
        if _normalize_for_quote_match(q) not in _normalize_for_quote_match(passage):
            return False, "evidence_quote not found verbatim in passage"
        # Section-header free-ride gate: if the quote (with leading/trailing dashes
        # stripped) does not appear in the passage body AFTER removing all
        # "--- ... ---" header lines, it's just echoing a section title.
        q_core = re.sub(r"^[-\s]+", "", q)
        q_core = re.sub(r"[-\s]+$", "", q_core)
        passage_body = re.sub(r"(?m)^\s*-{2,}.*-{2,}\s*$", "", passage)
        if q_core and _normalize_for_quote_match(q_core) not in _normalize_for_quote_match(passage_body):
            return False, "evidence_quote only echoes a section header (free-ride)"
    # Obs↔quote relevance: at least one non-stopword obs token must appear in quote,
    # to block "grounded section-header or title quote that doesn't support the obs".
    # Gate is two-track (ontology-aware): a card passes if EITHER the raw-token
    # overlap is non-empty, OR the ontology-node overlap is non-empty. This lets
    # tutorial/textbook sources pass when the obs says "charge transfer resistance"
    # and the quote says "R_ct" (previously rejected by raw-token track).
    _stop = {"the","a","an","and","or","of","to","in","on","for","with","by","is","are",
             "be","this","that","as","at","can","may","which","from","its","it","into",
             "not","such","between","when","than","then","also","rule","observation"}
    def _toks(s):
        return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _stop and len(t) > 2}
    obs_t = _toks(obj.get("observation", ""))
    q_t = _toks(q)
    raw_overlap = bool(obs_t & q_t)
    ontology_overlap_hit = False
    if obs_t and not raw_overlap:
        # Lazy import so --dry-run / test environments without the ontology file
        # still run (raw-token-only behavior, backward compatible).
        try:
            from pvgap_experiment.scripts.echem_ontology import ontology_overlap
            ontology_overlap_hit = ontology_overlap(obj.get("observation", ""), q) > 0
        except Exception:
            ontology_overlap_hit = False
    if obs_t and not (raw_overlap or ontology_overlap_hit):
        return False, "no observation token or ontology node appears in evidence_quote (weak grounding)"
    return True, ""


def parse_llm_output(raw: str) -> Iterable[dict]:
    # Strip markdown code fences (```json ... ``` or ``` ... ```).
    text = raw.strip()
    if text.startswith("```"):
        # drop first fence line and trailing fence
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    # Scan for balanced top-level JSON objects (handles pretty-printed multi-line JSON
    # and multiple objects separated by whitespace or commas).
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[i:j + 1]
                        try:
                            yield json.loads(chunk)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            break  # unbalanced — stop


# ───────────────────── driver ──────────────────────────────────────

def run(source_path: str, out_path: str, model: str, max_segments: int,
        source_key: str, dry_run: bool, mode: str = "standard") -> None:
    with open(source_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = segment_source(text)[:max_segments]
    print(f"segmented source into {len(chunks)} chunks (cap {max_segments})")

    if not dry_run:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
    accepted, rejected = 0, 0

    fh = None if dry_run else open(out_path, "w", encoding="utf-8")
    try:
        for i, chunk in enumerate(chunks):
            print(f"  [{i+1:02d}/{len(chunks)}] {chunk[:60]!r}...")
            if dry_run:
                print("    (dry-run; would call LLM with this passage)")
                continue
            raw = _call_llm(build_extraction_messages(chunk, source_key, mode), model)
            for obj in parse_llm_output(raw):
                ok, why = validate_card(obj, passage=chunk)
                if not ok:
                    rejected += 1
                    print(f"    REJECT {obj.get('rule_id','?')}: {why}")
                    continue
                # re-anchor rule_id with running counter; LLM cannot be trusted
                obj["rule_id"] = f"STAGED-{source_key}-{accepted+1:04d}"
                obj["_provenance"] = {"chunk_index": i, "source_path": source_path, "model": model}
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                accepted += 1
    finally:
        if fh is not None:
            fh.close()

    if dry_run:
        print("DRY-RUN complete — no API calls made.")
    else:
        print(f"staged {accepted} cards (rejected {rejected}) to {out_path}")
        print("→ next: human-review each card, then merge into echem_rules_seed.jsonl")


def main():
    ap = argparse.ArgumentParser(description="Mine EChemRules cards from a source text file.")
    ap.add_argument("--source", required=True, help="plain-text source file (UTF-8)")
    ap.add_argument("--out", default="pvgap_experiment/data/echem_rules/staging/mined.jsonl",
                    help="staging output JSONL path (will not touch seed.jsonl)")
    ap.add_argument("--model", default="gpt-4o", help="model key from config.MODELS")
    ap.add_argument("--source-key", default=None,
                    help="short key for sources field (default: basename of --source)")
    ap.add_argument("--max-segments", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true",
                    help="segment the source and print plan, but do not call the API")
    ap.add_argument("--mode", choices=["standard", "tutorial"], default="standard",
                    help="'standard' = research paper prompt; 'tutorial' = accepts "
                         "generic feature→mechanism teaching claims (textbook/review)")
    args = ap.parse_args()

    key = args.source_key or os.path.splitext(os.path.basename(args.source))[0]
    key = re.sub(r"[^A-Za-z0-9_-]", "_", key)[:20] or "SRC"
    run(args.source, args.out, args.model, args.max_segments, key, args.dry_run,
        mode=args.mode)


if __name__ == "__main__":
    main()
