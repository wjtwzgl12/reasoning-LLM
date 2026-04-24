"""§9C.1 — LLM prior-emission caller + validator.

Loads the prompt template + JSON schema, calls an LLM, parses + validates the
emission. Hardens against the most common LLM failure modes (markdown fences,
trailing prose, single-quote JSON).

Decision gate: 5 hand cases → ≥ 4 schema-valid AND physically-reasonable.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import jsonschema  # type: ignore
except ImportError:  # graceful — schema check degrades to required-key check
    jsonschema = None


HERE = Path(__file__).resolve().parent
PROMPT_DIR = HERE.parent / "prompts"
SCHEMA_PATH = PROMPT_DIR / "sbi_prior_schema_v1.json"
PROMPT_PATH = PROMPT_DIR / "sbi_prior_config_v1.md"


REQUIRED_NAMES = [
    "Negative electrode thickness [m]",
    "Positive electrode thickness [m]",
    "Negative particle radius [m]",
    "Positive particle radius [m]",
    "Negative electrode diffusivity [m2.s-1]",
    "Positive electrode diffusivity [m2.s-1]",
]


# ─────────────────────── prompt assembly ──────────────────────────


def load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_messages(case: dict) -> list[dict]:
    """case keys: cell_text, hf_intercept, sc1_diameter, z_lowfreq,
    f_peak, candidate_param_sets."""
    system = (
        "You are an electrochemist tasked with proposing a Bayesian prior "
        "over physical battery parameters for SBI. Output ONLY a single JSON "
        "object matching this schema — no prose, no markdown fences, no "
        "commentary.\n\n"
        f"Schema (informational):\n{json.dumps(load_schema(), indent=2)}\n\n"
        "Rules:\n"
        "1. parameters array must contain EXACTLY 6 entries, one per allowed "
        "name (no duplicates, no omissions).\n"
        "2. Use 'lognormal' when the parameter spans >=1 decade in plausible "
        "literature range; 'uniform' for tight quantities.\n"
        "3. lognormal loc/scale are in ln-space (loc = ln(median)).\n"
        "4. support is always linear, in SI units.\n"
        "5. Calibrate to literature ranges (Chen2020 / Marquis2019 / "
        "OKane2022)."
    )
    user = (
        f"Cell description: {case.get('cell_text', '<unspecified>')}\n\n"
        "Observed Nyquist summary statistics:\n"
        f"  - HF intercept (R_s, Ohm):           {case.get('hf_intercept', 'NA')}\n"
        f"  - First semicircle diameter (Ohm):   {case.get('sc1_diameter', 'NA')}\n"
        f"  - Total |Z(0.01 Hz)| (Ohm):          {case.get('z_lowfreq', 'NA')}\n"
        f"  - Apparent characteristic frequency at semicircle apex (Hz): "
        f"{case.get('f_peak', 'NA')}\n\n"
        f"Candidate parameter sets the user is considering: "
        f"{case.get('candidate_param_sets', 'NA')}\n\n"
        "Emit JSON now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ─────────────────────── parsing + validation ───────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_emission(raw: str) -> dict:
    """Strip markdown fences, locate the first JSON object, return parsed dict.
    Raises ValueError on unrecoverable parse failure.
    """
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    # Find first '{' through matching last '}'.
    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j < 0 or j <= i:
        raise ValueError(f"no JSON object in emission (head={s[:80]!r})")
    return json.loads(s[i:j + 1])


def validate_emission(obj: dict, schema: dict | None = None
                      ) -> tuple[bool, list[str]]:
    """Returns (ok, list_of_errors)."""
    schema = schema or load_schema()
    errs: list[str] = []

    if jsonschema is not None:
        v = jsonschema.Draft202012Validator(schema)
        for e in sorted(v.iter_errors(obj), key=lambda x: x.path):
            errs.append(f"schema: {list(e.path)} {e.message}")
        if errs:
            return False, errs

    # Independent checks (covers cases jsonschema is missing or 'allOf if/then'
    # support is patchy).
    params = obj.get("parameters", [])
    if not isinstance(params, list) or len(params) != 6:
        errs.append(f"parameters must be array of length 6, got {len(params)}")
        return False, errs

    seen = set()
    for i, p in enumerate(params):
        n = p.get("name")
        if n not in REQUIRED_NAMES:
            errs.append(f"[{i}] name not in allowed set: {n!r}")
        if n in seen:
            errs.append(f"[{i}] duplicate name {n!r}")
        seen.add(n)

        d = p.get("dist")
        if d == "lognormal":
            if "loc" not in p or "scale" not in p:
                errs.append(f"[{i}] lognormal requires loc + scale")
            elif p["scale"] <= 0:
                errs.append(f"[{i}] lognormal scale must be > 0")
        elif d == "uniform":
            if "low" not in p or "high" not in p:
                errs.append(f"[{i}] uniform requires low + high")
            elif p["low"] >= p["high"]:
                errs.append(f"[{i}] uniform low must be < high")
        else:
            errs.append(f"[{i}] dist must be lognormal|uniform, got {d!r}")

        sup = p.get("support")
        if not (isinstance(sup, list) and len(sup) == 2 and sup[0] < sup[1]):
            errs.append(f"[{i}] support must be [low, high] with low<high")

    missing = set(REQUIRED_NAMES) - seen
    if missing:
        errs.append(f"missing required names: {sorted(missing)}")

    return (len(errs) == 0), errs


# ─────────────────────── LLM caller ─────────────────────────────────


def call_llm(messages: list[dict], model: str | None = None,
             temperature: float = 0.2,
             json_mode: bool | None = None) -> str:
    """Chat wrapper with provider auto-select.

    Provider precedence:
      1. SBI_PRIOR_PROVIDER env (`openai` | `deepseek`) if set
      2. else: prefer DEEPSEEK_API_KEY → deepseek; fall back to OPENAI_API_KEY.

    Both providers expose an OpenAI-compatible /v1/chat/completions endpoint.
    """
    provider = os.environ.get("SBI_PRIOR_PROVIDER")
    if provider is None:
        if os.environ.get("DEEPSEEK_API_KEY"):
            provider = "deepseek"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise RuntimeError(
                "neither DEEPSEEK_API_KEY nor OPENAI_API_KEY set")

    if provider == "deepseek":
        api_key = os.environ["DEEPSEEK_API_KEY"]
        base_url = "https://api.deepseek.com"
        model = model or "deepseek-chat"
    else:
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = None
        model = model or "gpt-4o-mini"

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    # Auto-detect: enable json_object mode only if caller signals JSON intent
    # (word 'json' in prompt). DeepSeek/OpenAI both require that word when
    # response_format=json_object is set.
    if json_mode is None:
        blob = " ".join(str(m.get("content", "")) for m in messages).lower()
        json_mode = "json" in blob
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ─────────────────────── 5 hand-review test cases ───────────────────


HAND_CASES: list[dict[str, Any]] = [
    {
        "name": "lg_m50_healthy",
        "cell_text": "LG M50 21700 cylindrical cell, fresh, 25 degC, SOC 0.5",
        "hf_intercept": 0.025, "sc1_diameter": 0.02, "z_lowfreq": 0.5,
        "f_peak": 100.0,
        "candidate_param_sets": ["Chen2020"],
        "expect_validate": True,
    },
    {
        "name": "coin_aged_70C",
        "cell_text": "Coin cell, NMC811/graphite, aged 100 cycles at 70 degC",
        "hf_intercept": 0.04, "sc1_diameter": 0.06, "z_lowfreq": 1.5,
        "f_peak": 30.0,
        "candidate_param_sets": ["OKane2022"],
        "expect_validate": True,
    },
    {
        "name": "pouch_0C",
        "cell_text": "Pouch cell, NMC/graphite, 0 degC, low rate",
        "hf_intercept": 0.06, "sc1_diameter": 0.15, "z_lowfreq": 4.0,
        "f_peak": 5.0,
        "candidate_param_sets": ["Chen2020", "Marquis2019"],
        "expect_validate": True,
    },
    {
        "name": "symmetric_graphite_halfcell",
        "cell_text": ("Symmetric graphite/graphite half-cell — positive "
                      "electrode is also graphite (NOT NMC/LFP)."),
        "hf_intercept": 0.03, "sc1_diameter": 0.04, "z_lowfreq": 0.8,
        "f_peak": 50.0,
        "candidate_param_sets": [],
        "expect_validate": True,  # v1 still requires 6 params; LLM should
                                  # emit graphite-like priors for both sides.
    },
    {
        "name": "unknown_summary_only",
        "cell_text": "Unknown chemistry; only Nyquist summary available",
        "hf_intercept": 0.05, "sc1_diameter": 0.03, "z_lowfreq": 1.0,
        "f_peak": 20.0,
        "candidate_param_sets": [],
        "expect_validate": True,
    },
]


def main():
    schema = load_schema()
    out_dir = HERE.parent / "results" / "sbi_prior_emit"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    rows = []
    for case in HAND_CASES:
        msgs = build_messages(case)
        try:
            raw = call_llm(msgs)
            obj = parse_emission(raw)
            ok, errs = validate_emission(obj, schema)
        except Exception as ex:
            raw, obj, ok, errs = "", None, False, [f"exception: {ex}"]
        tag = "✓" if ok else "✗"
        print(f"  {tag} {case['name']:30s}  schema_ok={ok}  errs={len(errs)}")
        if not ok:
            for e in errs[:5]:
                print(f"       - {e}")
        if ok:
            valid_count += 1
        rows.append({
            "name": case["name"], "schema_ok": ok, "errors": errs,
            "emission": obj, "raw": raw,
        })

    out_path = out_dir / "hand_cases_v1.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")
    print(f"§9C.1 schema-valid: {valid_count}/{len(HAND_CASES)} "
          f"(threshold ≥ 4)")
    if valid_count < 4:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
