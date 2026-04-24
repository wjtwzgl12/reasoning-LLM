"""§9E.1 C3-full v4 — LLM structured-JSON candidate-parameter extractor.

Replaces the over-greedy regex mapper in `scenario_bridge.perturb_params_from_diagnosis`
for C3-full's per-candidate Chen2020 overrides. The regex mapper suffered from:

  * Recall-capped vocabulary — any phrasing outside `_MECH_KEYWORDS` collapsed
    to the default "healthy-ish" branch, so w_2 / w_4 could not discriminate.
  * Severity keyword match was brittle; "严重 LAM" and "SEI 严重增长" both
    collapsed to the same scale.
  * Candidates asserting mixed / staged degradation were always bucketed
    into `combined` with a single global severity.

This module provides `extract_overrides(cand_text)` that prompts a small
LLM call (DeepSeek-V3 by default, JSON mode) to emit:

    {
      "mechanism": "LAM_negative" | "LAM_positive" | "diffusion_degradation"
                   | "combined_degradation" | "SEI_growth" | "healthy"
                   | "unknown",
      "severity":  "severe" | "moderate" | "mild" | "none",
      "confidence": 0.0 – 1.0,
      "chen2020_overrides": {
          "Negative particle radius [m]": float | null,
          "Positive particle radius [m]": float | null,
          "Negative electrode diffusivity [m2.s-1]": float | null,
          "Positive electrode diffusivity [m2.s-1]": float | null,
          "Negative electrode thickness [m]": float | null,
          "Positive electrode thickness [m]": float | null
      },
      "rationale": "<= 80 char English/Chinese"
    }

The LLM returns *absolute* Chen2020 values (not relative scales) — we
validate each against hard physical bounds (2× expansion either side of
Chen defaults) and fall back to the regex mapper for any out-of-bounds
or missing fields.

Design notes
------------
  * One extractor call per candidate. Candidate texts are typically 300-600
    tokens, so this adds ~4 × (n_cand=4) = 16 calls per qid to C3-full.
  * Kept *deliberately* separate from `scenario_bridge.bridge()` so the
    dependency from C3-full → LLM stays opt-in (C0/C1/C3/C3div don't pay).
  * Safe fallback: if the LLM errors / JSON parse fails / bounds violated,
    we hand the candidate back to the old regex mapper so C3-full never
    blocks on extraction.
"""
from __future__ import annotations

import json
from typing import Any

from .sbi_posterior import PARAM_NAMES


# Chen2020 defaults (cached via scenario_bridge; re-use to avoid second pybamm
# import). Bounds = 0.1× … 10× Chen default (2 orders of magnitude total).
# v3.1 fix (§9E.1 pilot, 2026-04-22): LLM emits exactly {0.2×, 2×, 5×} Chen
# defaults (matching the severity→multiplier hint in the system prompt).
# Earlier bounds [0.2×, 5×] rejected those boundary values due to float
# comparison drift (e.g. 4e-15 * 0.2 != 8e-16 in IEEE-754). Widening to
# [0.1×, 10×] + a 1% relative epsilon guarantees LLM emissions validate.
_BOUND_FACTOR_LO = 0.1
_BOUND_FACTOR_HI = 10.0
_BOUND_REL_EPS = 1e-2  # 1% slack either side to absorb float drift


_SYSTEM_PROMPT = (
    "You are a Chen2020-parameter extractor for lithium-ion SPM simulation. "
    "Given a candidate diagnosis of an EIS spectrum (Chinese or English), "
    "emit a strict JSON object describing which Chen2020 parameters the "
    "candidate is implicitly committing to. Do NOT speculate — only extract "
    "what the candidate text asserts. If the candidate says 'healthy / no "
    "degradation', emit chen2020_overrides = null for every parameter and "
    "mechanism='healthy'. If the candidate asserts loss of negative active "
    "material (LAM_negative), shrink Negative particle radius [m] (because "
    "fewer active particles → higher surface-to-volume → larger effective R_p "
    "in Chen SPM convention). If diffusion_degradation, reduce the relevant "
    "diffusivity. Severity → multiplier: severe=5× away, moderate=2×, "
    "mild=1.3×. Always return absolute numeric values within [0.2×, 5×] of "
    "Chen defaults. Output MUST be valid json."
)


_CHEN_DEFAULTS_HINT = (
    "Chen2020 defaults you may perturb:\n"
    "  Negative electrode thickness [m]: 8.52e-5\n"
    "  Positive electrode thickness [m]: 7.56e-5\n"
    "  Negative particle radius [m]:     5.86e-6\n"
    "  Positive particle radius [m]:     5.22e-6\n"
    "  Negative electrode diffusivity [m2.s-1]: 3.3e-14\n"
    "  Positive electrode diffusivity [m2.s-1]: 4.0e-15"
)


SCHEMA_EXAMPLE = {
    "mechanism": "LAM_negative",
    "severity": "moderate",
    "confidence": 0.75,
    "chen2020_overrides": {
        "Negative particle radius [m]": 1.17e-5,
        "Positive particle radius [m]": None,
        "Negative electrode diffusivity [m2.s-1]": None,
        "Positive electrode diffusivity [m2.s-1]": None,
        "Negative electrode thickness [m]": None,
        "Positive electrode thickness [m]": None,
    },
    "rationale": "candidate 2× R_p_neg for moderate LAM_negative",
}


ALLOWED_MECHS = {
    "LAM_negative", "LAM_positive", "diffusion_degradation",
    "combined_degradation", "SEI_growth", "healthy", "unknown",
}
ALLOWED_SEV = {"severe", "moderate", "mild", "none"}


def _validate(obj: dict, chen: dict[str, float]
              ) -> tuple[bool, dict[str, float] | None, str]:
    """Return (ok, overrides_applied, reason). overrides_applied contains
    the full 6-param committed vector with Chen defaults filling any nulls."""
    if not isinstance(obj, dict):
        return False, None, "not a dict"
    mech = obj.get("mechanism")
    if mech not in ALLOWED_MECHS:
        return False, None, f"mechanism={mech!r} not in allowed"
    sev = obj.get("severity")
    if sev not in ALLOWED_SEV:
        return False, None, f"severity={sev!r} not in allowed"
    raw = obj.get("chen2020_overrides", {}) or {}
    if not isinstance(raw, dict):
        return False, None, "chen2020_overrides not a dict"

    out: dict[str, float] = {}
    for name in PARAM_NAMES:
        v = raw.get(name, None)
        if v is None:
            out[name] = chen[name]
            continue
        try:
            f = float(v)
        except Exception:
            return False, None, f"{name}: not numeric"
        lo = chen[name] * _BOUND_FACTOR_LO * (1.0 - _BOUND_REL_EPS)
        hi = chen[name] * _BOUND_FACTOR_HI * (1.0 + _BOUND_REL_EPS)
        if not (lo <= f <= hi):
            return False, None, f"{name}={f:.3g} outside [{lo:.3g}, {hi:.3g}]"
        out[name] = f
    return True, out, "ok"


def extract_overrides(
    cand_text: str, base_committed: dict[str, float],
    *, model: str | None = None, temperature: float = 0.0,
    fallback: Any = None,
) -> tuple[dict[str, float], dict]:
    """Extract Chen2020 overrides from a single candidate diagnosis text.

    Parameters
    ----------
    cand_text : str
        The full S1–S5 CoT of one C3 generator candidate.
    base_committed : dict[str, float]
        GT-derived committed dict from `scenario_bridge.bridge()`. Used ONLY
        as the source-of-truth for Chen defaults (same 6 keys).
    model : str | None
        Optional model override for `call_llm`. Default auto-selects per
        env (DeepSeek-V3 if DEEPSEEK_API_KEY present).
    fallback : callable | None
        Optional `perturb_params_from_diagnosis`-signature callable invoked
        when the LLM extractor fails. If None, returns `base_committed`
        unchanged with meta={"extractor": "none"}.

    Returns
    -------
    (overrides, meta) — identical shape to `perturb_params_from_diagnosis`,
    so this is a drop-in replacement at the call site in
    `run_9e_pilot.c3_full_predict`.
    """
    chen = {n: float(base_committed[n]) for n in PARAM_NAMES}
    user_msg = (
        f"{_CHEN_DEFAULTS_HINT}\n\n"
        f"Emit json in exactly this schema (fields in this order):\n"
        f"{json.dumps(SCHEMA_EXAMPLE, indent=2, ensure_ascii=False)}\n\n"
        f"Candidate diagnosis text:\n---\n{cand_text[:3000]}\n---\n"
        f"Emit the json object now (no markdown fences, no prose)."
    )

    from .sbi_prior_emit import call_llm  # local import to keep import graph clean

    try:
        raw = call_llm(
            [{"role": "system", "content": _SYSTEM_PROMPT},
             {"role": "user", "content": user_msg}],
            model=model, temperature=temperature, json_mode=True,
        )
    except Exception as ex:
        return _fallback(cand_text, base_committed, fallback,
                         meta_extra={"extractor": "llm_error",
                                     "err": str(ex)[:120]})

    try:
        obj = json.loads(raw)
    except Exception as ex:
        return _fallback(cand_text, base_committed, fallback,
                         meta_extra={"extractor": "json_parse_error",
                                     "err": str(ex)[:120],
                                     "raw_head": raw[:200]})

    ok, overrides, reason = _validate(obj, chen)
    if not ok or overrides is None:
        return _fallback(cand_text, base_committed, fallback,
                         meta_extra={"extractor": "validate_fail",
                                     "reason": reason,
                                     "llm_obj_head": str(obj)[:300]})

    mech_v = obj.get("mechanism")
    sev_v = obj.get("severity")
    meta = {
        "extractor": "llm_json",
        "mechanism": mech_v,
        "severity": sev_v,
        # legacy aliases — the pilot printer + older downstream readers
        # look for these v2-regex field names; keep both so no side is
        # blind to extractor output.
        "mech_detected": mech_v,
        "severity_detected": sev_v,
        "confidence": obj.get("confidence"),
        "rationale": (obj.get("rationale") or "")[:120],
    }
    return overrides, meta


def _fallback(cand_text: str, base_committed: dict[str, float],
              fallback, *, meta_extra: dict) -> tuple[dict[str, float], dict]:
    if fallback is None:
        return ({n: float(base_committed[n]) for n in PARAM_NAMES},
                {"extractor": "none", **meta_extra})
    try:
        params, meta_reg = fallback(cand_text, base_committed)
    except Exception as ex:
        return ({n: float(base_committed[n]) for n in PARAM_NAMES},
                {"extractor": "fallback_error",
                 "err": str(ex)[:120], **meta_extra})
    meta = {"extractor": "regex_fallback", **meta_extra}
    meta.update({f"regex_{k}": v for k, v in meta_reg.items()})
    return params, meta


__all__ = ["extract_overrides", "ALLOWED_MECHS", "ALLOWED_SEV"]
