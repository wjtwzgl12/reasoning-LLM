"""§9E.1 C3-full — bridge benchmark scenario → weaver_signals.extract_all case.

The benchmark rows have `ground_truth.physical_params` (e.g. temperature_K,
volume_fraction, diffusivity overrides) but NOT raw Nyquist traces. We
synthesise `observed_Z` on F_SUMMARY via `pybamm_eis_residual.simulate_Z`
so w_2 (residual), w_3 (lin-KK), w_4 (SBI match) have their inputs.

Per §4.14, regenerating from GT params is the accepted "simulated
observation" proxy used throughout §9. L1 questions are conceptual and
skip 5-signal (caller falls back to critic-only w_5).

API
---
  bridge(scenario: dict) -> dict | None
      Returns a weaver-signals-ready case, or None if scenario has no
      physical_params (L1 → skip).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .sbi_posterior import PARAM_NAMES, F_SUMMARY


# Chen2020 defaults — lazy cache so pybamm import only happens at first call.
_CHEN_DEFAULTS: dict[str, float] | None = None


def _chen_defaults() -> dict[str, float]:
    global _CHEN_DEFAULTS
    if _CHEN_DEFAULTS is not None:
        return _CHEN_DEFAULTS
    import pybamm
    pv = pybamm.ParameterValues("Chen2020")
    out = {n: float(pv[n]) for n in PARAM_NAMES}
    # Extras used by the condition-calibration gate.
    out["_T_default_K"] = 298.15
    _CHEN_DEFAULTS = out
    return out


def _commit_params(gt_physical: dict) -> dict[str, float]:
    """Build `committed_mechanism_params` with GT overrides on Chen2020.

    We apply:
      - explicit PARAM_NAMES matches
      - active-material-volume-fraction → particle-radius scale (approx
        surrogate: R_p_new = R_p_chen × (1 - εs_chen/εs_gt)^(1/3)). Keeps
        the 6-D param vector consistent with what SBI trained on.
    """
    chen = _chen_defaults()
    committed = {n: chen[n] for n in PARAM_NAMES}
    for k, v in (gt_physical or {}).items():
        if k in committed:
            committed[k] = float(v)
    # Heuristic: if GT says neg volume fraction = X, scale neg particle radius
    # as a rough surrogate (not physical, but keeps magnitude realistic).
    if "Negative electrode active material volume fraction" in (gt_physical or {}):
        eps_gt = float(gt_physical["Negative electrode active material volume fraction"])
        # Chen2020 default ≈ 0.75 (LG M50). Clamp to avoid div/0.
        eps_chen = 0.75
        scale = max((eps_gt / eps_chen) ** (1.0 / 3.0), 0.3)
        committed["Negative particle radius [m]"] *= (1.0 / scale)
    return committed


def _observation_summary(scenario: dict, z: np.ndarray) -> str:
    f = F_SUMMARY
    return (f"Synthesised Nyquist on F_SUMMARY 0.1 Hz–1 kHz. "
            f"HF intercept |Z({f[-1]:.0f}Hz)|={abs(z[-1]):.4f} Ω; "
            f"LF point |Z({f[0]:.1f}Hz)|={abs(z[0]):.4f} Ω.")


def bridge(scenario: dict) -> dict | None:
    """Convert a benchmark scenario → weaver-signals case dict.

    Returns None when scenario has no physical_params (e.g. L1 conceptual).
    """
    gt = scenario.get("ground_truth", {}) or {}
    phys = gt.get("physical_params") or {}
    # L4 NoOp rows inherit from an L3 counterpart — diagnosis present but
    # physical_params may be absent. If so, try to back-fill via
    # scenario.metadata.base_qid (if builder stamped it).
    if not phys:
        base_qid = (scenario.get("metadata") or {}).get("base_qid")
        if base_qid is None:
            return None
        # Caller is responsible for looking up base — we just refuse here.
        return None

    from .pybamm_eis_residual import simulate_Z

    committed = _commit_params(phys)
    T_K = float(phys.get("temperature_K", _chen_defaults()["_T_default_K"]))

    base = {
        "model_name": "SPM",
        "parameter_set": "Chen2020",
        "initial_soc": 0.5,
        "frequencies": F_SUMMARY,
        # simulate_Z applies these overrides on top of Chen2020.
        "param_overrides": {n: committed[n] for n in PARAM_NAMES},
    }
    try:
        z_obs = simulate_Z(base)
    except Exception as ex:
        # If pybamm forward fails (e.g. extreme param), return a structural
        # failure signal: callers should treat as "no 5-signal" and fall
        # back to critic-only.
        return {"_bridge_error": str(ex)[:120]}

    # Optional observation noise for KK realism (1% relative Gaussian).
    rng = np.random.default_rng(abs(hash(scenario.get("qid", "x"))) % (2**31))
    sigma_rel = 0.01
    eps = rng.normal(size=z_obs.shape) + 1j * rng.normal(size=z_obs.shape)
    z_obs = z_obs * (1.0 + sigma_rel * eps)

    case: dict[str, Any] = {
        # simulate_Z inputs (w_2 re-simulates at candidate params):
        "model_name": "SPM",
        "parameter_set": "Chen2020",
        "initial_soc": 0.5,
        "frequencies": F_SUMMARY,
        # Observation side:
        "observed_Z": z_obs,
        # Committed mechanism the candidate is asserting (w_2/w_4 compare
        # this against observed_Z):
        "committed_mechanism_params": {n: committed[n] for n in PARAM_NAMES},
        "param_overrides": {n: committed[n] for n in PARAM_NAMES},
        "observation_summary": _observation_summary(scenario, z_obs),
        # Condition-calibration gate keys (§9B.0):
        "observed_temperature_K": T_K,
        "observed_parameter_set": "Chen2020",
        "candidate_temperature_K": T_K,
        "candidate_parameter_set": "Chen2020",
    }
    return case


# ─────────── candidate CoT text → param overrides ──────────────────
#
# Per §9E.1 post-mortem v2: w_2/w_4 only discriminate candidates when each
# candidate maps to a *different* `committed_mechanism_params`. We parse
# the candidate's S3/S4 diagnosis (free Chinese/English text) into
# Chen2020 overrides via keyword+severity regex, then apply the same
# perturbations that built the synthesised observation.

_MECH_KEYWORDS = {
    "lam_neg": ["lam_negative", "负极活性材料损失", "负极lam",
                "负极活性物质损失", "lam_neg", "negative lam",
                "loss of active material (negative"],
    "lam_pos": ["lam_positive", "正极活性材料损失", "正极lam",
                "正极活性物质损失", "lam_pos"],
    "diff":    ["diffusion_degradation", "扩散退化", "固态扩散变慢",
                "扩散受限", "diffusion degradation", "扩散系数减小",
                "扩散系数下降"],
    "healthy": ["healthy", "电池健康", "无明显退化", "无退化"],
    "combined":["combined_degradation", "复合退化", "combined degradation",
                "同时存在"],
}
_SEV_KEYWORDS = {
    "severe":   ["severe", "严重", "重度"],
    "moderate": ["moderate", "中度", "中等程度"],
    "mild":     ["mild", "轻度", "轻微"],
}
_SEV_SCALE = {"severe": 0.2, "moderate": 0.5, "mild": 0.8, None: 0.5}


def _detect_mechanism(text: str) -> tuple[str | None, str | None]:
    t = (text or "").lower()
    sev = None
    for s, kws in _SEV_KEYWORDS.items():
        if any(k in t for k in kws):
            sev = s
            break
    for m in ("combined", "lam_neg", "lam_pos", "diff", "healthy"):
        if any(k in t for k in _MECH_KEYWORDS[m]):
            return m, sev
    return None, sev


def perturb_params_from_diagnosis(
    text: str, base_committed: dict[str, float]
) -> tuple[dict[str, float], dict]:
    """Map a candidate's S3/S4 diagnosis text → Chen2020 param overrides.

    `base_committed` is the GT-derived committed dict from `bridge()`.
    Returns (new_params, meta). If no mechanism is detected the function
    returns a *perturbed-toward-healthy* dict so w_2/w_4 still discriminate
    (otherwise all un-detectable candidates would collapse to identical
    scores again).
    """
    mech, sev = _detect_mechanism(text)
    s = _SEV_SCALE[sev]
    params = dict(base_committed)
    chen = _chen_defaults()

    if mech == "healthy" or mech is None:
        # Candidate implicitly asserts defaults → revert any GT overrides.
        params = {n: chen[n] for n in PARAM_NAMES}
    elif mech == "lam_neg":
        # eps_neg ↓ → R_p ↑ (inverse 1/3 scale, mirroring _commit_params).
        params["Negative particle radius [m]"] = (
            chen["Negative particle radius [m]"] * (1.0 / s) ** (1.0 / 3.0)
        )
    elif mech == "lam_pos":
        params["Positive particle radius [m]"] = (
            chen["Positive particle radius [m]"] * (1.0 / s) ** (1.0 / 3.0)
        )
    elif mech == "diff":
        params["Negative electrode diffusivity [m2.s-1]"] = (
            chen["Negative electrode diffusivity [m2.s-1]"] * s
        )
    elif mech == "combined":
        params["Negative particle radius [m]"] = (
            chen["Negative particle radius [m]"] * (1.0 / s) ** (1.0 / 3.0)
        )
        params["Negative electrode diffusivity [m2.s-1]"] = (
            chen["Negative electrode diffusivity [m2.s-1]"] * s
        )

    return params, {"mech_detected": mech, "severity_detected": sev,
                    "scale_applied": s}


__all__ = ["bridge", "perturb_params_from_diagnosis"]
