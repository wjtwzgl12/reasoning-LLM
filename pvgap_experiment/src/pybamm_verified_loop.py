"""
PyBaMM-Verified Loop (9B.2) — entry wrapper.
=========================================================================

This is the top-level dispatcher that §3.11 Weaver calls for the physical
forward-residual verifier. It enforces the §9B.0 condition-calibration gate
as a hard prerequisite: residual scoring is **never** returned for a case
the gate has ruled ABSTAIN on, and residual scoring is marked `low_trust`
when the gate returns FLAG.

Design contract (see Paper1_骨架.md §3.11 / §9B.0 / §9B.2):

    gate = evaluate(case, rules)
    ├── ABSTAIN  -> return {"score": None, "w2": 0.0, ...}
    │              never call residual_fn; Weaver zeros w_2
    ├── FLAG     -> return {"score": <computed>, "w2": 0.5, "low_trust": True, ...}
    │              residual is computed, Weaver halves w_2
    └── PASS     -> return {"score": <computed>, "w2": 1.0, "low_trust": False, ...}
                   standard path

The residual computation itself (`residual_fn`) is injected as a callback.
This keeps the wrapper independent of the PyBaMM-EIS environment (9B.1,
not yet wired) and testable with pure synthetic stubs. Once 9B.1 lands,
`residual_fn` will be bound to a real `pybamm_eis_residual(case) -> dict`
closure.

Returned score dict schema (when residual is computed):
    {
        "rho_real":    float,
        "rho_imag":    float,
        "rho_complex": float,
        "rho_logmag":  float,
    }
(keys provided by `residual_fn`; the wrapper does not validate their
contents — it only propagates them.)
"""

from __future__ import annotations
from typing import Callable

from .condition_calibration_gate import (
    GateVerdict,
    evaluate,
    load_gate_rules,
)


ResidualFn = Callable[[dict], dict]


def run_verified_loop(
    case: dict,
    *,
    residual_fn: ResidualFn | None = None,
    rules: list[dict] | None = None,
) -> dict:
    """Entry for §9B.2 physical verifier.

    Parameters
    ----------
    case : dict
        Observation + candidate metadata. See
        `condition_calibration_gate` module docstring for the full key list.
    residual_fn : callable(case) -> dict, optional
        Residual computation callback. Called only when the gate verdict is
        PASS or FLAG. If omitted and the gate does not ABSTAIN, the wrapper
        raises — this is intentional: a silent score of None on PASS/FLAG
        would be indistinguishable from ABSTAIN and break Weaver's w_2
        accounting.
    rules : list[dict], optional
        Pre-loaded gate rules. Defaults to `load_gate_rules()` (the
        packaged 15-rule CC corpus).

    Returns
    -------
    dict with keys: score, w2, reason, low_trust, gate.
    """
    verdict: GateVerdict = evaluate(case, rules=rules)

    if verdict.action == "ABSTAIN":
        return {
            "score": None,
            "w2": 0.0,
            "reason": "gate_abstain",
            "low_trust": False,
            "gate": verdict.to_dict(),
        }

    if residual_fn is None:
        raise RuntimeError(
            "residual_fn is required when gate verdict is PASS or FLAG. "
            "9B.1 (PyBaMM-EIS environment) must be wired before 9B.2 "
            "can score non-abstaining cases."
        )

    scores = residual_fn(case)

    if verdict.action == "FLAG":
        return {
            "score": scores,
            "w2": 0.5,
            "reason": "gate_flag_low_trust",
            "low_trust": True,
            "gate": verdict.to_dict(),
        }

    # PASS
    return {
        "score": scores,
        "w2": 1.0,
        "reason": "gate_pass",
        "low_trust": False,
        "gate": verdict.to_dict(),
    }


# ────────────────────────── CLI self-test ───────────────────────────


def _stub_residual(case: dict) -> dict:
    """Deterministic placeholder residual.

    Maps observed vs candidate |Z|_mean to a normalised residual signal.
    Not a physics model — only exists to let the wrapper self-test exercise
    the PASS/FLAG branches without a live PyBaMM installation.
    """
    z_obs = case.get("observed_abs_Z_mean", 0.0)
    z_cand = case.get("candidate_abs_Z_mean", 1.0)
    r = abs(z_obs - z_cand) / max(z_cand, 1e-9)
    return {
        "rho_real":    r,
        "rho_imag":    r * 0.9,
        "rho_complex": r * 1.05,
        "rho_logmag":  r * 0.8,
    }


def main():
    from .condition_calibration_gate import _synth_cases

    rules = load_gate_rules()
    print(f"loaded {len(rules)} gate rules\n")

    expected_dispatch = {
        "surface_form":         ("FLAG",    0.5, True,  False),  # score non-None
        "spm_vs_spme":          ("PASS",    1.0, False, False),
        "temperature_mismatch": ("ABSTAIN", 0.0, False, True),   # score is None
        "param_set_mismatch":   ("ABSTAIN", 0.0, False, True),
    }

    ok = 0
    total = 0
    for name, case, _expected_action in _synth_cases():
        total += 1
        out = run_verified_loop(case, residual_fn=_stub_residual, rules=rules)
        exp_action, exp_w2, exp_low_trust, exp_score_is_none = expected_dispatch[name]

        checks = [
            out["gate"]["action"] == exp_action,
            out["w2"] == exp_w2,
            out["low_trust"] == exp_low_trust,
            (out["score"] is None) == exp_score_is_none,
        ]
        match = "✓" if all(checks) else "✗"
        if all(checks):
            ok += 1
        print(f"  {match} {name:22s}  action={out['gate']['action']:8s}  "
              f"w2={out['w2']}  low_trust={out['low_trust']}  "
              f"score={'None' if out['score'] is None else 'dict'}")

    print(f"\nself-test: {ok}/{total} cases match §9B.2 dispatch contract")
    if ok != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
