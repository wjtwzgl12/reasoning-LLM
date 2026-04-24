"""§9B.2 integration test: 5 toy cases through the full verified loop.

Wires `pybamm_eis_residual` as `residual_fn` into `run_verified_loop`,
runs 5 hand-crafted (GT, candidates) cases, checks loop correctness:

  loop_correct iff the candidate equal to GT scores strictly lower
  ρ_complex than every perturbed candidate (when gate PASSes/FLAGs),
  OR the gate correctly ABSTAINs on a metadata mismatch.

Decision gate per Paper1_骨架.md §9B.2: ≥ 70% loop correctness → PASS.
With 5 cases the threshold is 4/5.
"""
from __future__ import annotations
import sys
import numpy as np

from pvgap_experiment.src.pybamm_eis_residual import (
    simulate_Z, pybamm_eis_residual,
)
from pvgap_experiment.src.pybamm_verified_loop import run_verified_loop
from pvgap_experiment.src.condition_calibration_gate import load_gate_rules


# Frequency grid shared by all cases.
F = np.logspace(-2, 4, 25)


def _gt_Z(model_name: str, parameter_set: str, soc: float,
          overrides: dict | None = None) -> np.ndarray:
    """Ground-truth impedance from PyBaMM."""
    return simulate_Z({
        "model_name":      model_name,
        "parameter_set":   parameter_set,
        "initial_soc":     soc,
        "frequencies":     F,
        "param_overrides": overrides,
    })


def _make_case(z_obs, model_name, parameter_set, soc,
               overrides=None,
               # gate-relevant metadata (mirrors P0.7 stressor schema):
               obs_temperature_C=25.0,
               cand_temperature_C=25.0,
               obs_param_set=None,
               cand_param_set=None,
               obs_abs_Z_mean=None,
               cand_abs_Z_mean=None) -> dict:
    case = {
        "model_name":      model_name,
        "parameter_set":   parameter_set,
        "initial_soc":     soc,
        "frequencies":     F,
        "observed_Z":      z_obs,
        "param_overrides": overrides,
        # metadata for condition-calibration gate
        "observed_temperature_C":   obs_temperature_C,
        "candidate_temperature_C":  cand_temperature_C,
        "observed_parameter_set":   obs_param_set or parameter_set,
        "candidate_parameter_set":  cand_param_set or parameter_set,
    }
    if obs_abs_Z_mean is not None:
        case["observed_abs_Z_mean"] = obs_abs_Z_mean
    if cand_abs_Z_mean is not None:
        case["candidate_abs_Z_mean"] = cand_abs_Z_mean
    return case


def main():
    rules = load_gate_rules()
    print(f"loaded {len(rules)} gate rules")
    print(f"frequency grid: {len(F)} pts, {F[0]:.2g}–{F[-1]:.2g} Hz\n")

    # Build GT impedance traces (cached after first call by _SIM_CACHE).
    print("=== building GT Z(ω) for 5 cases ===")
    gt_chen_p5  = _gt_Z("SPM", "Chen2020", 0.5)
    gt_chen_p7  = _gt_Z("SPM", "Chen2020", 0.7)
    gt_marq_p5  = _gt_Z("SPM", "Marquis2019", 0.5)
    gt_chen_p3  = _gt_Z("SPM", "Chen2020", 0.3)
    gt_spme_p5  = _gt_Z("SPMe", "Chen2020", 0.5)
    print("  done\n")

    # Perturbation amounts.
    L_neg_chen   = 8.52e-5
    R_part_chen  = 5.86e-6

    cases = [
        # 1. Correct candidate vs L_neg +15% perturbed: same metadata → gate
        #    PASS, residual must rank correct < perturbed.
        ("L_neg_perturbation_chen_p5",
         _make_case(gt_chen_p5, "SPM", "Chen2020", 0.5),
         _make_case(gt_chen_p5, "SPM", "Chen2020", 0.5,
                    overrides={"Negative electrode thickness [m]":
                               1.15 * L_neg_chen})),
        # 2. Correct vs R_particle +20% perturbed at SOC 0.7.
        ("R_particle_perturbation_chen_p7",
         _make_case(gt_chen_p7, "SPM", "Chen2020", 0.7),
         _make_case(gt_chen_p7, "SPM", "Chen2020", 0.7,
                    overrides={"Negative particle radius [m]":
                               1.20 * R_part_chen})),
        # 3. Correct vs SOC mismatch (0.5 vs 0.3) on same Chen2020.
        ("SOC_mismatch_chen",
         _make_case(gt_chen_p5, "SPM", "Chen2020", 0.5),
         _make_case(gt_chen_p5, "SPM", "Chen2020", 0.3)),
        # 4. Correct vs cross-parameter-set candidate Chen2020 vs Marquis2019.
        #    Gate metadata: observed_parameter_set='Chen2020',
        #    candidate_parameter_set='Marquis2019' → CC-003 should ABSTAIN.
        ("param_set_mismatch_chen_vs_marquis",
         _make_case(gt_chen_p5, "SPM", "Chen2020", 0.5,
                    obs_param_set="Chen2020",
                    cand_param_set="Chen2020"),
         _make_case(gt_chen_p5, "SPM", "Marquis2019", 0.5,
                    obs_param_set="Chen2020",
                    cand_param_set="Marquis2019")),
        # 5. SPMe-generated obs vs SPM candidate (model-form mismatch).
        #    Both metadata Chen2020 → gate may PASS or FLAG; residual should
        #    still penalise the SPM candidate relative to SPMe-correct.
        ("model_form_spme_vs_spm",
         _make_case(gt_spme_p5, "SPMe", "Chen2020", 0.5),
         _make_case(gt_spme_p5, "SPM",  "Chen2020", 0.5)),
    ]

    correct = 0
    for name, case_correct, case_wrong in cases:
        print(f"--- {name} ---")
        out_c = run_verified_loop(case_correct,
                                  residual_fn=pybamm_eis_residual,
                                  rules=rules)
        out_w = run_verified_loop(case_wrong,
                                  residual_fn=pybamm_eis_residual,
                                  rules=rules)

        gc, gw = out_c["gate"]["action"], out_w["gate"]["action"]
        print(f"  gate: correct={gc}  wrong={gw}")

        # Loop-correctness logic:
        #   (a) both PASS/FLAG: correct ρ_complex < wrong ρ_complex.
        #   (b) wrong ABSTAIN: loop refused to score wrong → correct.
        #   (c) correct ABSTAIN: pathological (loop refused on truth).
        if gw == "ABSTAIN":
            ok = True
            print(f"  → ABSTAIN on wrong candidate (loop correctly refused)")
        elif gc == "ABSTAIN":
            ok = False
            print(f"  ✗ ABSTAIN on CORRECT candidate (loop falsely refused)")
        else:
            rc = out_c["score"]["rho_complex"]
            rw = out_w["score"]["rho_complex"]
            ok = rc < rw
            tag = "✓" if ok else "✗"
            print(f"  ρ_complex correct={rc:.3e}  wrong={rw:.3e}  {tag}")

        if ok:
            correct += 1
        print()

    print(f"=== §9B.2 integration: {correct}/{len(cases)} loop-correct ===")
    threshold = 0.70
    rate = correct / len(cases)
    if rate >= threshold:
        print(f"PASS (rate={rate:.0%} ≥ {threshold:.0%})")
        return 0
    print(f"FAIL (rate={rate:.0%} < {threshold:.0%})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
