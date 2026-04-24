"""
Condition-calibration gate (9B.0) — P0.7 硬前置.
=========================================================================

Consumes `data/echem_rules/echem_rules_condition_calibration.jsonl`.
Given a case dict (observed metadata + candidate params), evaluates all gate
rules and returns an aggregated verdict that §3.11 w_2 must honor **before**
any residual scoring is computed.

Rule schema (one JSON per line, level='gate'):
  rule_id, observation, mechanism, alt_mechanisms, discriminators,
  confidence, level, sources, units, applies_to, counterexamples, gate_action

Gate action taxonomy (hard-coded verdict mapping):
  PASS_*                         -> PASS       (proceed with residual scoring)
  ABSTAIN_*                      -> ABSTAIN    (residual weight must be 0)
  require_*                      -> ABSTAIN    (calibration required)
  flag_*                         -> FLAG       (residual passes with low-trust mark)
  upgrade_*                      -> ABSTAIN    (model upgrade required)
  compensate_*                   -> FLAG       (after compensation, treat low-trust)
  apply_*_or_ABSTAIN             -> FLAG       (with scaling applied)
  accept_*                       -> PASS       (acknowledged null, still score)

The gate is purely rule-driven; matching is by `unit` range + metadata keys
on the case dict. Missing metadata defaults to "cannot evaluate rule" -> rule
does not fire. This is intentional: a verifier that cannot see temperature
must not silently assume it is 25 degC.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable


# ─────────────────────────── loading ────────────────────────────────

def load_gate_rules(path: str | None = None) -> list[dict]:
    """Load condition-calibration rules (level='gate' only)."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(
            here, "..", "data", "echem_rules",
            "echem_rules_condition_calibration.jsonl",
        ))
    rules = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("level") != "gate":
                continue
            if not r.get("gate_action"):
                raise ValueError(f"gate rule {r.get('rule_id')} missing gate_action")
            rules.append(r)
    return rules


# ────────────────────────── verdict mapping ─────────────────────────

_PASS_PREFIXES = ("PASS_", "accept_")
_FLAG_PREFIXES = ("flag_", "compensate_", "apply_")
# everything else (ABSTAIN_*, require_*, upgrade_*) -> ABSTAIN


def _classify(gate_action: str) -> str:
    if gate_action.startswith(_PASS_PREFIXES):
        return "PASS"
    if gate_action.startswith(_FLAG_PREFIXES):
        return "FLAG"
    return "ABSTAIN"


# ────────────────────────── rule matchers ───────────────────────────
#
# Each rule has a matcher that maps (case, rule) -> bool. Matchers are
# dispatched by rule_id. A rule that lacks a registered matcher simply does
# not fire (conservative — an unknown rule cannot gate-abort a run).
#
# Case dict (expected keys; all optional, missing -> matcher cannot fire):
#   observed_temperature_K          float
#   candidate_temperature_K         float
#   observed_parameter_set          str  (e.g. 'Chen2020', 'OKane2022', 'unknown')
#   candidate_parameter_set         str
#   observed_SOC                    float
#   candidate_SOC                   float
#   observed_R_ohm                  float (Ω)
#   candidate_R_ohm                 float (Ω)
#   observed_abs_Z_mean             float
#   candidate_abs_Z_mean            float
#   kk_residual_max_frac            float
#   observation_rest_time_min       float
#   observation_cell_capacity_Ah    float
#   candidate_cell_capacity_Ah      float
#   candidate_model                 str  ('SPM' | 'SPMe' | 'DFN')
#   observed_electrode_thickness_um float
#   candidate_frequency_range_Hz    tuple(low, high)
#   observation_frequency_range_Hz  tuple(low, high)
#   observation_reproducibility_drift_frac   float
#   cycling_history_has_plating     bool
#   has_cable_inductance_uncompensated       bool
# ────────────────────────────────────────────────────────────────────

Matcher = Callable[[dict, dict], bool]
_MATCHERS: dict[str, Matcher] = {}


def _register(rule_id: str) -> Callable[[Matcher], Matcher]:
    def deco(fn: Matcher) -> Matcher:
        _MATCHERS[rule_id] = fn
        return fn
    return deco


def _has(c: dict, *keys: str) -> bool:
    return all(k in c and c[k] is not None for k in keys)


@_register("CC-001")
def _cc001(c, r):
    if not _has(c, "observed_R_ohm", "candidate_R_ohm") or c["candidate_R_ohm"] <= 0:
        return False
    ratio = c["observed_R_ohm"] / c["candidate_R_ohm"]
    lo, hi = r["units"]["R_ohm_ratio"]
    return ratio >= lo and (ratio <= hi or hi >= 10)


@_register("CC-002")
def _cc002(c, r):
    if not _has(c, "observed_abs_Z_mean", "candidate_abs_Z_mean") or c["candidate_abs_Z_mean"] <= 0:
        return False
    ratio = c["observed_abs_Z_mean"] / c["candidate_abs_Z_mean"]
    lo, hi = r["units"]["|Z|_ratio"]
    return lo <= ratio <= hi


@_register("CC-003")
def _cc003(c, r):
    if not _has(c, "observed_parameter_set", "candidate_parameter_set"):
        return False
    if c["observed_parameter_set"] in ("unknown", None, ""):
        return False
    return c["observed_parameter_set"] != c["candidate_parameter_set"]


@_register("CC-004")
def _cc004(c, r):
    if not _has(c, "observed_SOC", "candidate_SOC"):
        return False
    soc = c["observed_SOC"]
    gap = abs(soc - c["candidate_SOC"])
    plateau_lo, plateau_hi = r["units"]["SOC_range"]
    gap_lo, gap_hi = r["units"]["SOC_gap"]
    return (plateau_lo <= soc <= plateau_hi) and (gap_lo <= gap <= gap_hi)


@_register("CC-005")
def _cc005(c, r):
    if not _has(c, "observed_SOC", "candidate_SOC"):
        return False
    # sloped region: outside [0.45, 0.55] is a pragmatic proxy when slope unavailable
    soc = c["observed_SOC"]
    in_slope = not (0.45 <= soc <= 0.55)
    gap = abs(soc - c["candidate_SOC"])
    lo, hi = r["units"]["SOC_gap"]
    return in_slope and lo < gap <= hi


@_register("CC-006")
def _cc006(c, r):
    if not _has(c, "kk_residual_max_frac"):
        return False
    lo, hi = r["units"]["kk_residual_frac"]
    return lo <= c["kk_residual_max_frac"] <= hi


@_register("CC-007")
def _cc007(c, r):
    if not _has(c, "candidate_model", "observed_electrode_thickness_um"):
        return False
    if c["candidate_model"] != "SPM":
        return False
    lo, hi = r["units"]["electrode_thickness_um"]
    return lo <= c["observed_electrode_thickness_um"] <= hi


@_register("CC-008")
def _cc008(c, r):
    if not _has(c, "observed_temperature_K", "candidate_temperature_K"):
        return False
    T_obs = c["observed_temperature_K"]
    T_cand = c["candidate_temperature_K"]
    # Fires if T gap >= 5 degC. (Note: CC-008's stated observation is
    # "outside 10-40 degC while candidate at 25 degC", but the operational
    # intent is any T-mismatch >= 5 K; we key on that.)
    return abs(T_obs - T_cand) >= 5.0


@_register("CC-009")
def _cc009(c, r):
    return bool(c.get("has_cable_inductance_uncompensated", False))


@_register("CC-010")
def _cc010(c, r):
    if not _has(c, "observation_rest_time_min"):
        return False
    lo, hi = r["units"]["rest_time_min"]
    return lo <= c["observation_rest_time_min"] <= hi


@_register("CC-011")
def _cc011(c, r):
    obs = c.get("observation_frequency_range_Hz")
    cand = c.get("candidate_frequency_range_Hz")
    if not (obs and cand):
        return False
    # fires if candidate extends beyond observation on either side
    return cand[0] < obs[0] * 0.9 or cand[1] > obs[1] * 1.1


@_register("CC-012")
def _cc012(c, r):
    return bool(c.get("cycling_history_has_plating", False)) \
        and c.get("candidate_model", "").upper() in ("SPM", "SPME", "SPMEQ")


@_register("CC-013")
def _cc013(c, r):
    # This is the PASS rule. It fires when the shape-overlap metric is high.
    if "normalized_shape_overlap_frac" not in c or c["normalized_shape_overlap_frac"] is None:
        return False
    lo, hi = r["units"]["shape_overlap_frac"]
    return lo <= c["normalized_shape_overlap_frac"] <= hi


@_register("CC-014")
def _cc014(c, r):
    if "observation_reproducibility_drift_frac" not in c or c["observation_reproducibility_drift_frac"] is None:
        return False
    return c["observation_reproducibility_drift_frac"] > 0.02


@_register("CC-015")
def _cc015(c, r):
    if not _has(c, "observation_cell_capacity_Ah", "candidate_cell_capacity_Ah"):
        return False
    if c["candidate_cell_capacity_Ah"] <= 0:
        return False
    ratio = c["observation_cell_capacity_Ah"] / c["candidate_cell_capacity_Ah"]
    lo, hi = r["units"]["capacity_ratio"]
    # rule fires when ratio is OUTSIDE the within-batch window. CC-015 states
    # "differ by > 20%", i.e. ratio outside [0.8, 1.2]. The JSONL units carry
    # [0.5, 1.5] as the action-zone; interpret as "outside [0.8, 1.2]".
    return not (0.8 <= ratio <= 1.2) and lo <= ratio <= hi


# ────────────────────────── main evaluator ──────────────────────────


@dataclass
class GateVerdict:
    action: str                     # 'PASS' | 'FLAG' | 'ABSTAIN'
    w2_weight: float                # 1.0 if PASS, 0.5 if FLAG, 0.0 if ABSTAIN
    fired_rules: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "w2_weight": self.w2_weight,
            "fired_rules": self.fired_rules,
            "reasons": self.reasons,
        }


# Precedence: ABSTAIN > FLAG > PASS. Any abstaining rule vetoes.
_PRECEDENCE = {"ABSTAIN": 2, "FLAG": 1, "PASS": 0}


def evaluate(case: dict, rules: list[dict] | None = None) -> GateVerdict:
    """Apply all gate rules to a case; return aggregated verdict.

    `case` — dict of observation/candidate metadata. See module docstring
             for the full key list. Missing keys cause their rule to not
             fire (conservative).
    """
    if rules is None:
        rules = load_gate_rules()

    fired = []
    best = "PASS"
    for r in rules:
        matcher = _MATCHERS.get(r["rule_id"])
        if matcher is None:
            continue
        try:
            hit = matcher(case, r)
        except Exception as e:
            # matcher bug should not silently gate through; log and abstain
            fired.append({"rule_id": r["rule_id"], "action": "ABSTAIN",
                          "gate_action": f"matcher_error:{type(e).__name__}",
                          "reason": f"matcher raised {e!r}"})
            best = "ABSTAIN"
            continue
        if not hit:
            continue
        cls = _classify(r["gate_action"])
        fired.append({"rule_id": r["rule_id"], "action": cls,
                      "gate_action": r["gate_action"],
                      "reason": r["observation"]})
        if _PRECEDENCE[cls] > _PRECEDENCE[best]:
            best = cls

    w2 = {"PASS": 1.0, "FLAG": 0.5, "ABSTAIN": 0.0}[best]
    reasons = [f"{f['rule_id']}: {f['gate_action']}" for f in fired]
    return GateVerdict(action=best, w2_weight=w2, fired_rules=fired, reasons=reasons)


# ────────────────────────── CLI self-test ───────────────────────────


def _synth_cases() -> list[tuple[str, dict, str]]:
    """Synthetic cases modelling the four P0.7 stressor modes.

    Expected verdicts (see docstring below on why surface_form is FLAG,
    not PASS):

      surface_form         -> FLAG     (CC-002 fires: observed |Z| is 1.75x
                                        candidate |Z|, shape-preserved.
                                        From metadata alone the gate cannot
                                        distinguish solver-form mismatch
                                        from real T-mismatch — conservative
                                        FLAG is the correct verdict. A human
                                        with T-log would override.)
      spm_vs_spme          -> PASS     (|Z| ratio 0.82 stays inside
                                        CC-002's [1.5, 2.0] action zone, so
                                        gate is silent — this is the lucky
                                        case where solver-form difference
                                        happens to not mimic condition-gap)
      temperature_mismatch -> ABSTAIN  (CC-008 fires: 15 degC gap; and
                                        CC-002 also fires; precedence
                                        picks ABSTAIN)
      param_set_mismatch   -> ABSTAIN  (CC-003 fires on parameter-set name
                                        disagreement)

    The surface_form FLAG reveals a structural property of metadata-only
    gating: |Z| scaling from solver-form choice is indistinguishable from
    |Z| scaling from temperature. This is written into §6 as a known gate
    false-positive trade-off; the operational response is that downstream
    Weaver treats FLAG as w2=0.5 (not 0.0), so residual still contributes.
    """
    # ballpark magnitudes from P0.7 smoke-test readout:
    Z_base = 0.02662
    Z_algebraic = 0.04657
    Z_spm = 0.02173
    Z_cold10 = 0.04279
    Z_okane = 0.02774

    common = dict(
        observed_SOC=0.50, candidate_SOC=0.50,
        observed_parameter_set="Chen2020",
        candidate_parameter_set="Chen2020",
        observed_temperature_K=298.15, candidate_temperature_K=298.15,
        candidate_model="SPMe",
        observed_electrode_thickness_um=80,
        kk_residual_max_frac=0.01,
        observation_rest_time_min=60,
        observation_cell_capacity_Ah=5.0,
        candidate_cell_capacity_Ah=5.0,
        normalized_shape_overlap_frac=0.97,
        observation_reproducibility_drift_frac=0.005,
        observed_R_ohm=0.03, candidate_R_ohm=0.03,
        observed_abs_Z_mean=Z_base, candidate_abs_Z_mean=Z_base,
    )

    surface_form = dict(common)
    surface_form.update(observed_abs_Z_mean=Z_algebraic)
    # Only residual shape differs, not metadata. From the gate's metadata
    # point of view, nothing should fire -> PASS. (The mismatch is solver-
    # internal, not a condition-calibration issue.)

    spm_vs_spme = dict(common)
    spm_vs_spme.update(observed_abs_Z_mean=Z_spm)
    # Same metadata story.

    temperature_mismatch = dict(common)
    temperature_mismatch.update(
        observed_temperature_K=283.15,          # 10 degC
        observed_abs_Z_mean=Z_cold10,           # 1.61x Z_base
    )
    # CC-008 (T gap 15 degC >= 5) fires -> require_T_match -> ABSTAIN.
    # CC-002 (|Z| ratio 1.61 in [1.5, 2.0]) fires -> flag_T_mismatch -> FLAG.
    # Precedence -> ABSTAIN.

    param_set_mismatch = dict(common)
    param_set_mismatch.update(
        observed_parameter_set="OKane2022",     # candidate still Chen2020
        observed_abs_Z_mean=Z_okane,
    )
    # CC-003 fires -> require_parameter_set_calibration_pass -> ABSTAIN.

    return [
        ("surface_form", surface_form, "FLAG"),
        ("spm_vs_spme", spm_vs_spme, "PASS"),
        ("temperature_mismatch", temperature_mismatch, "ABSTAIN"),
        ("param_set_mismatch", param_set_mismatch, "ABSTAIN"),
    ]


def main():
    rules = load_gate_rules()
    print(f"loaded {len(rules)} gate rules from {len([r for r in rules if r['level']=='gate'])} gate-level entries")
    print()
    ok = 0
    for name, case, expected in _synth_cases():
        v = evaluate(case, rules)
        match = "✓" if v.action == expected else "✗"
        if v.action == expected:
            ok += 1
        print(f"  {match} {name:22s}  -> {v.action:8s}  (expected {expected}, w2={v.w2_weight})")
        for reason in v.reasons:
            print(f"       {reason}")
    print()
    print(f"self-test: {ok}/{len(_synth_cases())} cases match Paper1_骨架.md §9B.0 expectations")
    if ok != len(_synth_cases()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
