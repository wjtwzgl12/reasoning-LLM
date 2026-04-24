"""
PyBaMM-EIS Residual (§9B.1) — physical forward + ρ scoring.
=========================================================================

Wraps `pybammeis.EISSimulation` to compute impedance Z(ω) from a candidate
(model + parameter set + SOC), then compares against an observed Nyquist
trace via four normalized residuals:

    ρ_real    : Re-axis residual,  ⟨(ReΔ)²⟩ / ⟨(Re Z_obs)²⟩
    ρ_imag    : Im-axis residual,  ⟨(ImΔ)²⟩ / ⟨(Im Z_obs)²⟩
    ρ_complex : full-complex,      ⟨|Δ|²⟩  / ⟨|Z_obs|²⟩
    ρ_logmag  : log-magnitude,     ⟨(log|Z_obs| − log|Z_sim|)²⟩

where Δ = Z_obs − Z_sim. All four are dimensionless; smaller = better fit.
The pluggable `residual_fn` contract in `pybamm_verified_loop.py` calls this
module *only* when the §9B.0 condition-calibration gate has not ABSTAIN-ed.

Case-dict schema (consumed here):
    {
        "model_name":     str,    # "SPM" | "SPMe" | "DFN"
        "parameter_set":  str,    # e.g. "Chen2020", "Marquis2019"
        "initial_soc":    float,  # 0–1
        "frequencies":    list[float] | np.ndarray,  # Hz
        "observed_Z":     list[complex] | np.ndarray, # Ω, same length
        # optional pybamm overrides:
        "param_overrides": dict[str, float] | None,
    }

Notes
-----
- Simulation builds (~1–5 s for SPM, ~30 s for DFN) are cached on a hash of
  (model_name, parameter_set, initial_soc, frozenset(param_overrides.items())).
  In the §3.11 Weaver loop the cache is hit ≥99% of the time because the
  same candidate is queried over many observations.
- `solve(frequencies, method="direct")` returns Z scaled to Ω by the
  underlying `z_scale`. For SPM with default LG-M50 params at SOC=0.5 the
  high-frequency intercept lands ~0.04 Ω as expected for a coin-cell.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import numpy as np


# Lazy imports — keep this module importable on machines without pybamm.
_pybamm = None
_pybammeis = None


def _lazy_import():
    global _pybamm, _pybammeis
    if _pybamm is None:
        import pybamm as _pb
        import pybammeis as _pbe
        _pybamm = _pb
        _pybammeis = _pbe
    return _pybamm, _pybammeis


_MODEL_REGISTRY = {
    "SPM":  "SPM",
    "SPMe": "SPMe",
    "DFN":  "DFN",
}


# LRU-capped sim cache. PyBaMM EISSimulation objects retain compiled CasADi
# graphs and JIT state; on §9C.2 10⁴-sim runs unbounded growth correlates with
# wall-time blowup (sim 2501→3001 jumped 4702s→13995s in the first attempt).
# Each entry is ~50 MB; cap at 64 → ~3 GB RSS ceiling.
from collections import OrderedDict
_SIM_CACHE_MAX = 64
_SIM_CACHE: "OrderedDict[str, Any]" = OrderedDict()


def _cache_key(model_name: str, parameter_set: str, initial_soc: float,
               param_overrides: dict | None) -> str:
    blob = json.dumps({
        "m": model_name, "p": parameter_set, "soc": float(initial_soc),
        "ov": sorted((param_overrides or {}).items()),
    }, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def _build_eis_sim(model_name: str, parameter_set: str, initial_soc: float,
                   param_overrides: dict | None):
    pybamm, pybammeis = _lazy_import()
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"unknown model_name {model_name!r}; "
                         f"expected one of {list(_MODEL_REGISTRY)}")

    Cls = getattr(pybamm.lithium_ion, _MODEL_REGISTRY[model_name])
    model = Cls(options={"surface form": "differential"})

    pv = pybamm.ParameterValues(parameter_set)
    if param_overrides:
        pv.update(param_overrides, check_already_exists=False)

    return pybammeis.EISSimulation(
        model, parameter_values=pv, initial_soc=initial_soc,
    )


def get_eis_sim(model_name: str, parameter_set: str, initial_soc: float,
                param_overrides: dict | None = None):
    """Cached EISSimulation factory."""
    k = _cache_key(model_name, parameter_set, initial_soc, param_overrides)
    sim = _SIM_CACHE.get(k)
    if sim is None:
        sim = _build_eis_sim(model_name, parameter_set, initial_soc,
                             param_overrides)
        _SIM_CACHE[k] = sim
        # LRU eviction: drop oldest when over cap.
        while len(_SIM_CACHE) > _SIM_CACHE_MAX:
            _SIM_CACHE.popitem(last=False)
    else:
        _SIM_CACHE.move_to_end(k)
    return sim


def simulate_Z(case: dict) -> np.ndarray:
    """Run PyBaMM-EIS forward at case['frequencies'] → complex Z (Ω)."""
    sim = get_eis_sim(
        case["model_name"], case["parameter_set"],
        float(case["initial_soc"]), case.get("param_overrides"),
    )
    f = np.asarray(case["frequencies"], dtype=float)
    Z = sim.solve(f, method="direct")
    return np.asarray(Z, dtype=complex)


def residuals(z_obs: np.ndarray, z_sim: np.ndarray) -> dict[str, float]:
    """Compute four normalized ρ residuals. Both arrays must align in ω."""
    z_obs = np.asarray(z_obs, dtype=complex)
    z_sim = np.asarray(z_sim, dtype=complex)
    if z_obs.shape != z_sim.shape:
        raise ValueError(f"shape mismatch z_obs {z_obs.shape} vs "
                         f"z_sim {z_sim.shape}")

    eps = 1e-30
    delta = z_obs - z_sim

    rho_real = float(np.mean(delta.real ** 2)
                     / max(np.mean(z_obs.real ** 2), eps))
    rho_imag = float(np.mean(delta.imag ** 2)
                     / max(np.mean(z_obs.imag ** 2), eps))
    rho_complex = float(np.mean(np.abs(delta) ** 2)
                        / max(np.mean(np.abs(z_obs) ** 2), eps))
    log_obs = np.log(np.abs(z_obs) + eps)
    log_sim = np.log(np.abs(z_sim) + eps)
    rho_logmag = float(np.mean((log_obs - log_sim) ** 2))

    return {
        "rho_real":    rho_real,
        "rho_imag":    rho_imag,
        "rho_complex": rho_complex,
        "rho_logmag":  rho_logmag,
    }


def pybamm_eis_residual(case: dict) -> dict[str, float]:
    """§9B.2 entrypoint: case → 4 ρ residuals.

    Bound as `residual_fn` in `pybamm_verified_loop.run_verified_loop`.
    """
    z_obs = np.asarray(case["observed_Z"], dtype=complex)
    z_sim = simulate_Z(case)
    return residuals(z_obs, z_sim)


# ────────────────────────── CLI smoke test ──────────────────────────


def _smoke():
    """SPM / Chen2020 / SOC=0.5: forward 30 freqs, self-residual = 0."""
    pybamm, _ = _lazy_import()
    f = np.logspace(-2, 4, 30)  # 0.01 Hz – 10 kHz
    case = {
        "model_name":    "SPM",
        "parameter_set": "Chen2020",
        "initial_soc":   0.5,
        "frequencies":   f,
    }
    print(f"building SPM/Chen2020/SOC=0.5 ...")
    z = simulate_Z(case)
    print(f"  Z range: |Z| ∈ [{np.abs(z).min():.4f}, {np.abs(z).max():.4f}] Ω")
    print(f"  Re(Z) HF intercept ≈ {z[-1].real:.5f} Ω (expect O(1e-2))")

    # Self-residual must be 0.
    case["observed_Z"] = z
    rho = pybamm_eis_residual(case)
    print(f"  self-residual: {rho}")
    assert all(v < 1e-20 for v in rho.values()), \
        f"self-residual not zero: {rho}"

    # Perturb candidate: change negative-electrode thickness +10% → expect
    # all ρ > 0.
    case_pert = dict(case)
    case_pert["param_overrides"] = {"Negative electrode thickness [m]":
                                    1.1 * 8.52e-5}
    case_pert.pop("observed_Z")
    print(f"\nperturbing L_neg +10% ...")
    z_pert = simulate_Z(case_pert)
    rho_pert = residuals(z, z_pert)
    print(f"  perturbed residual: {rho_pert}")
    assert rho_pert["rho_complex"] > 1e-6, \
        "perturbation produced no measurable residual"

    print(f"\n✓ §9B.1 smoke test PASS")


if __name__ == "__main__":
    _smoke()
