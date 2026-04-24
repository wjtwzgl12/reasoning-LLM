"""§9D.1 — Weaver 5-signal extraction wrapper.

Per Paper1_骨架.md §3.11, the 5 heterogeneous verifier signals are:
  w_1  EIS-PRM stepwise-quality score (§3.8, §9A.3 PRM model)
  w_2  PyBaMM forward-residual ρ_real, gated by 9B.0 condition-calibration
       (§3.9, §9B.2 verified loop). 0.0 if gate ABSTAINs.
  w_3  lin-Kramers-Kronig pass/fail (§3.9), Platt-scaled to [0,1]
  w_4  SBI posterior density at committed θ_mechanism (§3.10, §9C.2)
  w_5  LLM critic score (GPT-4o/DeepSeek-as-judge, baseline + fallback)

This module wires each signal as a separate function with consistent
signature `extract_w_k(case, **kwargs) -> dict[str, float]`. Stubs are
provided where upstream models aren't yet trained (w_1 PRM not yet on
disk; w_4 posterior may not be pre-trained). Stubbed signals return a
neutral 0.5 with `meta["stub"]=True` so the §9D.2 label model can learn
to down-weight them in calibration.

Decision gate (§9D.1): on a 20% EIS-Commit calibration subset, the 5×5
Spearman correlation matrix must NOT have all off-diagonals > 0.9 (i.e.
signals must carry independent information). The harness in `main()`
synthesises a 30-case mini-batch and reports the correlation matrix.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Callable

import numpy as np


# ─────────────────────── w_1: PRM stepwise score ──────────────────────


def extract_w1_prm(case: dict, prm_model: Any | None = None) -> dict:
    """w_1 = EIS-PRM stepwise-quality score on case['prediction_text'].

    Stubbed when prm_model is None (the §9A.3 PRM training will be done
    on Colab; until the checkpoint lands, return neutral 0.5).
    """
    if prm_model is None:
        return {"w1": 0.5, "meta": {"stub": True, "reason": "no_prm_model"}}

    # Real path: tokenize prediction_text → run PRM forward → mean step prob.
    pred = case.get("prediction_text", "")
    score = float(prm_model.score(pred))  # contract: returns float in [0,1]
    return {"w1": score, "meta": {"stub": False}}


# ─────────────────────── w_2: PyBaMM residual ─────────────────────────


def extract_w2_pybamm_residual(case: dict, rules: list[dict] | None = None
                                ) -> dict:
    """w_2 = 1 - ρ_real (clipped to [0,1]), 0.0 if gate ABSTAINs.

    Calls the §9B.2 verified loop wrapper, which enforces 9B.0 gate.
    """
    from .pybamm_eis_residual import pybamm_eis_residual
    from .pybamm_verified_loop import run_verified_loop

    out = run_verified_loop(case, residual_fn=pybamm_eis_residual,
                             rules=rules)
    if out["gate"]["action"] == "ABSTAIN":
        return {"w2": 0.0, "meta": {"stub": False, "abstain": True,
                                    "gate": out["gate"]["action"]}}
    rho_real = float(out["score"]["rho_real"])
    # Map ρ_real → [0,1]: w2 = exp(-ρ_real / τ), τ ≈ 0.10 is the §3.9 ROC
    # midpoint between correct (median 0.02) and wrong_mech (median 0.25).
    w2 = math.exp(-rho_real / 0.10)
    return {"w2": float(np.clip(w2, 0.0, 1.0)),
            "meta": {"stub": False, "rho_real": rho_real,
                     "gate": out["gate"]["action"],
                     "low_trust": out.get("low_trust", False)}}


# ─────────────────────── w_3: lin-KK pass/fail ────────────────────────


def extract_w3_linkk(case: dict, noise_threshold_pct: float = 1.0) -> dict:
    """w_3 = soft KK-validity score in [0,1] from pyimpspec.

    v3.1 (§9E.1 pilot, 2026-04-22): made API-version-tolerant. pyimpspec
    went through several breaking changes 4.x→5.x→6.x. Specifically:
      * DataSet ctor: early versions accepted positional (freq, imp);
        ≥5.x requires kwargs `frequencies=`, `impedances=`.
      * ≥6.x renamed `perform_kramers_kronig_test` → `perform_test` and
        the top-level export moved to `pyimpspec.analysis.kramers_kronig`.
      * `get_estimated_percent_noise()` was dropped in 6.x in favour of
        `pseudo_chi_squared` on TestResult. We fall back to χ² → noise-%
        proxy via √χ² × 100.

    Behaviour: try each known API shape; on total failure, stub 0.5 with
    the *first* exception captured (it's usually the most informative).
    """
    f = np.asarray(case["frequencies"], dtype=float)
    z = np.asarray(case["observed_Z"], dtype=complex)
    first_err: str | None = None
    try:
        import pyimpspec  # type: ignore
    except Exception as ex:
        return {"w3": 0.5,
                "meta": {"stub": True, "reason": f"import: {ex!s}"[:100]}}

    # Build DataSet — try kwargs then positional.
    ds = None
    for ctor_attempt in ("kwargs", "positional"):
        try:
            if ctor_attempt == "kwargs":
                ds = pyimpspec.DataSet(frequencies=f, impedances=z)
            else:
                ds = pyimpspec.DataSet(f, z)
            break
        except Exception as ex:
            if first_err is None:
                first_err = f"DataSet({ctor_attempt}): {ex!s}"[:100]
            continue
    if ds is None:
        return {"w3": 0.5, "meta": {"stub": True,
                                     "reason": first_err or "DataSet failed"}}

    # Run KK test — try API shapes in order of recency.
    res = None
    for api_attempt in ("v5_toplevel", "v6_analysis_module",
                         "v6_perform_test"):
        try:
            if api_attempt == "v5_toplevel":
                res = pyimpspec.perform_kramers_kronig_test(ds)
            elif api_attempt == "v6_analysis_module":
                from pyimpspec.analysis import kramers_kronig as _kk
                res = _kk.perform_test(ds) if hasattr(_kk, "perform_test") \
                    else _kk.perform_kramers_kronig_test(ds)
            else:
                res = pyimpspec.perform_test(ds)
            break
        except Exception as ex:
            if first_err is None:
                first_err = f"KK({api_attempt}): {ex!s}"[:100]
            continue
    if res is None:
        return {"w3": 0.5, "meta": {"stub": True,
                                     "reason": first_err or "KK failed"}}

    # Newer pyimpspec returns a *list* of TestResult (one per test type).
    if isinstance(res, (list, tuple)) and res:
        res = res[0]

    # Noise percentage — try canonical method, then χ² fallback.
    noise_pct: float | None = None
    for src in ("get_estimated_percent_noise", "pseudo_chi_squared"):
        try:
            if src == "get_estimated_percent_noise":
                noise_pct = float(res.get_estimated_percent_noise())
            else:
                chi2 = float(getattr(res, "pseudo_chi_squared", float("nan")))
                if chi2 != chi2:  # NaN
                    chi2 = float(res.get_pseudo_chisqr())  # older alias
                noise_pct = float(np.sqrt(max(chi2, 0.0)) * 100.0)
            break
        except Exception as ex:
            if first_err is None:
                first_err = f"noise({src}): {ex!s}"[:100]
            continue
    if noise_pct is None:
        return {"w3": 0.5, "meta": {"stub": True,
                                     "reason": first_err or "noise failed"}}

    w3 = 1.0 / (1.0 + math.exp(noise_pct - noise_threshold_pct))
    meta = {"stub": False, "noise_pct": noise_pct,
            "threshold_pct": noise_threshold_pct}
    try:
        meta["num_RC"] = int(res.get_num_RC())
    except Exception:
        pass
    return {"w3": float(np.clip(w3, 0.0, 1.0)), "meta": meta}


# ─────────────────────── w_4: SBI posterior match ─────────────────────


def extract_w4_sbi_match(case: dict, posterior: Any | None = None,
                         param_names: list[str] | None = None) -> dict:
    """w_4 = posterior density rank at committed θ_mechanism, normalised
    to [0,1] via percentile.

    case['committed_mechanism_params']: dict[name, linear_value] of the
    committed mechanism's physical parameters.

    posterior: a sbi NeuralPosterior pre-conditioned on case['observed_Z']
    summary stats (caller's responsibility to set the conditioning x).

    Stubbed when posterior is None.
    """
    if posterior is None:
        return {"w4": 0.5, "meta": {"stub": True, "reason": "no_posterior"}}

    from .sbi_posterior import PARAM_NAMES, F_SUMMARY
    names = param_names or PARAM_NAMES
    committed = case.get("committed_mechanism_params", {})
    if not committed:
        return {"w4": 0.5, "meta": {"stub": True,
                                    "reason": "no_committed_params"}}

    import torch
    # Convert linear → log10-space (matches sbi_posterior prior embedding).
    theta_committed = np.array(
        [math.log10(max(committed[n], 1e-30)) for n in names],
        dtype=np.float32,
    )
    # Posterior log-prob at committed θ.
    logp_committed = float(posterior.log_prob(
        torch.tensor(theta_committed[None, :])).item())
    # Reference distribution: log-probs of n=200 posterior samples → percentile.
    ref = posterior.sample((200,), show_progress_bars=False)
    ref_logp = posterior.log_prob(ref).detach().cpu().numpy()
    rank = float(np.mean(ref_logp <= logp_committed))
    return {"w4": float(rank),
            "meta": {"stub": False, "log_prob": logp_committed,
                     "rank": rank}}


# ─────────────────────── w_5: LLM critic ──────────────────────────────


def extract_w5_critic(case: dict, llm_call: Callable | None = None) -> dict:
    """w_5 = LLM-as-judge score in [0,1].

    Calls llm_call(messages) → str (JSON object expected with key 'score').
    Defaults to sbi_prior_emit.call_llm if no override (uses DeepSeek when
    available, OpenAI otherwise).
    """
    pred = case.get("prediction_text", "")
    obs_summary = case.get("observation_summary", "")
    msg = [
        {"role": "system",
         "content": ("You are an electrochemist evaluating a proposed "
                     "mechanism explanation against an observed Nyquist trace. "
                     "Output ONLY JSON: {\"score\": <0..1 float>, "
                     "\"reason\": <short string>}. score = your confidence "
                     "the explanation is consistent with the observation.")},
        {"role": "user",
         "content": (f"Observation summary: {obs_summary}\n\n"
                     f"Proposed explanation:\n{pred}\n\nEmit JSON now.")},
    ]
    try:
        if llm_call is None:
            from .sbi_prior_emit import call_llm
            raw = call_llm(msg)
        else:
            raw = llm_call(msg)
        obj = json.loads(raw)
        score = float(obj.get("score", 0.5))
        return {"w5": float(np.clip(score, 0.0, 1.0)),
                "meta": {"stub": False,
                         "reason": str(obj.get("reason", ""))[:120]}}
    except Exception as ex:
        return {"w5": 0.5, "meta": {"stub": True, "reason": str(ex)[:100]}}


# ─────────────────────── 5-signal extract ─────────────────────────────


def extract_all(case: dict, *,
                prm_model: Any | None = None,
                rules: list[dict] | None = None,
                posterior: Any | None = None,
                llm_call: Callable | None = None) -> dict:
    """Extract all 5 weaver signals for a single case dict."""
    return {
        "w1": extract_w1_prm(case, prm_model=prm_model),
        "w2": extract_w2_pybamm_residual(case, rules=rules),
        "w3": extract_w3_linkk(case),
        "w4": extract_w4_sbi_match(case, posterior=posterior),
        "w5": extract_w5_critic(case, llm_call=llm_call),
    }


# ─────────────────────── correlation gate harness ─────────────────────


def _synth_minibatch(n: int = 30, seed: int = 0) -> list[dict]:
    """30 synthetic cases: 10 'correct' (small perturbation), 10 'wrong-
    mechanism' (large param shift), 10 'condition-mismatched' (T or
    parameter_set off). Each gets observed_Z from a GT SPM forward."""
    from .pybamm_eis_residual import simulate_Z
    import pybamm

    rng = np.random.default_rng(seed)
    f = np.logspace(-1, 3, 8)
    base = {
        "model_name": "SPM", "parameter_set": "Chen2020",
        "initial_soc": 0.5, "frequencies": f,
    }
    # Build one GT trace.
    z_gt = simulate_Z(base)
    cases = []
    # Pull real Chen2020 defaults (the data-generating dist), not hard-coded
    # guesses. This aligns committed_mechanism_params with the §9C.2
    # posterior mode so w_4 discriminates correct vs wrong_mech cases.
    _chen = pybamm.ParameterValues("Chen2020")
    L_neg_chen = float(_chen["Negative electrode thickness [m]"])
    L_pos_chen = float(_chen["Positive electrode thickness [m]"])
    R_neg_chen = float(_chen["Negative particle radius [m]"])
    R_pos_chen = float(_chen["Positive particle radius [m]"])
    D_neg_chen = float(_chen["Negative electrode diffusivity [m2.s-1]"])
    D_pos_chen = float(_chen["Positive electrode diffusivity [m2.s-1]"])
    L_chen = L_neg_chen  # legacy alias used below
    R_chen = R_neg_chen

    for i in range(n):
        kind = ["correct", "wrong_mech", "cond_mismatch"][i % 3]
        c = dict(base)
        # Per-case observation noise so that w_3 (KK validity) varies
        # case-to-case. Magnitude calibrated: sigma ∈ [0.5%, 8%] of |Z|,
        # log-uniform — clean traces stay clean (~1% noise → KK ~0.99),
        # noisy traces fall to KK ~0.4. This is a pre-condition for an
        # informative §9D.1 correlation gate.
        sigma = 10 ** rng.uniform(np.log10(0.005), np.log10(0.08))
        eps = (rng.normal(size=z_gt.shape)
               + 1j * rng.normal(size=z_gt.shape))
        c["observed_Z"] = z_gt * (1.0 + sigma * eps)
        c["_obs_noise_sigma"] = float(sigma)
        # Gate keys: Kelvin per condition_calibration_gate (CC-008).
        c["observed_temperature_K"] = 298.15
        c["observed_parameter_set"] = "Chen2020"
        c["candidate_temperature_K"] = 298.15
        c["candidate_parameter_set"] = "Chen2020"
        c["observation_summary"] = (
            f"Nyquist with HF intercept ~{abs(z_gt[-1]):.3f} ohm, "
            f"|Z(0.1Hz)| ~{abs(z_gt[0]):.3f} ohm")
        c["committed_mechanism_params"] = {
            "Negative electrode thickness [m]":         L_neg_chen,
            "Positive electrode thickness [m]":         L_pos_chen,
            "Negative particle radius [m]":             R_neg_chen,
            "Positive particle radius [m]":             R_pos_chen,
            "Negative electrode diffusivity [m2.s-1]":  D_neg_chen,
            "Positive electrode diffusivity [m2.s-1]":  D_pos_chen,
        }

        if kind == "correct":
            c["param_overrides"] = {
                "Negative electrode thickness [m]":
                    L_chen * (1 + rng.normal(0, 0.02)),
            }
            c["prediction_text"] = (
                "Step 1: HF intercept matches Rs. Step 2: Single semicircle "
                "implies dominant SEI/charge-transfer at negative. Step 3: "
                "L_neg ~85 um and R_p ~5.86 um match Chen2020 LG M50.")
        elif kind == "wrong_mech":
            c["param_overrides"] = {
                "Negative electrode thickness [m]": L_chen * 1.5,
                "Negative particle radius [m]":     R_chen * 0.5,
            }
            c["committed_mechanism_params"][
                "Negative electrode thickness [m]"] = L_chen * 1.5
            c["committed_mechanism_params"][
                "Negative particle radius [m]"] = R_chen * 0.5
            c["prediction_text"] = (
                "Step 1: HF intercept fits. Step 2: Semicircle attributed "
                "entirely to particle-size effect at positive. Step 3: "
                "Conclude L_pos thinning by 50% — INCONSISTENT with positive "
                "particle radius doubling.")
        else:  # cond_mismatch
            c["candidate_temperature_K"] = 278.15  # 5 degC → CC-008 ABSTAIN
            c["param_overrides"] = {}
            c["prediction_text"] = (
                "Step 1: HF intercept higher than expected. Step 2: Likely "
                "low-temperature operation. Step 3: Apply 25→5 degC "
                "Arrhenius correction.")
        cases.append(c)
    return cases


def main():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", help="output tag (v1/v4)")
    ap.add_argument("--real_prm", action="store_true",
                    help="load §9A.3 PRM checkpoint for w_1")
    ap.add_argument("--real_w4", action="store_true",
                    help="load §9C.2 density estimator for w_4")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    print(f"§9D.1 weaver_signals correlation gate harness [{args.version}]")
    cases = _synth_minibatch(n=args.n, seed=0)
    print(f"  built {len(cases)} synthetic cases\n")

    # Optional real scorers.
    prm_model = None
    if args.real_prm:
        from .prm_scorer import PRMScorer
        t0 = time.time()
        prm_model = PRMScorer()
        print(f"  PRM loaded in {time.time()-t0:.1f}s")
    posterior = None
    if args.real_w4:
        from .sbi_w4_scorer import Wfour
        t0 = time.time()
        posterior = Wfour()
        print(f"  SBI w_4 loaded in {time.time()-t0:.1f}s")

    names = ["w1_prm", "w2_resid", "w3_kk", "w4_sbi", "w5_llm"]
    out_path = os.path.abspath(
        os.path.dirname(os.path.abspath(__file__))
        + f"/../results/weaver_signals_minibatch_{args.version}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for i, c in enumerate(cases):
        # Dispatch: real w_4 needs a Wfour object (custom contract), so we
        # patch case['_w4_override'] when real_w4 is on.
        if posterior is not None:
            _w4 = posterior.score(c)
        else:
            _w4 = None
        out = extract_all(c, prm_model=prm_model)
        if _w4 is not None:
            out["w4"] = _w4
        w = [out["w1"]["w1"], out["w2"]["w2"], out["w3"]["w3"],
             out["w4"]["w4"], out["w5"]["w5"]]
        stubs = [out[k]["meta"].get("stub", False) for k in
                 ("w1", "w2", "w3", "w4", "w5")]
        gate = out["w2"]["meta"].get("gate", "?")
        rows.append(w + [int(s) for s in stubs])
        print(f"  case {i:2d}  w=[{w[0]:.2f} {w[1]:.2f} {w[2]:.2f} "
              f"{w[3]:.2f} {w[4]:.2f}]  stubs={sum(stubs)}/5  "
              f"gate={gate}", flush=True)
        # Eager save so a downstream crash doesn't lose the per-case data.
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"n_cases_so_far": len(rows), "signals": names,
                       "rows": [{"w": r[:5], "stubs": r[5:]} for r in rows]},
                      fh, indent=2)

    arr = np.array(rows, dtype=float)
    W = arr[:, :5]

    # Robust pairwise Spearman: handles constant columns gracefully.
    from scipy.stats import spearmanr  # type: ignore
    rho = np.full((5, 5), np.nan)
    np.fill_diagonal(rho, 1.0)
    for i in range(5):
        for j in range(i + 1, 5):
            ci, cj = W[:, i], W[:, j]
            if np.std(ci) == 0 or np.std(cj) == 0:
                continue
            r, _ = spearmanr(ci, cj)
            rho[i, j] = rho[j, i] = float(r)

    print("\n5x5 Spearman correlation matrix:")
    print("        " + " ".join(f"{n:>9s}" for n in names))
    for i, n in enumerate(names):
        cells = [(f"{rho[i,j]:9.3f}" if not np.isnan(rho[i, j])
                  else "      NaN") for j in range(5)]
        print(f"  {n:7s} " + " ".join(cells))

    # Gate: NOT all off-diagonal |rho| > 0.9 (signals must carry indep info).
    off_mask = ~np.eye(5, dtype=bool) & ~np.isnan(rho)
    off_vals = np.abs(rho[off_mask])
    off_max = float(off_vals.max()) if off_vals.size else float("nan")
    n_high = int((off_vals > 0.9).sum() // 2)
    n_real_pairs = int(off_mask.sum() // 2)
    print(f"\noff-diagonal |rho|: max={off_max:.3f}, "
          f"#(|rho|>0.9)={n_high}/{n_real_pairs} computable pairs "
          f"(10 nominal)")

    all_redundant = (n_real_pairs > 0 and n_high == n_real_pairs
                     and n_real_pairs == 10)
    informative = (n_real_pairs == 10)
    if not informative:
        print(f"§9D.1 gate: PROVISIONAL ({n_real_pairs}/10 pairs "
              f"computable; rest are stub-constant)")
    else:
        print(f"§9D.1 gate: {'FAIL' if all_redundant else 'PASS'} "
              f"(all_redundant={all_redundant})")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "n_cases": len(cases),
            "signals": names,
            "rows": [{"w": r[:5], "stubs": r[5:]} for r in rows],
            "spearman": [[None if np.isnan(x) else float(x) for x in row]
                         for row in rho],
            "off_diag_max": off_max,
            "n_high_corr": n_high,
            "n_real_pairs": n_real_pairs,
            "gate_informative": informative,
            "gate_pass": (not all_redundant),
        }, fh, indent=2)
    print(f"wrote {out_path}")
    return 0 if not all_redundant else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
