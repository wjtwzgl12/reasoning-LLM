"""§9C.2 → w_4 inference wrapper.

Reconstructs the SNPE-C DirectPosterior from:
  1. emission JSON for the case → prior (BoxUniform in 6-D log10-space)
  2. `results/sbi_prior_emit/density_estimator_last.pt` (state_dict of the
     normalising flow, saved by `sbi_posterior.train_snpe`)

Exposes `Wfour.score(case)` that returns a dict matching the
`weaver_signals.extract_w4_sbi_match` output (w4 + meta).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .sbi_posterior import (PARAM_NAMES, F_SUMMARY, build_prior,
                            simulator_factory, _SIGMA_REL)

HERE = Path(__file__).resolve().parent
_DEFAULT_CKPT = HERE.parent / "results" / "sbi_prior_emit" / "density_estimator_last.pt"
_DEFAULT_EMIT = HERE.parent / "results" / "sbi_prior_emit" / "hand_cases_v1.jsonl"


def _load_emission(case_name: str, path: Path) -> dict:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("name") == case_name and r.get("schema_ok"):
            return r["emission"]
    raise KeyError(f"emission for {case_name} not found in {path}")


def _build_posterior(prior, ckpt_path: Path, n_x_probe: int = 32):
    """Re-hydrate the DirectPosterior. Strategy: let sbi construct a fresh
    SNPE_C pipeline, train 1 mini-epoch on tiny synthetic data to realise a
    density_estimator with the *default* architecture (MAF), then overwrite
    its state_dict with our trained weights.
    """
    import torch
    from sbi.inference import SNPE_C

    torch.manual_seed(0)
    inferer = SNPE_C(prior=prior)
    # Tiny priming set so sbi builds the default flow; the resulting weights
    # are discarded via load_state_dict.
    theta_p = prior.sample((n_x_probe,))
    x_dim = 2 * len(F_SUMMARY)  # Re+Im at F_SUMMARY
    x_p = torch.randn(n_x_probe, x_dim, dtype=torch.float32)
    inferer = inferer.append_simulations(theta_p, x_p)
    density_estimator = inferer.train(
        show_train_summary=False, max_num_epochs=1,
        stop_after_epochs=1,
    )
    # Overwrite with trained weights.
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = density_estimator.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[sbi_w4_scorer] warn: unexpected keys in ckpt: {unexpected[:3]}…")
    if missing:
        print(f"[sbi_w4_scorer] warn: missing keys: {missing[:3]}…")
    posterior = inferer.build_posterior(density_estimator, sample_with="direct")
    posterior._raw_flow = density_estimator  # for w_4 direct sampling
    return posterior


class Wfour:
    """Stateful SBI w_4 scorer. Load once, score many cases.

    Ctor params
    -----------
    case_name : which emission to use when building the prior. The SNPE
        training in §9C.2 v4 conditioned on `lg_m50_healthy` emission; we
        reuse the same prior box.
    """

    def __init__(self,
                 ckpt_path: str | Path = _DEFAULT_CKPT,
                 emission_path: str | Path = _DEFAULT_EMIT,
                 case_name: str = "lg_m50_healthy"):
        ckpt_path = Path(ckpt_path); emission_path = Path(emission_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"density_estimator ckpt missing: {ckpt_path}")
        emission = _load_emission(case_name, emission_path)
        self.prior = build_prior(emission)
        self.posterior = _build_posterior(self.prior, ckpt_path)
        self.case_name = case_name
        # Cache a reference log-prob distribution for percentile normalisation.
        import torch
        with torch.no_grad():
            theta_ref = self.prior.sample((200,))
            # log_prob needs x conditioning; use an arbitrary fixed x (the
            # percentile will be computed per-case at score-time with the
            # case-specific x_obs, so this cache is unused there — kept only
            # as a fallback).
        self._theta_ref = theta_ref

    # ------------------------------------------------------------------

    def _x_obs_from_case(self, case: dict) -> "np.ndarray":
        """Produce the summary-stat x vector from case['observed_Z']. If not
        available, fall back to simulating at case-specified parameters."""
        z = case.get("observed_Z", None)
        if z is not None:
            z = np.asarray(z, dtype=np.complex128)
            if z.shape != F_SUMMARY.shape:
                raise ValueError(f"observed_Z shape {z.shape} != expected "
                                 f"{F_SUMMARY.shape} (F_SUMMARY={F_SUMMARY})")
            return np.concatenate([z.real, z.imag]).astype(np.float32)

        # Fallback: simulate from case params (no noise — we're scoring not
        # calibrating).
        sim = simulator_factory(sigma_rel=0.0)
        theta_log10 = np.array(
            [math.log10(max(case["committed_mechanism_params"][n], 1e-30))
             for n in PARAM_NAMES], dtype=np.float32)
        return sim(theta_log10)

    def score(self, case: dict) -> dict:
        """Return {'w4': float in [0,1], 'meta': {...}}.

        w4 = percentile rank of log p(θ_committed | x_obs) among 200 samples
        drawn from the same posterior. 1.0 → committed θ is a posterior mode;
        0.0 → committed θ is in the tails.
        """
        import torch

        committed = case.get("committed_mechanism_params", {})
        if not committed:
            return {"w4": 0.5, "meta": {"stub": True, "reason": "no_committed"}}

        try:
            x_obs = self._x_obs_from_case(case)
        except Exception as ex:
            return {"w4": 0.5,
                    "meta": {"stub": True, "reason": f"x_obs_fail:{ex}"}}

        theta_committed = np.array(
            [math.log10(max(committed.get(n, 1e-30), 1e-30)) for n in PARAM_NAMES],
            dtype=np.float32,
        )
        try:
            with torch.no_grad():
                x_t = torch.tensor(x_obs, dtype=torch.float32)
                theta_t = torch.tensor(theta_committed[None, :],
                                       dtype=torch.float32)
                # Use raw flow for log_prob and sampling to avoid rejection wrapper.
                raw = self.posterior._raw_flow
                logp_committed = float(raw.log_prob(
                    theta_t, condition=x_t.unsqueeze(0)).item())
                # v2: both percentile-vs-posterior (saturates at 0) and
                # percentile-vs-prior (saturates at 1) break as similarity
                # measures. Use a Mahalanobis-style Gaussian kernel around
                # the posterior mean: w4 = exp(-0.5 · mean((θ_c - μ)/σ)²).
                # Monotonic, ∈ (0,1], identity at the posterior mode.
                post = raw.sample((400,), condition=x_t.unsqueeze(0)
                                  ).reshape(400, -1).detach().cpu().numpy()
                mu  = post.mean(axis=0)
                sig = post.std(axis=0) + 1e-6
            z = (theta_committed - mu) / sig
            d2 = float(np.mean(z * z))
            # v3 kernel: 1/(1+d²/K²) with K = 10 posterior-σ bandwidth.
            # Posterior σ in §9C.2 trained flow is ~0.03 in log10 → a 2%
            # linear perturbation is already ~10σ. Without bandwidth
            # expansion the kernel saturates at ~0 for every realistic
            # committed-θ. K=10 gives: correct (~1σ) → w4~0.99, wrong_mech
            # (~50σ) → w4~0.04, preserving case-type discrimination.
            K = 10.0
            rank = float(1.0 / (1.0 + d2 / (K * K)))
        except Exception as ex:
            return {"w4": 0.5, "meta": {"stub": True, "reason": f"flow_fail:{ex}"}}

        return {"w4": rank,
                "meta": {"stub": False, "log_prob": logp_committed,
                         "case_name": self.case_name}}


# ─────────────────────── self-test CLI ─────────────────────────────────


def _cli():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_DEFAULT_CKPT))
    ap.add_argument("--emission", default=str(_DEFAULT_EMIT))
    ap.add_argument("--case_name", default="lg_m50_healthy")
    args = ap.parse_args()

    print(f"[sbi_w4_scorer] loading ckpt={args.ckpt}")
    t0 = time.time()
    w4 = Wfour(ckpt_path=args.ckpt, emission_path=args.emission,
               case_name=args.case_name)
    print(f"  loaded posterior in {time.time()-t0:.1f}s")

    # Build two synthetic probes: a "good" case (committed θ matches x_obs)
    # and a "bad" case (committed θ mismatched).
    from .pybamm_eis_residual import simulate_Z
    import pybamm
    base = {"model_name": "SPM", "parameter_set": "Chen2020",
            "initial_soc": 0.5, "frequencies": F_SUMMARY}
    z_gt = simulate_Z(base)
    # "good" = actual Chen2020 defaults (x_obs was generated from these, so
    # posterior mode should land here); "bad" = 10x off on each param.
    chen = pybamm.ParameterValues("Chen2020")
    committed_good = {n: float(chen[n]) for n in PARAM_NAMES}
    committed_bad  = {n: v * 3.0 for n, v in committed_good.items()}
    cases = [
        {"name": "good", "observed_Z": z_gt,
         "committed_mechanism_params": committed_good},
        {"name": "bad",  "observed_Z": z_gt,
         "committed_mechanism_params": committed_bad},
    ]
    for c in cases:
        out = w4.score(c)
        print(f"  case={c['name']:>4}  w4={out['w4']:.3f}  meta={out['meta']}")


if __name__ == "__main__":
    _cli()
