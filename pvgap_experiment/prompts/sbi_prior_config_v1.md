# LLM Prior-Emission Prompt (§9C.1, v1)

## Purpose

Given a textual description of an electrochemical cell + a brief summary of an
observed Nyquist trace (and optionally a few candidate parameter sets the user
has in mind), the LLM emits a **structured JSON prior** over a fixed set of
PyBaMM-EIS parameters. This prior is the input distribution for the §9C.2
SNPE-C posterior estimator.

The prior must be machine-validated against `sbi_prior_schema_v1.json`
(strict). If the LLM's emission fails schema validation, the run aborts at
this step — we do not attempt to repair or guess. (Honest scoping: a prior we
cannot parse is a prior we cannot trust.)

## Parameter set under prior (v1, 6 params)

The v1 schema covers exactly these six PyBaMM parameter handles. Names match
PyBaMM's `ParameterValues` keys verbatim:

| key | physical meaning | rough O() |
|---|---|---|
| `Negative electrode thickness [m]` | L_neg | 1e-5 – 5e-4 |
| `Positive electrode thickness [m]` | L_pos | 1e-5 – 5e-4 |
| `Negative particle radius [m]` | R_p,neg | 1e-7 – 1e-5 |
| `Positive particle radius [m]` | R_p,pos | 1e-7 – 1e-5 |
| `Negative electrode diffusivity [m2.s-1]` | D_s,neg | 1e-16 – 1e-12 |
| `Positive electrode diffusivity [m2.s-1]` | D_s,pos | 1e-16 – 1e-12 |

Extending past 6 params is deferred to v2 — adding more dimensions without
matching SNPE-C training-set growth degrades coverage (see §6 item 28).

## Allowed distribution families (v1)

Per parameter, the LLM picks **one** of:

- `{"dist": "lognormal", "loc": <log-mean>, "scale": <log-std>}` — `loc`/`scale`
  are in **natural-log space** (i.e. `ln param`).
- `{"dist": "uniform", "low": <float>, "high": <float>}` — values in **linear**
  param units (m, m²/s, etc).

Plus mandatory `"support": [low, high]` truncation in linear units (always).

## JSON schema (machine-checked)

```json
{
  "type": "object",
  "required": ["parameters"],
  "additionalProperties": false,
  "properties": {
    "parameters": {
      "type": "array",
      "minItems": 6, "maxItems": 6,
      "items": {
        "type": "object",
        "required": ["name", "dist", "support"],
        "additionalProperties": true,
        "properties": {
          "name":    {"enum": [
            "Negative electrode thickness [m]",
            "Positive electrode thickness [m]",
            "Negative particle radius [m]",
            "Positive particle radius [m]",
            "Negative electrode diffusivity [m2.s-1]",
            "Positive electrode diffusivity [m2.s-1]"
          ]},
          "dist":    {"enum": ["lognormal", "uniform"]},
          "loc":     {"type": "number"},
          "scale":   {"type": "number", "exclusiveMinimum": 0},
          "low":     {"type": "number"},
          "high":    {"type": "number"},
          "support": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2, "maxItems": 2
          }
        }
      }
    }
  }
}
```

## Prompt template (system + user)

### System

> You are an electrochemist tasked with proposing a Bayesian prior over
> physical battery parameters for SBI. Output **only** a single JSON object
> matching the schema below — no prose, no markdown fences, no commentary.
>
> Rules:
> 1. The `parameters` array must contain **exactly 6 entries**, one per allowed
>    name. No duplicates, no omissions.
> 2. Use `lognormal` when the parameter spans ≥1 decade in plausible literature
>    range (e.g. diffusivities). Use `uniform` for tightly-bracketed quantities.
> 3. `loc` / `scale` for `lognormal` are in **ln-space** (so `loc = ln(median)`).
> 4. `support` is always linear, in SI units. It must enclose the bulk of your
>    distribution: for lognormal, place `support` at roughly
>    `[exp(loc - 3 scale), exp(loc + 3 scale)]`.
> 5. Calibrate to literature ranges — do not output supports tighter than the
>    cited Chen2020 / Marquis2019 / OKane2022 PyBaMM parameter sets unless the
>    user explicitly says the cell is one of those.

### User template

```
Cell description: {cell_text}

Observed Nyquist summary statistics:
  - HF intercept (R_s, Ω):           {hf_intercept}
  - First semicircle diameter (Ω):   {sc1_diameter}
  - Total |Z(0.01 Hz)| (Ω):          {z_lowfreq}
  - Apparent characteristic frequency at semicircle apex (Hz): {f_peak}

Candidate parameter sets the user is considering: {candidate_param_sets}

Emit JSON now.
```

## Reasonable-prior heuristic (used by §9C.1 reviewer)

Beyond schema validity, the human-in-the-loop reviewer (per §9C.1 Decision
gate) checks:

- Each `support` interval is contained within the broad O() ranges in the
  table above (±1 decade tolerance).
- For `lognormal`, `scale ∈ [0.1, 1.5]` (neither absurdly tight nor diffuse).
- The LLM does not propose `uniform` over more than 2 decades (that would
  effectively ignore physical-prior information).

## Five hand-review test cases (§9C.1 gate)

1. **LG M50 21700 cell, healthy** — expect tight priors near Chen2020 medians.
2. **Coin cell, NMC811 / graphite, 70°C aged 100 cycles** — expect wider
   diffusivity priors (degradation widens uncertainty).
3. **Pouch cell at 0°C** — expect lognormal D_s priors with low `loc` (slow
   diffusion at low T).
4. **Symmetric graphite/graphite half-cell** — expect `support` for positive-
   electrode params to collapse near zero or LLM should *flag* via comment-
   key (out-of-scope; v2 may add an `"absent"` dist).
5. **Unknown chemistry, only Nyquist summary given** — expect uniform priors
   spanning literature O() ranges, low confidence reflected in wide `support`.

Cases 1–3 / 5 must round-trip through `validate_prior(emission, schema)` →
`True`. Case 4 is allowed to FAIL schema validation (signalling v2 work).
PASS if ≥ 4 / 5 cases validate AND human reviewer marks them "physically
reasonable".
