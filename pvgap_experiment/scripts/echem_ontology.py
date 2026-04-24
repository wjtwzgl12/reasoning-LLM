"""
EIS/electrochemistry ontology dict.

Maps many surface-form synonyms (abbreviations, symbols, long names, a few
non-English equivalents) to a canonical concept node, so that the
observation↔evidence_quote token-overlap gate in mine_echem_rules.py does
not reject a card just because the quote says "R_ct" while the observation
field says "charge transfer resistance".

Usage:
    from pvgap_experiment.scripts.echem_ontology import ontology_tokens
    ontology_tokens("the high-frequency Rct")  # -> {"charge_transfer_resistance",
                                                #      "high_frequency"}

Keys are case-folded; values are canonical node strings (snake_case).
Add new entries as new papers surface new jargon — keep it broad but
electrochemistry-scoped.
"""
from __future__ import annotations
import re

# canonical_node -> list of surface forms (case-insensitive)
_NODES: dict[str, list[str]] = {
    # ── resistances ─────────────────────────────────────────────────
    "solution_resistance": ["rs", "r_s", "r s", "solution resistance",
                            "electrolyte resistance", "uncompensated resistance",
                            "ohmic resistance", "bulk electrolyte resistance"],
    "charge_transfer_resistance": ["rct", "r_ct", "r ct", "charge transfer resistance",
                                   "charge-transfer resistance",
                                   "polarization resistance", "rp", "r_p",
                                   "faradaic resistance"],
    "grain_boundary_resistance": ["rgb", "r_gb", "grain boundary resistance",
                                  "gb resistance", "intergranular resistance"],
    "bulk_resistance": ["rb", "r_b", "bulk resistance", "grain bulk resistance",
                        "grain interior resistance", "lattice resistance"],
    "sei_resistance": ["rsei", "r_sei", "rfilm", "r_film", "r sei",
                       "sei resistance", "sei film resistance",
                       "surface film resistance", "passive film resistance"],
    "contact_resistance": ["rc", "r_c", "contact resistance",
                           "particle-particle contact", "constriction resistance",
                           "current-collector resistance"],
    "recombination_resistance": ["rrec", "r_rec", "recombination resistance"],
    "transport_resistance": ["transport resistance", "mass transport resistance",
                             "gas transport resistance", "ion transport resistance"],

    # ── capacitances ────────────────────────────────────────────────
    "double_layer_capacitance": ["cdl", "c_dl", "edl capacitance",
                                 "double-layer capacitance", "double layer capacitance",
                                 "electric double layer"],
    "coating_capacitance": ["cc", "c_c", "coating capacitance", "film capacitance"],
    "geometric_capacitance": ["geometric capacitance", "cg", "c_g",
                              "dielectric capacitance"],
    "chemical_capacitance": ["cmu", "c_mu", "chemical capacitance"],

    # ── CPE / Warburg / fit elements ────────────────────────────────
    "constant_phase_element": ["cpe", "constant phase element", "constant-phase element",
                               "q element", "fractional capacitor"],
    "cpe_exponent": ["n", "alpha", "beta", "cpe exponent", "dispersion coefficient",
                     "cpe n", "cpe alpha"],
    "warburg": ["warburg", "warburg element", "warburg impedance",
                "zw", "z_w", "warburg slope", "warburg region", "warburg tail",
                "sloping line"],
    "finite_warburg": ["finite warburg", "finite-length warburg", "bounded warburg",
                       "reflective warburg", "transmissive warburg"],

    # ── EIS features / shapes ───────────────────────────────────────
    "high_frequency_arc": ["high-frequency arc", "high frequency arc", "hf arc",
                           "high-frequency semicircle", "high frequency semicircle",
                           "hf semicircle", "first semicircle", "first arc"],
    "low_frequency_arc": ["low-frequency arc", "low frequency arc", "lf arc",
                          "low-frequency semicircle", "low frequency semicircle",
                          "lf semicircle", "second semicircle", "second arc",
                          "large semicircle"],
    "mid_frequency_arc": ["mid-frequency arc", "medium-frequency arc",
                          "mid frequency arc", "mid-frequency semicircle",
                          "intermediate-frequency semicircle"],
    "semicircle": ["semicircle", "semi-circle", "depressed semicircle", "arc",
                   "flattened arc"],
    "45_degree_line": ["45°", "45 degree", "45-degree line", "45° line",
                       "45 degrees slope"],
    "porous_response": ["porous electrode", "porous structure", "porous transport",
                        "transmission line", "de levie", "pore impedance"],
    "inductive_loop": ["inductive loop", "inductive behavior", "inductance",
                       "inductive arc", "positive imaginary"],
    "nyquist_intercept": ["intercept", "high-frequency intercept",
                          "hf intercept", "z' intercept", "real axis intercept",
                          "x-intercept"],
    "phase_angle": ["phase angle", "phase shift", "phase", "-phase",
                    "bode phase"],
    "impedance_modulus": ["|z|", "impedance modulus", "modulus of impedance",
                          "absolute impedance"],

    # ── DRT specific ────────────────────────────────────────────────
    "drt_peak": ["drt peak", "relaxation peak", "peak in the drt",
                 "distribution of relaxation times", "gamma(tau)"],
    "relaxation_time": ["tau", "τ", "relaxation time", "time constant",
                        "characteristic time", "rc time constant"],
    "characteristic_frequency": ["characteristic frequency", "peak frequency",
                                 "f_max", "fmax", "summit frequency",
                                 "apex frequency"],

    # ── processes / mechanisms ──────────────────────────────────────
    "charge_transfer": ["charge transfer", "charge-transfer", "electron transfer",
                        "faradaic reaction"],
    "mass_diffusion": ["mass diffusion", "mass transport", "diffusion",
                       "solid-state diffusion", "solid-phase diffusion"],
    "ion_diffusion": ["ion diffusion", "ionic diffusion", "li+ diffusion",
                      "lithium diffusion", "na+ diffusion", "sodium diffusion"],
    "desolvation": ["desolvation", "desolvating", "solvation shell"],
    "gas_diffusion": ["gas diffusion", "gas transport", "gas permeability",
                      "gas phase transport"],
    "sei_formation": ["sei formation", "sei growth", "solid electrolyte interphase"],
    "lithium_plating": ["li plating", "lithium plating", "li deposition",
                        "lithium deposition", "dendrite", "dendritic"],
    "dissolution": ["dissolution", "pitting", "corrosion pit", "active corrosion"],
    "passivation": ["passivation", "passive film", "passive layer", "protective layer"],
    "water_uptake": ["water uptake", "water absorption", "electrolyte diffusion",
                     "water ingress", "moisture ingress"],
    "adhesion_loss": ["loss of adhesion", "delamination", "disbondment",
                      "interface separation"],
    "grain_boundary_process": ["grain boundary", "grain boundaries",
                               "intergranular", "gb process", "gb conduction"],
    "bulk_ion_transport": ["bulk conductivity", "grain lattice", "lattice conduction",
                           "bulk conduction", "grain interior conduction",
                           "hopping mechanism"],

    # ── operating-state descriptors ─────────────────────────────────
    "low_soc": ["low soc", "low state of charge", "discharged state",
                "soc < 20", "sub-20% soc", "end-of-discharge"],
    "high_soc": ["high soc", "fully charged", "100% soc", "charged state"],
    "low_temperature": ["low temperature", "sub-zero", "-40", "low-t",
                        "cryogenic", "cold temperature"],
    "aged": ["aged", "aging", "degraded", "cycled", "capacity fade"],
    "fresh": ["fresh", "pristine", "uncycled", "bol", "beginning of life"],

    # ── systems ─────────────────────────────────────────────────────
    "lithium_ion_battery": ["li-ion", "lithium-ion battery", "lib", "li-ion cell"],
    "sodium_ion_battery": ["na-ion", "sodium-ion battery", "nib", "na-ion cell"],
    "lithium_sulfur_battery": ["li-s", "lithium-sulfur", "li-s cell", "polysulfide"],
    "solid_oxide_fuel_cell": ["sofc", "solid oxide fuel cell"],
    "perovskite_solar_cell": ["perovskite solar", "perovskite cell", "psc"],
    "biosensor": ["biosensor", "bioelectrode", "label-free"],
    "coating": ["coating", "paint", "polymer coating", "protective film"],
}


def _build_lookup() -> dict[str, str]:
    """Flatten {node: [aliases]} → {alias_casefolded: node}."""
    lut: dict[str, str] = {}
    for node, aliases in _NODES.items():
        for a in aliases:
            key = a.strip().lower()
            # register both the phrase and its compact (no-space) form so
            # we can match "R_ct" ↔ "r ct" ↔ "rct".
            lut[key] = node
            lut[re.sub(r"\s+", "", key)] = node
    return lut


_LUT = _build_lookup()
# Sort phrase keys (multi-word) first so we match longest phrases first.
_PHRASES = sorted({k for k in _LUT if " " in k or "-" in k}, key=len, reverse=True)
_SINGLE_TOKENS = {k for k in _LUT if " " not in k and "-" not in k}


def ontology_tokens(text: str) -> set[str]:
    """Return the set of canonical ontology nodes the text references.

    Strategy:
      1. Lowercase the text.
      2. Walk known multi-word/hyphenated phrases longest-first, record node
         hits, then blank the matched span so a shorter alias does not
         double-count it.
      3. Tokenize the residue and map any single-token alias to its node.
    """
    if not text:
        return set()
    s = text.lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    hits: set[str] = set()
    # multi-word phrases first
    for phrase in _PHRASES:
        idx = s.find(phrase)
        if idx != -1:
            hits.add(_LUT[phrase])
            s = s[:idx] + " " * len(phrase) + s[idx + len(phrase):]
    # single tokens
    for tok in re.findall(r"[a-z0-9_]+", s):
        if tok in _SINGLE_TOKENS:
            hits.add(_LUT[tok])
    return hits


def ontology_overlap(a: str, b: str) -> int:
    """Count ontology nodes that appear in both strings."""
    return len(ontology_tokens(a) & ontology_tokens(b))
