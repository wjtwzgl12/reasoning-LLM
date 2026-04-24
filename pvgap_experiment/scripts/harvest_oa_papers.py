"""
9A.1 OA-paper harvester — OpenAlex → (arXiv|Unpaywall) PDF → text.
================================================================

Round-3 verdict (骨架 §9A.1) killed the "cheap 10× multiplier" hypotheses
(figtable, ontology gate relaxation). The remaining path to v1 scale
(5000 rules) is industrial OA harvest:

  OpenAlex search  →  OA PDF location  →  PDF → text
                                       →  downstream miner (mine_echem_rules)

At ~4-5 accepted rules per primary-lit OA paper (empirically measured
in C-round-1 and C-round-2), reaching 1000 rules needs ~200 OA papers
and reaching 5k needs ~1000. This script delivers the first link in
that chain: from a search query to a directory of plain-text bodies
ready for the existing miner.

Usage
-----
    python -m pvgap_experiment.scripts.harvest_oa_papers \\
        --query "electrochemical impedance spectroscopy nyquist" \\
        --max-papers 20 \\
        --out-dir pvgap_experiment/data/literature/raw/oa_harvest \\
        --email your@email

    # Then feed each .txt file into the existing miner:
    # for f in out-dir/*.txt; do
    #   python -m pvgap_experiment.scripts.mine_echem_rules --source $f ...
    # done

Design decisions (honest, not hidden)
-------------------------------------
- **Why not GROBID?** GROBID gives XML with section-typed paragraphs
  (better signal/noise than a raw PDF dump), but adds a Java+Docker
  dependency that will fail on the user's Windows machine without
  setup. We ship pdfminer.six as the text extractor and note the
  GROBID upgrade path in a TODO. Empirically (B-β/C rounds),
  pdfminer+segment_source yields enough for the 3-gate miner to work.
- **Why OpenAlex over CrossRef?** OpenAlex exposes OA PDF locations
  directly via `best_oa_location.pdf_url`; CrossRef forces a second
  Unpaywall roundtrip. We use OpenAlex primary, Unpaywall as fallback.
- **Why skip arXiv `html/`?** The LaTeXML HTML route we already have
  works for arXiv; this harvester targets the broader OA corpus
  (PMC, publisher OA, preprint servers) where PDF is the only form.
- **Relevance filter** is keyword-based ("nyquist", "impedance",
  "semicircle", "warburg", "charge transfer resistance"). Loose on
  purpose — later the miner's 3 gates + reviewer catch off-topic
  hits. This harvester's job is retrieval, not ranking.

Contract
--------
Writes per-paper artefacts to `out-dir/`:
  - `<source_key>.txt`   plain-text body (pdfminer)
  - `<source_key>.meta.json`  metadata: doi, title, year, venue, pdf_url
  - `_harvest_index.jsonl`  one row per attempted paper with success/failure

Does NOT touch `echem_rules_seed.jsonl` or staging — downstream miner
does that. This is purely a data-acquisition tool.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from typing import Iterator, Optional

# Lazy imports so `--help` works without requests/pdfminer installed.


# ───────────────────── OpenAlex search ──────────────────────────────

OPENALEX_BASE = "https://api.openalex.org/works"


def search_openalex(query: str, per_page: int, max_pages: int, email: str
                    ) -> Iterator[dict]:
    """Page through OpenAlex works matching `query` with OA filter.

    Yields normalized dicts: {doi, title, year, venue, pdf_url, source}.
    """
    import requests
    session = requests.Session()
    session.headers["User-Agent"] = (f"pvgap-echem-harvest/0.1 (mailto:{email})"
                                     if email else "pvgap-echem-harvest/0.1")
    cursor = "*"
    yielded = 0
    for page in range(max_pages):
        params = {
            "search": query,
            "filter": "open_access.is_oa:true,type:article",
            "per-page": str(per_page),
            "cursor": cursor,
        }
        if email:
            params["mailto"] = email
        r = session.get(OPENALEX_BASE, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  openalex page {page}: HTTP {r.status_code}; stopping")
            return
        data = r.json()
        results = data.get("results", [])
        if not results:
            return
        for w in results:
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            pdf = None
            loc = w.get("best_oa_location") or {}
            pdf = loc.get("pdf_url") or loc.get("url")
            if not pdf:
                # sometimes primary_location carries it
                loc2 = w.get("primary_location") or {}
                pdf = loc2.get("pdf_url")
            yield {
                "doi": doi,
                "title": (w.get("title") or "")[:300],
                "year": w.get("publication_year"),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                "pdf_url": pdf,
                "openalex_id": w.get("id"),
            }
            yielded += 1
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            return
        time.sleep(0.2)  # be polite; OpenAlex allows 10 req/s


def unpaywall_fallback(doi: str, email: str) -> Optional[str]:
    """If OpenAlex has no pdf_url, try Unpaywall directly."""
    if not doi:
        return None
    import requests
    url = f"https://api.unpaywall.org/v2/{doi}"
    try:
        r = requests.get(url, params={"email": email}, timeout=20)
        if r.status_code != 200:
            return None
        d = r.json()
        loc = d.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


# ───────────────────── PDF download + extraction ────────────────────

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def download_pdf(url: str, out_path: str) -> bool:
    import requests
    try:
        r = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=60,
                         allow_redirects=True)
        if r.status_code != 200:
            return False
        ct = r.headers.get("Content-Type", "")
        # Some publishers send HTML challenge page with 200; reject non-PDF.
        if "pdf" not in ct.lower() and not r.content.startswith(b"%PDF"):
            return False
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False


def pdf_to_text(pdf_path: str) -> str:
    """pdfminer.six body extraction. Returns empty string on failure."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        raise SystemExit("pdfminer.six not installed. pip install pdfminer.six")
    try:
        return extract_text(pdf_path) or ""
    except Exception as e:
        print(f"  pdfminer failed: {e!r}")
        return ""


# ───────────────────── relevance / cleanup ──────────────────────────

RELEVANCE_TERMS = [
    "nyquist", "impedance spectroscop", "semicircle",
    "warburg", "charge transfer resistance", "rct", "r_ct",
    "equivalent circuit", "bode plot", "cpe",
    "distribution of relaxation times",
]


def is_eis_relevant(text: str, min_hits: int = 2) -> tuple[bool, list[str]]:
    t = text.lower()
    hits = [kw for kw in RELEVANCE_TERMS if kw in t]
    return (len(hits) >= min_hits, hits)


def make_source_key(doi: str, openalex_id: str, title: str) -> str:
    base = doi or (openalex_id or "").rsplit("/", 1)[-1] or title[:40]
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    return base[:40] or "oa_paper"


# ───────────────────── driver ──────────────────────────────────────

def run(query: str, max_papers: int, out_dir: str, email: str,
        per_page: int, dry_run: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "_harvest_index.jsonl")
    existing_dois = set()
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                try:
                    existing_dois.add(json.loads(line).get("doi", ""))
                except Exception:
                    pass

    max_pages = (max_papers // per_page) + 3
    kept, attempted, skipped_dup, skipped_nokey = 0, 0, 0, 0
    index_fh = open(index_path, "a", encoding="utf-8")
    try:
        for meta in search_openalex(query, per_page, max_pages, email):
            if kept >= max_papers:
                break
            doi = meta["doi"]
            if doi and doi in existing_dois:
                skipped_dup += 1
                continue
            attempted += 1
            pdf_url = meta["pdf_url"]
            if not pdf_url and doi:
                pdf_url = unpaywall_fallback(doi, email)
            if not pdf_url:
                rec = {**meta, "status": "no_pdf_url"}
                index_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            source_key = make_source_key(doi, meta.get("openalex_id", ""),
                                         meta.get("title", ""))
            pdf_path = os.path.join(out_dir, f"{source_key}.pdf")
            txt_path = os.path.join(out_dir, f"{source_key}.txt")
            meta_path = os.path.join(out_dir, f"{source_key}.meta.json")
            print(f"  [{attempted:03d}] {source_key}  {(meta.get('title') or '')[:70]!r}")
            if dry_run:
                continue
            ok = download_pdf(pdf_url, pdf_path)
            if not ok:
                rec = {**meta, "status": "download_failed", "source_key": source_key}
                index_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            text = pdf_to_text(pdf_path)
            if len(text) < 2000:
                rec = {**meta, "status": "text_too_short",
                       "source_key": source_key, "text_len": len(text)}
                index_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # keep pdf for later retry
                continue
            relevant, hits = is_eis_relevant(text)
            if not relevant:
                rec = {**meta, "status": "not_relevant",
                       "source_key": source_key, "hits": hits}
                index_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                os.remove(pdf_path)
                continue
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            rec = {**meta, "status": "ok", "source_key": source_key,
                   "text_len": len(text), "hits": hits}
            index_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            index_fh.flush()
            kept += 1
    finally:
        index_fh.close()

    print(f"\nharvest complete: kept={kept} attempted={attempted} "
          f"skipped_dup={skipped_dup}")
    print(f"→ next: feed {out_dir}/*.txt into mine_echem_rules")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--query", required=True)
    ap.add_argument("--max-papers", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--email", default="",
                    help="contact email for OpenAlex/Unpaywall polite pool")
    ap.add_argument("--per-page", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true",
                    help="list candidate papers but do not download PDFs")
    args = ap.parse_args()
    run(args.query, args.max_papers, args.out_dir, args.email,
        args.per_page, args.dry_run)


if __name__ == "__main__":
    main()
