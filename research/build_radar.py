#!/usr/bin/env python3
"""
Research Radar v1 (polished)
- Pulls recent PubMed papers for one or more queries
- Summarizes abstracts (default: Ollama; optional: OpenAI)
- Writes docs/research/papers.json for GitHub Pages to render

Usage:
  python build_radar.py --days 7 --max-per-query 20 --summarizer ollama --ollama-model llama3
Env:
  NCBI_API_KEY        (optional)
  OLLAMA_BASE_URL     (default http://localhost:11434)
  OLLAMA_MODEL        (default llama3)
  OPENAI_API_KEY      (optional, if using --summarizer openai)
  OPENAI_MODEL        (default gpt-4o-mini)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Paper:
    pmid: str
    title: str
    journal: str
    pub_date: str
    authors: List[str]
    abstract: str
    url: str
    query_tag: str
    summary_bullets: List[str]


# -----------------------------
# Helpers
# -----------------------------
def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> bytes:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def normalize_month(m: str) -> str:
    """
    PubMed month can be numeric or abbreviations (Jan, Feb, ...).
    Convert to 2-digit month when possible; else return empty string.
    """
    m = clean_text(m)
    if not m:
        return ""
    if m.isdigit():
        mm = int(m)
        return f"{mm:02d}" if 1 <= mm <= 12 else ""
    lookup = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    key = m[:3].lower()
    return lookup.get(key, "")


def parse_date_sortkey(pub_date: str) -> Tuple[int, int, int]:
    """
    Best-effort date parsing for sorting newest-first.
    Returns (year, month, day). Unknown values become 0.
    Accepts:
      - YYYY-MM-DD
      - YYYY-MM
      - YYYY
      - MedlineDate strings (we salvage year if present)
    """
    s = clean_text(pub_date)
    if not s:
        return (0, 0, 0)

    # Grab a 4-digit year anywhere
    m = re.search(r"(19\d{2}|20\d{2})", s)
    year = int(m.group(1)) if m else 0

    # If it looks like YYYY-MM-DD
    m2 = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m2:
        return (int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))

    # If it looks like YYYY-MM
    m3 = re.match(r"^(\d{4})-(\d{2})$", s)
    if m3:
        return (int(m3.group(1)), int(m3.group(2)), 0)

    # If only YYYY
    m4 = re.match(r"^(\d{4})$", s)
    if m4:
        return (int(m4.group(1)), 0, 0)

    return (year, 0, 0)


# -----------------------------
# PubMed E-utilities
# -----------------------------
def pubmed_esearch(term: str, days: int, retmax: int, api_key: Optional[str] = None) -> List[str]:
    today = dt.date.today()
    start = today - dt.timedelta(days=days)

    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "pub+date",
        "mindate": start.isoformat(),
        "maxdate": today.isoformat(),
        "datetype": "pdat",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{EUTILS}/esearch.fcgi?{urlencode(params)}"
    raw = http_get(url)
    data = json.loads(raw.decode("utf-8"))
    return data.get("esearchresult", {}).get("idlist", [])


def pubmed_efetch_details(pmids: List[str], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key

    url = f"{EUTILS}/efetch.fcgi?{urlencode(params)}"
    raw = http_get(url)
    root = ET.fromstring(raw)

    records: List[Dict[str, Any]] = []
    for art in root.findall(".//PubmedArticle"):
        records.append(parse_pubmed_article(art))
    return records


def parse_pubmed_article(art: ET.Element) -> Dict[str, Any]:
    def find_text(path: str) -> str:
        el = art.find(path)
        return clean_text(el.text if el is not None else "")

    pmid = find_text(".//MedlineCitation/PMID")
    title = find_text(".//Article/ArticleTitle")
    journal = find_text(".//Article/Journal/Title")

    # Date: prefer ArticleDate; else JournalIssue PubDate; else MedlineDate
    pub_date = ""
    ad = art.find(".//Article/ArticleDate")
    if ad is not None:
        y = find_text(".//Article/ArticleDate/Year")
        m = normalize_month(find_text(".//Article/ArticleDate/Month"))
        d = find_text(".//Article/ArticleDate/Day")
        if y:
            pub_date = y
            if m:
                pub_date += f"-{m}"
                if d and d.isdigit():
                    pub_date += f"-{int(d):02d}"

    if not pub_date:
        y = find_text(".//JournalIssue/PubDate/Year")
        m = normalize_month(find_text(".//JournalIssue/PubDate/Month"))
        d = find_text(".//JournalIssue/PubDate/Day")
        medline = find_text(".//JournalIssue/PubDate/MedlineDate")
        if y:
            pub_date = y
            if m:
                pub_date += f"-{m}"
                if d and d.isdigit():
                    pub_date += f"-{int(d):02d}"
        else:
            pub_date = medline  # best we can do

    # Authors
    authors: List[str] = []
    for a in art.findall(".//Article/AuthorList/Author"):
        last = clean_text((a.findtext("LastName") or "").strip())
        fore = clean_text((a.findtext("ForeName") or "").strip())
        coll = clean_text((a.findtext("CollectiveName") or "").strip())
        if coll:
            authors.append(coll)
        else:
            name = clean_text(f"{fore} {last}".strip())
            if name:
                authors.append(name)

    # Abstract: join sections
    abs_parts: List[str] = []
    for ab in art.findall(".//Article/Abstract/AbstractText"):
        label = ab.attrib.get("Label", "")
        section = clean_text("".join(ab.itertext()))
        if label:
            abs_parts.append(f"{label}: {section}")
        else:
            abs_parts.append(section)
    abstract = clean_text(" ".join([p for p in abs_parts if p]))

    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

    return {
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "pub_date": pub_date,
        "authors": authors,
        "abstract": abstract,
        "url": url,
    }


# -----------------------------
# Summarizers
# -----------------------------
def extract_bullets(text: str, n: int = 3) -> List[str]:
    """
    Extract bullet points from model output.
    Strips common 'preamble' lines and leading bullet chars.
    Falls back to sentence splitting if needed.
    """
    t = clean_text(text)

    # Remove common boilerplate/preambles (best-effort)
    # e.g., "Here are three bullet points:" or "Summary:"
    t = re.sub(
        r"^(here are|below are|these are).{0,80}?(bullet|points|summary).{0,20}?:\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"^summary\s*:\s*", "", t, flags=re.IGNORECASE)

    # Split by newlines first
    lines = [clean_text(x) for x in re.split(r"[\r\n]+", t) if clean_text(x)]

    bullets: List[str] = []
    for ln in lines:
        # Remove leading bullet markers
        ln = re.sub(r"^[-•\u2022]\s*", "", ln).strip()
        # Remove accidental numbering
        ln = re.sub(r"^\d+[\).\s]+", "", ln).strip()

        # If the model jammed everything into one line with embedded bullets, split those too
        if "•" in ln:
            parts = [clean_text(p) for p in ln.split("•") if clean_text(p)]
            for p in parts:
                bullets.append(p)
                if len(bullets) >= n:
                    break
        else:
            if ln:
                bullets.append(ln)

        if len(bullets) >= n:
            break

    # Fallback: split into sentences
    if len(bullets) < n:
        sent = re.split(r"(?<=[.!?])\s+", t)
        sent = [clean_text(s) for s in sent if clean_text(s)]
        bullets = sent[:n]

    while len(bullets) < n:
        bullets.append("Summary unavailable.")
    return bullets[:n]


def summarize_with_ollama(
    text: str,
    model: str = "llama3",
    base_url: Optional[str] = None,
) -> List[str]:
    """
    Calls Ollama local REST endpoint: POST /api/generate
    base_url can be provided or read from OLLAMA_BASE_URL env var.
    """
    base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Cleaner prompt: force bullets only, no intro
    prompt = f"""
Summarize this biomedical abstract.

Return EXACTLY three bullet points.
Each bullet must be one short sentence.
Do NOT include any introductions, headings, or extra text.

Abstract:
{text}
""".strip()

    payload = {"model": model, "prompt": prompt, "stream": False}

    req = Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    raw = urlopen(req, timeout=180).read()
    data = json.loads(raw.decode("utf-8"))
    out = clean_text(data.get("response", ""))

    return extract_bullets(out, n=3)


def summarize_with_openai(text: str, model: str = "gpt-4o-mini") -> List[str]:
    """
    Optional: OpenAI API summarization (requires OPENAI_API_KEY).
    Kept dependency-free using urllib.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/responses"

    prompt = f"""
Summarize this biomedical abstract.

Return EXACTLY three bullet points.
Each bullet must be one short sentence.
Do NOT include any introductions, headings, or extra text.

Abstract:
{text}
""".strip()

    payload = {"model": model, "input": prompt}

    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    raw = urlopen(req, timeout=180).read()
    data = json.loads(raw.decode("utf-8"))

    out = ""
    try:
        out = data["output"][0]["content"][0]["text"]
    except Exception:
        out = json.dumps(data)[:2000]

    out = clean_text(out)
    return extract_bullets(out, n=3)


# -----------------------------
# Queries (your knobs)
# -----------------------------
DEFAULT_QUERIES: List[Tuple[str, str]] = [
    (
        "AI for biology",
        '("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract]) '
        'AND (genomics OR proteomics OR "drug discovery" OR "computational biology")',
    ),
    (
        "Bioinformatics methods",
        '(bioinformatics[Title/Abstract] OR "computational biology"[Title/Abstract] OR "genome analysis"[Title/Abstract])',
    ),
    (
        "Variant calling",
        '("variant calling"[Title/Abstract] OR "variant detection"[Title/Abstract])',
    ),
]


# -----------------------------
# Pipeline
# -----------------------------
def run(
    days: int,
    max_per_query: int,
    out_json: str,
    summarizer: str,
    ollama_model: str,
    openai_model: str,
    ncbi_api_key: Optional[str],
    abstract_char_limit: int = 2000,
) -> None:
    all_papers: List[Paper] = []
    seen_pmids: set[str] = set()

    for tag, term in DEFAULT_QUERIES:
        pmids = pubmed_esearch(term=term, days=days, retmax=max_per_query, api_key=ncbi_api_key)
        time.sleep(0.35)

        details = pubmed_efetch_details(pmids, api_key=ncbi_api_key)
        time.sleep(0.35)

        for rec in details:
            pmid = rec.get("pmid", "")
            if not pmid or pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            abstract = rec.get("abstract", "") or ""
            abstract = clean_text(abstract)

            # Truncate to keep Ollama fast/cheap and avoid very long prompts
            if abstract and abstract_char_limit and len(abstract) > abstract_char_limit:
                abstract = abstract[:abstract_char_limit].rstrip() + " …"

            bullets: List[str]
            if abstract:
                try:
                    if summarizer == "ollama":
                        bullets = summarize_with_ollama(abstract, model=ollama_model)
                    elif summarizer == "openai":
                        bullets = summarize_with_openai(abstract, model=openai_model)
                    else:
                        bullets = ["(summarizer disabled)", "(summarizer disabled)", "(summarizer disabled)"]
                except Exception as e:
                    bullets = [f"Summarization failed: {e}", "—", "—"]
            else:
                bullets = ["No abstract available.", "—", "—"]

            p = Paper(
                pmid=pmid,
                title=rec.get("title", "") or "",
                journal=rec.get("journal", "") or "",
                pub_date=rec.get("pub_date", "") or "",
                authors=rec.get("authors", []) or [],
                abstract=abstract,
                url=rec.get("url", "") or "",
                query_tag=tag,
                summary_bullets=bullets,
            )
            all_papers.append(p)

    # Sort newest-first (best-effort)
    all_papers.sort(key=lambda x: parse_date_sortkey(x.pub_date), reverse=True)

    # Trending topics
    topic_counts = Counter([p.query_tag for p in all_papers]).most_common(10)

    out = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "days": days,
        "max_per_query": max_per_query,
        "queries": [{"tag": t, "term": q} for (t, q) in DEFAULT_QUERIES],
        "trending_topics": [{"topic": t, "count": c} for (t, c) in topic_counts],
        "papers": [
            {
                "pmid": p.pmid,
                "title": p.title,
                "journal": p.journal,
                "pub_date": p.pub_date,
                "authors": p.authors[:10],
                "url": p.url,
                "query_tag": p.query_tag,
                "summary_bullets": p.summary_bullets,
            }
            for p in all_papers
        ],
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(all_papers)} papers → {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--max-per-query", type=int, default=15)
    ap.add_argument("--out", type=str, default="docs/research/papers.json")
    ap.add_argument("--summarizer", choices=["ollama", "openai", "none"], default="ollama")
    ap.add_argument("--ollama-model", type=str, default=os.environ.get("OLLAMA_MODEL", "llama3"))
    ap.add_argument("--openai-model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--ncbi-api-key", type=str, default=os.environ.get("NCBI_API_KEY", ""))
    ap.add_argument("--abstract-char-limit", type=int, default=2000)
    args = ap.parse_args()

    run(
        days=args.days,
        max_per_query=args.max_per_query,
        out_json=args.out,
        summarizer=args.summarizer,
        ollama_model=args.ollama_model,
        openai_model=args.openai_model,
        ncbi_api_key=args.ncbi_api_key.strip() or None,
        abstract_char_limit=args.abstract_char_limit,
    )


if __name__ == "__main__":
    main()
