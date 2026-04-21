#!/usr/bin/env python3
"""
Research Radar v2
- Pulls recent PubMed papers for one or more queries
- Summarizes abstracts (default: Ollama; optional: OpenAI)
- Adds a structured study-appraisal layer
- Writes papers.json for a static frontend

Usage:
  python build_radar_v2.py --days 7 --max-per-query 15 --summarizer ollama --ollama-model llama3
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
    study_appraisal: Dict[str, Any]


def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> bytes:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def normalize_month(m: str) -> str:
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
    return lookup.get(m[:3].lower(), "")


def parse_date_sortkey(pub_date: str) -> Tuple[int, int, int]:
    s = clean_text(pub_date)
    if not s:
        return (0, 0, 0)

    m = re.search(r"(19\d{2}|20\d{2})", s)
    year = int(m.group(1)) if m else 0

    m2 = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m2:
        return (int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))

    m3 = re.match(r"^(\d{4})-(\d{2})$", s)
    if m3:
        return (int(m3.group(1)), int(m3.group(2)), 0)

    m4 = re.match(r"^(\d{4})$", s)
    if m4:
        return (int(m4.group(1)), 0, 0)

    return (year, 0, 0)


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
            pub_date = medline

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


def strip_meta_bullets(bullets: List[str], n: int = 3) -> List[str]:
    bad_prefixes = (
        "here are",
        "these are",
        "below are",
        "summary:",
        "the following",
    )
    cleaned: List[str] = []
    for b in bullets:
        s = clean_text(b)
        if not s:
            continue
        lower = s.lower()
        if any(lower.startswith(prefix) for prefix in bad_prefixes):
            continue
        cleaned.append(s)

    while len(cleaned) < n:
        cleaned.append("Summary unavailable.")
    return cleaned[:n]


def extract_bullets(text: str, n: int = 3) -> List[str]:
    t = clean_text(text)
    t = re.sub(
        r"^(here are|below are|these are).{0,100}?(bullet|points|summary).{0,30}?:\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"^summary\s*:\s*", "", t, flags=re.IGNORECASE)

    lines = [clean_text(x) for x in re.split(r"[\r\n]+", t) if clean_text(x)]
    bullets: List[str] = []

    for ln in lines:
        ln = re.sub(r"^[-•\u2022]\s*", "", ln).strip()
        ln = re.sub(r"^\d+[\).\s]+", "", ln).strip()
        if "•" in ln:
            parts = [clean_text(p) for p in ln.split("•") if clean_text(p)]
            bullets.extend(parts)
        else:
            bullets.append(ln)
        if len(bullets) >= n:
            break

    if len(bullets) < n:
        sent = re.split(r"(?<=[.!?])\s+", t)
        sent = [clean_text(s) for s in sent if clean_text(s)]
        bullets = sent[:n]

    return strip_meta_bullets(bullets, n=n)


def summarize_with_ollama(
    text: str,
    model: str = "llama3",
    base_url: Optional[str] = None,
) -> List[str]:
    base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    prompt = f"""
Summarize this biomedical abstract.

Return EXACTLY three bullet points.
Each bullet must be one short sentence.
Do NOT include any introductions, headings, labels, or extra text.

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
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/responses"
    prompt = f"""
Summarize this biomedical abstract.

Return EXACTLY three bullet points.
Each bullet must be one short sentence.
Do NOT include any introductions, headings, labels, or extra text.

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

    return extract_bullets(clean_text(out), n=3)


def infer_study_appraisal(title: str, abstract: str, journal: str = "") -> Dict[str, Any]:
    text = clean_text(f"{title}. {abstract}")
    lower = text.lower()

    def contains(*terms: str) -> bool:
        return any(term in lower for term in terms)

    def confidence(explicit_hits: int) -> str:
        if explicit_hits >= 5:
            return "high"
        if explicit_hits >= 2:
            return "medium"
        return "low"

    explicit_hits = 0

    if contains("randomized", "randomised", "randomly assigned", "placebo-controlled", "placebo controlled"):
        primary_category = "randomized_experiment"
        is_randomized = True
        explicit_hits += 2
    elif contains("quasi-experimental", "quasi experimental", "nonrandomized", "non-randomized"):
        primary_category = "quasi_experimental"
        is_randomized = False
        explicit_hits += 2
    elif contains("cohort", "prospective", "retrospective"):
        primary_category = "cohort"
        is_randomized = False
        explicit_hits += 1
    elif contains("case-control", "case control"):
        primary_category = "case_control"
        is_randomized = False
        explicit_hits += 1
    elif contains("time series", "time-series", "timeseries"):
        primary_category = "time_series"
        is_randomized = False
        explicit_hits += 1
    elif contains("in vitro", "cell line", "cell culture", "organoid", "mice", "mouse", "murine", "rat", "zebrafish", "docking", "molecular dynamics"):
        primary_category = "lab_experiment"
        is_randomized = False
        explicit_hits += 1
    elif contains("pipeline", "workflow", "framework", "algorithm", "benchmark", "tool ", "software", "snakemake", "nextflow"):
        primary_category = "computational_method"
        is_randomized = False
        explicit_hits += 1
    elif contains("review", "commentary", "perspective", "opinion"):
        primary_category = "review_or_commentary"
        is_randomized = False
        explicit_hits += 1
    else:
        primary_category = "observational_or_unclear"
        is_randomized = False

    is_longitudinal = contains("longitudinal", "follow-up", "follow up", "over time", "serial")
    is_time_series = contains("time series", "time-series", "timeseries", "multiple time points", "repeated measures")
    is_lab_study = contains("in vitro", "cell line", "cell culture", "mouse", "mice", "murine", "rat", "organoid", "laboratory", "docking")
    is_quasi = primary_category == "quasi_experimental"
    is_mixed = contains("mixed methods", "combining", "integrating") and (is_lab_study or primary_category in {"computational_method", "cohort", "case_control"})

    if is_longitudinal:
        explicit_hits += 1
    if is_time_series:
        explicit_hits += 1
    if is_lab_study:
        explicit_hits += 1

    if contains("healthy controls", "control group", "controls", "compared with", "versus", "vs."):
        control_group = "Control or comparison group mentioned in abstract/title."
        explicit_hits += 1
    else:
        control_group = "Unknown from abstract."

    if contains("patients", "subjects", "participants", "volunteers", "cohort", "adults", "children"):
        population = "Human participants or clinical subjects."
        contains_humans = True
        sample_type = "selected_sample"
        explicit_hits += 1
    elif contains("mouse", "mice", "murine", "rat", "animal model"):
        population = "Animal or preclinical sample."
        contains_humans = False
        sample_type = "lab_material_only"
        explicit_hits += 1
    elif contains("cell line", "cell culture", "organoid"):
        population = "Cell line or laboratory material."
        contains_humans = False
        sample_type = "lab_material_only"
        explicit_hits += 1
    elif contains("dataset", "public data", "sequencing data", "tcga", "geo"):
        population = "Public dataset or previously collected data."
        contains_humans = "patient" in lower or "human" in lower
        sample_type = "public_dataset_only"
        explicit_hits += 1
    else:
        population = "Unknown from abstract."
        contains_humans = False
        sample_type = "unknown"

    measure_types: List[str] = []
    if contains("questionnaire", "survey", "self-report", "self report"):
        measure_types.append("self_report")
        explicit_hits += 1
    if contains("observed", "observation", "video-coded", "behavior"):
        measure_types.append("observation")
    if contains("gene", "genome", "genomic", "rna-seq", "scrna", "variant", "sequence", "proteomic", "multi-omics", "omics", "biomarker"):
        measure_types.append("biological")
        explicit_hits += 1
    if contains("sequence", "sequencing", "variant", "genome", "genomic"):
        measure_types.append("genetic_sequence")
    if contains("clinical", "electronic health record", "ehr", "hospital", "diagnosis"):
        measure_types.append("clinical_record")
    if contains("model", "classifier", "predict", "prediction", "auc", "accuracy", "framework", "pipeline", "algorithm"):
        measure_types.append("computational_score")

    if not measure_types:
        measure_types.append("unknown")

    if contains("multiple time points", "repeated measures", "longitudinal", "serial", "follow-up", "follow up"):
        measurement_frequency = "repeated"
    elif contains("baseline", "single time point", "cross-sectional", "cross sectional"):
        measurement_frequency = "single_time_point"
    else:
        measurement_frequency = "unknown_from_abstract"

    null_results: List[str] = []
    if contains("no significant", "not significant", "did not differ", "no difference", "failed to show", "was not associated"):
        explicit_hits += 1
        null_results.append("At least one null or non-significant result is explicitly mentioned.")
        null_quality = "clear"
    else:
        null_quality = "unclear"

    main_findings: List[str] = []
    if abstract:
        sents = re.split(r"(?<=[.!?])\s+", abstract)
        for sent in sents:
            sent = clean_text(sent)
            if not sent:
                continue
            if re.search(r"\b(found|identified|showed|demonstrated|revealed|improved|associated)\b", sent, re.I):
                main_findings.append(sent)
            if len(main_findings) >= 2:
                break

    if not main_findings:
        main_findings = ["See summary bullets."]

    limitations: List[str] = []
    if contains("single-center", "single center"):
        limitations.append("Single-center design may limit generalizability.")
    if contains("small sample", "small cohort", "pilot study"):
        limitations.append("Small sample may limit precision and robustness.")
    if primary_category in {"computational_method", "lab_experiment"}:
        limitations.append("May not generalize directly to clinical practice without downstream validation.")
    if not null_results:
        limitations.append("Null findings are not clearly reported in the abstract.")
    if sample_type in {"selected_sample", "public_dataset_only"}:
        limitations.append("Sample may not represent the general population.")

    internal_validity_notes = []
    external_validity_notes = []

    if is_randomized:
        internal_validity_notes.append("Randomization, if correctly implemented, supports stronger causal inference.")
    elif primary_category in {"cohort", "case_control", "observational_or_unclear"}:
        internal_validity_notes.append("Observational design leaves room for confounding and selection effects.")
    elif primary_category == "computational_method":
        internal_validity_notes.append("Internal validity depends heavily on data quality, labeling, and benchmark setup.")
    else:
        internal_validity_notes.append("Internal validity cannot be fully judged from abstract alone.")

    if sample_type == "selected_sample":
        external_validity_notes.append("Selected sample may limit generalizability beyond the studied group.")
    elif sample_type == "lab_material_only":
        external_validity_notes.append("Laboratory materials or animal systems may not generalize directly to humans.")
    elif sample_type == "public_dataset_only":
        external_validity_notes.append("Generalizability depends on how representative the source dataset is.")
    else:
        external_validity_notes.append("External validity is unclear from abstract alone.")

    control_notes = "Unknown from abstract."
    if contains("matched controls", "age-matched", "healthy controls"):
        control_notes = "Control group appears selected for comparison, but matching details may still be incomplete."
        explicit_hits += 1

    return {
        "study_design": {
            "primary_category": primary_category,
            "is_longitudinal": is_longitudinal,
            "is_time_series": is_time_series,
            "is_lab_study": is_lab_study,
            "is_randomized_experiment": is_randomized,
            "is_quasi_experimental": is_quasi,
            "is_mixed_design": is_mixed,
            "design_notes": "Heuristic appraisal generated from title and abstract; verify against full text for high-stakes use.",
        },
        "subjects": {
            "population": population,
            "sample_type": sample_type,
            "selection_criteria": "Unknown from abstract.",
            "control_group_description": control_group,
            "control_selection_notes": control_notes,
            "group_differences_other_than_variable_of_interest": [],
        },
        "measures": {
            "measure_types": sorted(set(measure_types)),
            "measurement_frequency": measurement_frequency,
            "measurement_notes": "Measure classification inferred from abstract wording.",
        },
        "results": {
            "main_findings": main_findings[:2],
            "null_results": null_results,
            "null_result_reporting_quality": null_quality,
            "results_notes": "Null findings are often underreported in abstracts; full text review is better.",
        },
        "validity": {
            "internal_validity_notes": " ".join(internal_validity_notes),
            "external_validity_notes": " ".join(external_validity_notes),
        },
        "limitations": {
            "limitations_summary": " ".join(limitations) if limitations else "No major limitations extracted from the abstract.",
            "items": limitations,
        },
        "evidence_flags": {
            "contains_human_subjects": contains_humans,
            "contains_control_group": control_group != "Unknown from abstract.",
            "contains_self_report": "self_report" in measure_types,
            "contains_biological_measure": any(m in measure_types for m in ("biological", "genetic_sequence")),
        },
        "confidence": {
            "extraction_confidence": confidence(explicit_hits),
            "reason": "Rule-based abstract parsing; stronger when the abstract explicitly names design, sample, and results features.",
        },
    }


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

            abstract = clean_text(rec.get("abstract", "") or "")
            if abstract and abstract_char_limit and len(abstract) > abstract_char_limit:
                abstract = abstract[:abstract_char_limit].rstrip() + " …"

            if abstract:
                try:
                    if summarizer == "ollama":
                        bullets = summarize_with_ollama(abstract, model=ollama_model)
                    elif summarizer == "openai":
                        bullets = summarize_with_openai(abstract, model=openai_model)
                    else:
                        bullets = ["Summarizer disabled.", "No abstract summary generated.", "Use study appraisal below."]
                except Exception as e:
                    bullets = [f"Summarization failed: {e}", "Review abstract manually.", "Study appraisal may still be useful."]
            else:
                bullets = ["No abstract available.", "Manual review likely needed.", "Study appraisal confidence will be limited."]

            bullets = strip_meta_bullets(bullets, n=3)
            appraisal = infer_study_appraisal(
                title=rec.get("title", "") or "",
                abstract=abstract,
                journal=rec.get("journal", "") or "",
            )

            all_papers.append(
                Paper(
                    pmid=pmid,
                    title=rec.get("title", "") or "",
                    journal=rec.get("journal", "") or "",
                    pub_date=rec.get("pub_date", "") or "",
                    authors=rec.get("authors", []) or [],
                    abstract=abstract,
                    url=rec.get("url", "") or "",
                    query_tag=tag,
                    summary_bullets=bullets,
                    study_appraisal=appraisal,
                )
            )

    all_papers.sort(key=lambda x: parse_date_sortkey(x.pub_date), reverse=True)
    topic_counts = Counter([p.query_tag for p in all_papers]).most_common(10)
    journal_counts = Counter([p.journal for p in all_papers if p.journal]).most_common(10)

    out = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "days": days,
        "max_per_query": max_per_query,
        "queries": [{"tag": t, "term": q} for (t, q) in DEFAULT_QUERIES],
        "paper_count": len(all_papers),
        "trending_topics": [{"topic": t, "count": c} for (t, c) in topic_counts],
        "top_journals": [{"journal": j, "count": c} for (j, c) in journal_counts],
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
                "study_appraisal": p.study_appraisal,
            }
            for p in all_papers
        ],
    }

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(all_papers)} papers → {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--max-per-query", type=int, default=15)
    ap.add_argument("--out", type=str, default="papers.json")
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
