"""
Production-Grade ISI Verification Pipeline

Phase 0 – Auto-discovery: pair FA with best-matching ISI using rapidfuzz
Phase 1 – Blueprint:      parse ISI into structured sections via gpt-4o + Pydantic
Phase 2 – Async extract:  blast FA pages to OpenAI concurrently, deduplicate fragments
Phase 3 – Scoring:        normalized, section-wise Coverage / Authenticity / F1
Phase 4 – Audit output:   JSON + CSV audit report with token cost tracking
"""

import asyncio
import csv
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from docx import Document
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ISISection(BaseModel):
    title: str = Field(description="Section heading or title")
    keywords: list[str] = Field(description="Key medical/safety terms in this section")
    content: str = Field(description="Full verbatim text of this section")


class ISIBlueprint(BaseModel):
    drug_name: str = Field(description="Drug brand name (e.g. Fabhalta, Lutathera)")
    audience: str = Field(description="Intended audience: Consumer or Professional")
    sections: list[ISISection] = Field(description="All ISI sections in document order")


class FAFragment(BaseModel):
    section_title: str = Field(description="The ISI section title this fragment belongs to")
    fragments: list[str] = Field(
        description="Sentences or phrases on this page that appear to be ISI safety content"
    )


class FAPageFragments(BaseModel):
    extractions: list[FAFragment] = Field(
        default_factory=list,
        description="ISI-like fragments found on this page, grouped by section",
    )


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf_pages(client: DocumentIntelligenceClient, file_path: str) -> dict[int, str]:
    """Return {page_number: raw_text} for every page in the PDF."""
    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=f.read()),
            output_content_format="markdown",
        )
    result = poller.result()
    print(f"  Pages extracted: {len(result.pages)}")

    pages: dict[int, str] = {}
    for i, page in enumerate(result.pages):
        lines = [line.content for line in (page.lines or [])]
        pages[i + 1] = "\n".join(lines)
    return pages


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


# ── Text cleaning & normalization ──────────────────────────────────────────────

def clean_markdown(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """NFKC Unicode, fix hyphenated line-breaks, collapse whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\n", "", text)       # "contra-\nindication" → "contraindication"
    text = re.sub(r"\xa0", " ", text)      # non-breaking spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    try:
        return [s.strip() for s in sent_tokenize(cleaned) if s.strip()]
    except LookupError:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


# ── Phase 0: Auto-discovery ────────────────────────────────────────────────────

def auto_select_isi(fa_text: str, isi_dir: str) -> tuple[str, float]:
    """Score every ISI .docx against the FA text; return (best_path, score)."""
    isi_files = list(Path(isi_dir).glob("*.docx"))
    if not isi_files:
        raise FileNotFoundError(f"No .docx files found in {isi_dir}")

    fa_sample = normalize_text(fa_text[:4000])
    best_path, best_score = "", -1.0

    for isi_path in isi_files:
        isi_text = extract_text_from_docx(str(isi_path))
        isi_sample = normalize_text(isi_text[:4000])
        score = fuzz.token_set_ratio(fa_sample, isi_sample)
        print(f"    {isi_path.name}: {score:.0f}")
        if score > best_score:
            best_score = score
            best_path = str(isi_path)

    return best_path, best_score


# ── Phase 1: ISI Blueprint via gpt-4o ─────────────────────────────────────────

async def generate_isi_blueprint(
    client: AsyncOpenAI | AsyncAzureOpenAI, isi_text: str, token_log: dict, model: str = "gpt-4o-mini"
) -> ISIBlueprint:
    """Parse the ISI into a structured Pydantic blueprint using gpt-4o."""

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call() -> ISIBlueprint:
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pharmaceutical regulatory expert. "
                        "Parse the given ISI document into structured sections, "
                        "preserving the exact verbatim content of each section."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Parse this ISI document:\n\n{isi_text}",
                },
            ],
            response_format=ISIBlueprint,
        )
        usage = response.usage
        token_log["blueprint_prompt_tokens"] = usage.prompt_tokens
        token_log["blueprint_completion_tokens"] = usage.completion_tokens
        return response.choices[0].message.parsed

    return await _call()


# ── Phase 2: Async FA page extraction ─────────────────────────────────────────

def _content_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


async def _extract_page_fragments(
    client: AsyncOpenAI | AsyncAzureOpenAI,
    page_num: int,
    page_text: str,
    section_titles: list[str],
    token_log: dict,
    model: str = "gpt-4o-mini",
) -> tuple[int, FAPageFragments]:
    """Extract ISI-like fragments from a single FA page (with retries)."""

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call() -> FAPageFragments:
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pharmaceutical compliance analyst. "
                        "Given a page from a Final Asset (FA) marketing material and a list of ISI section titles, "
                        "extract any text that appears to be ISI safety information. "
                        "Group fragments by the closest matching ISI section. "
                        "If nothing on this page looks like ISI content, return empty extractions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"ISI Sections: {json.dumps(section_titles)}\n\n"
                        f"FA Page {page_num}:\n{page_text}"
                    ),
                },
            ],
            response_format=FAPageFragments,
        )
        usage = response.usage
        token_log.setdefault("fa_prompt_tokens", 0)
        token_log.setdefault("fa_completion_tokens", 0)
        token_log["fa_prompt_tokens"] += usage.prompt_tokens
        token_log["fa_completion_tokens"] += usage.completion_tokens
        return response.choices[0].message.parsed

    return page_num, await _call()


async def extract_all_fa_pages(
    client: AsyncOpenAI | AsyncAzureOpenAI,
    pages: dict[int, str],
    blueprint: ISIBlueprint,
    token_log: dict,
    model: str = "gpt-4o-mini",
) -> list[tuple[int, FAPageFragments]]:
    """Concurrently process all non-empty FA pages."""
    section_titles = [s.title for s in blueprint.sections]
    tasks = [
        _extract_page_fragments(client, pnum, ptext, section_titles, token_log, model)
        for pnum, ptext in pages.items()
        if ptext.strip()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    good = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  Warning – page extraction failed: {r}")
        else:
            good.append(r)
    return good


def _resolve_section_title(raw_title: str, canonical_titles: list[str]) -> str:
    """Fuzzy-match a LLM-returned section title to the nearest blueprint title."""
    from rapidfuzz import process as rfprocess
    match = rfprocess.extractOne(raw_title, canonical_titles, scorer=fuzz.token_set_ratio)
    return match[0] if match else raw_title


def deduplicate_fragments(
    page_results: list[tuple[int, FAPageFragments]],
    canonical_titles: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Merge fragments across pages by section.
    - Fuzzy-resolves LLM section titles to canonical blueprint titles (prevents key mismatches).
    - Strips exact duplicates (hash) and near-duplicates (>90 fuzzy similarity).
    """
    section_map: dict[str, list[str]] = {}
    seen_hashes: set[str] = set()

    for _pnum, page_frags in page_results:
        for group in page_frags.extractions:
            section = (
                _resolve_section_title(group.section_title, canonical_titles)
                if canonical_titles
                else group.section_title
            )
            section_map.setdefault(section, [])
            for frag in group.fragments:
                h = _content_hash(frag)
                if h in seen_hashes:
                    continue
                norm_frag = normalize_text(frag)
                existing = section_map[section]
                if existing and max(
                    fuzz.token_set_ratio(norm_frag, normalize_text(e)) for e in existing
                ) > 90:
                    continue
                seen_hashes.add(h)
                section_map[section].append(frag)

    return section_map


# ── Phase 3: Section-wise scoring ─────────────────────────────────────────────

def _best_sentence_scores(source: list[str], target: list[str]) -> list[float]:
    scores = []
    for src in source:
        if len(src.split()) < 4:
            continue
        score = max(fuzz.token_set_ratio(src, tgt) for tgt in target)
        scores.append(score)
    return scores


def score_section(
    isi_section: ISISection,
    fa_fragments: list[str],
    threshold: float = 75.0,
) -> dict:
    isi_sents = [normalize_text(s) for s in _split_sentences(isi_section.content)]
    fa_sents = [normalize_text(s) for s in fa_fragments if s.strip()]

    if not isi_sents or not fa_sents:
        return {
            "coverage": 0.0,
            "authenticity": 0.0,
            "f1": 0.0,
            "isi_sentence_count": len(isi_sents),
            "fa_fragment_count": len(fa_sents),
        }

    # Coverage: ISI → FA (penalise weak matches)
    cov_scores = _best_sentence_scores(isi_sents, fa_sents)
    calibrated = [s if s >= threshold else s * 0.6 for s in cov_scores]
    coverage = round(mean(calibrated), 2) if calibrated else 0.0

    # Authenticity: FA → ISI (only count confident matches)
    auth_scores = _best_sentence_scores(fa_sents, isi_sents)
    valid_auth = [s for s in auth_scores if s >= threshold]
    authenticity = round(mean(valid_auth), 2) if valid_auth else 0.0

    f1 = (
        round(2 * coverage * authenticity / (coverage + authenticity), 2)
        if (coverage + authenticity)
        else 0.0
    )
    return {
        "coverage": coverage,
        "authenticity": authenticity,
        "f1": f1,
        "isi_sentence_count": len(isi_sents),
        "fa_fragment_count": len(fa_sents),
    }


# ── Phase 4: Audit report ──────────────────────────────────────────────────────

def build_audit_report(
    fa_path: str,
    isi_path: str,
    blueprint: ISIBlueprint,
    section_scores: dict[str, dict],
    fa_section_map: dict[str, list[str]],
    token_log: dict,
) -> dict:
    cov_vals = [v["coverage"] for v in section_scores.values()]
    auth_vals = [v["authenticity"] for v in section_scores.values()]
    overall_coverage = round(mean(cov_vals), 2) if cov_vals else 0.0
    overall_auth = round(mean(auth_vals), 2) if auth_vals else 0.0
    overall_f1 = (
        round(2 * overall_coverage * overall_auth / (overall_coverage + overall_auth), 2)
        if (overall_coverage + overall_auth)
        else 0.0
    )

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "fa": Path(fa_path).name,
        "isi": Path(isi_path).name,
        "drug": blueprint.drug_name,
        "audience": blueprint.audience,
        "overall": {
            "coverage": overall_coverage,
            "authenticity": overall_auth,
            "f1": overall_f1,
        },
        "sections": [
            {
                "title": sec.title,
                "scores": section_scores.get(
                    sec.title, {"coverage": 0.0, "authenticity": 0.0, "f1": 0.0}
                ),
                "fa_fragments": fa_section_map.get(sec.title, []),
            }
            for sec in blueprint.sections
        ],
        "token_usage": {**token_log, "total": sum(token_log.values())},
    }


def save_audit_report(report: dict, output_dir: str = ".") -> tuple[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = Path(report["fa"]).stem[:40]
    out = Path(output_dir)

    json_path = out / f"audit_{stem}_{ts}.json"
    csv_path = out / f"audit_{stem}_{ts}.csv"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "coverage", "authenticity", "f1", "isi_sentences", "fa_fragments"],
        )
        writer.writeheader()
        for sec in report["sections"]:
            s = sec["scores"]
            writer.writerow({
                "section": sec["title"],
                "coverage": s.get("coverage", 0),
                "authenticity": s.get("authenticity", 0),
                "f1": s.get("f1", 0),
                "isi_sentences": s.get("isi_sentence_count", 0),
                "fa_fragments": s.get("fa_fragment_count", 0),
            })

    return str(json_path), str(csv_path)


# ── Client helpers ─────────────────────────────────────────────────────────────

def _get_azure_client() -> DocumentIntelligenceClient:
    endpoint = os.getenv("ENDPOINT")
    key = os.getenv("KEY")
    if not endpoint or not key:
        raise ValueError("Missing Azure credentials. Set ENDPOINT and KEY in .env")
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def _get_openai_client() -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
    """
    Returns (client, model_or_deployment_name).

    Priority:
      1. OPENAI_API_KEY present → standard OpenAI, model "gpt-4o-mini"
      2. AZURE_OPENAI_ENDPOINT present → Azure OpenAI via Managed Identity,
         deployment from AZURE_OPENAI_DEPLOYMENT (defaults to "gpt-4o-mini")
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return AsyncOpenAI(api_key=api_key), "gpt-4o-mini"

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        credential = ManagedIdentityCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2025-01-01-preview",
        )
        return client, deployment

    raise ValueError(
        "No OpenAI credentials found. "
        "Set OPENAI_API_KEY for OpenAI, or AZURE_OPENAI_ENDPOINT for Azure OpenAI with Managed Identity."
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(
    fa_path: str,
    isi_path: Optional[str] = None,
    isi_dir: str = "isi",
    debug: bool = False,
) -> dict:
    azure_client = _get_azure_client()
    openai_client, model_name = _get_openai_client()
    token_log: dict = {}

    # Phase 1a: Extract FA (page-by-page via Azure DI)
    print(f"\n[Phase 1] Extracting FA: {Path(fa_path).name}")
    fa_pages = extract_text_from_pdf_pages(azure_client, fa_path)
    fa_full_text = clean_markdown("\n".join(fa_pages.values()))

    # Phase 0: Auto-select ISI if not provided
    if isi_path is None:
        print(f"\n[Phase 0] Auto-selecting ISI from {isi_dir}/")
        isi_path, match_score = auto_select_isi(fa_full_text, isi_dir)
        print(f"  → Selected: {Path(isi_path).name} (score: {match_score:.0f})")

    # Phase 1b: Extract ISI docx
    print(f"\n[Phase 1] Extracting ISI: {Path(isi_path).name}")
    isi_text = extract_text_from_docx(isi_path)

    # Phase 1c: Generate structured ISI Blueprint
    print("\n[Phase 1] Generating ISI Blueprint via gpt-4o...")
    blueprint = await generate_isi_blueprint(openai_client, isi_text, token_log, model=model_name)
    print(
        f"  Drug: {blueprint.drug_name} | Audience: {blueprint.audience} | "
        f"Sections: {len(blueprint.sections)}"
    )

    # Phase 2: Async page-by-page extraction + deduplication
    print(f"\n[Phase 2] Extracting ISI fragments from {len(fa_pages)} FA pages (async)...")
    page_results = await extract_all_fa_pages(openai_client, fa_pages, blueprint, token_log, model=model_name)
    canonical_titles = [s.title for s in blueprint.sections]
    fa_section_map = deduplicate_fragments(page_results, canonical_titles=canonical_titles)
    total_frags = sum(len(v) for v in fa_section_map.values())
    print(f"  {total_frags} unique fragments across {len(fa_section_map)} sections")

    # Phase 3: Section-wise scoring
    print("\n[Phase 3] Scoring sections...")
    section_scores: dict[str, dict] = {}
    for sec in blueprint.sections:
        frags = fa_section_map.get(sec.title, [])
        scores = score_section(sec, frags)
        section_scores[sec.title] = scores
        if debug:
            print(
                f"  [{sec.title[:50]:<50}] "
                f"Cov={scores['coverage']:5.1f}  "
                f"Auth={scores['authenticity']:5.1f}  "
                f"F1={scores['f1']:5.1f}"
            )

    # Phase 4: Audit report
    print("\n[Phase 4] Saving audit report...")
    report = build_audit_report(
        fa_path, isi_path, blueprint, section_scores, fa_section_map, token_log
    )
    json_path, csv_path = save_audit_report(report)
    print(f"  JSON → {Path(json_path).name}")
    print(f"  CSV  → {Path(csv_path).name}")

    return report


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    fa = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "finalassets/fa-11419769-fab-fabhalta-igan-patient-understanding-your-igan-digital-pi-update-3-25.pdf"
    )
    isi = sys.argv[2] if len(sys.argv) > 2 else None  # None → auto-select

    report = asyncio.run(run_pipeline(fa_path=fa, isi_path=isi, isi_dir="isi", debug=True))

    overall = report["overall"]
    print(f"\n{'=' * 52}")
    print(f"FA:           {report['fa']}")
    print(f"ISI:          {report['isi']}")
    print(f"Drug:         {report['drug']} ({report['audience']})")
    print(f"Coverage:     {overall['coverage']:.1f}%  (ISI → FA)")
    print(f"Authenticity: {overall['authenticity']:.1f}%  (FA → ISI)")
    print(f"F1 Score:     {overall['f1']:.1f}%")
    print(f"Tokens used:  {report['token_usage']['total']:,}")
    print(f"{'=' * 52}")
