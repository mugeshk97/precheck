"""
ISI Verification Pipeline

Phase 0 – Auto-discovery: pair FA with best-matching ISI using rapidfuzz
Phase 1 – Blueprint:      parse ISI into structured sections via LLM + Pydantic
Phase 2 – Async extract:  extract ISI fragments from FA pages concurrently
Phase 3 – Scoring:        section-wise Coverage / Authenticity / F1 via scorer.py
Phase 4 – Audit output:   JSON + CSV audit report with token usage
"""

import asyncio
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from docx import Document
from dotenv import load_dotenv
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI
from rapidfuzz import fuzz
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models import FAPageFragments, ISIBlueprint, normalize_text
from section_scorer import SectionScorer

load_dotenv()


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf_pages(
    azure_client: DocumentIntelligenceClient, fa_path: str
) -> dict[int, str]:
    """Return {page_number: raw_text} for every page in the FA PDF."""
    with open(fa_path, "rb") as pdf_file:
        poller = azure_client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=pdf_file.read()),
        )
    result = poller.result()
    print(f"  Pages extracted: {len(result.pages)}")

    page_texts: dict[int, str] = {}
    for page_index, page in enumerate(result.pages):
        lines = [line.content for line in (page.lines or [])]
        page_texts[page_index + 1] = "\n".join(lines)
    return page_texts


def extract_text_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_markdown(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Phase 0: Auto-discovery ────────────────────────────────────────────────────

def auto_select_isi(fa_full_text: str, isi_dir: str) -> tuple[str, float]:
    """Score every ISI .docx against the FA text; return (best_path, score)."""
    isi_files = list(Path(isi_dir).glob("*.docx"))
    if not isi_files:
        raise FileNotFoundError(f"No .docx files found in {isi_dir}")

    fa_sample = normalize_text(fa_full_text[:4000])
    best_path, best_score = "", -1.0

    for candidate_path in isi_files:
        candidate_text = extract_text_from_docx(str(candidate_path))
        candidate_sample = normalize_text(candidate_text[:4000])
        match_score = fuzz.token_set_ratio(fa_sample, candidate_sample)
        print(f"    {candidate_path.name}: {match_score:.0f}")
        if match_score > best_score:
            best_score = match_score
            best_path = str(candidate_path)

    return best_path, best_score


# ── Phase 1: ISI Blueprint ─────────────────────────────────────────────────────

async def generate_isi_blueprint(
    openai_client: AsyncOpenAI | AsyncAzureOpenAI,
    isi_text: str,
    token_log: dict,
    model: str = "gpt-4o-mini",
) -> ISIBlueprint:
    """Parse the ISI into a structured Pydantic blueprint."""

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call() -> ISIBlueprint:
        response = await openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pharmaceutical regulatory expert. "
                        "Parse the given ISI document into structured sections. "
                        "CRITICAL RULES:\n"
                        "- DO NOT summarize, paraphrase, or condense any text.\n"
                        "- Preserve the EXACT wording, casing, and punctuation of every section.\n"
                        "- Every word from the source document must appear in exactly one section's content.\n"
                        "- If unsure which section a sentence belongs to, include it in the nearest preceding section."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Parse this ISI document:\n\n{isi_text}",
                },
            ],
            response_format=ISIBlueprint,
        )
        token_log["blueprint_prompt_tokens"] = response.usage.prompt_tokens
        token_log["blueprint_completion_tokens"] = response.usage.completion_tokens
        return response.choices[0].message.parsed

    return await _call()


# ── Phase 2: Async FA page extraction ─────────────────────────────────────────

async def _extract_page_fragments(
    openai_client: AsyncOpenAI | AsyncAzureOpenAI,
    page_number: int,
    page_text: str,
    isi_sections: list[dict],
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
        response = await openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pharmaceutical compliance analyst. "
                        "You are given the verbatim ISI (Important Safety Information) sections and a page from a Final Asset (FA). "
                        "Your task: find text on the FA page that matches or closely corresponds to any ISI sentence. "
                        "Copy the text EXACTLY as it appears on the FA page — do not paraphrase or summarize. "
                        "Group each extracted fragment under the ISI section it belongs to. "
                        "If nothing on this page matches any ISI content, return empty extractions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"ISI Sections (verbatim content):\n{json.dumps(isi_sections, indent=2)}\n\n"
                        f"FA Page {page_number}:\n{page_text}"
                    ),
                },
            ],
            response_format=FAPageFragments,
        )
        token_log.setdefault("fa_prompt_tokens", 0)
        token_log.setdefault("fa_completion_tokens", 0)
        token_log["fa_prompt_tokens"] += response.usage.prompt_tokens
        token_log["fa_completion_tokens"] += response.usage.completion_tokens
        return response.choices[0].message.parsed

    return page_number, await _call()


async def extract_all_fa_pages(
    openai_client: AsyncOpenAI | AsyncAzureOpenAI,
    page_texts: dict[int, str],
    blueprint: ISIBlueprint,
    token_log: dict,
    model: str = "gpt-4o-mini",
) -> list[tuple[int, FAPageFragments]]:
    """Concurrently extract ISI fragments from all non-empty FA pages."""
    isi_sections = [
        {"title": section.title, "content": section.content}
        for section in blueprint.sections
    ]
    tasks = [
        _extract_page_fragments(openai_client, page_number, page_text, isi_sections, token_log, model)
        for page_number, page_text in page_texts.items()
        if page_text.strip()
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_extractions = []
    for page_result in all_results:
        if isinstance(page_result, Exception):
            print(f"  Warning – page extraction failed: {page_result}")
        else:
            successful_extractions.append(page_result)
    return successful_extractions


# ── Phase 4: Audit report ──────────────────────────────────────────────────────

def build_audit_report(
    fa_path: str,
    isi_path: str,
    blueprint: ISIBlueprint,
    comparison_result: dict,
    token_log: dict,
) -> dict:
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "fa": Path(fa_path).name,
        "isi": Path(isi_path).name,
        "drug": blueprint.drug_name,
        "audience": blueprint.audience,
        "match_category": comparison_result["match_category"],
        "overall": comparison_result["overall"],
        "sections": comparison_result["sections"],
        "token_usage": {**token_log, "total": sum(token_log.values())},
    }


def save_audit_report(report: dict, output_dir: str = ".") -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fa_stem = Path(report["fa"]).stem[:40]
    output_path = Path(output_dir)

    json_path = output_path / f"audit_{fa_stem}_{timestamp}.json"
    csv_path = output_path / f"audit_{fa_stem}_{timestamp}.csv"

    with open(json_path, "w") as json_file:
        json.dump(report, json_file, indent=2)

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["section", "coverage", "authenticity", "f1", "isi_sentences", "fa_fragments", "mismatches"],
        )
        writer.writeheader()
        for section in report["sections"]:
            writer.writerow({
                "section":       section["title"],
                "coverage":      section.get("coverage", 0),
                "authenticity":  section.get("authenticity", 0),
                "f1":            section.get("f1", 0),
                "isi_sentences": section.get("isi_sentence_count", 0),
                "fa_fragments":  section.get("fa_fragment_count", 0),
                "mismatches":    len(section.get("mismatches", [])),
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
    Returns (client, model_name).

    Priority:
      1. OPENAI_API_KEY present → standard OpenAI, model "gpt-4o-mini"
      2. AZURE_OPENAI_ENDPOINT present → Azure OpenAI via Managed Identity,
         deployment from AZURE_OPENAI_DEPLOYMENT (defaults to "gpt-4o-mini")
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return AsyncOpenAI(api_key=openai_api_key), "gpt-4o-mini"

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        credential = ManagedIdentityCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2025-01-01-preview",
        )
        return azure_client, deployment_name

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

    # Phase 1a: Extract FA text page-by-page via Azure Document Intelligence
    print(f"\n[Phase 1] Extracting FA: {Path(fa_path).name}")
    fa_page_texts = extract_text_from_pdf_pages(azure_client, fa_path)
    fa_full_text = clean_markdown("\n".join(fa_page_texts.values()))

    # Phase 0: Auto-select ISI if not provided
    if isi_path is None:
        print(f"\n[Phase 0] Auto-selecting ISI from {isi_dir}/")
        isi_path, match_score = auto_select_isi(fa_full_text, isi_dir)
        print(f"  → Selected: {Path(isi_path).name} (score: {match_score:.0f})")

    # Phase 1b: Extract ISI text from docx
    print(f"\n[Phase 1] Extracting ISI: {Path(isi_path).name}")
    isi_text = extract_text_from_docx(isi_path)

    # Phase 1c: Generate structured ISI Blueprint
    print("\n[Phase 1] Generating ISI Blueprint...")
    blueprint = await generate_isi_blueprint(openai_client, isi_text, token_log, model=model_name)
    print(
        f"  Drug: {blueprint.drug_name} | Audience: {blueprint.audience} | "
        f"Sections: {len(blueprint.sections)}"
    )

    # Word-count checksum: warn if LLM dropped more than 10% of the ISI content
    isi_word_count = len(isi_text.split())
    blueprint_word_count = sum(len(s.content.split()) for s in blueprint.sections)
    retention_pct = blueprint_word_count / isi_word_count * 100 if isi_word_count else 100.0
    print(f"  ISI words: {isi_word_count} | Blueprint words: {blueprint_word_count} | Retained: {retention_pct:.1f}%")
    if retention_pct < 90.0:
        print(f"  WARNING: Blueprint retained only {retention_pct:.1f}% of ISI words — LLM may have summarized content.")

    # Phase 2: Async page-by-page ISI fragment extraction
    print(f"\n[Phase 2] Extracting ISI fragments from {len(fa_page_texts)} FA pages (async)...")
    page_results = await extract_all_fa_pages(openai_client, fa_page_texts, blueprint, token_log, model=model_name)
    print(f"  Extracted from {len(page_results)} pages successfully")

    # Phase 3: Section-wise scoring
    print("\n[Phase 3] Scoring sections...")
    comparison_result = SectionScorer().compare(blueprint, page_results)
    if debug:
        for section in comparison_result["sections"]:
            print(
                f"  [{section['title'][:50]:<50}] "
                f"Cov={section['coverage']:5.1f}  "
                f"Auth={section['authenticity']:5.1f}  "
                f"F1={section['f1']:5.1f}  "
                f"Mismatches={len(section['mismatches'])}"
            )

    # Phase 4: Audit report
    print("\n[Phase 4] Saving audit report...")
    report = build_audit_report(fa_path, isi_path, blueprint, comparison_result, token_log)
    json_path, csv_path = save_audit_report(report)
    print(f"  JSON → {Path(json_path).name}")
    print(f"  CSV  → {Path(csv_path).name}")

    return report


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    fa_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "finalassets/fa-11419769-fab-fabhalta-igan-patient-understanding-your-igan-digital-pi-update-3-25.pdf"
    )
    isi_path = sys.argv[2] if len(sys.argv) > 2 else None  # None → auto-select

    report = asyncio.run(run_pipeline(fa_path=fa_path, isi_path=isi_path, isi_dir="isi", debug=True))

    overall = report["overall"]
    print(f"\n{'=' * 52}")
    print(f"FA:           {report['fa']}")
    print(f"ISI:          {report['isi']}")
    print(f"Drug:         {report['drug']} ({report['audience']})")
    print(f"Match:        {report['match_category']}")
    print(f"Coverage:     {overall['coverage']:.1f}%  (ISI → FA)")
    print(f"Authenticity: {overall['authenticity']:.1f}%  (FA → ISI)")
    print(f"F1 Score:     {overall['f1']:.1f}%")
    print(f"Tokens used:  {report['token_usage']['total']:,}")
    print(f"{'=' * 52}")
