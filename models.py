"""
Shared Pydantic models and text utilities used by pipeline.py and section_scorer.py.
"""

import re
import unicodedata

from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field


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


# ── Text utilities ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """NFKC Unicode, fix hyphenated line-breaks, collapse whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\n", "", text)     # "contra-\nindication" → "contraindication"
    text = re.sub(r"\xa0", " ", text)   # non-breaking spaces
    text = re.sub(r"[\u00b7\u2022\u2023\u25cf\u25aa\u25ab\uf0b7\u2043\u204c\u204d]", " ", text)  # PDF bullet artifacts
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    try:
        return [sentence.strip() for sentence in sent_tokenize(cleaned) if sentence.strip()]
    except LookupError:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]
