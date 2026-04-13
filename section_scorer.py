"""
Section-wise ISI-FA scorer.

Coverage     (ISI → full FA text)       : is the ISI content present in the FA?
Authenticity (FA extracted ISI → ISI)   : does the extracted content trace back to the ISI?

Both use rapidfuzz.fuzz.partial_ratio — best-window match, clean + sentence-tokenized.
"""

import hashlib
import logging
from dataclasses import asdict, dataclass
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

from models import FAPageFragments, ISIBlueprint, ISISection, _split_sentences, normalize_text

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 75.0
MATCH_THRESHOLD = 75.0
MIN_WORDS = 4


def _prepare(text: str) -> str:
    """Clean and sentence-tokenize text into a single comparable string."""
    return " ".join(
        normalize_text(s) for s in _split_sentences(text) if s.strip()
    )


def _fragment_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def _resolve_section_title(raw_title: str, canonical_titles: list[str]) -> str:
    match = rfprocess.extractOne(raw_title, canonical_titles, scorer=fuzz.token_set_ratio)
    return match[0] if match else raw_title


@dataclass
class SentenceMismatch:
    isi_sentence: str
    closest_fa_text: str
    closest_fa_page: int | None
    score: float
    diff: str


@dataclass
class SectionResult:
    title: str
    coverage: float
    authenticity: float
    f1: float
    isi_sentence_count: int
    fa_fragment_count: int
    mismatches: list[dict]
    fa_fragments: list[dict]


class SectionScorer:

    def __init__(
        self,
        score_threshold: float = SCORE_THRESHOLD,
        match_threshold: float = MATCH_THRESHOLD,
    ):
        self.score_threshold = score_threshold
        self.match_threshold = match_threshold

    def _build_section_map(
        self,
        page_results: list[tuple[int, FAPageFragments]],
        canonical_titles: list[str],
    ) -> dict[str, list[tuple[str, int]]]:
        section_map: dict[str, list[tuple[str, int]]] = {}
        seen_hashes: set[str] = set()

        for page_number, page_frags in page_results:
            for group in page_frags.extractions:
                title = _resolve_section_title(group.section_title, canonical_titles)
                section_map.setdefault(title, [])
                for fragment in group.fragments:
                    fhash = _fragment_hash(fragment)
                    if fhash in seen_hashes:
                        continue
                    norm = normalize_text(fragment)
                    existing = [normalize_text(t) for t, _ in section_map[title]]
                    if existing and max(fuzz.token_set_ratio(norm, e) for e in existing) > 90:
                        continue
                    seen_hashes.add(fhash)
                    section_map[title].append((fragment, page_number))

        return section_map

    def _score_section(
        self,
        isi_section: ISISection,
        fa_entries: list[tuple[str, int]],
        fa_raw_text: str,
    ) -> SectionResult:
        isi_sentences    = [normalize_text(s) for s in _split_sentences(isi_section.content)]
        fa_frags         = [(normalize_text(t), p) for t, p in fa_entries if t.strip()]
        fa_texts         = [t for t, _ in fa_frags]
        page_by_fragment = {t: p for t, p in fa_frags}

        if not isi_sentences:
            return SectionResult(
                title=isi_section.title,
                coverage=0.0, authenticity=0.0, f1=0.0,
                isi_sentence_count=0, fa_fragment_count=len(fa_texts),
                mismatches=[],
                fa_fragments=[{"text": t, "page": p} for t, p in fa_entries],
            )

        isi_full = " ".join(isi_sentences)
        fa_extracted = " ".join(fa_texts)

        coverage     = float(fuzz.partial_ratio(isi_full, fa_raw_text))
        authenticity = float(fuzz.partial_ratio(fa_extracted, isi_full)) if fa_extracted.strip() else 0.0
        f1           = round(2 * coverage * authenticity / (coverage + authenticity), 2) if (coverage + authenticity) else 0.0

        mismatches = []
        for sent in isi_sentences:
            if len(sent.split()) < MIN_WORDS:
                continue
            score = fuzz.partial_ratio(sent, fa_raw_text)
            if score < self.score_threshold:
                closest = max(fa_texts, key=lambda f: fuzz.partial_ratio(sent, f)) if fa_texts else ""
                mismatches.append(asdict(SentenceMismatch(
                    isi_sentence=sent,
                    closest_fa_text=closest,
                    closest_fa_page=page_by_fragment.get(closest),
                    score=round(score, 1),
                    diff=f"- {sent}\n+ {closest}",
                )))

        return SectionResult(
            title=isi_section.title,
            coverage=coverage, authenticity=authenticity, f1=f1,
            isi_sentence_count=len(isi_sentences), fa_fragment_count=len(fa_texts),
            mismatches=mismatches,
            fa_fragments=[{"text": t, "page": p} for t, p in fa_entries],
        )

    def compare(
        self,
        blueprint: ISIBlueprint,
        page_results: list[tuple[int, FAPageFragments]],
        fa_page_texts: dict[int, str],
    ) -> dict[str, Any]:
        fa_raw_text = _prepare(" ".join(fa_page_texts.values()))
        canonical_titles = [s.title for s in blueprint.sections]
        section_map = self._build_section_map(page_results, canonical_titles)
        results = [
            self._score_section(s, section_map.get(s.title, []), fa_raw_text)
            for s in blueprint.sections
        ]

        # Overall coverage: sentence-wise ISI → FA matching.
        # Each ISI sentence is independently scored against the full FA text.
        # This avoids dependence on how the LLM splits sections in the blueprint.
        all_isi_sentences = [
            normalize_text(s)
            for sec in blueprint.sections
            for s in _split_sentences(sec.content)
            if len(normalize_text(s).split()) >= MIN_WORDS
        ]
        if all_isi_sentences:
            sent_scores = [float(fuzz.partial_ratio(s, fa_raw_text)) for s in all_isi_sentences]
            overall_cov = round(sum(sent_scores) / len(sent_scores), 2)
        else:
            overall_cov = 0.0

        # Overall authenticity: still section-averaged over active sections
        active = [r for r in results if r.fa_fragment_count > 0]
        auth   = [r.authenticity for r in active]
        overall_auth = round(sum(auth) / len(auth), 2) if auth else 0.0
        overall_f1   = round(2 * overall_cov * overall_auth / (overall_cov + overall_auth), 2) if (overall_cov + overall_auth) else 0.0

        return {
            "match_category": "Closest Match" if overall_cov >= self.match_threshold else "Not Matched",
            "overall": {"coverage": overall_cov, "authenticity": overall_auth, "f1": overall_f1},
            "sections": [asdict(r) for r in results],
        }
