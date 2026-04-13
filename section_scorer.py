"""
Section-wise ISI-FA scorer.

Two-way comparison per blueprint section:
  Coverage     (ISI → FA)  : is every ISI sentence present in the FA?
  Authenticity (FA → ISI)  : does every FA fragment trace back to the ISI?

Both use difflib.SequenceMatcher (order + spelling sensitive).
Low-scoring sentences include a word-level diff for human review.
rapidfuzz is used only for section title resolution and deduplication.
"""

import difflib
import hashlib
import logging
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

from models import FAPageFragments, ISIBlueprint, ISISection, _split_sentences, normalize_text

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 75.0          # sentences below this appear in the mismatch report
PARTIAL_MATCH_THRESHOLD = 75.0  # overall coverage required for "Closest Match"
PENALTY = 0.6                   # applied to coverage scores below threshold
MIN_WORDS = 4                   # skip very short sentences (headers / labels)


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _similarity_score(text_a: str, text_b: str) -> float:
    """difflib sequence match ratio scaled to 0–100."""
    return difflib.SequenceMatcher(None, text_a, text_b).ratio() * 100


def _word_level_diff(isi_sentence: str, fa_sentence: str) -> str:
    """
    Word-level diff between an ISI sentence and the closest FA match.
      '- word'  in ISI but missing/changed in FA
      '+ word'  in FA but not in ISI
      '  word'  unchanged
    """
    return " ".join(
        token for token in difflib.ndiff(isi_sentence.split(), fa_sentence.split())
        if not token.startswith("?")
    )


def _fragment_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def _resolve_section_title(raw_title: str, canonical_titles: list[str]) -> str:
    """Fuzzy-match a LLM-returned section title to the nearest blueprint title."""
    match = rfprocess.extractOne(raw_title, canonical_titles, scorer=fuzz.token_set_ratio)
    return match[0] if match else raw_title


# ── Data classes ───────────────────────────────────────────────────────────────

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
    coverage: float             # ISI → FA  (0–100)
    authenticity: float         # FA → ISI  (0–100)
    f1: float
    isi_sentence_count: int
    fa_fragment_count: int
    mismatches: list[dict]      # ISI sentences not well-matched in FA, with diffs
    fa_fragments: list[dict]    # [{"text": str, "page": int}, ...]


# ── Orchestrator ───────────────────────────────────────────────────────────────

class SectionScorer:
    """
    Usage:
        result = SectionScorer().compare(blueprint, page_results)

        blueprint    — ISIBlueprint  from pipeline.py Phase 1
        page_results — list[tuple[int, FAPageFragments]]  from pipeline.py Phase 2
    """

    def __init__(
        self,
        score_threshold: float = SCORE_THRESHOLD,
        partial_match_threshold: float = PARTIAL_MATCH_THRESHOLD,
    ):
        self.score_threshold = score_threshold
        self.partial_match_threshold = partial_match_threshold

    # ── Build section map with page numbers ───────────────────────────────────

    def _build_section_map(
        self,
        page_results: list[tuple[int, FAPageFragments]],
        canonical_titles: list[str],
    ) -> dict[str, list[tuple[str, int]]]:
        """Returns {section_title: [(fragment_text, page_number), ...]}."""
        section_map: dict[str, list[tuple[str, int]]] = {}
        seen_hashes: set[str] = set()

        for page_number, page_frags in page_results:
            for group in page_frags.extractions:
                section_title = _resolve_section_title(group.section_title, canonical_titles)
                section_map.setdefault(section_title, [])
                for fragment in group.fragments:
                    fragment_hash = _fragment_hash(fragment)
                    if fragment_hash in seen_hashes:
                        continue
                    normalized_fragment = normalize_text(fragment)
                    existing_fragments = [normalize_text(text) for text, _ in section_map[section_title]]
                    if existing_fragments and max(
                        fuzz.token_set_ratio(normalized_fragment, existing)
                        for existing in existing_fragments
                    ) > 90:
                        continue
                    seen_hashes.add(fragment_hash)
                    section_map[section_title].append((fragment, page_number))

        return section_map

    # ── Score one section ──────────────────────────────────────────────────────

    def _score_section(
        self,
        isi_section: ISISection,
        fa_entries: list[tuple[str, int]],
    ) -> SectionResult:
        isi_sentences = [normalize_text(s) for s in _split_sentences(isi_section.content)]
        fa_sentences  = [normalize_text(text) for text, _ in fa_entries if text.strip()]
        page_by_fragment = {normalize_text(text): page for text, page in fa_entries}

        empty_result = SectionResult(
            title=isi_section.title,
            coverage=0.0, authenticity=0.0, f1=0.0,
            isi_sentence_count=len(isi_sentences),
            fa_fragment_count=len(fa_sentences),
            mismatches=[],
            fa_fragments=[{"text": text, "page": page} for text, page in fa_entries],
        )
        if not isi_sentences or not fa_sentences:
            return empty_result

        threshold = self.score_threshold

        # Coverage: ISI → FA
        coverage_scores, mismatches = [], []
        for isi_sentence in isi_sentences:
            if len(isi_sentence.split()) < MIN_WORDS:
                continue
            scored_fragments = [(fa_sentence, _similarity_score(isi_sentence, fa_sentence)) for fa_sentence in fa_sentences]
            best_matching_fragment, match_score = max(scored_fragments, key=lambda x: x[1])
            coverage_scores.append(match_score if match_score >= threshold else match_score * PENALTY)
            if match_score < threshold:
                mismatches.append(asdict(SentenceMismatch(
                    isi_sentence=isi_sentence,
                    closest_fa_text=best_matching_fragment,
                    closest_fa_page=page_by_fragment.get(best_matching_fragment),
                    score=round(match_score, 1),
                    diff=_word_level_diff(isi_sentence, best_matching_fragment),
                )))

        coverage = round(mean(coverage_scores), 2) if coverage_scores else 0.0

        # Authenticity: FA → ISI
        authenticity_scores = []
        for fa_sentence in fa_sentences:
            if len(fa_sentence.split()) < MIN_WORDS:
                continue
            best_match_score = max(_similarity_score(fa_sentence, isi_sentence) for isi_sentence in isi_sentences)
            authenticity_scores.append(best_match_score if best_match_score >= threshold else best_match_score * PENALTY)

        authenticity = round(mean(authenticity_scores), 2) if authenticity_scores else 0.0

        f1 = (
            round(2 * coverage * authenticity / (coverage + authenticity), 2)
            if (coverage + authenticity) else 0.0
        )

        return SectionResult(
            title=isi_section.title,
            coverage=coverage,
            authenticity=authenticity,
            f1=f1,
            isi_sentence_count=len(isi_sentences),
            fa_fragment_count=len(fa_sentences),
            mismatches=mismatches,
            fa_fragments=[{"text": text, "page": page} for text, page in fa_entries],
        )

    # ── Main entry point ───────────────────────────────────────────────────────

    def compare(
        self,
        blueprint: ISIBlueprint,
        page_results: list[tuple[int, FAPageFragments]],
    ) -> dict[str, Any]:
        """
        Returns:
        {
            "match_category": "Closest Match" | "Not Matched",
            "overall": {"coverage": float, "authenticity": float, "f1": float},
            "sections": [
                {
                    "title": str,
                    "coverage": float,       # ISI → FA
                    "authenticity": float,   # FA → ISI
                    "f1": float,
                    "isi_sentence_count": int,
                    "fa_fragment_count": int,
                    "mismatches": [
                        {
                            "isi_sentence": str,
                            "closest_fa_text": str,
                            "closest_fa_page": int | None,
                            "score": float,
                            "diff": str,     # "- removed + added   unchanged"
                        }
                    ],
                    "fa_fragments": [{"text": str, "page": int}]
                }
            ]
        }
        """
        canonical_titles = [section.title for section in blueprint.sections]
        section_map = self._build_section_map(page_results, canonical_titles)
        section_results = [
            self._score_section(section, section_map.get(section.title, []))
            for section in blueprint.sections
        ]

        coverage_values  = [section_result.coverage for section_result in section_results]
        authenticity_values = [section_result.authenticity for section_result in section_results]
        overall_coverage     = round(mean(coverage_values), 2)     if coverage_values     else 0.0
        overall_authenticity = round(mean(authenticity_values), 2) if authenticity_values else 0.0
        overall_f1 = (
            round(2 * overall_coverage * overall_authenticity / (overall_coverage + overall_authenticity), 2)
            if (overall_coverage + overall_authenticity) else 0.0
        )

        # A section with ISI sentences but zero coverage means content is completely absent in FA.
        # Averaging would hide this — force Not Matched if any such section exists.
        has_missing_section = any(
            sr.coverage == 0.0 and sr.isi_sentence_count > 0
            for sr in section_results
        )

        return {
            "match_category": (
                "Closest Match"
                if overall_coverage >= self.partial_match_threshold and not has_missing_section
                else "Not Matched"
            ),
            "overall": {
                "coverage":     overall_coverage,
                "authenticity": overall_authenticity,
                "f1":           overall_f1,
            },
            "sections": [asdict(section_result) for section_result in section_results],
        }
