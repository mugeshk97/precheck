"""
Section-wise ISI-FA scorer.

Coverage     (ISI → full FA text)       : is the ISI content present in the FA?
Authenticity (FA extracted ISI → ISI)   : does the extracted content trace back to the ISI?

Both use rapidfuzz.fuzz.partial_ratio — best-window match, clean + sentence-tokenized.
"""

import difflib
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

        coverage = float(fuzz.partial_ratio(isi_full, fa_raw_text))

        # Fragment-wise authenticity: score each FA fragment individually against ISI section
        scorable_frags = [t for t in fa_texts if len(t.split()) >= MIN_WORDS]
        if scorable_frags:
            frag_scores = [float(fuzz.partial_ratio(f, isi_full)) for f in scorable_frags]
            authenticity = round(sum(frag_scores) / len(frag_scores), 2)
        else:
            authenticity = 0.0

        f1 = round(2 * coverage * authenticity / (coverage + authenticity), 2) if (coverage + authenticity) else 0.0

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

        # Overall authenticity: fragment-wise FA → full ISI matching.
        # Each FA fragment is scored against the full ISI text (not section-scoped),
        # so fragments assigned to the wrong section still get credit.
        isi_full_text = _prepare(" ".join(sec.content for sec in blueprint.sections))
        all_fa_frags = [
            normalize_text(frag)
            for entries in section_map.values()
            for frag, _ in entries
            if len(normalize_text(frag).split()) >= MIN_WORDS
        ]
        if all_fa_frags:
            frag_auth_scores = [float(fuzz.partial_ratio(f, isi_full_text)) for f in all_fa_frags]
            overall_auth = round(sum(frag_auth_scores) / len(frag_auth_scores), 2)
        else:
            overall_auth = 0.0

        overall_f1 = round(2 * overall_cov * overall_auth / (overall_cov + overall_auth), 2) if (overall_cov + overall_auth) else 0.0

        return {
            "match_category": "Closest Match" if overall_cov >= self.match_threshold else "Not Matched",
            "overall": {"coverage": overall_cov, "authenticity": overall_auth, "f1": overall_f1},
            "sections": [asdict(r) for r in results],
            # Keep raw data for debug report generation
            "_debug_data": {
                "all_isi_sentences": all_isi_sentences,
                "sent_scores": [float(fuzz.partial_ratio(s, fa_raw_text)) for s in all_isi_sentences] if all_isi_sentences else [],
                "section_map": section_map,
                "fa_raw_text": fa_raw_text,
                "isi_full_text": isi_full_text,
                "all_fa_frags": all_fa_frags,
                "frag_auth_scores": frag_auth_scores if all_fa_frags else [],
            },
        }

    def generate_debug_report(
        self,
        blueprint: ISIBlueprint,
        comparison_result: dict,
    ) -> str:
        """Generate a human-readable debug report explaining score gaps.

        Shows:
        - Coverage gaps: ISI sentences not fully found in the FA, with word-level diffs
        - Authenticity gaps: FA fragments that don't trace back cleanly to the ISI
        """
        debug_data = comparison_result.get("_debug_data", {})
        fa_raw_text = debug_data.get("fa_raw_text", "")
        section_map = debug_data.get("section_map", {})
        all_isi_sentences = debug_data.get("all_isi_sentences", [])
        sent_scores = debug_data.get("sent_scores", [])

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("DEBUG REPORT — Score Gap Analysis")
        lines.append("=" * 80)

        overall = comparison_result.get("overall", {})
        lines.append(f"\nOverall Coverage:     {overall.get('coverage', 0):.1f}")
        lines.append(f"Overall Authenticity: {overall.get('authenticity', 0):.1f}")
        lines.append(f"Overall F1:           {overall.get('f1', 0):.1f}")
        lines.append(f"Match Category:       {comparison_result.get('match_category', '?')}")

        # ── Coverage Gaps (ISI → FA) ─────────────────────────────────────────
        lines.append(f"\n{'─' * 80}")
        lines.append("COVERAGE GAPS — ISI sentences not fully found in the FA")
        lines.append(f"{'─' * 80}")

        # Group sentences by section for readability
        sentence_idx = 0
        total_gaps = 0
        for section in blueprint.sections:
            sec_sentences = [
                normalize_text(s)
                for s in _split_sentences(section.content)
                if len(normalize_text(s).split()) >= MIN_WORDS
            ]
            sec_gaps = []
            for sent in sec_sentences:
                if sentence_idx < len(sent_scores):
                    score = sent_scores[sentence_idx]
                    if score < 100.0:
                        # Find the best matching window in FA text
                        best_fa_match = self._find_best_match(sent, fa_raw_text)
                        diff = _word_diff(sent, best_fa_match)
                        sec_gaps.append((sent, best_fa_match, score, diff))
                    sentence_idx += 1

            if sec_gaps:
                total_gaps += len(sec_gaps)
                lines.append(f"\n  Section: {section.title}")
                lines.append(f"  Gaps: {len(sec_gaps)} / {len(sec_sentences)} sentences")
                for sent, fa_match, score, diff in sec_gaps:
                    lines.append(f"\n    Score: {score:.1f}")
                    lines.append(f"    ISI: \"{sent}\"")
                    lines.append(f"    FA:  \"{fa_match}\"")
                    lines.append(f"    Diff:")
                    for d in diff:
                        lines.append(f"      {d}")

        if total_gaps == 0:
            lines.append("\n  No coverage gaps — all ISI sentences found in FA.")
        else:
            lines.append(f"\n  Total coverage gaps: {total_gaps} / {len(all_isi_sentences)} sentences")

        # ── Authenticity Gaps (FA → ISI) ─────────────────────────────────────
        lines.append(f"\n{'─' * 80}")
        lines.append("AUTHENTICITY GAPS — FA fragments that don't trace back to the ISI")
        lines.append(f"{'─' * 80}")

        isi_full_text = debug_data.get("isi_full_text", "")
        total_auth_gaps = 0
        total_auth_frags = 0
        for section in blueprint.sections:
            fa_entries = section_map.get(section.title, [])
            if not fa_entries:
                continue

            sec_auth_gaps = []
            for frag_text, page_num in fa_entries:
                frag_norm = normalize_text(frag_text)
                if not frag_norm.strip() or len(frag_norm.split()) < MIN_WORDS:
                    continue
                total_auth_frags += 1
                # Score against full ISI text (matches overall scoring)
                score = float(fuzz.partial_ratio(frag_norm, isi_full_text))
                if score < 100.0:
                    best_isi_match = self._find_best_match(frag_norm, isi_full_text)
                    diff = _word_diff(best_isi_match, frag_norm)
                    sec_auth_gaps.append((frag_text, page_num, best_isi_match, score, diff))

            if sec_auth_gaps:
                total_auth_gaps += len(sec_auth_gaps)
                lines.append(f"\n  Section: {section.title}")
                lines.append(f"  Gaps: {len(sec_auth_gaps)} / {len(fa_entries)} fragments")
                for frag, page, isi_match, score, diff in sec_auth_gaps:
                    lines.append(f"\n    Score: {score:.1f} | Page: {page}")
                    lines.append(f"    FA fragment: \"{frag}\"")
                    lines.append(f"    ISI match:   \"{isi_match}\"")
                    lines.append(f"    Diff (ISI → FA):")
                    for d in diff:
                        lines.append(f"      {d}")

        if total_auth_gaps == 0:
            lines.append("\n  No authenticity gaps — all FA fragments trace back to ISI.")
        else:
            lines.append(f"\n  Total authenticity gaps: {total_auth_gaps} / {total_auth_frags} fragments")

        # ── Per-section summary table ────────────────────────────────────────
        lines.append(f"\n{'─' * 80}")
        lines.append("SECTION SUMMARY")
        lines.append(f"{'─' * 80}")
        lines.append(f"\n  {'Section':<50} {'Cov':>5} {'Auth':>5} {'F1':>5} {'ISI#':>5} {'FA#':>5} {'Gaps':>5}")
        lines.append(f"  {'─' * 80}")
        for sec in comparison_result.get("sections", []):
            lines.append(
                f"  {sec['title'][:50]:<50} "
                f"{sec['coverage']:5.1f} {sec['authenticity']:5.1f} {sec['f1']:5.1f} "
                f"{sec['isi_sentence_count']:5d} {sec['fa_fragment_count']:5d} "
                f"{len(sec['mismatches']):5d}"
            )

        lines.append(f"\n{'=' * 80}")
        return "\n".join(lines)

    def _find_best_match(self, query: str, text: str, window_words: int = 0) -> str:
        """Find the best matching substring in text for the query using sliding window."""
        query_words = query.split()
        text_words = text.split()
        if not text_words or not query_words:
            return ""

        win = window_words or max(len(query_words), 8)
        best_score = -1.0
        best_window = ""

        for i in range(max(1, len(text_words) - win + 1)):
            window = " ".join(text_words[i : i + win])
            score = fuzz.partial_ratio(query, window)
            if score > best_score:
                best_score = score
                best_window = window
            if score == 100.0:
                break

        return best_window


def _word_diff(expected: str, actual: str) -> list[str]:
    """Produce a human-readable word-level diff between two strings.

    Returns lines like:
      MISSING: "strong"         (in ISI but not in FA)
      ADDED:   "moderate"       (in FA but not in ISI)
      CHANGED: "inhibitors" → "inhibitor"
    """
    exp_words = expected.split()
    act_words = actual.split()
    sm = difflib.SequenceMatcher(None, exp_words, act_words)

    result: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "delete":
            missing = " ".join(exp_words[i1:i2])
            result.append(f"MISSING: \"{missing}\"")
        elif tag == "insert":
            added = " ".join(act_words[j1:j2])
            result.append(f"ADDED:   \"{added}\"")
        elif tag == "replace":
            old = " ".join(exp_words[i1:i2])
            new = " ".join(act_words[j1:j2])
            result.append(f"CHANGED: \"{old}\" → \"{new}\"")
    return result if result else ["(no word-level differences detected)"]
