# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses `uv` for dependency management.

```bash
uv sync                              # Install dependencies
uv run python v2.py                  # Run with default FA, auto-selects ISI
uv run python v2.py path/to/fa.pdf   # Custom FA, auto-selects ISI
uv run python v2.py fa.pdf isi.docx  # Explicit FA + ISI pair
uv add <package>                     # Add a dependency
```

## Configuration

`.env` must define:
- `KEY` — Azure Document Intelligence API key
- `ENDPOINT` — Azure Document Intelligence endpoint URL
- `OPENAI_API_KEY` — OpenAI API key (required for Blueprint + extraction phases)

## Architecture

A 4-phase compliance pipeline that checks whether a pharmaceutical Final Asset (FA) PDF correctly reproduces its Important Safety Information (ISI).

### Phase 0 — Auto-discovery
`auto_select_isi()` scores every `isi/*.docx` against the FA text using `rapidfuzz.fuzz.token_set_ratio` and selects the highest-scoring file as the Ground Truth. Skipped if an ISI path is passed explicitly.

### Phase 1 — Extraction & Blueprint
- **FA**: Azure AI Document Intelligence (`prebuilt-layout` model) extracts text page-by-page into `{page_num: str}`.
- **ISI**: `python-docx` extracts the `.docx` as plain text.
- **Blueprint**: The ISI text is sent once to `gpt-4o-mini` with a strict Pydantic schema (`ISIBlueprint → list[ISISection]`). Each section has `title`, `keywords`, and verbatim `content`. Structured outputs guarantee valid JSON even on edge cases.

### Phase 2 — Async extraction & deduplication
`asyncio.gather` fires one `gpt-4o-mini` call per FA page concurrently. Each call returns `FAPageFragments` (a Pydantic model): ISI-like sentences grouped by the matching ISI section title. Every LLM call is wrapped in a `@retry` decorator (tenacity, exponential backoff, 4 attempts). After gathering, `deduplicate_fragments()` merges results across pages using MD5 hashing for exact duplicates and `>90` fuzzy similarity for near-duplicates (eliminates repeated footers).

### Phase 3 — Section-wise scoring
For each `ISISection`, `score_section()` compares the ISI section's sentences against the FA fragments extracted for that section:
- **Coverage** (ISI → FA): average best fuzzy match per ISI sentence; scores below 75 are penalised ×0.6.
- **Authenticity** (FA → ISI): average best fuzzy match per FA fragment, only counting scores ≥ 75.
- **F1**: harmonic mean of the two.
Text is normalized before scoring: NFKC Unicode, hyphenated line-break repair (`contra-\nindication`), non-breaking space removal, lowercased.

### Phase 4 — Audit output
`build_audit_report()` aggregates section scores into overall Coverage/Authenticity/F1 and logs all token counts. `save_audit_report()` writes two files to the working directory:
- `audit_<fa_stem>_<timestamp>.json` — full section-level breakdown with FA fragments
- `audit_<fa_stem>_<timestamp>.csv` — one row per section for spreadsheet review

### Key data models (Pydantic)
| Model | Purpose |
|---|---|
| `ISIBlueprint` | Structured parse of the full ISI (drug name, audience, sections) |
| `ISISection` | One ISI section: title, keywords, verbatim content |
| `FAPageFragments` | LLM output for one FA page: fragments grouped by ISI section |
| `FAFragment` | Section title + list of matching sentences from one FA page |

### Notes
- `langchain*` packages in `pyproject.toml` are unused — leftovers from earlier experiments.
- Sentence splitting uses NLTK `sent_tokenize` with a regex fallback if punkt data is unavailable.
- Short sentences (< 4 words) are skipped during scoring to avoid noise from headers/labels.
