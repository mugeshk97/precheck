# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management & Setup

This project uses `uv` for dependency management.

```bash
uv sync                                              # Install dependencies
uv add <package>                                     # Add a dependency
```

## Running the pipeline

```bash
uv run python v2.py                          # Default FA (hardcoded in __main__), auto-selects ISI
uv run python v2.py path/to/fa.pdf           # Custom FA, auto-selects ISI
uv run python v2.py path/to/fa.pdf isi.docx  # Explicit FA + ISI pair
```

Input files live in `finalassets/` (PDFs) and `isi/` (`.docx` files). Pass `debug=True` to `run_pipeline()` to print per-section Coverage/Authenticity/F1 scores during Phase 3.

There are no automated tests in this repository.

## Configuration

`.env` must define:
- `KEY` — Azure Document Intelligence API key
- `ENDPOINT` — Azure Document Intelligence endpoint URL

For LLM calls (Blueprint + extraction), one of:
- `OPENAI_API_KEY` — standard OpenAI key *(takes priority if set)*
- `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_DEPLOYMENT` — Azure OpenAI via Managed Identity *(used as fallback when `OPENAI_API_KEY` is absent)*

## Architecture

A 4-phase compliance pipeline (`v2.py`) that checks whether a pharmaceutical Final Asset (FA) PDF correctly reproduces its Important Safety Information (ISI).

### Phase 0 — Auto-discovery
`auto_select_isi()` scores every `isi/*.docx` against the first 4000 chars of FA text using `rapidfuzz.fuzz.token_set_ratio` and picks the highest-scoring file as Ground Truth. Skipped if an ISI path is passed explicitly.

### Phase 1 — Extraction & Blueprint
- **FA**: Azure AI Document Intelligence (`prebuilt-layout` model) extracts text page-by-page into `dict[int, str]`.
- **ISI**: `python-docx` extracts the `.docx` as plain text.
- **Blueprint**: ISI text is sent once to the configured LLM (default model: `o3-mini`) with a strict Pydantic `response_format=ISIBlueprint`. Returns drug name, audience, and a list of `ISISection` objects each containing `title`, `keywords`, and verbatim `content`.

### Phase 2 — Async extraction & deduplication
`asyncio.gather` fires one LLM call per FA page concurrently. Each call returns `FAPageFragments`: ISI-like sentences grouped by section title. Every LLM call is wrapped in `@retry` (tenacity, exponential backoff, 4 attempts). `deduplicate_fragments()` then:
1. **Fuzzy-resolves** LLM-returned section titles to the canonical blueprint titles via `rapidfuzz.process.extractOne` (handles casing/wording drift like `"Indication and Usage"` → `"INDICATION"`).
2. Deduplicates fragments using MD5 hash (exact) and `>90` fuzzy similarity (near-duplicates / repeated footers).

### Phase 3 — Section-wise scoring
`score_section()` compares each `ISISection.content` against the FA fragments mapped to that section:
- **Coverage** (ISI → FA): best fuzzy match per ISI sentence, averaged; scores below 75 penalised ×0.6.
- **Authenticity** (FA → ISI): best fuzzy match per FA fragment; only scores ≥ 75 counted.
- **F1**: harmonic mean. Short sentences (< 4 words) are skipped to avoid noise from headers/labels.

Text normalization before scoring: NFKC Unicode, hyphenated line-break repair (`contra-\nindication`), non-breaking space removal, lowercased.

### Phase 4 — Audit output
`build_audit_report()` aggregates section scores into overall Coverage/Authenticity/F1 and logs token counts. `save_audit_report()` writes to the working directory:
- `audit_<fa_stem>_<timestamp>.json` — full section-level breakdown with FA fragments
- `audit_<fa_stem>_<timestamp>.csv` — one row per section for spreadsheet review

### Key data models (Pydantic)
| Model | Purpose |
|---|---|
| `ISIBlueprint` | Structured parse of the full ISI: drug name, audience, sections |
| `ISISection` | One section: title, keywords, verbatim content |
| `FAPageFragments` | LLM output for one FA page: fragments grouped by section |
| `FAFragment` | Section title + list of ISI-like sentences from one FA page |

### Notes
- Sentence splitting uses NLTK `sent_tokenize` with a regex fallback (`(?<=[.!?])\s+`) if punkt data is unavailable.
- `langchain*` and `spacy` packages in `pyproject.toml` are unused leftovers from earlier experiments.
- `_get_openai_client()` returns `(client, model_name)`. If `OPENAI_API_KEY` is set it returns `AsyncOpenAI` with model `"o3-mini"`; otherwise it builds `AsyncAzureOpenAI` using `ManagedIdentityCredential` + `get_bearer_token_provider` with deployment from `AZURE_OPENAI_DEPLOYMENT` (defaults to `"o3-mini"`). The `model_name` is threaded through `generate_isi_blueprint` → `extract_all_fa_pages` → `_extract_page_fragments` via a `model=` parameter. Note: individual function signatures still show `gpt-4o-mini` as a default but this is always overridden at the call site.
- The `token_log` dict is mutated in-place and shared across all async calls; additions use `+=` with `setdefault` to avoid overwrites on the FA extraction counts.
