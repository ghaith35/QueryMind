# QueryMind — Day 1 Sanity Check Kit

Three pre-flight scripts. Run in order. Each must pass before you write a single line of production code.

**Total runtime:** ~15–20 minutes
**Purpose:** De-risk the three assumptions that could sink QueryMind after Day 2.

---

## Setup (once)

```bash
cd ~/querymind  # or wherever you've scaffolded the project
python -m venv .venv
source .venv/bin/activate

pip install pymupdf sentence-transformers torch numpy google-generativeai
```

Get a Gemini API key from https://aistudio.google.com/apikey, then:
```bash
export GEMINI_API_KEY=your_key_here
```

Collect one PDF per language and place them somewhere findable:
- `test_pdfs/arabic.pdf` — any Arabic-language document
- `test_pdfs/french.pdf` — any French medical/technical document
- `test_pdfs/english.pdf` — any English research paper

---

## Check #1 — PDF Extraction (5 min)

Validates PyMuPDF can extract Arabic, French, and English without garbling Unicode, breaking accents, or producing � replacement chars.

```bash
python sanity_check_01_extraction.py \
    test_pdfs/arabic.pdf \
    test_pdfs/french.pdf \
    test_pdfs/english.pdf
```

**Pass criteria:**
- Printable char ratio ≥ 0.90 per PDF
- Zero or near-zero � replacement characters
- Visual inspection of first 400 chars looks correct (RTL for Arabic, accents preserved for French)

**If it fails:** Try `pdfplumber` as a fallback extractor, or for scanned PDFs, accept that OCR is out of scope for v1.

---

## Check #2 — Embeddings (5 min, first run downloads ~470MB)

Validates that `intfloat/multilingual-e5-small` produces semantically aligned embeddings across Arabic/French/English, and that MPS acceleration is working on the M1.

```bash
python sanity_check_02_embeddings.py
```

**Pass criteria:**
- MPS available and used
- Paraphrase similarity ≥ 0.75 across all language pairs
- Unrelated sentences show similarity < 0.60
- 16-sentence batch embeds in < 2s on MPS

**If it fails:** Either (a) swap to `intfloat/multilingual-e5-base` (larger, stronger), or (b) drop Arabic from scope and ship EN+FR only.

---

## Check #3 — Citation Prompt (5 min)

The single most important check. Validates Gemini can produce parseable citations that match provided chunk IDs, refuse when content is missing, and handle cross-lingual Q&A.

```bash
python sanity_check_03_citation_prompt.py
```

**Pass criteria:**
- 4/5 or 5/5 test cases pass
- Zero hallucinated chunk IDs across all tests
- Refusal test produces actual refusal (not hallucinated facts)

**If it fails:** This is a fork in the road. Decide Day 1 evening:
- Tighten the prompt with few-shot examples
- Switch to Claude API (costs $10–20 for portfolio demo)
- Reduce scope (e.g., drop Arabic Q&A)

Do not enter Day 2 until you have a passing citation strategy.

---

## After All Three Pass

1. Commit the sanity checks to the repo — they're also your regression tests
2. Write `docs/ADR-001-metadata-rag.md` from the provided template
3. Begin Day 1 Hour 3: the chunker module (see master plan Section 3)

---

## If Something Fails

Do not rationalize. The purpose of these checks is to catch architecture-breaking assumptions on Day 1, when changing course costs hours. On Day 3, it costs the project.
