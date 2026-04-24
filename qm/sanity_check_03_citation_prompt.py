"""
QueryMind Day 1 — Sanity Check #3: Gemini Citation Prompt Validation

Purpose: The citation prompt is the riskiest component of QueryMind.
This script validates, on Day 1 Hour 2-3 (not Day 2), that Gemini can:
    1. Accept chunks with [CHUNK_ID: xxx] markers
    2. Cite using those exact IDs in the [CITE: xxx] format
    3. Refuse to answer when chunks are irrelevant
    4. Handle Arabic queries against English/French chunks

If this fails on Day 1, we switch LLM provider or change prompt strategy
BEFORE investing 8 hours in Day 2 on a doomed architecture.

Setup:
    export GEMINI_API_KEY=your_key_here
    pip install google-generativeai

Usage:
    python sanity_check_03_citation_prompt.py
"""

import os
import re
import sys
import json
import time
from typing import List, Dict

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed")
    print("Run: pip install google-genai")
    sys.exit(1)


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable not set")
    print("Get key from: https://aistudio.google.com/apikey")
    print("Then: export GEMINI_API_KEY=your_key_here")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"  # use this; 2.0-flash and 2.0-flash-lite have 0 free quota on this project


CITATION_SYSTEM_PROMPT = """You are QueryMind, a document intelligence assistant. You answer questions using ONLY the provided document chunks. Every factual claim MUST be followed by a citation in the exact format: [CITE: chunk_id]

RULES (non-negotiable):
1. Use ONLY chunks provided below. Do not use outside knowledge.
2. Every sentence containing a fact MUST end with [CITE: chunk_id] where chunk_id matches EXACTLY one of the provided IDs.
3. If the chunks do not contain the answer, respond: "The provided documents do not contain information to answer this question."
4. Multiple citations allowed: [CITE: abc_0001] [CITE: abc_0042]
5. Respond in the same language as the user's question.
6. Do not invent chunk_ids. Do not modify chunk_ids. Copy them character-for-character.

AVAILABLE CHUNKS:
{chunks}
"""


# Test chunks — realistic medical content with known chunk IDs
TEST_CHUNKS = {
    "medical_en": [
        {
            "chunk_id": "med_a1b2_0001",
            "document_name": "cardiology_2024.pdf",
            "page_number": 12,
            "paragraph_index": 3,
            "language": "en",
            "text": "ACE inhibitors are first-line therapy for hypertension in patients over 60. Common side effects include dry cough, which occurs in approximately 10% of patients, and hyperkalemia, particularly in patients with renal impairment.",
        },
        {
            "chunk_id": "med_a1b2_0014",
            "document_name": "cardiology_2024.pdf",
            "page_number": 28,
            "paragraph_index": 1,
            "language": "en",
            "text": "In elderly patients, monitoring of serum creatinine and potassium is essential during the first month of ACE inhibitor therapy. Dose reduction is required if eGFR falls below 30 mL/min.",
        },
        {
            "chunk_id": "med_a1b2_0022",
            "document_name": "cardiology_2024.pdf",
            "page_number": 45,
            "paragraph_index": 2,
            "language": "en",
            "text": "Calcium channel blockers are an alternative when ACE inhibitors are contraindicated. Amlodipine is commonly used at 5-10mg daily, with the most frequent side effect being peripheral edema.",
        },
    ],
    "irrelevant_en": [
        {
            "chunk_id": "bio_xyz_0001",
            "document_name": "marine_biology.pdf",
            "page_number": 3,
            "paragraph_index": 1,
            "language": "en",
            "text": "Coral reefs in the Indo-Pacific region host the greatest marine biodiversity on Earth. The Great Barrier Reef alone contains over 1,500 species of fish.",
        },
    ],
}


def format_chunks(chunks: List[Dict]) -> str:
    """Render chunks for prompt injection."""
    formatted = []
    for c in chunks:
        formatted.append(
            f"[CHUNK_ID: {c['chunk_id']}]\n"
            f"[SOURCE: {c['document_name']}, page {c['page_number']}, paragraph {c['paragraph_index']}]\n"
            f"[LANGUAGE: {c['language']}]\n"
            f"---\n"
            f"{c['text']}\n"
            f"---"
        )
    return "\n\n".join(formatted)


def parse_citations(answer: str, valid_chunk_ids: set) -> tuple[list, list]:
    """Extract cited IDs and split into valid vs hallucinated."""
    pattern = re.compile(r'\[\s*CITE\s*[:\s]\s*([a-zA-Z0-9_]+)\s*\]', re.IGNORECASE)
    cited = pattern.findall(answer)
    valid = [c for c in cited if c in valid_chunk_ids]
    hallucinated = [c for c in cited if c not in valid_chunk_ids]
    return valid, hallucinated


def ask_gemini(question: str, chunks: List[Dict]) -> str:
    """Run one question through Gemini with citation prompt. Retries on 429."""
    system_prompt = CITATION_SYSTEM_PROMPT.format(chunks=format_chunks(chunks))
    user_prompt = f"USER QUESTION: {question}\n\nANSWER:"
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 20 + attempt * 10
                print(f"  Rate limited — waiting {wait}s before retry {attempt+2}/3 …")
                time.sleep(wait)
            else:
                raise


def run_test(name: str, question: str, chunks: List[Dict],
             expect_refusal: bool = False, expect_language: str = "en") -> dict:
    """Run a single test case and grade the response."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print('=' * 60)
    print(f"Question: {question}")
    print(f"Chunks provided: {len(chunks)} ({[c['chunk_id'] for c in chunks]})")
    print(f"Expected: {'refusal' if expect_refusal else f'answer with citations in {expect_language}'}")

    try:
        answer = ask_gemini(question, chunks)
    except Exception as e:
        print(f"❌ API ERROR: {e}")
        return {"name": name, "passed": False, "error": str(e)}

    print(f"\n--- Gemini's answer ---")
    print(answer)
    print("--- End answer ---")

    valid_ids = {c["chunk_id"] for c in chunks}
    valid_citations, hallucinated = parse_citations(answer, valid_ids)

    print(f"\nValid citations:        {valid_citations}")
    print(f"Hallucinated citations: {hallucinated}")

    # Grading
    passed = True
    issues = []

    if expect_refusal:
        refusal_phrases = [
            "do not contain",
            "cannot answer",
            "no information",
            "ne contient",
            "لا تحتوي",
        ]
        refused = any(p in answer.lower() for p in refusal_phrases)
        if not refused:
            passed = False
            issues.append("Expected refusal but got an answer")
        if valid_citations:
            passed = False
            issues.append("Refusal should not contain citations")
    else:
        if not valid_citations:
            passed = False
            issues.append("Expected citations but found none")
        if hallucinated:
            passed = False
            issues.append(f"HALLUCINATED IDs: {hallucinated}")

    # Language check (simple heuristic)
    if expect_language == "ar" and not any(
        '\u0600' <= c <= '\u06FF' for c in answer
    ):
        passed = False
        issues.append("Expected Arabic response, got non-Arabic")

    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}")
    for issue in issues:
        print(f"   ⚠ {issue}")

    return {
        "name": name,
        "passed": passed,
        "issues": issues,
        "valid_citations": valid_citations,
        "hallucinated": hallucinated,
        "answer": answer,
    }


def main():
    print("QueryMind Citation Prompt Sanity Check")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    results = []
    INTER_TEST_DELAY = 5  # seconds between calls to avoid per-minute rate limits

    # Test 1: Standard English question with relevant chunks
    results.append(run_test(
        name="English question with relevant chunks",
        question="What are the side effects of ACE inhibitors in elderly patients?",
        chunks=TEST_CHUNKS["medical_en"],
        expect_language="en",
    ))

    time.sleep(INTER_TEST_DELAY)
    # Test 2: Refusal — irrelevant chunks
    results.append(run_test(
        name="Refusal when chunks are irrelevant",
        question="What are the side effects of ACE inhibitors?",
        chunks=TEST_CHUNKS["irrelevant_en"],
        expect_refusal=True,
    ))

    time.sleep(INTER_TEST_DELAY)
    # Test 3: Arabic question against English chunks
    results.append(run_test(
        name="Arabic question on English chunks (cross-lingual)",
        question="ما هي الآثار الجانبية لمثبطات الإنزيم المحول للأنجيوتنسين؟",
        chunks=TEST_CHUNKS["medical_en"],
        expect_language="ar",
    ))

    time.sleep(INTER_TEST_DELAY)
    # Test 4: Multi-chunk synthesis
    results.append(run_test(
        name="Multi-chunk synthesis with multiple citations",
        question="What should be monitored when prescribing ACE inhibitors to a 70-year-old patient?",
        chunks=TEST_CHUNKS["medical_en"],
        expect_language="en",
    ))

    time.sleep(INTER_TEST_DELAY)
    # Test 5: Adversarial — question mixes covered and uncovered facts
    results.append(run_test(
        name="Mixed — partial answer available",
        question="What is the dose of amlodipine and what is its half-life?",
        chunks=TEST_CHUNKS["medical_en"],  # dose is present, half-life is NOT
        expect_language="en",
    ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for r in results if r.get("passed"))
    total = len(results)

    for r in results:
        status = "✅" if r.get("passed") else "❌"
        print(f"  {status} {r['name']}")
        for issue in r.get("issues", []):
            print(f"      ⚠ {issue}")

    print(f"\nPassed: {passed_count}/{total}")

    # Hallucination summary — most important metric
    total_hallucinated = sum(len(r.get("hallucinated", [])) for r in results)
    if total_hallucinated == 0:
        print("✅ ZERO HALLUCINATED CITATIONS — prompt strategy viable")
    else:
        print(f"⚠ {total_hallucinated} hallucinated citations across {total} tests")
        print("   Consider: prompt reinforcement, switch to Claude API, or add retry logic")

    if passed_count == total and total_hallucinated == 0:
        print("\n✅ CITATION PROMPT PASSES — proceed to Day 2 with confidence")
        sys.exit(0)
    elif passed_count >= total - 1 and total_hallucinated <= 1:
        print("\n⚠ MOSTLY PASSING — review failures, likely proceed with caution")
        sys.exit(0)
    else:
        print("\n❌ CITATION PROMPT UNRELIABLE — fix BEFORE Day 2")
        print("\nRemediation paths:")
        print("  - Tighten system prompt with more explicit examples")
        print("  - Add one-shot or few-shot demonstration in prompt")
        print("  - Switch primary LLM to Claude API")
        print("  - Add structured output mode (Gemini JSON mode)")
        sys.exit(1)


if __name__ == "__main__":
    main()
