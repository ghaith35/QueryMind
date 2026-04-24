"""
Phase 4 tests.

Unit tests (no API, no PDF):
    - Citation parser: valid, hallucinated, de-duplication, refusal detection
    - Prompt builder: chunk formatting, history sliding window
    - Retriever: inject synthetic chunks, query, verify hydration

Integration tests (require GEMINI_API_KEY):
    - 3 known chunks → answer with correct citations
    - Adversarial: irrelevant chunks → refusal
    - Arabic question on English chunks → Arabic answer with EN citations
"""

import json
import os
import sys
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.rag.citation_parser import (
    parse_citations,
    strip_hallucinated_citations,
    is_refusal,
)
from backend.rag.prompt import format_chunks, format_history, build_system_prompt
from backend.rag.retriever import RetrievedChunk

GEMINI_AVAILABLE = bool(os.environ.get("GEMINI_API_KEY"))


# ── Helpers ───────────────────────────────────────────────────────

def _make_retrieved(chunk_id: str, text: str, lang: str = "en") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_name="test.pdf",
        document_id="deadbeef00001111",
        document_set_id="set-001",
        page_number=1,
        paragraph_index=0,
        text=text,
        entities=[],
        char_start=0,
        char_end=len(text),
        word_count=len(text.split()),
        chunk_index_in_document=0,
        language=lang,
        timestamp_indexed=datetime(2026, 4, 22, tzinfo=timezone.utc),
        relevance_score=0.9,
    )


# ── Citation Parser ───────────────────────────────────────────────

CHUNKS = [
    _make_retrieved("abc_0001", "ACE inhibitors are first-line therapy for hypertension."),
    _make_retrieved("abc_0002", "Common side effects include dry cough in 10% of patients."),
    _make_retrieved("abc_0003", "Calcium channel blockers are an alternative therapy."),
]


def test_parse_citations_valid():
    answer = "ACE inhibitors treat hypertension [CITE: abc_0001] and cause cough [CITE: abc_0002]."
    _, valid, hallucinated = parse_citations(answer, CHUNKS)
    assert len(valid) == 2
    assert hallucinated == []
    assert valid[0].chunk_id == "abc_0001"
    assert valid[1].chunk_id == "abc_0002"


def test_parse_citations_hallucinated():
    answer = "Some fact [CITE: abc_0001] and fake [CITE: xyz_9999]."
    _, valid, hallucinated = parse_citations(answer, CHUNKS)
    assert len(valid) == 1
    assert "xyz_9999" in hallucinated


def test_parse_citations_deduplication():
    answer = "Fact A [CITE: abc_0001]. Also, fact A again [CITE: abc_0001]."
    _, valid, hallucinated = parse_citations(answer, CHUNKS)
    assert len(valid) == 1  # de-duplicated
    assert hallucinated == []


def test_parse_citations_no_citations():
    answer = "This answer has no citation markers at all."
    _, valid, hallucinated = parse_citations(answer, CHUNKS)
    assert valid == []
    assert hallucinated == []


def test_parse_citations_case_insensitive():
    answer = "Fact [cite: abc_0001] and [CITE:abc_0002]."
    _, valid, hallucinated = parse_citations(answer, CHUNKS)
    assert len(valid) == 2


def test_parse_citations_permissive_spacing():
    answer = "Fact [CITE abc_0001] and [ Cite : abc_0002 ]."
    _, valid, hallucinated = parse_citations(answer, CHUNKS)
    assert len(valid) == 2
    assert hallucinated == []


def test_parse_citations_excerpt_max_150():
    long_chunk = _make_retrieved("abc_0004", "word " * 100)
    answer = "Fact [CITE: abc_0004]."
    _, valid, _ = parse_citations(answer, [long_chunk])
    assert len(valid[0].excerpt) <= 150


def test_parse_citations_relevance_score_preserved():
    chunk = _make_retrieved("abc_0001", "text")
    chunk.relevance_score = 0.876
    answer = "Fact [CITE: abc_0001]."
    _, valid, _ = parse_citations(answer, [chunk])
    assert valid[0].relevance_score == 0.876


def test_strip_hallucinated_citations():
    answer = "Fact [CITE: abc_0001] and bad [CITE: xyz_9999]."
    cleaned = strip_hallucinated_citations(answer, ["xyz_9999"])
    assert "xyz_9999" not in cleaned
    assert "[citation omitted]" in cleaned
    assert "abc_0001" in cleaned  # valid citation untouched


def test_is_refusal_english():
    assert is_refusal("The provided documents do not contain information to answer this question.")
    assert not is_refusal("ACE inhibitors treat hypertension [CITE: abc_0001].")


def test_is_refusal_arabic():
    assert is_refusal("لا تحتوي الوثائق المقدمة على معلومات للإجابة على هذا السؤال.")


def test_is_refusal_french():
    assert is_refusal("Les documents fournis ne contient pas les informations nécessaires.")


# ── Prompt Builder ────────────────────────────────────────────────

def test_format_chunks_includes_chunk_ids():
    formatted = format_chunks(CHUNKS[:2])
    assert "[CHUNK_ID: abc_0001]" in formatted
    assert "[CHUNK_ID: abc_0002]" in formatted
    assert "ACE inhibitors" in formatted


def test_format_chunks_includes_source():
    formatted = format_chunks(CHUNKS[:1])
    assert "test.pdf" in formatted
    assert "page 1" in formatted


def test_format_history_empty():
    result = format_history([])
    assert "no prior conversation" in result


def test_format_history_respects_budget():
    turns = [
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "x" * 9000},  # exceeds budget alone
        {"role": "user", "content": "latest question"},
    ]
    # With a small budget, only newest turns should appear
    result = format_history(turns, max_tokens=50)
    assert "latest question" in result


def test_format_history_chronological_order():
    turns = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    result = format_history(turns)
    assert result.index("first") < result.index("second") < result.index("third")


def test_build_system_prompt_contains_chunks_and_history():
    prompt = build_system_prompt(CHUNKS[:1], [{"role": "user", "content": "hello"}])
    assert "[CHUNK_ID: abc_0001]" in prompt
    assert "CONVERSATION HISTORY" in prompt
    assert "USER: hello" in prompt


def test_build_system_prompt_strict_adds_warning():
    normal = build_system_prompt(CHUNKS[:1], [], strict=False)
    strict = build_system_prompt(CHUNKS[:1], [], strict=True)
    assert "STRICT MODE" in strict
    assert "STRICT MODE" not in normal


# ── Retriever (synthetic data in temp DB + ChromaDB) ─────────────

@pytest.fixture
def rag_env(tmp_path):
    """Sets up a complete RAG environment with 3 known chunks."""
    from backend.db.connection import init_db, get_db, insert_chunk, upsert_document_set, close_thread_connection
    from backend.indexing.pipeline import init_chroma, _get_collection
    from backend.indexing.embedder import embed_chunks

    db_path = tmp_path / "test.db"
    chroma_path = tmp_path / "chroma"
    init_db(db_path)
    init_chroma(chroma_path)

    doc_set_id = "rag-test-set-001"
    doc_id = "deadbeef00001111"

    with get_db() as conn:
        upsert_document_set(conn, doc_set_id, "test_rag.pdf")
        conn.commit()

    texts = [
        "ACE inhibitors are first-line therapy for hypertension in patients over 60.",
        "Common side effects include dry cough, which occurs in 10% of patients.",
        "Calcium channel blockers are an alternative when ACE inhibitors are contraindicated.",
    ]
    chunk_ids = ["deadbeef00001111_0000", "deadbeef00001111_0001", "deadbeef00001111_0002"]

    embeddings = embed_chunks(texts)
    collection = _get_collection(doc_set_id)
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "chunk_id": cid,
            "document_id": doc_id,
            "document_set_id": doc_set_id,
            "page_number": i + 1,
            "language": "en",
            "entities_csv": "",
        } for i, cid in enumerate(chunk_ids)],
    )

    with get_db() as conn:
        for i, (cid, text) in enumerate(zip(chunk_ids, texts)):
            conn.execute("""
                INSERT INTO chunks (
                    chunk_id, document_set_id, document_id, document_name,
                    page_number, paragraph_index, text, entities_json,
                    char_start, char_end, word_count, chunk_index_in_document,
                    language, timestamp_indexed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cid, doc_set_id, doc_id, "test_rag.pdf",
                  i + 1, 0, text, "[]", 0, len(text),
                  len(text.split()), i, "en", "2026-04-22T10:00:00"))
        conn.commit()

    yield {"doc_set_id": doc_set_id, "chunk_ids": chunk_ids, "texts": texts}

    # Teardown: close thread-local connection so next test gets a fresh one
    close_thread_connection()


def test_retrieve_returns_results(rag_env):
    from backend.rag.retriever import retrieve
    results = retrieve("What are the side effects of ACE inhibitors?", rag_env["doc_set_id"])
    assert len(results) > 0
    assert all(hasattr(r, "relevance_score") for r in results)
    assert all(0 <= r.relevance_score <= 1 for r in results)


def test_retrieve_top_result_is_relevant(rag_env):
    from backend.rag.retriever import retrieve
    results = retrieve("side effects of ACE inhibitors cough", rag_env["doc_set_id"])
    # The "cough" chunk should rank high
    top = results[0]
    assert "cough" in top.text or "ACE" in top.text


def test_retrieve_hydrates_full_text(rag_env):
    from backend.rag.retriever import retrieve
    results = retrieve("hypertension treatment", rag_env["doc_set_id"])
    for r in results:
        assert r.text  # no empty text
        assert r.document_name == "test_rag.pdf"


def test_retrieve_document_overview_returns_broader_context(rag_env):
    from backend.rag.retriever import retrieve
    results = retrieve("What is this document about? Give me an overview.", rag_env["doc_set_id"])
    assert len(results) >= 3
    assert any("ACE inhibitors" in r.text for r in results)
    assert any("Calcium channel blockers" in r.text for r in results)


def test_retrieve_empty_collection(tmp_path):
    """Retrieval on a non-existent document set should return empty list."""
    from backend.db.connection import init_db
    from backend.indexing.pipeline import init_chroma, _get_collection
    from backend.rag.retriever import retrieve

    init_db(tmp_path / "test.db")
    init_chroma(tmp_path / "chroma")

    # Create an empty collection
    col = _get_collection("empty-set-001")
    # ChromaDB returns error if n_results > count — retrieve should handle gracefully
    results = retrieve("anything", "empty-set-001")
    assert results == []


# ── Integration tests (require GEMINI_API_KEY) ────────────────────

MEDICAL_CHUNKS_DATA = [
    {
        "chunk_id": "med_a1b2_0001",
        "text": "ACE inhibitors are first-line therapy for hypertension in patients over 60. Common side effects include dry cough in approximately 10% of patients, and hyperkalemia in patients with renal impairment.",
        "lang": "en",
    },
    {
        "chunk_id": "med_a1b2_0014",
        "text": "In elderly patients, monitoring of serum creatinine and potassium is essential during the first month of ACE inhibitor therapy. Dose reduction is required if eGFR falls below 30 mL/min.",
        "lang": "en",
    },
    {
        "chunk_id": "med_a1b2_0022",
        "text": "Calcium channel blockers are an alternative when ACE inhibitors are contraindicated. Amlodipine is commonly used at 5-10mg daily with the most frequent side effect being peripheral edema.",
        "lang": "en",
    },
]

MARINE_CHUNKS_DATA = [
    {
        "chunk_id": "bio_xyz_0001",
        "text": "Coral reefs in the Indo-Pacific region host the greatest marine biodiversity on Earth. The Great Barrier Reef alone contains over 1,500 species of fish.",
        "lang": "en",
    },
]


def _build_retrieved(chunks_data) -> List[RetrievedChunk]:
    return [
        _make_retrieved(d["chunk_id"], d["text"], d["lang"])
        for d in chunks_data
    ]


@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="GEMINI_API_KEY not set")
def test_integration_english_question_valid_citations():
    """3 known medical chunks → answer cites only those chunk IDs."""
    from backend.rag.llm_client import generate
    from backend.rag.prompt import build_system_prompt

    chunks = _build_retrieved(MEDICAL_CHUNKS_DATA)
    prompt = build_system_prompt(chunks, [])
    answer = generate("What are the side effects of ACE inhibitors in elderly patients?", prompt)

    _, valid, hallucinated = parse_citations(answer, chunks)
    assert len(valid) >= 1, f"Expected at least 1 citation, got 0. Answer:\n{answer}"
    assert hallucinated == [], f"Hallucinated IDs: {hallucinated}"


@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="GEMINI_API_KEY not set")
def test_integration_refusal_on_irrelevant_chunks():
    """Marine biology chunks → must refuse to answer medical question."""
    from backend.rag.llm_client import generate
    from backend.rag.prompt import build_system_prompt

    chunks = _build_retrieved(MARINE_CHUNKS_DATA)
    prompt = build_system_prompt(chunks, [])
    answer = generate("What are the side effects of ACE inhibitors?", prompt)

    assert is_refusal(answer), f"Expected refusal, got:\n{answer}"
    _, valid, hallucinated = parse_citations(answer, chunks)
    assert hallucinated == [], f"Refusal should have no hallucinated IDs: {hallucinated}"


@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="GEMINI_API_KEY not set")
def test_integration_arabic_question_gives_arabic_answer():
    """Arabic question against English chunks → answer in Arabic, citations in EN chunk IDs."""
    from backend.rag.llm_client import generate
    from backend.rag.prompt import build_system_prompt

    chunks = _build_retrieved(MEDICAL_CHUNKS_DATA)
    prompt = build_system_prompt(chunks, [])
    arabic_q = "ما هي الآثار الجانبية لمثبطات الإنزيم المحول للأنجيوتنسين؟"
    answer = generate(arabic_q, prompt)

    # Answer should contain Arabic characters
    arabic_chars = sum(1 for c in answer if "؀" <= c <= "ۿ")
    assert arabic_chars > 10, f"Expected Arabic answer, got:\n{answer}"

    _, valid, hallucinated = parse_citations(answer, chunks)
    assert hallucinated == [], f"Hallucinated IDs in Arabic answer: {hallucinated}"
    assert len(valid) >= 1, f"Expected citations in Arabic answer, got 0"
