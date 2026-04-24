# QueryMind — Master Implementation Plan

**Author:** Staff-Level AI Systems Architect (for Ghaith)
**Target:** 4-day build on M1 MacBook Air (8GB / 256GB)
**Portfolio Target:** Mozn, Lucidya, Elm

---

## SECTION 1 — Problem Framing & Architecture Decision Justification

### The 3 Hardest Engineering Problems in QueryMind

**Problem 1 — Citation Fidelity Under Generation Pressure.**
The LLM must produce fluent natural-language answers while emitting machine-parseable citations that point to *exactly* the chunks it actually used — not chunks it retrieved but ignored, not hallucinated page numbers, not paraphrased document names. The failure mode isn't "wrong answer"; it's "correct-looking answer with fabricated `[Page 47]` that doesn't exist." This destroys trust instantly on a demo. The solution is a closed-loop architecture: chunks enter the prompt with explicit `[CHUNK_ID: xyz]` tags, the LLM is instructed to cite using those exact IDs, and a post-generation validator rejects any citation not in the retrieved set.

**Problem 2 — Real-Time Graph Coherence During Indexing.**
The knowledge graph must *grow visibly* during indexing — nodes appearing, edges forming — while the indexing pipeline runs at ~5–10 chunks/sec on an M1 with MPS acceleration. Pushing a node event per entity per chunk floods the WebSocket and causes D3 to re-simulate the force layout on every tick, freezing the UI. The solution is batched graph diffs: every 20 chunks (or 500ms), the backend emits `{added_nodes, added_edges, updated_nodes}` as a single payload, and the frontend applies a D3 `enter/update/exit` transition without restarting the simulation.

**Problem 3 — Arabic Through the Full Stack.**
PyMuPDF extracts Arabic correctly only if encoding is preserved; sentence-transformers must use a multilingual model; ChromaDB must store Unicode cleanly; the LLM must answer in Arabic when asked in Arabic; and the frontend must render RTL chat bubbles. Any one layer that drops Arabic silently (common with PyMuPDF on ligatures) poisons the whole demo. Solution: test Arabic end-to-end on Day 1 with a single Arabic PDF before building anything else.

### Why Citation Accuracy Is Non-Negotiable

Recruiters at Mozn/Lucidya/Elm have seen 50 ChatPDF clones. What they haven't seen is a system that *refuses* to answer when retrieved chunks are insufficient, and that makes every citation clickable and verifiable. Citation accuracy is the **only** feature that separates a toy from infrastructure. The architecture must prevent hallucinated citations structurally, not via prompt begging.

**How this architecture prevents hallucinated citations:**
1. Every retrieved chunk is injected into the prompt with an explicit `[CHUNK_ID: abc123]` marker.
2. The system prompt mandates the LLM cite *only* using those IDs.
3. A post-parse validator intersects cited IDs with retrieved IDs. Any non-matching citation triggers an auto-retry with a stricter prompt; second failure → citation stripped and answer flagged.
4. The frontend only renders citations that resolve to real chunks.

### Why Metadata-Rich RAG (Option B) Over True Graph RAG

True Graph RAG (Microsoft's GraphRAG, LightRAG) requires: LLM-based entity extraction per chunk (~$0.01/chunk × thousands of chunks = real money), relationship extraction with LLM calls, community detection, and hierarchical summarization. For a 4-day build on free-tier Gemini with rate limits, this is infeasible. More importantly, the *visible output* of Graph RAG to a recruiter watching a 2-minute demo is indistinguishable from metadata-rich RAG with a D3 co-occurrence graph. The recruiter sees: nodes glow when an answer is generated. They cannot tell whether edges were LLM-extracted relations or co-occurrence weights. We ship the visible 90% in 4 days instead of the invisible 100% in 4 weeks.

### What Makes QueryMind Genuinely Impressive

- **Verifiable citations** — click any citation, jump to exact chunk, see highlighted excerpt
- **Live-building graph** — recruiter watches the knowledge structure emerge during upload
- **Answer path highlighting** — the graph nodes that contributed glow in real-time
- **Multilingual** — demo a French medical PDF, ask in Arabic, get an Arabic answer with French citations
- **Auto re-indexing** — drop a new PDF in the folder, graph updates live
- **Zero cloud dependencies** — runs on the M1 Air the recruiter is watching

ChatPDF has none of these except basic citations. This is the demo differential.

### The Single Most Important Thing on Day 1

**Arabic text extraction from a real Algerian PDF must work end-to-end before anything else is built.** If PyMuPDF mangles Arabic ligatures, or multilingual-e5-small doesn't embed Arabic semantically, or the chunk storage corrupts Unicode, the entire project's differentiator collapses. Test with one Arabic PDF, one French medical PDF, one English research paper on Day 1 Hour 1. If any fails, architecture changes *today*, not Day 3.

### 🔨 Build Checklist — Section 1

- [ ] Write a 20-line script: PyMuPDF extract → print text → visually verify Arabic, French accents, English
- [ ] Load `intfloat/multilingual-e5-small`, embed one Arabic sentence + one French sentence, cosine-similarity — confirm semantic alignment
- [ ] Commit architectural decision log as `docs/ADR-001-metadata-rag.md`
- [ ] Set up project structure: `backend/`, `frontend/`, `docs/`, `data/pdfs/`, `data/chroma/`, `data/sqlite/`
- [ ] Install Python 3.11 via pyenv (sentence-transformers is stable here); create venv
- [ ] `pip install fastapi uvicorn pymupdf pdfplumber sentence-transformers chromadb langchain langchain-google-genai spacy watchdog python-multipart websockets`
- [ ] `python -m spacy download en_core_web_sm fr_core_news_sm`
- [ ] Get Gemini API key, test with a single "Hello" call
- [ ] Buy `ghaith.com` on Cloudflare Registrar (can be done in parallel)

---

## SECTION 2 — Data Contracts (Complete Specification)

### Chunk Schema (Source of Truth)

```python
# backend/schemas/chunk.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Tuple, Literal

class Chunk(BaseModel):
    chunk_id: str  # format: "{doc_hash[:8]}_{chunk_index:04d}" e.g. "a3f9c2b1_0042"
    document_name: str  # original filename, UI-facing
    document_id: str  # sha256(filename + filesize)[:16]
    document_set_id: str  # UUID of the set this chunk belongs to
    page_number: int  # 1-indexed, matches PDF viewer convention
    paragraph_index: int  # 0-indexed within the page
    text: str  # raw chunk text, max 1200 chars
    entities: List[str] = Field(default_factory=list)  # max 15 per chunk
    char_start: int  # position in original page text
    char_end: int
    word_count: int
    chunk_index_in_document: int  # 0-indexed, globally within doc
    language: Literal["en", "fr", "ar", "mixed"]
    timestamp_indexed: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "a3f9c2b1_0042",
                "document_name": "cardiology_guidelines_2024.pdf",
                "document_id": "a3f9c2b1d4e5f6a7",
                "document_set_id": "550e8400-e29b-41d4-a716-446655440000",
                "page_number": 12,
                "paragraph_index": 3,
                "text": "Hypertension management in adults over 60...",
                "entities": ["hypertension", "cardiology", "ACE inhibitor"],
                "char_start": 450,
                "char_end": 1620,
                "word_count": 187,
                "chunk_index_in_document": 42,
                "language": "en",
                "timestamp_indexed": "2026-04-22T10:30:00Z"
            }
        }
```

**Note:** `embedding` is NOT in the Pydantic schema — it lives in ChromaDB's vector store, referenced by `chunk_id`. Keeping Pydantic models lean avoids serialization bloat over WebSocket.

**Note on `relationships`:** Removed from schema. The original prompt specified it but co-occurrence edges are derived at graph-build time from `entities` lists across chunks — storing them per chunk is redundant and wastes metadata budget.

### ChromaDB Metadata (40KB Limit Handling)

ChromaDB has a ~40KB limit per metadata record. Storing full entity lists + text in metadata blows this fast. Solution: **metadata stores only what's needed for retrieval filtering; full chunk data lives in SQLite.**

```python
# ChromaDB metadata (stripped, filter-only)
{
    "chunk_id": "a3f9c2b1_0042",
    "document_id": "a3f9c2b1d4e5f6a7",
    "document_set_id": "550e8400-...",
    "page_number": 12,
    "language": "en",
    "entities_csv": "hypertension,cardiology,ACE inhibitor"  # CSV, max 500 chars, truncated if needed
}
# Full chunk text + complete data → SQLite, keyed by chunk_id
```

### Graph Node Schema

```python
# backend/schemas/graph.py
class GraphNode(BaseModel):
    id: str  # sha256(entity_label.lower())[:12]
    label: str  # canonical display form
    type: Literal["concept", "person", "organization", "location", "technique"]
    document_sources: List[str]  # unique document names
    chunk_ids: List[str]  # all chunks mentioning this entity
    frequency: int  # len(chunk_ids)
    is_active: bool = False  # set true during answer highlighting
```

### Graph Edge Schema

```python
class GraphEdge(BaseModel):
    source: str  # node_id
    target: str  # node_id
    relation_type: Literal["co_occurrence"] = "co_occurrence"
    weight: float  # normalized 0-1: (co_occur_count / max_co_occur_in_set)
    co_occurrence_count: int  # raw count
    is_active: bool = False
```

### Conversation Schema (SQLite)

```sql
-- backend/db/schema.sql
CREATE TABLE IF NOT EXISTS document_sets (
    id TEXT PRIMARY KEY,  -- UUID
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,  -- UUID from browser localStorage
    document_set_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_set_id) REFERENCES document_sets(id)
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,  -- UUID
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations_json TEXT,  -- JSON-serialized List[Citation]
    retrieved_chunk_ids_json TEXT,  -- JSON-serialized List[str]
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_set_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    paragraph_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    entities_json TEXT,  -- full entity list (no 500-char truncation here)
    char_start INTEGER,
    char_end INTEGER,
    word_count INTEGER,
    chunk_index_in_document INTEGER,
    language TEXT,
    timestamp_indexed TIMESTAMP
);

CREATE INDEX idx_chunks_doc_set ON chunks(document_set_id);
CREATE INDEX idx_chunks_doc ON chunks(document_id);
CREATE INDEX idx_turns_session ON turns(session_id, timestamp);
```

### Citation Schema

```python
class Citation(BaseModel):
    document_name: str
    page_number: int
    paragraph_index: int
    chunk_id: str
    relevance_score: float  # 0-1, from retrieval
    excerpt: str  # first 150 chars of chunk.text
```

### WebSocket Message Protocol

All WS messages share an envelope:
```typescript
interface WSMessage<T> {
    type: string;
    timestamp: string;  // ISO 8601
    payload: T;
}
```

**Message types:**

```typescript
// type: "index_progress"
{
    job_id: string,
    document_name: string,
    chunks_processed: number,
    total_chunks_estimated: number,
    stage: "extracting" | "chunking" | "embedding" | "storing" | "complete",
    percent: number
}

// type: "graph_update" (batched every 20 chunks or 500ms)
{
    document_set_id: string,
    added_nodes: GraphNode[],
    added_edges: GraphEdge[],
    updated_nodes: { id: string, frequency: number }[]
}

// type: "answer_stream"
{
    session_id: string,
    turn_id: string,
    token: string,  // partial text
    is_final: boolean
}

// type: "answer_complete"
{
    session_id: string,
    turn_id: string,
    full_answer: string,
    citations: Citation[],
    active_node_ids: string[],
    active_edge_ids: string[]  // format "source_id->target_id"
}

// type: "error"
{
    code: "INDEX_FAILED" | "LLM_FAILED" | "CITATION_HALLUCINATED" | "INVALID_PDF",
    message: string,
    recoverable: boolean
}
```

### 🔨 Build Checklist — Section 2

- [ ] Create `backend/schemas/` with `chunk.py`, `graph.py`, `conversation.py`, `websocket.py`
- [ ] Run `sqlite3 data/sqlite/querymind.db < backend/db/schema.sql`
- [ ] Write `backend/db/connection.py` with connection pool + migration runner
- [ ] Unit test: serialize/deserialize each Pydantic model, round-trip
- [ ] Unit test: insert chunk with max-size entity list, verify ChromaDB accepts it
- [ ] Document the WS protocol in `docs/websocket-protocol.md` for the frontend dev (you)

---

## SECTION 3 — Indexing Pipeline

### Chunking Strategy

**Parameters (locked):**
- Target chunk size: **900 characters** (~150 words, empirically optimal for multilingual-e5-small)
- Max chunk size: **1200 characters** (hard ceiling)
- Overlap: **150 characters** (~25 words)
- Boundary rule: **never split mid-paragraph if chunk would be >400 chars; split on sentence boundary otherwise**

**Chunking algorithm:**
```python
def chunk_page(page_text: str, page_number: int, doc_hash: str,
               start_index: int) -> List[Chunk]:
    paragraphs = re.split(r'\n\s*\n', page_text)
    chunks = []
    buffer = ""
    buffer_start_char = 0
    paragraph_idx = 0
    current_para_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph keeps us under 1200, append
        if len(buffer) + len(para) + 2 <= 1200:
            if buffer:
                buffer += "\n\n"
            buffer += para
        else:
            # Flush current buffer as chunk
            if buffer:
                chunks.append(make_chunk(buffer, page_number,
                    current_para_idx, buffer_start_char, doc_hash,
                    start_index + len(chunks)))

            # If paragraph itself > 1200, sentence-split
            if len(para) > 1200:
                chunks.extend(sentence_split_chunks(para, ...))
                buffer = ""
            else:
                buffer = para
                current_para_idx = paragraph_idx

        paragraph_idx += 1

    if buffer:
        chunks.append(make_chunk(buffer, ...))

    # Apply overlap: prepend last 150 chars of chunk[i-1] to chunk[i]
    return apply_overlap(chunks, overlap_chars=150)
```

### PyMuPDF Extraction with Page/Paragraph Accuracy

```python
import fitz  # PyMuPDF

def extract_pdf(path: str) -> List[Dict]:
    """Returns list of {page_number, text, blocks} with accurate indexing."""
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        # "text" mode preserves reading order; "blocks" gives paragraph structure
        text = page.get_text("text")
        blocks = page.get_text("blocks")  # [(x0,y0,x1,y1,text,block_no,block_type)]

        # Filter text blocks only (type 0 = text, type 1 = image)
        text_blocks = [b for b in blocks if b[6] == 0]

        pages.append({
            "page_number": page_num,
            "text": text,
            "blocks": [{"text": b[4], "bbox": b[:4]} for b in text_blocks]
        })
    doc.close()
    return pages
```

**Arabic/RTL note:** PyMuPDF's `"text"` mode handles Arabic shaping correctly as of v1.23+. Verify on Day 1 with a known Arabic PDF. Fallback: `pdfplumber` with `cluster_chars_by_line`.

### Entity Extraction (Speed-First)

Use spaCy rule-based + NER, NOT LLM-based, for 4-day timeline:

```python
import spacy

nlp_en = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
nlp_fr = spacy.load("fr_core_news_sm", disable=["parser", "lemmatizer"])
# Arabic: no good free spaCy model. Strategy: when lang=="ar",
# use entity list from retrieved chunks (English entities extracted from
# Arabic chunks via lang-detect on entity surface forms that transliterate).
# For v1: Arabic chunks contribute chunks to retrieval but don't add new nodes.

ENTITY_TYPE_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location", "LOC": "location",
    "PRODUCT": "concept", "WORK_OF_ART": "concept",
    "EVENT": "concept", "LAW": "concept",
}

def extract_entities(text: str, lang: str) -> List[Tuple[str, str]]:
    """Returns [(label, type), ...] deduped, max 15."""
    if lang == "en":
        doc = nlp_en(text)
    elif lang == "fr":
        doc = nlp_fr(text)
    else:
        return []  # Arabic: no nodes added from this chunk

    entities = []
    seen = set()
    for ent in doc.ents:
        label = ent.text.strip()
        if len(label) < 3 or len(label) > 60:
            continue
        if label.lower() in seen:
            continue
        etype = ENTITY_TYPE_MAP.get(ent.label_, "concept")
        entities.append((label, etype))
        seen.add(label.lower())
        if len(entities) >= 15:
            break
    return entities
```

### ChromaDB Collection (Per Document Set)

```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./data/chroma",
    settings=Settings(anonymized_telemetry=False)
)

def get_or_create_collection(document_set_id: str):
    return client.get_or_create_collection(
        name=f"set_{document_set_id}",
        metadata={"hnsw:space": "cosine"}
    )
```

### Duplicate Prevention (Watchdog)

```python
# backend/indexing/watcher.py
from watchdog.events import FileSystemEventHandler
import hashlib

class PDFWatcher(FileSystemEventHandler):
    def __init__(self, indexer):
        self.indexer = indexer
        self.processing = set()  # in-flight file paths

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return
        file_hash = self._hash_file(event.src_path)
        if self.indexer.document_exists(file_hash):
            print(f"Skip: {event.src_path} already indexed")
            return
        if file_hash in self.processing:
            return
        self.processing.add(file_hash)
        try:
            self.indexer.index_document(event.src_path, file_hash)
        finally:
            self.processing.discard(file_hash)

    def _hash_file(self, path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
```

**Duplicate check in indexer:**
```python
def document_exists(self, doc_id: str) -> bool:
    cur = self.db.execute(
        "SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1", (doc_id,))
    return cur.fetchone() is not None
```

### Embedding with MPS Acceleration

```python
from sentence_transformers import SentenceTransformer
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)

# e5 requires "passage: " prefix for documents, "query: " for queries
def embed_chunks(texts: List[str]) -> List[List[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=16,  # M1 8GB sweet spot
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
```

### Progress Streaming (Batched)

```python
class IndexingProgressEmitter:
    def __init__(self, ws_manager, job_id):
        self.ws = ws_manager
        self.job_id = job_id
        self.pending_nodes = []
        self.pending_edges = []
        self.last_emit = time.time()

    async def on_chunk_indexed(self, chunk, new_entities, new_edges):
        self.pending_nodes.extend(new_entities)
        self.pending_edges.extend(new_edges)

        # Emit every 20 chunks or 500ms
        if len(self.pending_nodes) >= 20 or \
           (time.time() - self.last_emit) > 0.5:
            await self.ws.broadcast({
                "type": "graph_update",
                "payload": {
                    "added_nodes": self.pending_nodes,
                    "added_edges": self.pending_edges,
                    ...
                }
            })
            self.pending_nodes = []
            self.pending_edges = []
            self.last_emit = time.time()
```

### Realistic Performance on M1 Air 8GB

| Stage | Time (130-page PDF, ~800 chunks) |
|---|---|
| PyMuPDF extraction | 3-5s |
| Chunking | <1s |
| spaCy NER (all chunks) | 15-25s |
| Embedding (MPS, batch 16) | 45-70s |
| ChromaDB insert | 5-10s |
| SQLite insert | 2-3s |
| **Total** | **~75-115s per 130-page PDF** |

**If embedding dominates on CPU-only (MPS unavailable):** 3-4x slower, ~4-5 min. If this happens, batch size drops to 8 and indexing runs async (see Section 5).

### 🔨 Build Checklist — Section 3

- [ ] `backend/indexing/extractor.py` — PyMuPDF extractor with page/paragraph indexing
- [ ] `backend/indexing/chunker.py` — 900-char chunker with 150-char overlap
- [ ] `backend/indexing/entity_extractor.py` — spaCy EN/FR wrapper
- [ ] `backend/indexing/embedder.py` — e5 model loader with MPS
- [ ] `backend/indexing/watcher.py` — watchdog file observer
- [ ] `backend/indexing/pipeline.py` — orchestrator calling all above
- [ ] Test: index one 130-page English PDF, measure time, verify chunk count
- [ ] Test: index one Arabic PDF, verify text integrity via SQL inspection
- [ ] Test: drop a duplicate PDF in watch folder, verify skip

---

## SECTION 4 — Retrieval and Answer Generation

### Retrieval Pipeline (Vector-Only for v1)

**Hybrid BM25 + vector:** skip for v1. BM25 adds index complexity and multilingual BM25 tokenization is non-trivial for Arabic. Vector similarity with e5 is strong enough for portfolio demo.

**Reranking:** skip. Cross-encoder rerankers add 200-500ms latency and require another model load. Ship v1 without; add in v1.1 if citation precision is weak.

```python
def retrieve(query: str, document_set_id: str, k: int = 8) -> List[RetrievedChunk]:
    query_embedding = model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    collection = get_or_create_collection(document_set_id)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "distances"]
    )
    # Hydrate full chunk data from SQLite using chunk_ids
    chunk_ids = [m["chunk_id"] for m in results["metadatas"][0]]
    full_chunks = sqlite_fetch_chunks(chunk_ids)

    # relevance_score = 1 - cosine_distance
    for chunk, dist in zip(full_chunks, results["distances"][0]):
        chunk.relevance_score = max(0, 1 - dist)
    return full_chunks
```

**k=8 justification:** Gemini free tier has ~1M token context, but injecting 8 chunks × ~900 chars + system prompt + history fits comfortably and keeps the LLM focused. More than 10 chunks degrades citation precision empirically.

### THE CITATION PROMPT (Most Critical)

```python
CITATION_SYSTEM_PROMPT = """You are QueryMind, a document intelligence assistant. You answer questions using ONLY the provided document chunks. Every factual claim MUST be followed by a citation in the exact format: [CITE: chunk_id]

RULES (non-negotiable):
1. Use ONLY chunks provided below. Do not use outside knowledge.
2. Every sentence containing a fact MUST end with [CITE: chunk_id] where chunk_id matches EXACTLY one of the provided IDs.
3. If the chunks do not contain the answer, respond: "The provided documents do not contain information to answer this question."
4. Multiple citations allowed: [CITE: abc_0001] [CITE: abc_0042]
5. Respond in the same language as the user's question.
6. Do not invent chunk_ids. Do not modify chunk_ids. Copy them character-for-character.

AVAILABLE CHUNKS:
{retrieved_chunks_formatted}

CONVERSATION HISTORY (most recent):
{history_formatted}
"""

CHUNK_INJECTION_FORMAT = """
[CHUNK_ID: {chunk_id}]
[SOURCE: {document_name}, page {page_number}, paragraph {paragraph_index}]
[LANGUAGE: {language}]
---
{text}
---
"""
```

**Example of rendered prompt section:**
```
[CHUNK_ID: a3f9c2b1_0042]
[SOURCE: cardiology_2024.pdf, page 12, paragraph 3]
[LANGUAGE: en]
---
Hypertension in adults over 60 requires first-line ACE inhibitors...
---
```

### History Injection (Token-Budget Sliding Window)

```python
def format_history(session_id: str, max_tokens: int = 2000) -> str:
    turns = fetch_turns(session_id, limit=20)  # newest-first
    # Rough estimate: 1 token ≈ 4 chars for English, 2 chars for Arabic
    formatted = []
    budget = max_tokens * 4  # char budget
    for turn in turns:
        line = f"{turn.role.upper()}: {turn.content}"
        if budget - len(line) < 0:
            break
        formatted.insert(0, line)  # maintain chronological order
        budget -= len(line)
    return "\n".join(formatted) if formatted else "(no prior conversation)"
```

### Citation Parsing

```python
import re

CITATION_REGEX = re.compile(r'\[CITE:\s*([a-f0-9_]+)\]', re.IGNORECASE)

def parse_citations(answer_text: str, retrieved_chunks: List[Chunk]) -> Tuple[str, List[Citation], List[str]]:
    """
    Returns: (cleaned_answer, valid_citations, hallucinated_ids)
    """
    retrieved_map = {c.chunk_id: c for c in retrieved_chunks}
    cited_ids = CITATION_REGEX.findall(answer_text)

    valid = []
    hallucinated = []
    for cid in cited_ids:
        if cid in retrieved_map:
            chunk = retrieved_map[cid]
            valid.append(Citation(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                paragraph_index=chunk.paragraph_index,
                chunk_id=chunk.chunk_id,
                relevance_score=chunk.relevance_score,
                excerpt=chunk.text[:150]
            ))
        else:
            hallucinated.append(cid)

    return answer_text, valid, hallucinated
```

### Hallucination Detection & Recovery

```python
async def generate_answer_with_validation(query, chunks, history, max_retries=1):
    for attempt in range(max_retries + 1):
        answer = await llm_call(query, chunks, history,
                                strict_mode=(attempt > 0))
        _, valid, hallucinated = parse_citations(answer, chunks)

        if not hallucinated:
            return answer, valid, []

        if attempt < max_retries:
            # Retry with stricter prompt
            continue
        else:
            # Strip hallucinated citations, flag answer
            for bad_id in hallucinated:
                answer = answer.replace(f"[CITE: {bad_id}]", "[citation omitted]")
            return answer, valid, hallucinated
```

### Active Graph Path Computation

After answer is finalized, compute which graph nodes/edges to highlight:

```python
def compute_active_graph(valid_citations, graph_nodes, graph_edges):
    cited_chunk_ids = {c.chunk_id for c in valid_citations}
    active_node_ids = [
        n.id for n in graph_nodes
        if any(cid in n.chunk_ids for cid in cited_chunk_ids)
    ]
    active_node_set = set(active_node_ids)
    active_edge_ids = [
        f"{e.source}->{e.target}" for e in graph_edges
        if e.source in active_node_set and e.target in active_node_set
    ]
    return active_node_ids, active_edge_ids
```

### 🔨 Build Checklist — Section 4

- [ ] `backend/rag/retriever.py` — vector retrieval with SQLite hydration
- [ ] `backend/rag/prompt.py` — system prompt + formatters
- [ ] `backend/rag/llm_client.py` — Gemini wrapper with streaming
- [ ] `backend/rag/citation_parser.py` — regex + validator
- [ ] `backend/rag/answer_generator.py` — orchestrator with retry loop
- [ ] Test: inject 3 known chunks, ask question, assert citations match chunks
- [ ] Test: ask an adversarial question ("What is the capital of Mars?"), assert refusal response
- [ ] Test: Arabic question on French document, assert Arabic answer with French citations

---

## SECTION 5 — Backend API Specification

### FastAPI Routes

```python
# backend/main.py (structure)
from fastapi import FastAPI, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="QueryMind API", version="1.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://ghaith.com"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
```

### POST /documents/upload

```python
@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    document_set_id: str = Form(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:  # 50MB hard limit
        raise HTTPException(413, "File exceeds 50MB")
    if len(contents) < 100:
        raise HTTPException(400, "File appears empty")

    # Save to watched folder
    save_path = f"./data/pdfs/{document_set_id}/{file.filename}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(contents)

    job_id = str(uuid.uuid4())
    # Fire-and-forget background task
    asyncio.create_task(
        indexing_pipeline.run(save_path, document_set_id, job_id)
    )
    return UploadResponse(job_id=job_id, filename=file.filename,
                          status="queued")
```

**Schemas:**
```python
class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: Literal["queued", "indexing", "complete", "failed"]
```

### POST /documents/set

```python
@app.post("/documents/set", response_model=DocumentSet)
def create_document_set(req: CreateSetRequest):
    set_id = str(uuid.uuid4())
    db.execute("INSERT INTO document_sets (id, name) VALUES (?, ?)",
               (set_id, req.name))
    return DocumentSet(id=set_id, name=req.name, chunk_count=0,
                       created_at=datetime.utcnow())

class CreateSetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
```

### GET /documents/sets

```python
@app.get("/documents/sets", response_model=List[DocumentSetWithStats])
def list_sets():
    rows = db.execute("""
        SELECT ds.id, ds.name, ds.created_at,
               COUNT(DISTINCT c.document_id) AS doc_count,
               COUNT(c.chunk_id) AS chunk_count
        FROM document_sets ds
        LEFT JOIN chunks c ON c.document_set_id = ds.id
        GROUP BY ds.id
        ORDER BY ds.created_at DESC
    """).fetchall()
    return [DocumentSetWithStats(**dict(r)) for r in rows]
```

### DELETE /documents/{doc_id}

```python
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    # Get set_id first for collection cleanup
    row = db.execute(
        "SELECT document_set_id FROM chunks WHERE document_id = ? LIMIT 1",
        (doc_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Document not found")

    set_id = row[0]
    # Get all chunk_ids for this doc
    chunk_ids = [r[0] for r in db.execute(
        "SELECT chunk_id FROM chunks WHERE document_id = ?", (doc_id,))]

    # Remove from ChromaDB
    collection = get_or_create_collection(set_id)
    collection.delete(ids=chunk_ids)

    # Remove from SQLite
    db.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    db.commit()
    return {"deleted": True, "chunks_removed": len(chunk_ids)}
```

### POST /chat

```python
@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Triggers RAG pipeline. Streams answer via WebSocket (session_id keyed).
    Returns immediately with turn_id so frontend can correlate WS events.
    """
    turn_id = str(uuid.uuid4())
    asyncio.create_task(
        run_rag_pipeline(req.session_id, req.document_set_id,
                         req.message, turn_id)
    )
    return {"turn_id": turn_id, "status": "streaming"}

class ChatRequest(BaseModel):
    session_id: str
    document_set_id: str
    message: str = Field(..., min_length=1, max_length=2000)
```

### GET /chat/history/{session_id}

```python
@app.get("/chat/history/{session_id}", response_model=List[Turn])
def get_history(session_id: str, limit: int = 50):
    rows = db.execute("""
        SELECT turn_id, role, content, citations_json,
               retrieved_chunk_ids_json, timestamp
        FROM turns
        WHERE session_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
    """, (session_id, limit)).fetchall()
    return [Turn.from_row(r) for r in rows]
```

### GET /graph/{document_set_id}

```python
@app.get("/graph/{document_set_id}", response_model=GraphResponse)
def get_graph(document_set_id: str,
              min_frequency: int = 1,
              entity_types: Optional[List[str]] = Query(None)):
    nodes, edges = graph_service.build_graph(document_set_id,
                                             min_frequency, entity_types)
    return GraphResponse(nodes=nodes, edges=edges,
                         node_count=len(nodes), edge_count=len(edges))
```

### GET /health

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chromadb": chromadb_client.heartbeat() is not None,
        "sqlite": db.execute("SELECT 1").fetchone() is not None,
        "llm_configured": bool(os.getenv("GEMINI_API_KEY")),
    }
```

### WebSocket /ws/{session_id}

```python
class WSManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.connections[session_id] = ws

    def disconnect(self, session_id: str):
        self.connections.pop(session_id, None)

    async def send(self, session_id: str, message: dict):
        ws = self.connections.get(session_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(session_id)

    async def broadcast_to_set(self, document_set_id: str, message: dict):
        # For graph updates — find all sessions on this set
        for sid, ws in list(self.connections.items()):
            # simple impl: broadcast to all; frontend filters
            await self.send(sid, message)

ws_manager = WSManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws_manager.connect(session_id, ws)
    try:
        while True:
            # Client can send heartbeat or explicit disconnect
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except Exception:
        ws_manager.disconnect(session_id)
```

### Indexing: Async vs Sync Decision

**Async (background task), justification:**
- Indexing a 130-page PDF takes ~90s. Blocking the HTTP request is unacceptable.
- FastAPI's `asyncio.create_task` is sufficient — no need for Celery/Redis (4-day build, no cloud services).
- Progress is pushed via WebSocket, so the frontend UX stays responsive.
- Risk: if the server restarts mid-indexing, that job is lost. Mitigation: on startup, scan `data/pdfs/` for PDFs without corresponding SQLite entries → re-queue.

### File Validation

```python
def validate_pdf(contents: bytes) -> None:
    if contents[:5] != b"%PDF-":
        raise HTTPException(400, "Not a valid PDF (missing magic bytes)")
    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        if doc.page_count == 0:
            raise HTTPException(400, "PDF has no pages")
        if doc.page_count > 500:
            raise HTTPException(413, "PDF exceeds 500 pages")
        doc.close()
    except Exception as e:
        raise HTTPException(400, f"Corrupt PDF: {e}")
```

### 🔨 Build Checklist — Section 5

- [ ] `backend/main.py` — FastAPI app + CORS
- [ ] `backend/routes/documents.py` — upload, set, delete, list
- [ ] `backend/routes/chat.py` — chat, history
- [ ] `backend/routes/graph.py` — graph endpoint
- [ ] `backend/routes/health.py`
- [ ] `backend/ws/manager.py` — WSManager class
- [ ] `backend/services/orphan_scanner.py` — startup re-queue logic
- [ ] Test: `curl` upload a PDF, watch indexing complete via logs
- [ ] Test: `websocat ws://localhost:8000/ws/test` + upload → see graph_update events

---

## SECTION 6 — Frontend Specification

### Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   ├── client.ts          # axios wrapper
│   │   └── endpoints.ts       # typed route calls
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useSession.ts
│   │   └── useGraph.ts
│   ├── components/
│   │   ├── ChatPanel.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── CitationBadge.tsx
│   │   ├── SourcePanel.tsx
│   │   ├── KnowledgeGraph.tsx    # D3 container
│   │   ├── UploadZone.tsx
│   │   └── DocumentSetSelector.tsx
│   ├── pages/
│   │   ├── UploadPage.tsx
│   │   ├── ChatPage.tsx
│   │   └── DocumentsPage.tsx
│   ├── store/
│   │   └── appStore.ts        # Zustand store
│   ├── types/
│   │   └── api.ts             # mirrors backend schemas
│   └── App.tsx
├── tailwind.config.js
└── vite.config.ts
```

### State Management: Zustand (not Redux)

Redux is overkill for 4 days. Zustand is 1KB and covers all needs:

```typescript
// src/store/appStore.ts
import { create } from 'zustand';

interface AppStore {
  sessionId: string;
  currentSetId: string | null;
  messages: Message[];
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  activeNodeIds: Set<string>;
  activeEdgeIds: Set<string>;
  indexingJobs: Record<string, IndexProgress>;

  addMessage: (m: Message) => void;
  appendStreamingToken: (turnId: string, token: string) => void;
  mergeGraphUpdate: (update: GraphUpdate) => void;
  setActivePath: (nodes: string[], edges: string[]) => void;
  clearActivePath: () => void;
}

export const useAppStore = create<AppStore>((set, get) => ({
  sessionId: localStorage.getItem('qm_session') || crypto.randomUUID(),
  // ... implementation
  mergeGraphUpdate: (update) => set((state) => ({
    graphNodes: mergeNodes(state.graphNodes, update.added_nodes, update.updated_nodes),
    graphEdges: mergeEdges(state.graphEdges, update.added_edges),
  })),
}));
```

### D3.js Force-Directed Graph

```typescript
// src/components/KnowledgeGraph.tsx
import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

export function KnowledgeGraph({ nodes, edges, activeNodeIds, activeEdgeIds, onNodeClick }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<any, any> | null>(null);

  // Initialize once
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const width = svgRef.current!.clientWidth;
    const height = svgRef.current!.clientHeight;

    svg.selectAll('*').remove();

    // Zoom behavior
    const g = svg.append('g');
    svg.call(d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on('zoom', (e) => g.attr('transform', e.transform))
    );

    // Containers
    g.append('g').attr('class', 'edges');
    g.append('g').attr('class', 'nodes');

    // Force simulation (tuned for readability)
    simulationRef.current = d3.forceSimulation()
      .force('link', d3.forceLink().id((d: any) => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => 8 + d.frequency * 2))
      .alphaDecay(0.02); // slower decay = smoother incremental adds
  }, []);

  // Update on data change
  useEffect(() => {
    if (!simulationRef.current) return;
    const svg = d3.select(svgRef.current);
    const sim = simulationRef.current;

    // EDGES
    const edgeSel = svg.select('g.edges')
      .selectAll('line')
      .data(edges, (d: any) => `${d.source.id || d.source}->${d.target.id || d.target}`);

    edgeSel.exit().transition().duration(300).attr('opacity', 0).remove();

    const edgeEnter = edgeSel.enter().append('line')
      .attr('stroke', 'var(--edge-color)')
      .attr('stroke-width', (d: any) => 0.5 + d.weight * 2)
      .attr('opacity', 0);
    edgeEnter.transition().duration(500).attr('opacity', (d: any) => 0.3 + d.weight * 0.5);

    const allEdges = edgeEnter.merge(edgeSel as any)
      .classed('edge-active', (d: any) =>
        activeEdgeIds.has(`${d.source.id || d.source}->${d.target.id || d.target}`));

    // NODES
    const nodeSel = svg.select('g.nodes')
      .selectAll('g.node')
      .data(nodes, (d: any) => d.id);

    nodeSel.exit().transition().duration(300).attr('opacity', 0).remove();

    const nodeEnter = nodeSel.enter().append('g')
      .attr('class', 'node')
      .attr('opacity', 0)
      .call(d3.drag<any, any>()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      )
      .on('click', (e, d) => onNodeClick(d));

    nodeEnter.append('circle')
      .attr('r', (d: any) => 6 + Math.min(d.frequency, 10) * 1.5)
      .attr('fill', (d: any) => nodeColorByType(d.type));

    nodeEnter.append('text')
      .text((d: any) => d.label)
      .attr('dy', (d: any) => -(12 + Math.min(d.frequency, 10) * 1.5))
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-1)')
      .attr('font-size', '11px')
      .attr('font-family', 'var(--font-mono)');

    nodeEnter.transition().duration(500).attr('opacity', 1);

    const allNodes = nodeEnter.merge(nodeSel as any)
      .classed('node-active', (d: any) => activeNodeIds.has(d.id));

    // Update simulation
    sim.nodes(nodes);
    (sim.force('link') as any).links(edges);
    sim.alpha(0.3).restart();

    sim.on('tick', () => {
      allEdges
        .attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y);
      allNodes.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });
  }, [nodes, edges, activeNodeIds, activeEdgeIds]);

  return <svg ref={svgRef} className="w-full h-full" />;
}
```

### Active Node Glow (CSS)

```css
/* src/styles/graph.css */
.node-active circle {
  filter: drop-shadow(0 0 8px var(--accent-glow))
          drop-shadow(0 0 16px var(--accent-glow));
  animation: pulse 1.6s ease-in-out infinite;
}

.edge-active {
  stroke: var(--accent-glow) !important;
  stroke-width: 2.5 !important;
  opacity: 1 !important;
  filter: drop-shadow(0 0 4px var(--accent-glow));
}

@keyframes pulse {
  0%, 100% { filter: drop-shadow(0 0 8px var(--accent-glow))
                     drop-shadow(0 0 16px var(--accent-glow)); }
  50%      { filter: drop-shadow(0 0 12px var(--accent-glow))
                     drop-shadow(0 0 24px var(--accent-glow)); }
}
```

### Citation Rendering

```typescript
// src/components/MessageBubble.tsx
function renderContentWithCitations(content: string, citations: Citation[]) {
  const parts = content.split(/(\[CITE:\s*[a-f0-9_]+\])/gi);
  const cMap = Object.fromEntries(citations.map(c => [c.chunk_id, c]));

  return parts.map((part, i) => {
    const match = part.match(/\[CITE:\s*([a-f0-9_]+)\]/i);
    if (match) {
      const citation = cMap[match[1]];
      if (!citation) return null; // hallucinated, stripped
      return <CitationBadge key={i} citation={citation} />;
    }
    return <span key={i}>{part}</span>;
  });
}
```

```typescript
// src/components/CitationBadge.tsx
export function CitationBadge({ citation }: { citation: Citation }) {
  const [hovering, setHovering] = useState(false);
  return (
    <span
      className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 rounded-md
                 bg-[var(--accent-subtle)] text-[var(--accent)]
                 border border-[var(--accent-border)]
                 font-mono text-xs cursor-pointer hover:bg-[var(--accent-hover)]"
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
      onClick={() => useAppStore.getState().openSourcePanel(citation.chunk_id)}
    >
      📄 p.{citation.page_number}
      {hovering && (
        <div className="absolute z-50 mt-6 p-3 max-w-sm bg-[var(--surface-2)]
                        border border-[var(--border)] rounded-lg shadow-xl text-xs">
          <div className="text-[var(--text-2)] font-mono mb-1">
            {citation.document_name} · p.{citation.page_number} · ¶{citation.paragraph_index}
          </div>
          <div className="text-[var(--text-1)]">{citation.excerpt}...</div>
        </div>
      )}
    </span>
  );
}
```

### useWebSocket Hook

```typescript
// src/hooks/useWebSocket.ts
export function useWebSocket(sessionId: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const store = useAppStore();

  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case 'index_progress':
          store.updateIndexProgress(msg.payload);
          break;
        case 'graph_update':
          store.mergeGraphUpdate(msg.payload);
          break;
        case 'answer_stream':
          store.appendStreamingToken(msg.payload.turn_id, msg.payload.token);
          break;
        case 'answer_complete':
          store.finalizeAnswer(msg.payload);
          store.setActivePath(msg.payload.active_node_ids, msg.payload.active_edge_ids);
          setTimeout(() => store.clearActivePath(), 6000);
          break;
        case 'error':
          store.showError(msg.payload);
          break;
      }
    };

    ws.onclose = () => {
      // Reconnect with backoff
      setTimeout(() => { /* recreate */ }, 2000);
    };

    const heartbeat = setInterval(() => ws.send('ping'), 30000);
    return () => { clearInterval(heartbeat); ws.close(); };
  }, [sessionId]);
}
```

### Upload UX

```typescript
// src/components/UploadZone.tsx
export function UploadZone({ documentSetId }) {
  const [files, setFiles] = useState<UploadFile[]>([]);

  const onDrop = async (acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map(f => ({
      id: crypto.randomUUID(),
      file: f,
      status: 'pending',
      progress: 0,
    }));
    setFiles(prev => [...prev, ...newFiles]);

    for (const uf of newFiles) {
      const fd = new FormData();
      fd.append('file', uf.file);
      fd.append('document_set_id', documentSetId);
      await axios.post('/documents/upload', fd, {
        onUploadProgress: (e) => {
          const pct = Math.round((e.loaded * 100) / (e.total || 1));
          setFiles(prev => prev.map(f => f.id === uf.id ? { ...f, progress: pct } : f));
        },
      });
      // After upload, indexing progress comes via WebSocket
    }
  };

  // ... react-dropzone setup, render progress bars
}
```

### Source Panel

Sidebar that opens when a citation is clicked. Shows the full chunk with the excerpt highlighted, document context (page, paragraph), and a "jump to next citation" button.

### 🔨 Build Checklist — Section 6

- [ ] `npm create vite@latest frontend -- --template react-ts`
- [ ] Install: `tailwindcss d3 zustand axios react-dropzone react-router-dom`
- [ ] `tailwind.config.js` with custom CSS variables
- [ ] Build `KnowledgeGraph.tsx` with a hard-coded 10-node test dataset first
- [ ] Build `ChatPanel.tsx` with mock messages + citations before wiring API
- [ ] Build `useWebSocket.ts` with reconnect logic
- [ ] Wire Zustand store, connect to real backend
- [ ] Test: upload PDF, watch graph grow in real-time, ask question, watch glow

---

## SECTION 7 — Knowledge Graph Design

### Entity Extraction by Language

| Language | Model | Notes |
|---|---|---|
| English | spaCy `en_core_web_sm` | PERSON, ORG, GPE, LOC, PRODUCT, EVENT |
| French | spaCy `fr_core_news_sm` | Same tags, coverage slightly weaker |
| Arabic | **None (by design)** | Arabic chunks contribute to retrieval; entity extraction skipped. See Decision 3: graph stays English/French for visual clarity. |

### Entity Types → Node Colors

```javascript
const nodeColorByType = (type) => ({
  'person':       '#c084fc',  // violet-400
  'organization': '#60a5fa',  // blue-400
  'location':     '#34d399',  // emerald-400
  'concept':      '#fbbf24',  // amber-400
  'technique':    '#f87171',  // red-400
}[type] || '#94a3b8');        // slate-400 fallback
```

### Co-Occurrence Edge Computation

```python
def compute_edges_for_chunk(chunk: Chunk, all_nodes: Dict[str, GraphNode]):
    """Called per chunk during indexing."""
    entity_ids = [entity_to_node_id(e) for e in chunk.entities]
    new_edges = []
    for i, src in enumerate(entity_ids):
        for tgt in entity_ids[i+1:]:
            edge_key = tuple(sorted([src, tgt]))
            existing = edge_store.get(edge_key)
            if existing:
                existing.co_occurrence_count += 1
            else:
                new_edges.append(GraphEdge(
                    source=edge_key[0], target=edge_key[1],
                    co_occurrence_count=1, weight=0.0
                ))
    return new_edges

def normalize_weights(edges: List[GraphEdge]):
    max_co = max(e.co_occurrence_count for e in edges) if edges else 1
    for e in edges:
        e.weight = e.co_occurrence_count / max_co
```

### D3 Force Parameters (Tuned)

| Parameter | Value | Rationale |
|---|---|---|
| `forceLink.distance` | 80 | Enough spacing to read labels |
| `forceManyBody.strength` | -200 | Moderate repulsion; higher → fly apart on small graphs |
| `forceCollide.radius` | `8 + frequency * 2` | Prevents label overlap on hubs |
| `alphaDecay` | 0.02 | Slow = smoother incremental updates |
| `velocityDecay` | 0.4 (default) | Standard |

### Graph Evolution Animation

- New nodes fade in over 500ms (`opacity: 0 → 1`)
- New edges fade in over 500ms
- Simulation `alpha` bumped to 0.3 on each update (not restart from 1.0) — smoother

### Filtering Controls

Top-right overlay on graph canvas:
```
┌──────────────────────────────┐
│ 🔍 Filter                    │
│ Min frequency: [====●====] 2 │
│ ☑ Concept  ☑ Person          │
│ ☑ Organization  ☐ Location   │
│ ☑ Technique                  │
│ Document: [All ▼]            │
└──────────────────────────────┘
```

Filtering calls `GET /graph/{set_id}?min_frequency=2&entity_types=concept,person`. Backend returns filtered graph; frontend replaces nodes/edges. Active path preserved if nodes still visible.

### Answer Path Highlighting

- Duration: **6 seconds** after `answer_complete`
- Behavior: Active nodes pulse with glow; active edges brighten; non-active fade to 40% opacity
- Reset: timer auto-clears; or on new question, clear immediately

### Graph as Navigation

Click on node → opens side panel listing all chunks where entity appears, each clickable → opens source panel with chunk text.

### Large Graph Handling (1000+ Nodes)

At 1000+ nodes, D3 force layout becomes sluggish. Strategy:
1. Default filter: `min_frequency ≥ 2` (removes single-occurrence noise, usually cuts 60%)
2. If still >500 nodes: show top 300 by frequency, display "showing top 300 of 1247 — adjust filter to see more"
3. Do not implement clustering/community detection in v1 (too much work).

### 🔨 Build Checklist — Section 7

- [ ] `backend/graph/builder.py` — co-occurrence edge computation
- [ ] `backend/graph/filters.py` — frequency + type filtering
- [ ] Test: index 3 PDFs, verify node count roughly matches manual entity spotting
- [ ] Tune D3 forces interactively until readable at 50, 200, 500 nodes

---

## SECTION 8 — Conversation Memory Architecture

### Session Identity

```typescript
// src/hooks/useSession.ts
export function useSession() {
  const [sessionId] = useState(() => {
    const existing = localStorage.getItem('qm_session');
    if (existing) return existing;
    const fresh = crypto.randomUUID();
    localStorage.setItem('qm_session', fresh);
    return fresh;
  });
  return sessionId;
}
```

One session per browser, persists across refreshes. User can "reset session" via button → clears localStorage + calls `DELETE /chat/session/{id}`.

### History Injection Strategy: Token-Budget Sliding Window

**Not "last N turns"** — a single long answer can eat the budget. Instead:

```python
MAX_HISTORY_TOKENS = 2000

def build_history_window(session_id: str) -> List[Turn]:
    turns = fetch_turns(session_id, order_desc=True)  # newest first
    selected = []
    budget = MAX_HISTORY_TOKENS

    for turn in turns:
        estimated = estimate_tokens(turn.content)
        if estimated > budget:
            break
        selected.append(turn)
        budget -= estimated

    return list(reversed(selected))  # restore chronological order

def estimate_tokens(text: str) -> int:
    # Rough: 1 token ≈ 4 chars for latin, 2 chars for arabic
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin_chars = len(text) - arabic_chars
    return latin_chars // 4 + arabic_chars // 2
```

### Cross-Turn Reasoning

The LLM has the history in context. For questions like "and what about the second study?", the prompt includes the previous turn's answer, so the LLM naturally has reference. No special handling needed — this is the strength of sticking with simple LangChain over custom reasoning.

### Document Set Scoping

Each session is bound to one active `document_set_id`. Switching sets:
1. Frontend calls `POST /chat/session/switch` with new `set_id`
2. Backend creates a fresh logical conversation (new turns from here on are scoped to new set)
3. Old turns remain queryable for history viewing but aren't injected into prompt for new set

**Implementation shortcut:** Keep it simple — when user switches sets, create a new `session_id` (new UUID) and update localStorage. Old session remains in DB but frontend no longer uses it. Cleanup via cron/manual.

### Session Cleanup

On startup:
```sql
DELETE FROM turns WHERE session_id IN (
    SELECT id FROM sessions WHERE last_activity < datetime('now', '-30 days')
);
DELETE FROM sessions WHERE last_activity < datetime('now', '-30 days');
```

No user-facing cleanup UI for v1.

### 🔨 Build Checklist — Section 8

- [ ] `backend/services/conversation.py` — history window builder
- [ ] `backend/services/session.py` — session create/fetch/touch
- [ ] `frontend/src/hooks/useSession.ts`
- [ ] Test: ask 3 related questions, verify 2nd and 3rd include prior context
- [ ] Test: switch document set, verify conversation context resets

---

## SECTION 9 — Visual Design System

### Typography

- **Display:** `Inter` (400, 500, 600, 700) — UI, chat content
- **Mono (citations + chunk text):** `JetBrains Mono` (400, 500) — citation badges, source panel chunks, graph node labels
- **Arabic fallback:** `Noto Sans Arabic` (400, 500, 700)

Load via Google Fonts in `index.html`:
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+Arabic:wght@400;500;700&display=swap" rel="stylesheet">
```

### CSS Variables (Dark Aesthetic)

```css
/* src/styles/tokens.css */
:root {
  /* Backgrounds */
  --bg-0: #0a0e1a;          /* page background, near-black navy */
  --bg-1: #0f1525;          /* chat area */
  --surface-1: #151c30;     /* cards, message bubbles */
  --surface-2: #1d2540;     /* hover, elevated */
  --surface-3: #252d4a;     /* active states */

  /* Borders */
  --border: #2a3352;
  --border-subtle: #1e2640;

  /* Text */
  --text-0: #f1f5ff;        /* primary */
  --text-1: #b8c2e0;        /* secondary */
  --text-2: #6e7a9f;        /* tertiary, metadata */
  --text-muted: #4a5478;

  /* Accent (purple-cyan gradient theme) */
  --accent: #8b5cf6;        /* violet-500, primary action */
  --accent-hover: #a78bfa;
  --accent-subtle: rgba(139, 92, 246, 0.1);
  --accent-border: rgba(139, 92, 246, 0.3);
  --accent-glow: #c4b5fd;   /* violet-300, for graph glow */

  /* Semantic */
  --success: #10b981;
  --warning: #f59e0b;
  --error: #ef4444;

  /* Graph specifics */
  --edge-color: rgba(110, 122, 159, 0.3);
  --edge-hover: rgba(196, 181, 253, 0.6);

  /* Fonts */
  --font-display: 'Inter', 'Noto Sans Arabic', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.4);
  --shadow-glow: 0 0 24px rgba(139, 92, 246, 0.35);
}
```

### Visual Signature Element

**The graph's answer-path pulse.** When an answer completes, the contributing nodes pulse violet over black for 6 seconds with the edges between them brightening simultaneously. This is the single screen-recordable moment that says "this is not ChatPDF." Tune the pulse to feel like a signal traveling through the graph.

### Chat Bubble Design

```
┌─────────────────────────────────────────────┐
│ You                                         │
│ ┌───────────────────────────────────────┐   │
│ │ What are the side effects of         │   │
│ │ ACE inhibitors?                      │   │
│ └───────────────────────────────────────┘   │
│                                             │
│ QueryMind                                   │
│ ┌───────────────────────────────────────┐   │
│ │ ACE inhibitors commonly cause dry    │   │
│ │ cough [📄 p.12] and may cause        │   │
│ │ hyperkalemia in renal patients       │   │
│ │ [📄 p.14].                           │   │
│ │                                      │   │
│ │ In elderly patients, monitoring     │   │
│ │ creatinine is essential [📄 p.28].  │   │
│ └───────────────────────────────────────┘   │
│ ⚡ Answered from 3 sources · 1.2s           │
└─────────────────────────────────────────────┘
```

- User bubbles: `--surface-1`, right-aligned, max-width 70%
- AI bubbles: transparent with `--border-subtle` left-border accent, left-aligned, max-width 85%
- Citation badges: inline, `var(--accent-subtle)` background
- Metadata footer: `text-xs text-[var(--text-2)]`

### Graph Visual Identity

- Node size: `6 + min(frequency, 10) * 1.5` px radius
- Edge opacity: `0.3 + weight * 0.5`
- Labels: mono font, 11px, positioned above node
- Active node: glow filter + pulse animation
- Hover: tooltip with entity name + document count + top-3 chunk excerpts

### Source Panel Style

Right-side slide-in panel (400px wide):
- Header: document name + page reference in mono
- Body: full chunk text with the excerpt highlighted (`<mark>` with `--accent-subtle`)
- Footer: "Open PDF at page 12" button (future: links to pdf.js viewer)

### 🔨 Build Checklist — Section 9

- [ ] `src/styles/tokens.css` with all CSS variables
- [ ] `src/styles/globals.css` with typography base
- [ ] `tailwind.config.js` extending with CSS var colors
- [ ] Build a "style guide" page temporarily to visualize all tokens
- [ ] Record a 10-second screen test of the graph pulse — does it feel impressive?

---

## SECTION 10 — Risks & Failure Cases

### 1. ChromaDB Metadata Size Overflow

**Cause:** Entity list of a dense chunk exceeds ~40KB when serialized.
**Detection:** `chromadb.add()` raises `ValueError: metadata too large`.
**Recovery:**
```python
def safe_metadata(chunk):
    entities_csv = ",".join(chunk.entities)
    if len(entities_csv) > 500:
        entities_csv = ",".join(chunk.entities[:10])  # keep top 10
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_set_id": chunk.document_set_id,
        "page_number": chunk.page_number,
        "language": chunk.language,
        "entities_csv": entities_csv,
    }
# Full entity list still lives in SQLite untruncated.
```

### 2. LLM Malformed Citations

**Cause:** Gemini occasionally produces `[CITE abc_0042]` (missing colon), `[Cite: abc_0042]`, or citations inside prose.
**Detection:** Regex parse returns fewer citations than the answer implies.
**Recovery:**
```python
# Use permissive regex
CITATION_REGEX = re.compile(
    r'\[\s*CITE\s*[:\s]\s*([a-f0-9_]+)\s*\]',
    re.IGNORECASE
)
# If still nothing matched but answer length > 100 chars:
#   → log warning, retry once with reinforced prompt
```

### 3. Garbled Text from Scanned PDFs

**Cause:** PyMuPDF extracts junk from image-only PDFs.
**Detection:** Post-extraction check:
```python
def is_garbled(text: str) -> bool:
    if len(text) < 50:
        return True
    printable_ratio = sum(1 for c in text if c.isprintable()) / len(text)
    return printable_ratio < 0.85
```
**Recovery:** Reject upload with clear error: `"This PDF appears to be scanned. OCR is not supported in this version."` (v1.1 could add Tesseract; out of 4-day scope.)

### 4. Too Many Graph Nodes

**Cause:** 10 PDFs × 100 chunks × 10 entities = 10,000 raw entities. Even after deduplication, may exceed 1000 nodes.
**Detection:** Count nodes after indexing; warn if >500.
**Recovery:** Default `min_frequency=2` filter in graph endpoint. Show message: "Graph filtered to concepts appearing in 2+ chunks. Adjust filter to see all."

### 5. WebSocket Drops During Indexing

**Cause:** User closes tab, network drop, laptop sleep.
**Detection:** `ws.onclose` in frontend; backend's `ws_manager.send()` catches.
**Recovery:**
- Backend: indexing continues regardless of WS connection. Completion is persisted to SQLite.
- Frontend on reconnect: fetch current state via `GET /graph/{set_id}` + `GET /documents/jobs/{job_id}` to catch up.
- No mid-index progress replay — just show current snapshot.

### 6. Duplicate Chunks on Re-indexing

**Cause:** Same PDF uploaded twice; watcher fires twice.
**Detection:**
```python
def document_exists(doc_id):
    return db.execute("SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1", (doc_id,)).fetchone()
```
**Recovery:** Skip silently if exists. Log: `"Skipped duplicate: {filename}"`. If user explicitly requests re-index, call `DELETE /documents/{doc_id}` first.

### 7. Cross-Document Retrieval Returns Irrelevant Chunks

**Cause:** Query matches topical similarity in wrong domain (e.g., "treatment" matches medical AND legal PDFs).
**Detection:** User sees irrelevant citations.
**Recovery (v1):** Set similarity threshold:
```python
MIN_RELEVANCE = 0.35
chunks = [c for c in retrieved if c.relevance_score > MIN_RELEVANCE]
if not chunks:
    return "No relevant content found for this question."
```
**Recovery (v1.1):** User can filter chat by document subset — not for v1.

### 8. Conversation Context Overflow

**Cause:** Long conversation, history exceeds 2000 tokens.
**Detection:** Budget exceeded in `build_history_window`.
**Recovery:** Sliding window naturally drops oldest turns. Already handled in Section 8.

### 9. sentence-transformers Slow on CPU

**Cause:** MPS unavailable (PyTorch version mismatch, macOS < 12.3).
**Detection:** Benchmark on startup:
```python
def benchmark_embedding():
    start = time.time()
    model.encode(["test sentence"] * 16)
    elapsed = time.time() - start
    if elapsed > 5:  # MPS should be <1s
        logger.warning("Embedding slow — MPS likely unavailable. Indexing will be 3-4x slower.")
```
**Recovery:** Reduce batch size to 8, show user estimated time proportionally. If really unusable, swap model to `intfloat/multilingual-e5-small` with quantized ONNX build (add only if needed).

### 10. Hallucinated Citations

**Cause:** LLM invents chunk_ids not in retrieved set.
**Detection:** `parse_citations()` returns non-empty `hallucinated` list.
**Recovery:** Auto-retry once with stricter prompt:
```
STRICT MODE: Your previous response included citation IDs that do not exist in the provided chunks. This is unacceptable. You MUST copy chunk_id values character-for-character from the AVAILABLE CHUNKS section. If you are unsure which chunk supports a claim, do not make the claim. Retry:
```
If retry still hallucinates → strip citations, display answer with warning banner: `"⚠️ Citation verification failed for this answer."`

### 🔨 Build Checklist — Section 10

- [ ] `backend/errors.py` — typed error codes matching WS protocol
- [ ] `backend/monitoring/benchmark.py` — startup MPS benchmark
- [ ] Wire each failure case to a user-visible error state in the frontend
- [ ] Test: upload a scanned PDF, verify graceful rejection
- [ ] Test: simulate WS drop mid-index, verify reconnect shows current state

---

## SECTION 11 — Implementation Roadmap (4-Day Build)

### Day 1 — Indexing Pipeline End-to-End (8-10 hours)

**Hour 1-2: Sanity checks (non-negotiable)**
- Arabic + French + English PDF extraction via PyMuPDF → print to terminal, visually verify
- Embed one sentence in each language, compute cross-language similarity → confirm semantic alignment
- Gemini API "Hello" call → confirm key works

**Hour 3-5: Core pipeline**
- PyMuPDF extractor with page/paragraph indexing
- Chunker (900 char, 150 overlap, paragraph-aware)
- spaCy entity extractor (EN + FR)
- sentence-transformers embedder with MPS

**Hour 6-8: Storage**
- SQLite schema + connection wrapper
- ChromaDB collection management
- Full pipeline orchestrator: PDF → chunks → entities → embeddings → DB

**Hour 9-10: Watcher + verification**
- watchdog file watcher
- Duplicate prevention
- **Deliverable: run `python -m backend.indexing.pipeline path/to/pdf` and see chunks in SQLite, embeddings in ChromaDB**

**End of Day 1 checkpoint:** Index one 130-page English PDF, one 50-page French PDF, one 30-page Arabic PDF. Query ChromaDB manually with a test question, verify sensible chunks returned.

### Day 2 — Retrieval + Citation Generation (8-10 hours)

**Hour 1-3: Retrieval**
- Vector retrieval with SQLite hydration
- Relevance scoring
- Unit test: 3 test questions against indexed docs

**Hour 4-7: THE CITATION PROMPT**
- System prompt with strict citation rules
- Chunk injection formatter
- History window builder
- Citation regex parser
- Hallucination validator + retry logic
- **This is the highest-risk component.** If by end of Hour 7 citations aren't reliable, stay here until 10 PM.

**Hour 8-9: FastAPI routes**
- `/documents/upload`, `/documents/sets`, `/chat`, `/graph`, `/health`
- WebSocket manager

**Hour 10: End-to-end backend test**
- `curl` upload → wait for indexing → `curl` chat question → verify valid citations in response
- **Deliverable: backend-only demo via terminal — the hardest work is done**

### Day 3 — Frontend (8-10 hours)

**Hour 1-2: Scaffolding**
- Vite + React + TS + Tailwind setup
- CSS variables + design tokens
- Zustand store
- API client with TypeScript types mirroring backend schemas

**Hour 3-5: Chat UI**
- MessageBubble with citation rendering
- CitationBadge with hover excerpt
- useWebSocket hook
- Streaming token display

**Hour 6-8: Knowledge Graph**
- D3 force simulation
- Node/edge rendering with enter/update/exit
- Active-path glow animation
- Hook up to WebSocket graph updates

**Hour 9-10: Upload + Source Panel**
- UploadZone with drag-and-drop
- Per-file progress bars
- Source panel with chunk display
- **Deliverable: full app functional end-to-end on localhost**

### Day 4 — Polish, Demo, Deploy (6-8 hours)

**Hour 1-2: Polish**
- Fix every visible bug from Day 3 exploration
- Refine graph forces until readable
- Tune glow animation intensity
- Ensure Arabic RTL rendering is clean

**Hour 3-4: Demo content preparation**
- Curate sample set: 2 medical PDFs, 2 research papers, 1 legal/regulatory
- Pre-index sample set; verify great demo questions
- Write 3-question demo script

**Hour 5-6: Demo video**
- Record 2-minute screen video:
  1. (0:00) Upload 3 PDFs, graph builds live
  2. (0:45) Ask cross-document question, see streaming answer with citations
  3. (1:15) Click citation → source panel opens
  4. (1:30) Ask Arabic question → Arabic answer with French citations
  5. (1:45) Show graph path highlight
- Encode, upload to YouTube unlisted

**Hour 7-8: Deployment prep**
- Install cloudflared on M1 Air
- Set up Cloudflare Tunnel to `localhost:8000` (backend) and `localhost:5173` (frontend build served via nginx)
- Configure `querymind.ghaith.com` DNS
- Don't launch 24/7 yet; verify it works, then shut down until needed for a specific application

### Highest-Risk Component

**Citation prompt engineering (Day 2, Hours 4-7).** Everything else is standard web dev. This is the novel part that determines demo credibility. Budget an entire day for this if needed — cut elsewhere.

### Cut List (If Behind Schedule)

**Must-have (ship even if exhausted):**
1. PDF upload + indexing
2. Cited answers (citations must be verifiable)
3. Basic knowledge graph (even static, no animation)
4. Demo video

**Nice-to-have (cut in order):**
1. ❌ Auto re-indexing (watchdog) → manual re-index button instead
2. ❌ Answer path glow animation → static colored nodes
3. ❌ Arabic support → English + French only
4. ❌ Source panel → just show citations inline with hover
5. ❌ Multi-document-set management → single hardcoded set
6. ❌ Conversation memory → stateless Q&A
7. ❌ Graph filtering UI

**Do not cut:** PDF upload, cited answers, basic graph rendering, demo video.

### 2-Minute Demo Script

```
[0:00] "QueryMind is a document intelligence platform I built in 4 days.
        Let me show you what it does."

[0:05] [Open QueryMind, click Upload, drag 3 PDFs: cardiology guidelines,
        drug interaction study, algerian pharma regulation]

[0:15] [Indexing starts. Graph panel animates: nodes appear for 'hypertension',
        'ACE inhibitor', 'CNAS', 'Ministry of Health', edges form between them]

[0:40] "As documents index, the knowledge graph builds live — concepts become
        nodes, edges show co-occurrence across chunks."

[0:50] [Click chat, type: "How should hypertension be treated in elderly
        Algerian patients, considering insurance coverage?"]

[1:05] [Answer streams with citations: "ACE inhibitors are first-line [📄 p.12
        cardiology.pdf]...CNAS covers most options [📄 p.3 pharma-reg.pdf]..."]

[1:20] [Graph: 6 nodes pulse violet — hypertension, ACE inhibitor, elderly,
        CNAS, coverage, contraindication. Edges between them brighten.]

[1:30] [Click citation badge — source panel opens, shows exact chunk with
        excerpt highlighted, document and page visible]

[1:45] [Switch chat language, ask in Arabic: "ما هي الآثار الجانبية؟"]

[1:55] [Arabic answer streams with French citations]

[2:00] "Full stack is FastAPI, ChromaDB, D3, running entirely on this
        MacBook. Portfolio link in description."
```

### 🔨 Build Checklist — Section 11

- [ ] Block 4 consecutive days on calendar
- [ ] Day 1 Hour 1-2 sanity test script ready
- [ ] Demo PDFs collected (medical, research, legal) before Day 1
- [ ] OBS or QuickTime installed + screen recording tested
- [ ] Cloudflare account + domain ready

---

## SECTION 12 — Self-Critique & Refinement

### Vague Decisions I'm Calling Out

**Weakness 1: "Gemini free tier handles Arabic well" is unverified.**
The Gemini free tier (as of the plan) has rate limits and the quality claim is based on general reputation, not your specific test. **Fix:** Hour 1 of Day 1, run exactly 5 Arabic questions through Gemini against 5 known Arabic chunks, grade the outputs. If quality <70%, fallback to Claude API (budget $20/month for portfolio — acceptable).

**Weakness 2: "spaCy EN/FR entity extraction is good enough" is optimistic.**
spaCy's small models miss domain-specific entities ("tafsir", "wilaya", "CNAS") entirely. **Fix:** Add a `CUSTOM_ENTITIES` regex pattern list for Algerian-domain terms applied alongside spaCy. Load from `backend/data/custom_entities.json`. Takes 30 minutes; massive quality gain.

**Weakness 3: "Batch every 20 chunks" graph update cadence is a guess.**
Untested. Might be too slow (UI feels lifeless for 10 seconds) or too fast (UI thrashes). **Fix:** Make it configurable via env var (`GRAPH_BATCH_SIZE=20`, `GRAPH_BATCH_MS=500`). Tune empirically on Day 3.

**Weakness 4: "min_frequency=2 default filter" may hide most interesting entities.**
Rare entities are often the most interesting (e.g., a specific drug name mentioned once). **Fix:** Default `min_frequency=1` but cap total nodes at 400. Sort by frequency desc, take top 400. Still readable, preserves rare-but-retrievable concepts.

**Weakness 5: "Answer path highlighting after answer_complete" could feel disconnected from streaming answer.**
Ideal UX: nodes light up progressively as citations stream in, not all at the end. **Fix:** Each citation, as parsed from stream, fires a `node_activate` event. Nodes light up in sequence. But this adds complexity — OK to defer to v1.1 if Day 3 runs long.

**Weakness 6: "relevance_score threshold 0.35" is arbitrary.**
Different embedding models produce different score distributions. **Fix:** Calibrate on Day 2 against the test PDFs. If too strict → drop to 0.25. If too lenient → raise to 0.45.

**Weakness 7: Indexing pipeline is single-threaded.**
If M1 is slower than expected, indexing a 400-page PDF could take 6 minutes. **Fix:** Parallelize embedding (batch parallelism already in sentence-transformers) but keep NER single-threaded for memory. On 8GB RAM, don't try ProcessPoolExecutor — memory will blow up.

**Weakness 8: "WebSocket manager" is global singleton.**
Fine for dev, fragile under real load. But for a 4-day portfolio project with 1 concurrent user (you), it's correct. **Not fixing.**

### The One Assumption Most Likely Wrong

**Assumption:** Gemini free tier rate limits are sufficient for live demo.

**Reality:** Gemini free tier is ~15 RPM for flash models. A recruiter rapid-firing 8 questions in the demo will hit limits instantly, showing 429 errors. This destroys the demo.

**Fallback plan:**
- Pre-cache answers for the 3 demo-script questions as SQLite fixtures. If Gemini returns 429, serve from cache with a `cached=true` flag (demo stays smooth).
- In parallel: load $10 on OpenRouter as a paid fallback that auto-routes to Claude or Gemini paid endpoints. Set budget alerts.

### Confidence Score

**78% confident QueryMind ships demo-ready in 4 days.**

Breakdown:
- Day 1 (indexing): 90% — standard engineering, well-scoped
- Day 2 (citations + API): 65% — citation engineering is novel and risky
- Day 3 (frontend + D3): 75% — D3 graph animation is fiddly
- Day 4 (polish + demo): 90% — if Days 1-3 are done

The 22% risk: citation reliability takes 1.5 days instead of 0.5, eating into frontend time. Mitigation: on Day 1 evening, spend 30 minutes on a prototype prompt to pre-validate the citation approach before Day 2 is all-in.

### The Single Most Important Sentence in This Document

**Citations are the product; the graph is the signature; everything else is table stakes — if citations don't verify against retrieved chunks, ship nothing.**

---

## 🔨 MASTER BUILD CHECKLIST — Pre-Flight

Before starting Day 1:

- [ ] 4 consecutive days blocked on calendar
- [ ] M1 Air has ≥40GB free disk space (ChromaDB + models)
- [ ] Python 3.11 via pyenv, verified
- [ ] Node 20+ via nvm, verified
- [ ] Gemini API key obtained + tested with 1 call
- [ ] 3 test PDFs collected: 1 Arabic, 1 French medical, 1 English research (before Day 1)
- [ ] 5 demo PDFs curated: 2 medical, 2 research, 1 legal (before Day 4)
- [ ] Cloudflare account created + `ghaith.com` purchased (can be parallel)
- [ ] OBS Studio installed for screen recording
- [ ] `docs/ADR-001-metadata-rag.md` written (justify the architectural choice in writing — forces clarity)
- [ ] Read every section of this plan once more before `git init`

---

**End of Master Plan. Go build.**
