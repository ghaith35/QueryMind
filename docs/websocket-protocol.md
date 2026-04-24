# QueryMind WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws/{session_id}`  
**Encoding:** JSON (UTF-8, full Unicode support including Arabic/RTL)

---

## Envelope

Every message sent by the server shares this envelope:

```typescript
interface WSMessage<T> {
    type: string;       // discriminant — see message types below
    timestamp: string;  // ISO 8601 UTC, e.g. "2026-04-22T10:30:00.000000+00:00"
    payload: T;
}
```

---

## Server → Client Messages

### `index_progress`

Emitted during PDF indexing. Sent at each pipeline stage transition and approximately every 20 chunks.

```typescript
{
    type: "index_progress",
    payload: {
        job_id: string,
        document_name: string,
        chunks_processed: number,
        total_chunks_estimated: number,
        stage: "extracting" | "chunking" | "embedding" | "storing" | "complete",
        percent: number  // 0-100
    }
}
```

**Frontend behaviour:** update progress bar; on `stage = "complete"`, dismiss progress UI.

---

### `graph_update`

Batched graph diff. Emitted every **20 chunks processed** or **500ms**, whichever comes first. Never emitted more than once per 100ms (debounced server-side).

```typescript
{
    type: "graph_update",
    payload: {
        document_set_id: string,
        added_nodes: GraphNode[],
        added_edges: GraphEdge[],
        updated_nodes: { id: string, frequency: number }[]
    }
}
```

**GraphNode shape:**
```typescript
interface GraphNode {
    id: string;       // sha256(label.lower())[:12]
    label: string;
    type: "concept" | "person" | "organization" | "location" | "technique";
    document_sources: string[];
    chunk_ids: string[];
    frequency: number;
    is_active: boolean;
}
```

**GraphEdge shape:**
```typescript
interface GraphEdge {
    source: string;   // node id
    target: string;   // node id
    relation_type: "co_occurrence";
    weight: number;   // 0-1 normalized
    co_occurrence_count: number;
    is_active: boolean;
}
```

**Frontend behaviour:** apply D3 `enter/update/exit` transitions without restarting the force simulation.

---

### `answer_stream`

Streaming tokens during LLM generation. May arrive at 10–50 tokens/sec.

```typescript
{
    type: "answer_stream",
    payload: {
        session_id: string,
        turn_id: string,
        token: string,       // partial text fragment
        is_final: boolean    // true on last token before answer_complete
    }
}
```

**Frontend behaviour:** append `token` to the current answer bubble. On `is_final = true`, stop appending and wait for `answer_complete`.

---

### `answer_complete`

Sent once after all `answer_stream` tokens. Contains full validated answer, citations, and graph highlight instructions.

```typescript
{
    type: "answer_complete",
    payload: {
        session_id: string,
        turn_id: string,
        full_answer: string,
        citations: Citation[],
        active_node_ids: string[],
        active_edge_ids: string[]  // format: "sourceId->targetId"
    }
}
```

**Citation shape:**
```typescript
interface Citation {
    document_name: string;
    page_number: number;
    paragraph_index: number;
    chunk_id: string;
    relevance_score: number;  // 0-1
    excerpt: string;          // first 150 chars of chunk text
}
```

**Frontend behaviour:**
1. Replace streamed answer with `full_answer` (already identical, but guarantees consistency)
2. Render inline `[CITE: chunk_id]` markers as clickable badges
3. Set `is_active = true` on nodes in `active_node_ids` → glow animation
4. Set `is_active = true` on edges in `active_edge_ids` → highlight edges

---

### `error`

```typescript
{
    type: "error",
    payload: {
        code: "INDEX_FAILED" | "LLM_FAILED" | "CITATION_HALLUCINATED" | "INVALID_PDF",
        message: string,
        recoverable: boolean
    }
}
```

**Error codes:**
| Code | Cause | Recoverable |
|---|---|---|
| `INDEX_FAILED` | Pipeline crash during indexing | true — retry upload |
| `LLM_FAILED` | Gemini API error or timeout | true — retry question |
| `CITATION_HALLUCINATED` | Post-parse validator found non-retrieved chunk IDs after 2 retries | false — answer stripped |
| `INVALID_PDF` | PyMuPDF couldn't open or extracted < 200 chars | false — reject file |

---

## Client → Server Messages

The client sends plain JSON over the same WebSocket:

```typescript
// Ask a question
{
    type: "ask",
    payload: {
        session_id: string,
        document_set_id: string,
        question: string  // supports Arabic, French, English
    }
}

// Start indexing a previously uploaded PDF
{
    type: "index",
    payload: {
        document_set_id: string,
        filename: string
    }
}
```

---

## Connection Lifecycle

1. Browser opens `ws://localhost:8000/ws/{session_id}`
2. Server confirms with first `index_progress` or waits for client message
3. Session ID is stored in `localStorage` — reconnect reuses the same session
4. Server streams `answer_stream` tokens, then sends `answer_complete`
5. On disconnect, server buffers any in-progress `graph_update` and delivers on reconnect

---

## Arabic / RTL Notes

- All JSON is UTF-8; Arabic strings are transmitted as-is (no escaping)
- `full_answer` may be right-to-left; frontend must detect script direction and apply `dir="rtl"` to the chat bubble
- `excerpt` in citations may be Arabic — render accordingly
