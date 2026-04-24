# ADR-001: Metadata-Rich RAG Over True Graph RAG

**Status:** Accepted
**Date:** 2026-04-22
**Author:** Ghaith
**Project:** QueryMind

---

## Context

QueryMind is a document intelligence platform with a live knowledge graph as its central visual differentiator. The graph must show concepts as nodes, relationships as edges, and highlight the reasoning path when an answer is generated.

Three architectural options were considered for how the graph backs the retrieval system:

**Option A — Pure Vector RAG (no graph):**
Standard retrieval-augmented generation with chunk embeddings. Graph is a post-hoc visualization with no retrieval role.

**Option B — Metadata-Rich RAG + Visual Graph (chosen):**
Standard vector similarity retrieval. Entities extracted per chunk are stored as chunk metadata. Graph is built from chunk metadata at query time — nodes are extracted concepts, edges are co-occurrence. The graph is visually central and navigable, but retrieval is driven by vector similarity, not graph traversal.

**Option C — True Graph RAG (Microsoft GraphRAG / LightRAG style):**
LLM-based entity and relationship extraction per chunk. Knowledge graph drives retrieval via graph traversal, community detection, hierarchical summarization. Answers generated from graph subgraphs, not raw chunks.

## Decision

**Option B (Metadata-Rich RAG + Visual Graph) is adopted.**

## Rationale

### Why Not Option C (True Graph RAG)

True Graph RAG requires an LLM call per chunk to extract entities and relationships with structured output. For a document set of 10 PDFs averaging 80 chunks each = 800 LLM calls just for indexing. On Gemini free tier this blows through the rate limit; on paid Claude/GPT-4 this is real money per user session.

Beyond cost, GraphRAG adds:
- Community detection (Leiden/Louvain algorithms) on the entity graph
- Hierarchical summarization of communities by an LLM
- Query-time graph traversal logic
- Retrieval scoring that blends vector and graph-proximity signals

Estimated implementation: **3–4 weeks minimum.** For a 4-day portfolio build, this is infeasible.

### Why Not Option A (Pure Vector RAG)

The graph is not cosmetic — it is the portfolio differentiator. A recruiter at Mozn, Lucidya, or Elm has seen fifty ChatPDF clones. What they have not seen is:
- A live-building concept graph during indexing
- An answer path that highlights contributing nodes
- Visual confirmation that the system actually "understands" the documents' conceptual structure

Removing the graph reduces QueryMind to a commodity RAG app and eliminates the single most screen-recordable moment of the demo.

### Why Option B Wins

Option B preserves the visible 90% of the "graph-powered" experience while eliminating the engineering cost of the invisible 10%:

| Dimension | Option B | Option C | Delta |
|---|---|---|---|
| Indexing LLM calls per chunk | 0 | 1–2 | Large cost savings |
| Citation reliability | High (vector-retrieved chunks) | Medium (graph-summarized) | B is safer |
| Visual graph in UI | Yes, co-occurrence edges | Yes, LLM-extracted edges | Visually indistinguishable to viewer |
| Answer path highlighting | Yes | Yes | Identical UX |
| Implementation time | ~4 days | ~3–4 weeks | Ships on schedule |
| Failure modes | Well-understood | Emerging, under-documented | B is debuggable |

**Critical insight:** A recruiter watching a 2-minute demo video cannot distinguish LLM-extracted relationships from co-occurrence edges. Both show nodes, edges, pulses. The differentiator is *visual presence of a working graph*, not the algorithm underneath.

### Citation Reliability Argument

True Graph RAG summarizes subgraphs into context, then generates answers from summaries. This introduces a second hop where citation precision can erode — the citation points to a summary, and tracing back to the exact source chunk requires extra bookkeeping.

Option B retrieves chunks directly. Every citation maps 1:1 to a retrieved chunk with exact page/paragraph. For a system whose primary differentiator is verifiable citations, the shorter path from query → chunk → answer → citation is architecturally safer.

## Consequences

### Positive

- Ships in 4 days on the stated hardware (M1 Air, 8GB RAM)
- Zero LLM cost during indexing
- Citation pipeline is straightforward and auditable
- Graph is a UI layer, so iterating on its visual design doesn't require re-indexing
- Works offline after initial model downloads

### Negative / Trade-offs

- Edges carry no semantic relationship labels (only "co-occurrence"). We cannot answer questions like "what techniques defeat which attacks" from graph structure alone.
- Questions requiring multi-hop reasoning across documents depend on vector retrieval catching the linking chunk. If no chunk explicitly bridges two concepts, the connection is lost.
- The graph's "intelligence" appearance is a UI illusion backed by frequency statistics, not genuine relation inference. Honest demo framing matters — do not oversell as "Graph RAG."

### Mitigations

- In demo script, describe the system as "a concept graph derived from document content, with answer path visualization," not "Graph RAG."
- For v1.1, upgrade-in-place path exists: add relationship extraction as an optional LLM pass post-indexing, upgrading edge `relation_type` from `"co_occurrence"` to semantic labels without changing the schema.

## Alternatives Revisited (Upgrade Path)

If QueryMind moves from portfolio to product, the migration path to Option C is clean:
1. Add `relationship_type` field to edge schema (already reserved).
2. Run an offline batch job: LLM-extract relationships for existing edges.
3. Update edges in-place; no re-indexing of chunks required.
4. Add graph-traversal retrieval as a parallel path to vector retrieval; blend scores.

No wasted work in Option B.

## References

- Microsoft GraphRAG: https://microsoft.github.io/graphrag/
- LightRAG: https://github.com/HKUDS/LightRAG
- `intfloat/multilingual-e5-small`: https://huggingface.co/intfloat/multilingual-e5-small
- QueryMind Master Plan, Section 1 (Problem Framing)
