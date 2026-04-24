# ADR-001: Metadata-Rich RAG Over True Graph RAG

**Date:** 2026-04-22  
**Status:** Accepted

## Context

QueryMind needs a knowledge graph visualization that grows live during PDF indexing, with semantic search, citation verification, and multilingual support (Arabic/French/English). The question is whether to implement True Graph RAG (LLM-extracted entity relationships) or Metadata-Rich RAG (co-occurrence graph derived from NLP entity extraction).

## Decision

Use **Metadata-Rich RAG** with a co-occurrence knowledge graph rather than True Graph RAG (Microsoft GraphRAG / LightRAG).

## Rationale

**True Graph RAG requires:**
- LLM API call per chunk for entity extraction (~$0.01/chunk × thousands = real cost)
- LLM API call per chunk for relationship extraction (double the cost)
- Community detection post-processing (hours of computation)
- Hierarchical summarization passes

**Metadata-Rich RAG delivers the same visible output:**
- spaCy NER extracts entities locally (zero API cost, ~200 chunks/sec)
- Co-occurrence edges weighted by shared-chunk frequency
- D3 force graph renders nodes + edges identically to GraphRAG output
- Answer path highlighting (nodes that contributed to an answer glow) is purely frontend logic

**The recruiter demo test:** A person watching a 2-minute demo cannot distinguish LLM-extracted relations from co-occurrence edges. Both produce a glowing graph where nodes light up when an answer is generated. We ship the visible 90% in 4 days instead of invisible 100% in 4 weeks.

## Citation Fidelity Architecture

The core differentiator is **structurally-enforced citation accuracy**:

1. Retrieved chunks injected into prompt with explicit `[CHUNK_ID: <uuid>]` markers
2. System prompt mandates LLM cite only those exact IDs
3. Post-generation validator intersects cited IDs with retrieved IDs
4. Mismatch → auto-retry with stricter prompt; second failure → citation stripped, answer flagged
5. Frontend renders only citations that resolve to real chunks in the store

This prevents hallucinated citations architecturally, not via prompt engineering.

## Consequences

- **Pro:** Feasible in 4 days on free-tier Gemini with rate limits
- **Pro:** Zero API cost for graph construction
- **Pro:** Graph grows live during indexing (spaCy is fast enough for real-time display)
- **Con:** Edges are co-occurrence weights, not semantic relationships — a graph analyst would notice
- **Con:** Cross-document relationship discovery is limited to shared entity names, not inferred concepts

## Alternatives Rejected

- **LightRAG:** Requires LLM calls for every chunk, exceeds free-tier rate limits in < 1 hour of indexing
- **Microsoft GraphRAG:** Requires OpenAI GPT-4, not compatible with Gemini free tier
- **Embedding-only RAG (no graph):** Misses the core demo differentiator entirely
