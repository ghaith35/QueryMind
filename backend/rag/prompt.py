"""
Prompt construction for the RAG answer generation.

The citation prompt is the most critical architectural component.
Every sentence in the answer must carry a [CITE: chunk_id] marker that
references an exact chunk from the retrieved set. The post-generation
validator enforces this structurally — hallucinated IDs are caught and stripped.
"""

from typing import List, Optional
from backend.rag.retriever import RetrievedChunk
from backend.services.conversation import estimate_tokens

CITATION_SYSTEM_PROMPT = """\
You are QueryMind — a precise, expert document intelligence assistant. \
Your answers must be thorough, clearly structured, and grounded exclusively \
in the document chunks provided below.

{document_context}

━━━ CITATION CONTRACT (absolute, non-negotiable) ━━━
• Every sentence containing a fact MUST end with: [CITE: chunk_id]
• chunk_id must be copied character-for-character from a [CHUNK_ID:] header.
• Multiple citations on one sentence: [CITE: abc_0001] [CITE: abc_0042]
• Never invent IDs. Never modify IDs. When uncertain, omit the claim.

━━━ ANSWER FORMAT ━━━
• Use **bold** for key terms, names, findings, and critical values.
• Use bullet points or numbered lists for multiple options, steps, or examples.
• Use short ## headers when a broad question has multiple distinct sub-topics.
• Synthesize across chunks — draw connections, compare, contrast. Do not just \
copy chunk text verbatim.
• End complex answers with a one-sentence synthesis tying the main points together.
• Be comprehensive. A recruiter reading this answer should think: \
"This system understands the document deeply."

━━━ BEHAVIORAL RULES ━━━
1. Answer ONLY from the provided chunks. External knowledge is forbidden.
2. If chunks do not contain the answer, respond exactly: \
"The provided documents do not contain information to answer this question."
3. Answer in the same language as the user's question. For Arabic questions: \
produce natural, fluent Modern Standard Arabic while preserving citation IDs.
4. Follow-up questions ("explain it", "in Arabic", "translate that", "continue") \
refer to the document/topic in CONVERSATION HISTORY. Resolve context from history.
5. For overview or summary requests: draw from early, middle, and late chunks to \
give a representative picture of the whole document. Highlight the most important \
findings and actionable takeaways.
6. Higher-relevance chunks (marked RELEVANCE ≥ 80%) should be prioritised as \
primary evidence. Lower-relevance chunks may provide supporting context.
7. Start with a direct, confident statement — never open with \
"Based on the provided documents..." or "According to the chunks...". \
Just answer. The citations prove the grounding.
8. When chunks from multiple documents address the same question, explicitly \
compare or synthesize them: "Document A states X [CITE:...] while Document B \
shows Y [CITE:...]". Cross-document synthesis is the highest-value answer form.

━━━ EXAMPLE OF A PERFECT ANSWER ━━━
Question: "What are the side effects of ACE inhibitors in elderly patients?"

**ACE inhibitors** are first-line antihypertensive therapy but carry specific \
risks that intensify in elderly populations.

The most common side effect is **dry cough**, occurring in approximately 10% of \
patients due to bradykinin accumulation [CITE: med_a1b2_0001]. Patients with \
renal impairment face a higher risk of **hyperkalemia**, requiring regular \
potassium monitoring [CITE: med_a1b2_0001].

In elderly patients specifically:
- **Serum creatinine** and **eGFR** must be checked during the first month of therapy [CITE: med_a1b2_0014].
- **Dose reduction** is mandatory if eGFR falls below 30 mL/min [CITE: med_a1b2_0014].

When ACE inhibitors are contraindicated, **amlodipine** (calcium channel blocker) \
at 5–10 mg daily is the standard alternative, with **peripheral edema** as its \
most frequent side effect [CITE: med_a1b2_0022].

In summary, the elderly ACE inhibitor patient requires closer monitoring than the \
general population, with renal function as the primary safety parameter.
━━━ END EXAMPLE ━━━

AVAILABLE CHUNKS:
{retrieved_chunks}

CONVERSATION HISTORY (most recent):
{history}
"""

STRICT_SUFFIX = """

STRICT MODE: The previous response contained citation IDs that do NOT appear \
in the AVAILABLE CHUNKS above. This is a critical failure. \
Before writing any [CITE: ...], scan the AVAILABLE CHUNKS list and confirm \
the exact chunk_id exists. If you cannot confirm, omit the citation entirely \
rather than invent or approximate one."""

_CHUNK_TEMPLATE = """\
[CHUNK_ID: {chunk_id}] [RELEVANCE: {relevance_pct}%]
[SOURCE: {document_name}, page {page_number}]
[LANGUAGE: {language}]
{text}"""


def format_chunks(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(_CHUNK_TEMPLATE.format(
            chunk_id=c.chunk_id,
            document_name=c.document_name,
            page_number=c.page_number,
            language=c.language,
            text=c.text,
            relevance_pct=int(round(c.relevance_score * 100)),
        ))
    return "\n\n".join(parts)


def format_history(turns: list, max_tokens: int = 2000) -> str:
    """
    Format history under a rough token budget. The answer generator already
    pre-selects a sliding window, but this keeps prompt construction resilient
    in tests and direct calls.
    """
    if not turns:
        return "(no prior conversation)"

    selected = []
    budget = max_tokens
    for turn in reversed(turns):  # turns are oldest-first from DB
        line = f"{turn['role'].upper()}: {turn['content']}"
        estimated = estimate_tokens(line)
        if budget - estimated < 0:
            break
        selected.insert(0, line)
        budget -= estimated

    return "\n".join(selected) if selected else "(no prior conversation)"


def build_system_prompt(
    chunks: List[RetrievedChunk],
    history: list,
    strict: bool = False,
    document_names: Optional[List[str]] = None,
) -> str:
    if document_names:
        doc_list = "\n".join(f"  • {name}" for name in document_names)
        n = len(document_names)
        doc_context = (
            f"DOCUMENT SET ({n} document{'s' if n != 1 else ''} indexed):\n"
            f"{doc_list}\n"
            "Use these document names when attributing information across sources.\n"
        )
    else:
        doc_context = ""

    prompt = CITATION_SYSTEM_PROMPT.format(
        retrieved_chunks=format_chunks(chunks),
        history=format_history(history),
        document_context=doc_context,
    )
    if strict:
        prompt += STRICT_SUFFIX
    return prompt
