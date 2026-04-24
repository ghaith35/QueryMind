"""
QueryMind Day 1 — Sanity Check #2: Multilingual Embedding Alignment

Purpose: Verify that intfloat/multilingual-e5-small produces semantically
aligned embeddings across Arabic, French, and English. If cross-lingual
similarity is weak, the entire "ask in Arabic, retrieve from French docs"
feature collapses.

Also verifies:
    - MPS (Apple Silicon GPU) acceleration is working
    - Embedding batch performance is acceptable for the 4-day timeline

Usage:
    python sanity_check_02_embeddings.py

Pass criteria:
    - MPS available on M1
    - Batch of 16 sentences embeds in < 2s on MPS (< 8s on CPU fallback)
    - Cross-lingual cosine similarity >= 0.75 for paraphrase pairs
    - Unrelated sentences show similarity < 0.60 (semantic discrimination)
"""

import sys
import time
from typing import List

try:
    import torch
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    print(f"ERROR: missing dependency — {e}")
    print("Run: pip install sentence-transformers torch numpy")
    sys.exit(1)


MODEL_NAME = "intfloat/multilingual-e5-small"

# Paraphrase pairs across languages (same meaning, different language)
# Used to check cross-lingual alignment.
PARAPHRASE_PAIRS = [
    {
        "id": "medical",
        "en": "Hypertension in elderly patients requires careful monitoring of blood pressure.",
        "fr": "L'hypertension chez les patients âgés nécessite une surveillance attentive de la tension artérielle.",
        "ar": "يتطلب ارتفاع ضغط الدم لدى المرضى المسنين مراقبة دقيقة لضغط الدم.",
    },
    {
        "id": "tech",
        "en": "Machine learning models can detect anomalies in network traffic.",
        "fr": "Les modèles d'apprentissage automatique peuvent détecter des anomalies dans le trafic réseau.",
        "ar": "يمكن لنماذج التعلم الآلي اكتشاف الحالات الشاذة في حركة مرور الشبكة.",
    },
    {
        "id": "finance",
        "en": "The bank reported higher quarterly profits than expected.",
        "fr": "La banque a déclaré des bénéfices trimestriels plus élevés que prévu.",
        "ar": "أعلن البنك عن أرباح فصلية أعلى من المتوقع.",
    },
]

# Unrelated sentences — should have LOW similarity to the above.
UNRELATED_SENTENCES = {
    "en": "The cat sat on the windowsill watching birds.",
    "fr": "Le chat s'est assis sur le rebord de la fenêtre en regardant les oiseaux.",
    "ar": "جلست القطة على حافة النافذة تراقب الطيور.",
}


def setup_device() -> str:
    """Detect best available device. Prefers MPS on Apple Silicon."""
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) is available")
        return "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA GPU is available")
        return "cuda"
    else:
        print("⚠ No GPU — falling back to CPU (slower)")
        return "cpu"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (both assumed L2-normalized)."""
    return float(np.dot(a, b))


def embed_with_prefix(model: SentenceTransformer, texts: List[str],
                      role: str = "passage") -> np.ndarray:
    """e5 models require 'query: ' or 'passage: ' prefix for best performance."""
    prefixed = [f"{role}: {t}" for t in texts]
    return model.encode(
        prefixed,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def benchmark_embedding(model: SentenceTransformer, device: str) -> float:
    """Embed 16 sentences, measure wall time."""
    sample = ["This is a benchmark test sentence for embedding performance."] * 16
    # Warm-up (first call loads weights to device)
    _ = embed_with_prefix(model, sample[:2])

    start = time.time()
    _ = embed_with_prefix(model, sample)
    elapsed = time.time() - start
    return elapsed


def test_cross_lingual_alignment(model: SentenceTransformer):
    """For each paraphrase triple, verify EN<>FR, EN<>AR, FR<>AR similarities."""
    print("\n" + "=" * 60)
    print("Cross-Lingual Alignment Test")
    print("=" * 60)

    all_pass = True
    THRESHOLD_PARAPHRASE = 0.75
    THRESHOLD_UNRELATED_MAX = 0.60

    for pair in PARAPHRASE_PAIRS:
        texts = [pair["en"], pair["fr"], pair["ar"]]
        embeddings = embed_with_prefix(model, texts, role="passage")

        en_fr = cosine_similarity(embeddings[0], embeddings[1])
        en_ar = cosine_similarity(embeddings[0], embeddings[2])
        fr_ar = cosine_similarity(embeddings[1], embeddings[2])

        print(f"\nTopic: {pair['id']}")
        print(f"  EN <-> FR: {en_fr:.3f}  {'✅' if en_fr >= THRESHOLD_PARAPHRASE else '❌'}")
        print(f"  EN <-> AR: {en_ar:.3f}  {'✅' if en_ar >= THRESHOLD_PARAPHRASE else '❌'}")
        print(f"  FR <-> AR: {fr_ar:.3f}  {'✅' if fr_ar >= THRESHOLD_PARAPHRASE else '❌'}")

        if min(en_fr, en_ar, fr_ar) < THRESHOLD_PARAPHRASE:
            all_pass = False

    # Discrimination test: paraphrases should be MORE similar than unrelated
    print("\n" + "=" * 60)
    print("Semantic Discrimination Test")
    print("=" * 60)
    print("(Unrelated sentences should show LOWER similarity than paraphrases)")

    medical = PARAPHRASE_PAIRS[0]
    all_texts = [
        medical["en"], medical["fr"], medical["ar"],
        UNRELATED_SENTENCES["en"],
        UNRELATED_SENTENCES["fr"],
        UNRELATED_SENTENCES["ar"],
    ]
    embeds = embed_with_prefix(model, all_texts, role="passage")

    # medical_en vs cat_en (same language, different topic)
    med_en_vs_cat_en = cosine_similarity(embeds[0], embeds[3])
    # medical_en vs cat_ar (different lang, different topic)
    med_en_vs_cat_ar = cosine_similarity(embeds[0], embeds[5])

    print(f"\n  Medical-EN vs Cat-EN (same lang, diff topic): {med_en_vs_cat_en:.3f}")
    print(f"  Medical-EN vs Cat-AR (diff lang, diff topic): {med_en_vs_cat_ar:.3f}")
    print(f"\n  Discrimination threshold: unrelated sim should be < {THRESHOLD_UNRELATED_MAX}")

    if med_en_vs_cat_en >= THRESHOLD_UNRELATED_MAX:
        print(f"  ⚠ WARNING: unrelated sentences show high similarity ({med_en_vs_cat_en:.3f})")
        print("    The model may be weak at discrimination — retrieval quality risk")
        all_pass = False
    else:
        print("  ✅ Good discrimination between unrelated topics")

    return all_pass


def test_query_passage_asymmetry(model: SentenceTransformer):
    """e5 models are trained asymmetric: 'query:' prefix for questions,
    'passage:' for documents. Verify that using the correct prefix improves
    retrieval-style similarity."""
    print("\n" + "=" * 60)
    print("Query/Passage Prefix Asymmetry Test")
    print("=" * 60)

    question = "What are the side effects of ACE inhibitors?"
    passage = "ACE inhibitors commonly cause dry cough and may lead to hyperkalemia in patients with renal impairment."

    # Correct: query/passage
    q_embed = embed_with_prefix(model, [question], role="query")[0]
    p_embed = embed_with_prefix(model, [passage], role="passage")[0]
    correct_sim = cosine_similarity(q_embed, p_embed)

    # Incorrect: both as passage
    both_passage = embed_with_prefix(model, [question, passage], role="passage")
    wrong_sim = cosine_similarity(both_passage[0], both_passage[1])

    print(f"  With correct 'query:'/'passage:' prefixes: {correct_sim:.3f}")
    print(f"  With both as 'passage:':                    {wrong_sim:.3f}")
    print(f"  Asymmetry benefit: {correct_sim - wrong_sim:+.3f}")

    if correct_sim < 0.70:
        print("  ⚠ Lower than expected — check e5 model version")
    else:
        print("  ✅ Query/passage retrieval alignment working")


def main():
    print("QueryMind Embedding Sanity Check")
    print("=" * 60)

    # Device
    device = setup_device()

    # Load model
    print(f"\nLoading {MODEL_NAME} on {device}...")
    load_start = time.time()
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        print("Fix: check internet connection and disk space (~470MB)")
        sys.exit(1)
    load_elapsed = time.time() - load_start
    print(f"✅ Model loaded in {load_elapsed:.1f}s")

    # Benchmark
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    elapsed = benchmark_embedding(model, device)
    print(f"Embedding 16 sentences on {device}: {elapsed:.2f}s")

    threshold = 2.0 if device == "mps" else (1.0 if device == "cuda" else 8.0)
    if elapsed > threshold:
        print(f"⚠ SLOWER THAN EXPECTED (threshold {threshold}s for {device})")
        print("  Indexing will be slow. Consider:")
        print("  - Reducing batch size")
        print("  - Verifying PyTorch version supports MPS (torch >= 2.0)")
    else:
        print(f"✅ Performance acceptable (< {threshold}s threshold)")

    # Estimate indexing time for a 130-page PDF (~800 chunks)
    chunks_estimate = 800
    batches = chunks_estimate // 16
    est_indexing = batches * elapsed
    print(f"\nEstimated embedding time for 130-page PDF (~800 chunks): {est_indexing:.0f}s")
    if est_indexing > 180:
        print("  ⚠ Indexing will take > 3 min per large PDF. UX consideration.")
    else:
        print("  ✅ Acceptable indexing time")

    # Cross-lingual alignment
    alignment_ok = test_cross_lingual_alignment(model)

    # Query/passage asymmetry
    test_query_passage_asymmetry(model)

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if alignment_ok:
        print("✅ EMBEDDINGS PASS — multilingual retrieval is viable")
        print("   Proceed to Day 1 Hour 3 (chunker + pipeline)")
        sys.exit(0)
    else:
        print("❌ EMBEDDINGS WEAK — cross-lingual retrieval may be unreliable")
        print("\nRemediation paths:")
        print("  - Try larger variant: intfloat/multilingual-e5-base (~1.1GB)")
        print("  - Try: BAAI/bge-m3 (larger but strong cross-lingual)")
        print("  - Reconsider scope: drop Arabic support, keep EN+FR only")
        sys.exit(1)


if __name__ == "__main__":
    main()
