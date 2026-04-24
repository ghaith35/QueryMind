"""
QueryMind Day 1 — Sanity Check #1: Multilingual PDF Extraction

Purpose: Verify PyMuPDF extracts Arabic, French, and English text cleanly.
If any language fails here, the entire project's multilingual claim collapses
and architecture must change TODAY.

Usage:
    python sanity_check_01_extraction.py path/to/arabic.pdf \\
                                          path/to/french.pdf \\
                                          path/to/english.pdf

Pass criteria:
    - Arabic: text present, no � replacement chars, RTL characters intact
    - French: accents preserved (é, è, à, ç, ô, etc.)
    - English: clean extraction, no weird line breaks inside words
    - All three: printable_ratio >= 0.90
"""

import sys
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)


ARABIC_RANGE = (0x0600, 0x06FF)
FRENCH_ACCENTS = set("éèêëàâäôöûüùçîïÉÈÊËÀÂÄÔÖÛÜÙÇÎÏ")


def detect_languages(text: str) -> dict:
    """Rough language signal detection."""
    arabic_count = sum(1 for c in text if ARABIC_RANGE[0] <= ord(c) <= ARABIC_RANGE[1])
    french_accent_count = sum(1 for c in text if c in FRENCH_ACCENTS)
    ascii_alpha_count = sum(1 for c in text if c.isalpha() and ord(c) < 128)

    total_chars = len(text)
    return {
        "arabic_chars": arabic_count,
        "arabic_ratio": arabic_count / total_chars if total_chars else 0,
        "french_accent_chars": french_accent_count,
        "ascii_alpha_chars": ascii_alpha_count,
        "total_chars": total_chars,
    }


def check_garbled(text: str) -> tuple[bool, float]:
    """Returns (is_garbled, printable_ratio)."""
    if len(text) < 50:
        return True, 0.0
    # Printable = letters, digits, spaces, punctuation, Arabic, French accents
    printable = sum(
        1 for c in text
        if c.isprintable() or c in "\n\t"
    )
    ratio = printable / len(text)
    return ratio < 0.90, ratio


def check_replacement_chars(text: str) -> int:
    """Count � (U+FFFD) replacement characters — sign of encoding failure."""
    return text.count("\ufffd")


def check_word_fragmentation(text: str) -> int:
    """Count suspicious mid-word line breaks (English/French heuristic).

    Pattern: lowercase letter + newline + lowercase letter = likely broken.
    """
    return len(re.findall(r"[a-záéèêëàâäôöûüùçîï]\n[a-záéèêëàâäôöûüùçîï]", text))


def extract_pdf(path: Path) -> tuple[str, int]:
    """Extract full text. Returns (text, page_count)."""
    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    page_count = doc.page_count
    doc.close()
    return "\n\n".join(pages), page_count


def analyze_pdf(path: Path) -> dict:
    """Run all checks on a single PDF."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {path.name}")
    print('=' * 60)

    if not path.exists():
        return {"file": str(path), "error": "File not found"}

    try:
        text, page_count = extract_pdf(path)
    except Exception as e:
        return {"file": str(path), "error": f"Extraction failed: {e}"}

    lang_signals = detect_languages(text)
    is_garbled, printable_ratio = check_garbled(text)
    replacement_chars = check_replacement_chars(text)
    fragmented_words = check_word_fragmentation(text)

    # Determine primary language
    if lang_signals["arabic_ratio"] > 0.3:
        primary_lang = "ar"
    elif lang_signals["french_accent_chars"] > 10:
        primary_lang = "fr (likely)"
    else:
        primary_lang = "en (likely)"

    result = {
        "file": path.name,
        "pages": page_count,
        "total_chars": lang_signals["total_chars"],
        "primary_language_guess": primary_lang,
        "arabic_chars": lang_signals["arabic_chars"],
        "french_accent_chars": lang_signals["french_accent_chars"],
        "printable_ratio": round(printable_ratio, 3),
        "is_garbled": is_garbled,
        "replacement_chars": replacement_chars,
        "fragmented_words": fragmented_words,
    }

    # Report
    print(f"Pages:                    {result['pages']}")
    print(f"Total chars extracted:    {result['total_chars']}")
    print(f"Primary language guess:   {result['primary_language_guess']}")
    print(f"Arabic chars:             {result['arabic_chars']}")
    print(f"French accent chars:      {result['french_accent_chars']}")
    print(f"Printable ratio:          {result['printable_ratio']}")
    print(f"Replacement chars (�):    {result['replacement_chars']}")
    print(f"Fragmented words:         {result['fragmented_words']}")

    # Verdict
    passed = True
    issues = []

    if is_garbled:
        passed = False
        issues.append(f"GARBLED (printable_ratio {printable_ratio:.2f} < 0.90)")
    if replacement_chars > 5:
        passed = False
        issues.append(f"ENCODING FAILURE ({replacement_chars} replacement chars)")
    if result["total_chars"] < 200:
        passed = False
        issues.append("TOO LITTLE TEXT (likely scanned PDF, OCR needed)")
    if fragmented_words > (page_count * 3):
        issues.append(f"WORD FRAGMENTATION warning ({fragmented_words} suspicious breaks)")

    print()
    if passed:
        print(f"✅ PASS: {path.name}")
    else:
        print(f"❌ FAIL: {path.name}")

    for issue in issues:
        print(f"   ⚠ {issue}")

    # Print a sample for eyeball check
    print("\n--- First 400 chars (visual inspection) ---")
    print(text[:400])
    print("--- End sample ---")

    result["passed"] = passed
    result["issues"] = issues
    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: Provide at least one PDF path.")
        sys.exit(1)

    pdf_paths = [Path(p) for p in sys.argv[1:]]
    results = [analyze_pdf(p) for p in pdf_paths]

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for r in results:
        if r.get("error"):
            print(f"  {r['file']}: ERROR — {r['error']}")
            all_passed = False
        elif r["passed"]:
            print(f"  ✅ {r['file']} ({r['primary_language_guess']})")
        else:
            print(f"  ❌ {r['file']} — {'; '.join(r['issues'])}")
            all_passed = False

    print()
    if all_passed:
        print("✅ ALL CHECKS PASSED — proceed to sanity check #2 (embeddings)")
        sys.exit(0)
    else:
        print("❌ FAILURES DETECTED — do not proceed to Day 1 Hour 3 until fixed")
        print("\nRemediation paths:")
        print("  - Arabic garbled: try pdfplumber as alternative extractor")
        print("  - Scanned PDF (low char count): add Tesseract OCR or reject format")
        print("  - Encoding replacement chars: update PyMuPDF (pip install -U pymupdf)")
        sys.exit(1)


if __name__ == "__main__":
    main()
