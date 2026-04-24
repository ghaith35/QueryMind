import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.indexing.extractor import is_garbled, validate_extraction


def test_is_garbled_short_text():
    assert is_garbled("tiny text")


def test_is_garbled_printable_text_passes():
    assert not is_garbled("This is a normal extracted PDF paragraph with readable content.")


def test_validate_extraction_rejects_scanned_pdf_like_output():
    ok, reason = validate_extraction([
        {"page_number": 1, "text": "abc", "char_count": 3, "blocks": []},
    ])

    assert ok is False
    assert "scanned" in reason.lower()
