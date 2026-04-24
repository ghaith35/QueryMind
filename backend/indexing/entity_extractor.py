"""
Named entity extraction using spaCy EN + FR models.

Arabic: no free spaCy model with adequate quality. Arabic chunks are fully
indexed and retrievable but contribute no graph nodes in v1. This is a
documented limitation, not a silent failure.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)

ENTITY_TYPE_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "PRODUCT": "concept",
    "WORK_OF_ART": "concept",
    "EVENT": "concept",
    "LAW": "concept",
    "NORP": "concept",     # Nationality / religion / political
    "LANGUAGE": "concept",
}

_nlp_en = None
_nlp_fr = None
_custom_patterns: list[tuple[re.Pattern[str], str]] | None = None

CUSTOM_ENTITIES_PATH = Path(__file__).resolve().parent.parent / "data" / "custom_entities.json"


def _load_models() -> None:
    global _nlp_en, _nlp_fr, _custom_patterns
    if _nlp_en is None:
        import spacy
        _nlp_en = spacy.load(
            "en_core_web_sm",
            disable=["parser", "lemmatizer", "attribute_ruler"],
        )
        log.info("Loaded en_core_web_sm")
    if _nlp_fr is None:
        import spacy
        _nlp_fr = spacy.load(
            "fr_core_news_sm",
            disable=["parser", "lemmatizer", "attribute_ruler"],
        )
        log.info("Loaded fr_core_news_sm")
    if _custom_patterns is None:
        try:
            raw = json.loads(CUSTOM_ENTITIES_PATH.read_text())
            _custom_patterns = [
                (re.compile(item["pattern"], re.IGNORECASE), item["type"])
                for item in raw
                if item.get("pattern") and item.get("type")
            ]
            log.info("Loaded %d custom entity patterns", len(_custom_patterns))
        except Exception as exc:
            log.warning("Failed to load custom entity patterns: %s", exc)
            _custom_patterns = []


def extract_entities(text: str, lang: str) -> List[Tuple[str, str]]:
    """
    Returns [(label, entity_type), ...], deduplicated, max 15.

    entity_type is one of: concept, person, organization, location, technique.
    Arabic chunks return [] — documented limitation.
    """
    _load_models()

    if lang == "en":
        nlp = _nlp_en
    elif lang == "fr":
        nlp = _nlp_fr
    else:
        # Arabic, mixed, or unknown: no NER
        return []

    doc = nlp(text)
    entities: List[Tuple[str, str]] = []
    seen: set = set()

    for ent in doc.ents:
        label = ent.text.strip()
        # Filter noise: too short, too long, or purely numeric
        if len(label) < 3 or len(label) > 60:
            continue
        if label.replace(" ", "").isnumeric():
            continue
        normalized = label.lower()
        if normalized in seen:
            continue
        etype = ENTITY_TYPE_MAP.get(ent.label_, "concept")
        entities.append((label, etype))
        seen.add(normalized)
        if len(entities) >= 15:
            break

    for pattern, entity_type in _custom_patterns or []:
        for match in pattern.finditer(text):
            label = match.group(0).strip()
            normalized = label.lower()
            if normalized in seen:
                continue
            entities.append((label, entity_type))
            seen.add(normalized)
            if len(entities) >= 15:
                return entities

    return entities


def entity_labels(entities: List[Tuple[str, str]]) -> List[str]:
    """Extract just the label strings from entity tuples."""
    return [label for label, _ in entities]


def serialise_entities(entities: List[Tuple[str, str]]) -> str:
    """
    Persist typed entity metadata in SQLite while staying backward-compatible
    with older rows that only stored label strings.
    """
    payload = [{"label": label, "type": entity_type} for label, entity_type in entities]
    return json.dumps(payload, ensure_ascii=False)


def parse_stored_entities(raw: str | None) -> List[Tuple[str, str]]:
    """
    Parse entities_json from SQLite.

    Old rows may contain `["Paris", "UN"]`; newer rows store
    `[{"label": "Paris", "type": "location"}]`.
    """
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    entities: List[Tuple[str, str]] = []
    seen: set[str] = set()

    for item in data:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            entity_type = str(item.get("type", "concept")).strip() or "concept"
        else:
            label = str(item).strip()
            entity_type = "concept"

        if not label:
            continue

        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append((label, entity_type))

    return entities


def stored_entity_labels(raw: str | None) -> List[str]:
    """Return only the entity labels from a stored entities_json payload."""
    return [label for label, _ in parse_stored_entities(raw)]
