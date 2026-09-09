"""Deterministic de-identification and near-duplicate utilities."""

from __future__ import annotations

from html import unescape
import hashlib
import re
import unicodedata
from typing import Iterable


_HTML = re.compile(r"<[^>]+>")
_EMAIL = re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
_HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,32}\b")
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[\w']+", re.UNICODE)


def pii_counts(text: str) -> dict[str, int]:
    value = str(text or "")
    return {
        "email": len(_EMAIL.findall(value)),
        "url": len(_URL.findall(value)),
        "phone": len(_PHONE.findall(value)),
        "handle": len(_HANDLE.findall(value)),
    }


def deidentify(text: str) -> str:
    value = unicodedata.normalize("NFC", unescape(str(text or "")))
    value = _HTML.sub(" ", value)
    value = _EMAIL.sub("[EMAIL]", value)
    value = _URL.sub("[URL]", value)
    value = _PHONE.sub("[PHONE]", value)
    value = _HANDLE.sub("[HANDLE]", value)
    return _WHITESPACE.sub(" ", value).strip()


def normalized_text(text: str) -> str:
    return " ".join(_TOKEN.findall(deidentify(text).casefold()))


def content_fingerprint(*parts: str) -> str:
    normalized = "\n".join(normalized_text(part) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def token_shingles(text: str, size: int = 3) -> frozenset[tuple[str, ...]]:
    tokens = normalized_text(text).split()
    if not tokens:
        return frozenset()
    if len(tokens) < size:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def jaccard_similarity(left: Iterable[object], right: Iterable[object]) -> float:
    first, second = set(left), set(right)
    union = first | second
    return len(first & second) / len(union) if union else 1.0
