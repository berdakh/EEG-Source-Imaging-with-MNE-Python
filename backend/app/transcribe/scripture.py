"""Detect scripture references like 'John 3:16' or '1 Corinthians 13:4-7' in text.

Verification against a local public-domain Bible text is a follow-up; this module
only finds and normalizes references so the UI can flag them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BOOKS = [
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms",
    "Psalm",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation",
]

# Longest names first so "1 John" wins over "John".
_BOOK_ALT = "|".join(re.escape(b) for b in sorted(BOOKS, key=len, reverse=True))
_REF = re.compile(
    rf"\b(?P<book>{_BOOK_ALT})\s+(?P<chapter>\d{{1,3}})(?::(?P<verse>\d{{1,3}})"
    rf"(?:\s*[-–]\s*(?P<verse_end>\d{{1,3}}))?)?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScriptureRef:
    book: str
    chapter: int
    verse: int | None = None
    verse_end: int | None = None

    def __str__(self) -> str:
        s = f"{self.book} {self.chapter}"
        if self.verse is not None:
            s += f":{self.verse}"
            if self.verse_end is not None:
                s += f"-{self.verse_end}"
        return s


def _canonical_book(raw: str) -> str:
    low = raw.lower()
    for b in BOOKS:
        if b.lower() == low:
            return "Psalms" if b == "Psalm" else b
    return raw.title()


def find_scripture_refs(text: str) -> list[ScriptureRef]:
    refs: list[ScriptureRef] = []
    seen: set[ScriptureRef] = set()
    for m in _REF.finditer(text):
        ref = ScriptureRef(
            book=_canonical_book(m.group("book")),
            chapter=int(m.group("chapter")),
            verse=int(m.group("verse")) if m.group("verse") else None,
            verse_end=int(m.group("verse_end")) if m.group("verse_end") else None,
        )
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs
