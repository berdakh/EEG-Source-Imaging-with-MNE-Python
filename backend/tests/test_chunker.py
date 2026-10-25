import pytest

from app.ingest import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_short_text_is_one_chunk():
    assert chunk_text("a b c", size=10, overlap=2) == ["a b c"]


def test_chunks_overlap():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words), size=10, overlap=3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert " ".join(chunks[-1].split()[-1:]) == "w24"


def test_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b", size=5, overlap=5)
