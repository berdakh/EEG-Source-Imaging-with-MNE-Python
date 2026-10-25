from app.transcribe import find_scripture_refs


def test_finds_simple_reference():
    refs = find_scripture_refs("As it says in John 3:16, God so loved the world.")
    assert [str(r) for r in refs] == ["John 3:16"]


def test_numbered_books_and_ranges():
    refs = find_scripture_refs("Turn to 1 Corinthians 13:4-7 and then 1 John 4:8.")
    assert [str(r) for r in refs] == ["1 Corinthians 13:4-7", "1 John 4:8"]


def test_chapter_only_and_psalm_alias():
    refs = find_scripture_refs("We read Psalm 23 and Matthew 7.")
    assert [str(r) for r in refs] == ["Psalms 23", "Matthew 7"]


def test_deduplicates():
    refs = find_scripture_refs("Romans 8:28. Again, Romans 8:28.")
    assert len(refs) == 1
