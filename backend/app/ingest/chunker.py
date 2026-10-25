"""Word-based chunking with overlap. Simple on purpose; tune after real documents."""


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks
