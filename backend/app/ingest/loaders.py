"""Turn files into a list of pages of plain text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


@dataclass
class Page:
    text: str
    page: int | None = None


def load_file(path: Path) -> list[Page]:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return [Page(path.read_text(encoding="utf-8", errors="replace"))]
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, p in enumerate(reader.pages, start=1):
            text = p.extract_text() or ""
            if text.strip():
                pages.append(Page(text, page=i))
        return pages
    if ext == ".docx":
        import docx

        d = docx.Document(str(path))
        text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
        return [Page(text)]
    raise ValueError(f"Unsupported file type: {ext}")
