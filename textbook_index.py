"""
Textbook index for RAG-style grounding (Step 3 of the agent upgrade).

Extracts text from grade12math.pdf once, stores page-level chunks in
textbook_chunks.json, and provides a lightweight keyword search.

No embedding API is used on purpose:
- zero token cost for retrieval
- works offline and on Streamlit Cloud
- good enough to ground lessons on real textbook wording

Run directly to (re)build the index:
    python textbook_index.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_FILE = BASE_DIR / "grade12math.pdf"
CHUNKS_FILE = BASE_DIR / "textbook_chunks.json"

# Words too common to be useful for scoring
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "with", "that", "this", "it", "as", "be", "by", "at", "from", "we", "you",
    "your", "can", "will", "what", "how", "when", "which", "each", "their",
}

_MAX_CHUNK_CHARS = 1600


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return [w for w in words if w not in _STOPWORDS]


def build_index(pdf_path: Path | None = None, out_file: Path | None = None) -> int:
    """
    Extract text per PDF page and save chunks to JSON.

    Returns:
        Number of chunks written.
    """
    import fitz  # pymupdf; imported lazily so the app can run without it

    pdf = Path(pdf_path) if pdf_path else PDF_FILE
    out = Path(out_file) if out_file else CHUNKS_FILE
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    chunks: list[dict] = []
    doc = fitz.open(pdf)
    try:
        for page_number, page in enumerate(doc, start=1):
            text = re.sub(r"\s+", " ", page.get_text() or "").strip()
            if not text:
                continue
            # Split long pages into smaller chunks so results stay focused
            for i in range(0, len(text), _MAX_CHUNK_CHARS):
                piece = text[i : i + _MAX_CHUNK_CHARS].strip()
                if len(piece) < 80:
                    continue
                chunks.append({"page": page_number, "text": piece})
    finally:
        doc.close()

    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    return len(chunks)


def load_chunks(chunks_file: Path | None = None) -> list[dict]:
    path = Path(chunks_file) if chunks_file else CHUNKS_FILE
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def search_chunks(query: str, k: int = 3, chunks_file: Path | None = None) -> list[dict]:
    """
    Keyword-overlap search over the textbook chunks.

    Returns up to ``k`` chunks sorted by score (best first). Each result is
    ``{"page": int, "text": str, "score": float}``.
    """
    chunks = load_chunks(chunks_file)
    query_tokens = set(_tokenize(query))
    if not chunks or not query_tokens:
        return []

    scored: list[tuple[float, dict]] = []
    for chunk in chunks:
        tokens = _tokenize(chunk.get("text", ""))
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in query_tokens)
        if hits == 0:
            continue
        # Normalize by chunk length so short focused chunks rank fairly
        score = hits / (len(tokens) ** 0.5)
        scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"page": c["page"], "text": c["text"], "score": round(s, 4)}
        for s, c in scored[:k]
    ]


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks from {PDF_FILE.name} -> {CHUNKS_FILE.name}")
