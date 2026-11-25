"""Reference loader + retrieval (RAG) for Mode 2.
Reads PDFs, Word files, text files and web links, splits them into passages, and picks the
passages most relevant to the question (TF-IDF similarity). Those passages become the reference
that the Gemini judge checks claims against."""

import io
import re

import requests
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_WORDS = 180      # words per passage
OVERLAP_WORDS = 40     # overlap so sentences cut at a boundary are not lost
MAX_REF_WORDS = 1200   # cap on how much text goes to the judge
TOP_K = 5


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ------------------------------------------------------------------ loaders
def load_pdf(name: str, data: bytes) -> list:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append({"source": name, "page": i, "text": text})
    return pages


def load_docx(name: str, data: bytes) -> list:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    text = _clean(" ".join(parts))
    return [{"source": name, "page": None, "text": text}] if text else []


def load_txt(name: str, data: bytes) -> list:
    text = _clean(data.decode("utf-8", errors="ignore"))
    return [{"source": name, "page": None, "text": text}] if text else []


def load_url(url: str) -> list:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (PS86 reference loader)"})
    resp.raise_for_status()
    if "pdf" in resp.headers.get("Content-Type", "").lower() or url.lower().endswith(".pdf"):
        return load_pdf(url, resp.content)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = _clean(main.get_text(" "))
    return [{"source": url, "page": None, "text": text}] if text else []


def load_file(name: str, data: bytes) -> list:
    n = name.lower()
    if n.endswith(".pdf"):
        return load_pdf(name, data)
    if n.endswith(".docx"):
        return load_docx(name, data)
    if n.endswith((".txt", ".md")):
        return load_txt(name, data)
    raise ValueError(f"Unsupported file type: {name} (use PDF, DOCX, TXT or MD)")


# ------------------------------------------------------------------ chunk + retrieve
def chunk(docs: list) -> list:
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for d in docs:
        words = d["text"].split()
        for start in range(0, max(1, len(words)), step):
            piece = words[start:start + CHUNK_WORDS]
            if len(piece) < 15 and start > 0:
                break
            chunks.append({"source": d["source"], "page": d["page"], "text": " ".join(piece)})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


def retrieve(question: str, chunks: list, k: int = TOP_K, max_words: int = MAX_REF_WORDS) -> list:
    """Return the passages most relevant to the question, each with a similarity score."""
    if not chunks:
        return []
    total = sum(len(c["text"].split()) for c in chunks)
    if total <= max_words:  # small reference: use everything, in original order
        return [dict(c, score=None) for c in chunks]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform([c["text"] for c in chunks] + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    picked, words = [], 0
    for i in ranked[:k]:
        if scores[i] <= 0:  # no word overlap with the question: not relevant
            break
        n = len(chunks[i]["text"].split())
        if words + n > max_words and picked:
            break
        picked.append(dict(chunks[i], score=round(float(scores[i]), 3)))
        words += n
    return picked


def label(c: dict) -> str:
    return f"{c['source']}" + (f", p. {c['page']}" if c.get("page") else "")


def build_reference(passages: list) -> str:
    """Join passages into one reference text. Source labels go in brackets so evidence quotes
    still match the passage text exactly."""
    return "\n\n".join(f"[{label(p)}] {p['text']}" for p in passages)