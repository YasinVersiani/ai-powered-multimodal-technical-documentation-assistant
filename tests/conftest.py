"""
Shared test fixtures.

OpenAI is never called: ChatOpenAI and OpenAIEmbeddings are replaced with small fakes,
so the tests run offline and for free. Everything else (PyMuPDF, LangChain, Chroma) is real.
"""
import hashlib
import os
import re
import sys

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import app  # noqa: E402


# ---------------------------------------------------------------------------- fakes
class FakeChat:
    """Stands in for ChatOpenAI and records every request."""
    calls = []

    def __init__(self, model=None, **kwargs):
        self.model = model

    def invoke(self, messages):
        FakeChat.calls.append(messages)
        return AIMessage(content="Hydraulic circuit diagram showing pump XR-200 connected in a loop (p. 2).")


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding, so retrieval behaves like keyword search."""
    def _embed(self, text):
        vec = [0.0] * 256
        for word in re.findall(r"[a-z0-9-]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Run every test in a temp folder with OpenAI replaced by fakes."""
    FakeChat.calls = []
    monkeypatch.setattr(app, "ChatOpenAI", FakeChat)
    monkeypatch.setattr(app, "OpenAIEmbeddings", lambda **kw: KeywordEmbeddings())
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------- sample PDF
def _png(width, height, rgb):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.set_rect(pix.irect, rgb)
    pix.set_rect(pymupdf.IRect(width // 4, height // 4, width // 2, height // 2), (255, 255, 255))
    return pix.tobytes("png")


@pytest.fixture
def sample_pdf(tmp_path):
    """
    4-page manual:
      p1  text + photo with caption 'Figure 1-1' + logo
      p2  text + vector schematic with caption 'Figure 2-1' + logo + footer line
      p3  scanned page (image only, no text layer)
      p4  text only + logo
    Expected visuals: p1 raster, p2 vector, p3 scan. The logo and footer line must be ignored.
    """
    doc = pymupdf.open()
    logo = _png(120, 120, (200, 0, 0))
    body = "This manual describes the XR-200 hydraulic pump and its maintenance. " * 6

    page = doc.new_page()
    page.insert_image(pymupdf.Rect(500, 20, 560, 80), stream=logo)
    page.insert_textbox(pymupdf.Rect(50, 100, 550, 200), body, fontsize=10)
    page.insert_image(pymupdf.Rect(100, 220, 400, 420), stream=_png(300, 200, (0, 90, 200)))
    page.insert_text((100, 440), "Figure 1-1: XR-200 pump assembly overview", fontsize=10)

    page = doc.new_page()
    page.insert_image(pymupdf.Rect(500, 20, 560, 80), stream=logo)
    page.insert_textbox(pymupdf.Rect(50, 100, 550, 200), body, fontsize=10)
    for i in range(6):
        page.draw_rect(pymupdf.Rect(100 + i * 60, 250, 140 + i * 60, 290))
        if i < 5:
            page.draw_line((140 + i * 60, 270), (160 + i * 60, 270))
    page.draw_line((120, 290), (120, 400))
    page.draw_line((120, 400), (420, 400))
    page.draw_line((420, 400), (420, 290))
    page.insert_text((100, 430), "Figure 2-1: Hydraulic circuit schematic", fontsize=10)
    page.draw_line((50, 780), (550, 780))

    scan_src = pymupdf.open()
    scan_page = scan_src.new_page()
    scan_page.insert_textbox(scan_page.rect + (50, 50, -50, -50),
                             "Maintenance table: oil change every 500 h.", fontsize=14)
    page = doc.new_page()
    page.insert_image(page.rect, stream=scan_page.get_pixmap(dpi=72).tobytes("png"))

    page = doc.new_page()
    page.insert_image(pymupdf.Rect(500, 20, 560, 80), stream=logo)
    page.insert_textbox(pymupdf.Rect(50, 100, 550, 400), body * 2, fontsize=10)

    path = tmp_path / "sample.pdf"
    doc.save(path)
    return str(path)
