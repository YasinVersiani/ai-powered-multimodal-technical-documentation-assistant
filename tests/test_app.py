"""Run with:  pytest -q"""
import os

from conftest import FakeChat

import app


def build(pdf_path, vision_enabled=True):
    return app.build_vector_index(pdf_path, "sample.pdf", vision_enabled, "gpt-5-mini")


# --- VISUAL EXTRACTION ---
def test_extracts_raster_vector_and_scanned_page(sample_pdf):
    visuals, truncated = app.extract_visuals(sample_pdf, app.IMAGE_DIR)
    found = sorted((v["page"] + 1, v["kind"]) for v in visuals)
    assert found == [(1, "raster"), (2, "vector"), (3, "scan")]   # Logo + footer line ignored
    assert not truncated
    assert all(os.path.exists(v["path"]) for v in visuals)


def test_finds_captions(sample_pdf):
    visuals, _ = app.extract_visuals(sample_pdf, app.IMAGE_DIR)
    captions = {v["kind"]: v["caption"] for v in visuals}
    assert captions["raster"].startswith("Figure 1-1")
    assert captions["vector"].startswith("Figure 2-1")


def test_respects_visual_limit(sample_pdf, monkeypatch):
    monkeypatch.setattr(app, "MAX_VISUALS", 1)
    visuals, truncated = app.extract_visuals(sample_pdf, app.IMAGE_DIR)
    assert len(visuals) == 1 and truncated


# --- VISION ENGINE ---
def test_descriptions_are_cached(sample_pdf):
    visuals, _ = app.extract_visuals(sample_pdf, app.IMAGE_DIR)
    first, errors = app.describe_visuals(visuals, "gpt-5-mini")
    assert len(first) == 3 and errors == [] and len(FakeChat.calls) == 3

    second, _ = app.describe_visuals(visuals, "gpt-5-mini")
    assert second == first and len(FakeChat.calls) == 3          # No new API calls

    image_part = FakeChat.calls[0][0].content[1]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_vision_errors_are_reported(sample_pdf, monkeypatch):
    class BrokenChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            raise RuntimeError("Unsupported value: 'temperature'")

    monkeypatch.setattr(app, "ChatOpenAI", BrokenChat)
    db, stats = build(sample_pdf)
    assert stats["visuals"] == 0 and stats["failed"] == 3
    assert "temperature" in stats["first_error"]


def test_models_use_supported_temperature():
    assert app.LLM_TEMPERATURE == 1


# --- RAG PIPELINE ---
def test_index_contains_text_and_visuals(sample_pdf):
    db, stats = build(sample_pdf)
    assert stats["text_chunks"] > 0 and stats["visuals"] == 3
    assert {m["type"] for m in db.get()["metadatas"]} == {"text", "visual"}


def test_text_only_mode_skips_visuals(sample_pdf):
    db, stats = build(sample_pdf, vision_enabled=False)
    assert stats["visuals"] == 0 and FakeChat.calls == []
    assert {m["type"] for m in db.get()["metadatas"]} == {"text"}


def test_answer_attaches_retrieved_images(sample_pdf):
    db, _ = build(sample_pdf)
    answer, docs = app.generate_answer("Explain the hydraulic circuit diagram", db, "gpt-5-mini", attach_images=True)
    assert answer and len(docs) == app.TOP_K
    system, user = FakeChat.calls[-1]
    assert "[Source 1 |" in system.content
    images = [p for p in user.content if isinstance(p, dict) and p.get("type") == "image_url"]
    assert 1 <= len(images) <= app.MAX_IMAGES_PER_ANSWER


def test_answer_without_images_sends_plain_text(sample_pdf):
    db, _ = build(sample_pdf)
    app.generate_answer("Explain the hydraulic circuit diagram", db, "gpt-5-mini", attach_images=False)
    assert isinstance(FakeChat.calls[-1][1].content, str)


def test_reindexing_does_not_mix_documents(sample_pdf):
    first, _ = build(sample_pdf)
    count = len(first.get()["ids"])
    first.delete_collection()
    second, _ = build(sample_pdf)
    assert len(second.get()["ids"]) == count


# --- UI ---
def test_app_starts_and_handles_query_without_document():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(os.path.dirname(app.__file__), "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert any(t.value == "Multimodal Retrieval Engine" for t in at.title)
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert "ingest a document" in at.error[0].value