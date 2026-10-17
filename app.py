import sys
# SQLite patch for Streamlit Cloud (ChromaDB needs SQLite >= 3.35)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Local machines usually ship a recent SQLite

import streamlit as st
import os
from dotenv import load_dotenv
import shutil
import base64
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- ARCHITECTURE STACK ---
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

# --- MULTIMODAL STACK ---
import pymupdf  # Image extraction, vector diagram detection & page rendering


# Load environment variables
load_dotenv()

# --- SYSTEM PARAMETERS ---
MODEL_OPTIONS = ["gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-5.1"]
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_TEMPERATURE = 1            # GPT-5 models only accept the default temperature (1)

CHUNK_SIZE = 1000              # Characters per text chunk
CHUNK_OVERLAP = 250            # Overlap for context continuity between chunks
TOP_K = 4                      # Chunks retrieved per query (text or visual)

TEMP_FILE = "./temp_manual.pdf"
IMAGE_DIR = "./extracted_images"
CACHE_FILE = "./image_cache/descriptions.json"

MIN_IMAGE_PX = 100             # Smaller images are icons/bullets
LOGO_PAGE_COUNT = 3            # Image repeated on this many pages = logo/header
MIN_PAGE_TEXT = 50             # Less text than this = scanned page
MIN_DRAWING_SHARE = 0.04       # Vector region must cover >= 4% of the page
MAX_DRAWING_SHARE = 0.90       # ...but < 90% (otherwise it's a page frame)
MIN_DRAWING_SIDE = 30          # Thinner regions are separator lines
RENDER_DPI = 150               # Resolution for rendered diagrams & scans
MAX_VISUALS = 60               # Cost guard per document
VISION_WORKERS = 4             # Parallel vision API calls
MAX_IMAGES_PER_ANSWER = 3      # Original images attached to one answer
CAPTION_PREFIXES = ("fig", "figure", "table", "diagram", "abb", "tab.")

VISUAL_LABELS = {"raster": "Figure", "vector": "Diagram", "scan": "Scanned Page"}

# Chat avatars (monochrome Material icons)
AVATARS = {"user": ":material/person:", "assistant": ":material/description:"}

# --- PROMPTS ---
# System Prompt for Technical Persona
SYSTEM_PROMPT = (
    "You are an expert Technical Solutions Architect specializing in industrial systems. "
    "Use the provided context to answer the user's technical query accurately. "
    "Maintain a professional, concise, and engineering-focused tone. "
    "Keep values, units, part numbers and warnings exactly as written. "
    "Some context entries describe figures, diagrams or scanned pages, and the original images may be attached - use them to verify details. "
    "Cite the page for every fact, e.g. (p. 12). "
    "If the answer is not in the context, state that information is missing from the provided document. "
    "Context: {context}"
)

# Vision Prompt for indexing visuals
VISION_PROMPT = (
    "You are indexing a visual element from a technical manual so engineers can find it with search. "
    "Describe it precisely and factually:\n"
    "1. Type of visual (wiring diagram, exploded view, flowchart, table, chart, photo, screenshot, scanned text page, ...).\n"
    "2. Transcribe ALL readable text: labels, part numbers, pin names, values with units, table cells, warnings.\n"
    "3. Explain what it shows: components and connections, procedure steps, chart trends.\n"
    "Never guess unreadable values - write 'illegible'. No introduction."
)


# --- VISUAL EXTRACTION ENGINE ---
def find_caption(page, rect, max_gap=60.0):
    """Find the caption above/below a visual ('Figure ...' blocks win over plain text)."""
    candidates = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        text = " ".join(text.split())
        if not text or x0 > rect.x1 or x1 < rect.x0:
            continue
        if y0 >= rect.y1:
            gap = y0 - rect.y1  # Below the visual
        elif y1 <= rect.y0:
            gap = rect.y0 - y1  # Above the visual
        else:
            continue  # Inside the visual = labels, not a caption
        if gap <= max_gap:
            is_caption = text.lower().startswith(CAPTION_PREFIXES)
            candidates.append((not is_caption, gap, text))
    return min(candidates)[2][:300] if candidates else ""


def extract_visuals(pdf_path, out_dir):
    """Extract embedded images, vector diagrams and scanned pages as PNGs. Returns (visuals, truncated)."""
    os.makedirs(out_dir, exist_ok=True)
    visuals, seen = [], set()

    def keep(png, page_no, kind, caption):
        digest = hashlib.sha256(png).hexdigest()[:16]
        if digest in seen:  # Skip duplicate images
            return
        seen.add(digest)
        path = os.path.join(out_dir, f"p{page_no + 1}_{kind}_{digest}.png")
        with open(path, "wb") as f:
            f.write(png)
        visuals.append({"path": path, "hash": digest, "page": page_no, "kind": kind, "caption": caption})

    with pymupdf.open(pdf_path) as pdf:
        # Count pages per image to detect logos/headers
        pages_using = {}
        for page in pdf:
            for xref in {img[0] for img in page.get_images(full=True)}:
                pages_using[xref] = pages_using.get(xref, 0) + 1

        for page in pdf:
            if len(visuals) >= MAX_VISUALS:
                return visuals, True

            # Scanned page -> render the full page
            if len(page.get_text().strip()) < MIN_PAGE_TEXT:
                keep(page.get_pixmap(dpi=RENDER_DPI).tobytes("png"), page.number, "scan", "")
                continue

            # Embedded raster images
            for img in page.get_images(full=True):
                xref, width, height = img[0], img[2], img[3]
                if pages_using[xref] >= LOGO_PAGE_COUNT or min(width, height) < MIN_IMAGE_PX:
                    continue
                try:
                    pix = pymupdf.Pixmap(pdf, xref)
                    if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    png = pix.tobytes("png")
                except Exception:
                    continue  # Skip unsupported image encodings
                placements = page.get_image_rects(xref)
                caption = find_caption(page, placements[0]) if placements else ""
                keep(png, page.number, "raster", caption)

            # Vector diagrams -> cluster drawing paths and render each region
            page_area = page.rect.width * page.rect.height
            try:
                regions = page.cluster_drawings()
            except Exception:
                regions = []
            for rect in regions:
                share = (rect.width * rect.height) / page_area
                if not (MIN_DRAWING_SHARE <= share < MAX_DRAWING_SHARE):
                    continue
                if min(rect.width, rect.height) < MIN_DRAWING_SIDE:
                    continue
                clip = (rect + (-8, -8, 8, 8)) & page.rect  # Small margin, stay on page
                png = page.get_pixmap(dpi=RENDER_DPI, clip=clip).tobytes("png")
                keep(png, page.number, "vector", find_caption(page, rect))

    return visuals, False


# --- VISION ENGINE ---
def image_block(path):
    """Encode a PNG as an OpenAI image message block."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}", "detail": "high"}}


def describe_visual(llm, visual):
    """Generate a searchable text description of one visual."""
    hint = f"Caption or nearby text: {visual['caption']}" if visual["caption"] else "No caption found."
    message = HumanMessage(content=[
        {"type": "text", "text": f"{VISION_PROMPT}\n\n{hint}"},
        image_block(visual["path"]),
    ])
    return llm.invoke([message]).content


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def describe_visuals(visuals, vision_model, on_progress=None):
    """Describe all visuals in parallel, re-using cached results. Returns ({hash: description}, [error messages])."""
    cache = load_cache()
    key = lambda v: f"{vision_model}:{v['hash']}"  # Same image + model = never paid twice
    todo = [v for v in visuals if key(v) not in cache]
    done, errors = len(visuals) - len(todo), []
    if on_progress:
        on_progress(done, len(visuals))

    if todo:
        llm = ChatOpenAI(model=vision_model, temperature=LLM_TEMPERATURE)
        with ThreadPoolExecutor(max_workers=VISION_WORKERS) as pool:
            jobs = {pool.submit(describe_visual, llm, v): v for v in todo}
            for job in as_completed(jobs):
                try:
                    cache[key(jobs[job])] = job.result()
                except Exception as e:
                    errors.append(str(e))
                    print(f"[Vision] Failed on page {jobs[job]['page'] + 1}: {e}")  # Visible in the terminal
                done += 1
                if on_progress:
                    on_progress(done, len(visuals))
        save_cache(cache)

    descriptions = {v["hash"]: cache[key(v)] for v in visuals if key(v) in cache}
    return descriptions, errors


def visual_documents(visuals, descriptions, source):
    """Wrap descriptions as Documents. The image path is kept in metadata for evidence & answers."""
    docs = []
    for v in visuals:
        if v["hash"] not in descriptions:
            continue
        text = f"{VISUAL_LABELS[v['kind']]} (page {v['page'] + 1})\n"
        if v["caption"]:
            text += f"Caption: {v['caption']}\n"
        text += descriptions[v["hash"]]
        docs.append(Document(
            page_content=text,
            metadata={"source": source, "page": v["page"], "type": "visual",
                      "kind": v["kind"], "image_path": v["path"]},
        ))
    return docs


# --- RAG PIPELINE ---
def build_vector_index(pdf_path, filename, vision_enabled, vision_model, on_progress=None):
    """PDF -> text chunks + visual descriptions -> ChromaDB index. Returns (vector_db, stats)."""
    # Load and Chunk
    docs = PyPDFLoader(pdf_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    for split in splits:
        split.metadata.update(source=filename, type="text")
    stats = {"text_chunks": len(splits), "visuals": 0, "failed": 0, "first_error": None, "truncated": False}

    # Extract and Describe Visuals
    visual_docs = []
    if vision_enabled:
        shutil.rmtree(IMAGE_DIR, ignore_errors=True)
        visuals, truncated = extract_visuals(pdf_path, IMAGE_DIR)
        descriptions, errors = describe_visuals(visuals, vision_model, on_progress)
        visual_docs = visual_documents(visuals, descriptions, filename)
        stats.update(visuals=len(visual_docs), failed=len(errors),
                     first_error=errors[0] if errors else None, truncated=truncated)

    all_docs = splits + visual_docs
    if not all_docs:
        raise ValueError("No text or visuals could be extracted from this PDF.")

    # Generate Embeddings and Index (fresh in-memory collection per document)
    vector_db = Chroma.from_documents(
        documents=all_docs,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=f"manual_{uuid.uuid4().hex[:12]}",
    )
    return vector_db, stats


def is_visual(doc):
    return doc.metadata.get("type") == "visual"


def page_of(doc):
    return doc.metadata.get("page", 0) + 1


def build_messages(question, docs, attach_images):
    """Text context in the system prompt + original images of retrieved visuals in the user message."""
    context = "\n\n---\n\n".join(
        f"[Source {i} | {'visual' if is_visual(d) else 'text'} | p. {page_of(d)}]\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    content = [{"type": "text", "text": question}]
    if attach_images:
        for doc in [d for d in docs if is_visual(d)][:MAX_IMAGES_PER_ANSWER]:
            path = doc.metadata.get("image_path")
            if path and os.path.exists(path):
                content.append({"type": "text", "text": f"Original image from page {page_of(doc)}:"})
                content.append(image_block(path))
    user_message = HumanMessage(content=content if len(content) > 1 else question)
    return [SystemMessage(content=SYSTEM_PROMPT.format(context=context)), user_message]


def generate_answer(question, vector_db, model, attach_images):
    """Retrieve top-k context and generate the answer. Returns (answer, docs)."""
    docs = vector_db.similarity_search(question, k=TOP_K)
    # All GPT-5 models require default temperature=1 (cannot be customized)
    llm = ChatOpenAI(model=model, temperature=LLM_TEMPERATURE)
    response = llm.invoke(build_messages(question, docs, attach_images))
    return response.content, docs


def evidence_from(docs):
    """Convert retrieved Documents into plain dicts so evidence survives in chat history."""
    return [{
        "label": VISUAL_LABELS.get(d.metadata.get("kind"), "Figure") if is_visual(d) else "Text Chunk",
        "page": page_of(d),
        "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
        "image_path": d.metadata.get("image_path"),
    } for d in docs]


def render_evidence(evidence):
    # Display Evidence (White-box AI)
    with st.expander("View Retrieval Context (Evidence)"):
        for i, item in enumerate(evidence, start=1):
            st.caption(f"Source {i}: {item['label']} · Page {item['page']}")
            if item.get("image_path") and os.path.exists(item["image_path"]):
                st.image(item["image_path"], width=420)
            st.code(item["snippet"], language="text", wrap_lines=True)


def drop_index():
    """Delete the current Chroma collection from memory."""
    if st.session_state.get("vector_db") is not None:
        try:
            st.session_state.vector_db.delete_collection()
        except Exception:
            pass
    st.session_state.vector_db = None
    st.session_state.current_file = None
    st.session_state.index_stats = None


# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI-Powered Multimodal Technical Documentation Assistant",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in environment variables. Please check your .env file or Streamlit secrets.")
    st.stop()

# Custom CSS for cleaner UI
st.markdown("""
<style>
    header {visibility: hidden;}
    .main {padding-top: 2rem;}
    div.stToast { background-color: #2e7bcf; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "index_stats" not in st.session_state:
    st.session_state.index_stats = None

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("System Controls")

    # Connectivity check
    st.success("API Gateway: Online")

    st.divider()

    # Inference Model Selection
    st.subheader("Inference Architecture")
    model_choice = st.selectbox(
        "Select Model:",
        MODEL_OPTIONS,
        index=0,
        help="Select 'Nano' for fastest/cheapest, 'Mini' for speed, '5' for balanced, or '5.1' for complex reasoning."
    )

    st.info(f"Active Configuration: {model_choice}")
    st.divider()

    # Multimodal Settings
    st.subheader("Vision Pipeline")
    vision_enabled = st.toggle(
        "Multimodal Ingestion",
        value=True,
        help="Index embedded images, vector diagrams and scanned pages using a vision model."
    )
    vision_model = st.selectbox(
        "Vision Model:",
        MODEL_OPTIONS,
        index=0,
        disabled=not vision_enabled,
        help="Runs once per new image during ingestion. Descriptions are cached."
    )
    attach_images = st.toggle(
        "Send Images to Inference Model",
        value=True,
        disabled=not vision_enabled,
        help="Attach retrieved figures to the query for more accurate answers (uses more tokens)."
    )
    st.divider()

    # Data Ingestion
    st.subheader("Knowledge Base")
    uploaded_file = st.file_uploader("Ingest Technical PDF", type="pdf")

    if st.session_state.index_stats:
        stats = st.session_state.index_stats
        st.caption(f"Indexed: {stats['text_chunks']} text chunks · {stats['visuals']} visuals")

    # Clear Context
    if st.button("Clear Session Context", use_container_width=True):
        drop_index()
        st.session_state.messages = []
        # Clean up temp files and extracted images (description cache is kept to save API cost)
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)
        shutil.rmtree(IMAGE_DIR, ignore_errors=True)
        st.rerun()

# --- MAIN DASHBOARD ---
st.title("Multimodal Retrieval Engine")
st.caption(
    f"Architecture: Multimodal RAG | Model: {model_choice} | "
    f"Vision: {vision_model if vision_enabled else 'Off'} | Semantic Search Enabled"
)

# --- ETL PIPELINE ---
if uploaded_file is not None:
    # Re-index when the file content or vision settings change
    file_bytes = uploaded_file.getvalue()
    ingestion_key = f"{hashlib.sha256(file_bytes).hexdigest()[:16]}|{vision_enabled}|{vision_model}"

    if st.session_state.current_file != ingestion_key:
        with st.spinner("Processing Document Vectors & Building Index..."):
            try:
                # Save temp file
                with open(TEMP_FILE, "wb") as f:
                    f.write(file_bytes)

                # Progress bar for the vision pipeline
                progress_bar = st.progress(0.0, text="Scanning visuals...") if vision_enabled else None

                def update_progress(done, total):
                    progress_bar.progress(done / total if total else 1.0,
                                          text=f"Analyzing visuals: {done}/{total}")

                drop_index()
                vector_db, stats = build_vector_index(
                    TEMP_FILE, uploaded_file.name, vision_enabled, vision_model,
                    on_progress=update_progress if progress_bar else None
                )
                if progress_bar:
                    progress_bar.empty()

                st.session_state.vector_db = vector_db
                st.session_state.current_file = ingestion_key
                st.session_state.index_stats = stats
                st.toast(f"Ingestion Complete: {stats['text_chunks']} text chunks + {stats['visuals']} visuals indexed.")

                if stats["truncated"]:
                    st.warning(f"Only the first {MAX_VISUALS} visuals were analyzed (cost limit).")
                if stats["failed"]:
                    st.warning(f"{stats['failed']} visual(s) could not be analyzed and were skipped. "
                               f"Reason: {stats['first_error']}")

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.session_state.vector_db = None

# --- RENDER CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message.get("evidence"):
            render_evidence(message["evidence"])

# --- INFERENCE ENGINE ---
if prompt := st.chat_input("Query the knowledge base..."):
    # Append User Query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    # Ensure DB is active
    if st.session_state.vector_db is not None:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analyzing context...")

            try:
                # Execute Multimodal RAG
                full_response, retrieved_docs = generate_answer(
                    prompt, st.session_state.vector_db, model_choice,
                    attach_images=vision_enabled and attach_images
                )

                # Render Response
                message_placeholder.markdown(full_response)
                evidence = evidence_from(retrieved_docs)
                render_evidence(evidence)

                # Update History
                st.session_state.messages.append({"role": "assistant", "content": full_response, "evidence": evidence})

            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

    else:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.error("System Standby: Please ingest a document to begin analysis.")