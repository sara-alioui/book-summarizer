"""
BookSum Demo — Interface Streamlit
Pipeline Map-Reduce avec PEGASUS + LoRA
"""

import streamlit as st
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ─────────────────────────────────────────────
#  CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BookSum — Automatic Summarization",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Crimson Pro', Georgia, serif !important;
}

/* ── BACKGROUND ── */
.stApp {
    background: #0f0e17 !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #1a1826 !important;
    border-right: 1px solid #2d2b3d !important;
}
section[data-testid="stSidebar"] * {
    color: #c4bfd8 !important;
}

/* ── HERO BANNER ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0533 0%, #0d1f4a 50%, #001a2e 100%);
    border: 1px solid #3d2c6b;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(138,79,255,0.15) 0%, transparent 70%);
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,180,255,0.08) 0%, transparent 70%);
}
.hero-title {
    font-size: 42px;
    font-weight: 300;
    color: #f0eaff;
    letter-spacing: -1px;
    margin-bottom: 8px;
    line-height: 1.1;
}
.hero-title span {
    color: #a78bfa;
    font-style: italic;
}
.hero-sub {
    font-size: 16px;
    color: #8b85a8;
    font-weight: 300;
    letter-spacing: 0.5px;
    margin-bottom: 24px;
}
.hero-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.pill {
    background: rgba(167,139,250,0.12);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 99px;
    padding: 4px 16px;
    font-size: 11px;
    font-weight: 500;
    color: #c4b5fd;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}

/* ── STAT CARDS ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.stat-card {
    background: #1a1826;
    border: 1px solid #2d2b3d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #5b4a9e; }
.stat-num {
    font-size: 32px;
    font-weight: 300;
    color: #a78bfa;
    font-style: italic;
    line-height: 1;
    margin-bottom: 6px;
}
.stat-lbl {
    font-size: 11px;
    color: #5e5a78;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── SECTION TITLES ── */
.section-title {
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    color: #5e5a78;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2d2b3d;
}

/* ── INPUT AREA ── */
.stTextArea textarea {
    background: #1a1826 !important;
    border: 1px solid #2d2b3d !important;
    border-radius: 10px !important;
    color: #e0daf0 !important;
    font-family: 'Crimson Pro', serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}
.stTextArea textarea:focus {
    border-color: #7c5cbf !important;
    box-shadow: 0 0 0 3px rgba(124,92,191,0.15) !important;
}

/* ── SUMMARY OUTPUT ── */
.summary-box {
    background: linear-gradient(135deg, #130e2a, #0d1a2e);
    border: 1px solid #3d2c6b;
    border-radius: 12px;
    padding: 28px 32px;
    margin-top: 8px;
}
.summary-text {
    font-size: 18px;
    font-weight: 300;
    color: #e8e0ff;
    line-height: 1.85;
    font-style: italic;
}
.summary-label {
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    color: #7c5cbf;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 16px;
}

/* ── LOG BOX ── */
.log-box {
    background: #0d0c17;
    border: 1px solid #1e1c2e;
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #6b6585;
    line-height: 1.8;
    white-space: pre-wrap;
}

/* ── STATS BAR ── */
.stats-bar {
    background: #1a1826;
    border: 1px solid #2d2b3d;
    border-radius: 8px;
    padding: 12px 20px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #8b85a8;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 16px;
}
.stats-bar span { color: #a78bfa; font-weight: 500; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #5b4a9e, #3b5998) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(91,74,158,0.4) !important;
}

/* ── SLIDERS ── */
.stSlider [data-baseweb="slider"] {
    padding-top: 8px;
}

/* ── FILE UPLOADER ── */
.stFileUploader {
    border: 1px dashed #2d2b3d !important;
    border-radius: 10px !important;
    background: #1a1826 !important;
}

/* ── HIDE STREAMLIT DEFAULT ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CHEMINS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "google/pegasus-xsum"
MODEL_DIR  = os.path.join(BASE_DIR, "..", "notebooks", "model_finetuned")
DATA_PATH  = os.path.join(BASE_DIR, "..", "notebooks", "donnees_propres.json")

# ─────────────────────────────────────────────
#  CHARGEMENT DU MODELE (cache)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model      = PeftModel.from_pretrained(base_model, MODEL_DIR)
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)   # ✅ depuis HuggingFace
    model      = model.to(device)
    model.eval()
    return model, tokenizer, device

@st.cache_data
def load_examples():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            donnees = json.load(f)
        return [donnees[i]["chunks"][0] for i in range(3)]
    return ["", "", ""]

# ─────────────────────────────────────────────
#  FONCTIONS PIPELINE
# ─────────────────────────────────────────────
def chunker(text: str, chunk_size: int = 512, overlap: int = 50):
    words = text.split()
    step  = max(1, chunk_size - overlap)
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), step) if words[i:i+chunk_size]]

def resumer_chunk(model, tokenizer, device, text: str, max_input: int = 512, max_output: int = 128) -> str:
    inputs = tokenizer(
        text,
        max_length=max_input,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output,
            max_length=None,            # ✅ FIX : supprime le conflit max_length/max_new_tokens
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_pdf_text(pdf_file) -> str:
    try:
        import fitz
        data = pdf_file.read()
        doc  = fitz.open(stream=data, filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        return "❌ PyMuPDF non installé : pip install pymupdf"
    except Exception as e:
        return f"❌ Erreur PDF : {e}"

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📚 Book<span>Sum</span></div>
    <div class="hero-sub">Automatic Literary Summarization · PEGASUS + LoRA · Map-Reduce Pipeline</div>
    <div class="hero-pills">
        <span class="pill">PEGASUS-xsum</span>
        <span class="pill">LoRA r=16</span>
        <span class="pill">ROUGE-1 31.44%</span>
        <span class="pill">BookSum Dataset</span>
        <span class="pill">Map-Reduce</span>
        <span class="pill">GPU T4</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STAT CARDS
# ─────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-num">PEGASUS</div>
        <div class="stat-lbl">Base model + LoRA</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">31.4%</div>
        <div class="stat-lbl">ROUGE-1 score</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">2000</div>
        <div class="stat-lbl">BookSum chapters</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">×3</div>
        <div class="stat-lbl">vs baseline gain</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR — PARAMETRES
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline settings")
    st.markdown("---")

    chunk_size = st.slider("Chunk size (tokens)", 128, 512, 512, step=64)
    overlap    = st.slider("Overlap (tokens)", 0, 100, 50, step=10)
    max_output = st.slider("Max output length", 64, 256, 128, step=32)

    st.markdown("---")
    st.markdown("### 📌 Quick examples")

    examples = load_examples()
    if st.button("📖 Chapter 1", use_container_width=True):
        st.session_state["input_text"] = examples[0]
    if st.button("📖 Chapter 2", use_container_width=True):
        st.session_state["input_text"] = examples[1]
    if st.button("📖 Chapter 3", use_container_width=True):
        st.session_state["input_text"] = examples[2]

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#4a4760; line-height:1.6;">
    <b style="color:#6b5ea8;">Map-Reduce Pipeline</b><br>
    1. Split text into chunks<br>
    2. Summarize each chunk → Map<br>
    3. Merge summaries → Reduce<br><br>
    <b style="color:#6b5ea8;">Fine-tuning</b><br>
    LoRA r=16 · 3 epochs · GPU T4<br>
    BookSum dataset · 500 examples
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN — INPUT
# ─────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="section-title">Text input</div>', unsafe_allow_html=True)
    input_text = st.text_area(
        label="",
        value=st.session_state.get("input_text", ""),
        placeholder="Paste a book chapter here...",
        height=220,
        key="text_input_area",
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<div class="section-title">PDF upload</div>', unsafe_allow_html=True)
    pdf_file = st.file_uploader(
        "",
        type=["pdf"],
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("✦ Summarize", use_container_width=True)
    clear = st.button("Clear", use_container_width=True)
    if clear:
        st.session_state["input_text"] = ""
        st.rerun()

# ─────────────────────────────────────────────
#  MAIN — PIPELINE
# ─────────────────────────────────────────────
if run:
    text = input_text

    # PDF override
    if pdf_file is not None:
        with st.spinner("Extracting PDF..."):
            text = extract_pdf_text(pdf_file)
        if text.startswith("❌"):
            st.error(text)
            st.stop()

    if not text or not text.strip():
        st.warning("Paste some text or upload a PDF first.")
        st.stop()

    # Chargement modèle
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model()

    chunks = chunker(text, chunk_size, overlap)

    st.markdown('<div class="section-title">Pipeline output</div>', unsafe_allow_html=True)

    # Progress bar
    progress = st.progress(0, text=f"Processing {len(chunks)} chunk(s)...")
    log_lines = [f"📄 {len(chunks)} chunk(s) · chunk_size={chunk_size} · overlap={overlap}\n"]

    intermediaires = []
    for i, chunk in enumerate(chunks):
        resume = resumer_chunk(model, tokenizer, device, chunk, max_input=512, max_output=max_output)
        intermediaires.append(resume)
        log_lines.append(f"  ✅ Chunk {i+1}/{len(chunks)} → {len(resume.split())} mots")
        progress.progress((i + 1) / (len(chunks) + 1), text=f"Chunk {i+1}/{len(chunks)}...")

    # Reduce
    progress.progress(len(chunks) / (len(chunks) + 1), text="Reduce — merging summaries...")
    resume_final = resumer_chunk(
        model, tokenizer, device,
        " ".join(intermediaires),
        max_input=512,
        max_output=150
    )
    progress.progress(1.0, text="Done!")
    log_lines.append(f"\n🔀 Reduce terminé → {len(resume_final.split())} mots")

    # ── RÉSUMÉ FINAL ──
    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-label">✦ Final summary — Map-Reduce</div>
        <div class="summary-text">{resume_final}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── STATS BAR ──
    nb_in  = len(text.split())
    nb_out = len(resume_final.split())
    ratio  = round(nb_in / max(nb_out, 1), 1)
    st.markdown(f"""
    <div class="stats-bar">
        Input <span>{nb_in} words</span> &nbsp;·&nbsp;
        Output <span>{nb_out} words</span> &nbsp;·&nbsp;
        Compression <span>{ratio}×</span> &nbsp;·&nbsp;
        Chunks <span>{len(chunks)}</span> &nbsp;·&nbsp;
        Device <span>{device.upper()}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── LOG ──
    with st.expander("🔍 Pipeline log"):
        st.markdown(f'<div class="log-box">{"<br>".join(log_lines)}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ABOUT (expander)
# ─────────────────────────────────────────────
with st.expander("ℹ️ About this project"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**BookSum — Deep Learning Project**

*Team:* Personne A (preprocessing + pipeline) · Personne B (model + fine-tuning)

*Dataset:* BookSum — 2000 literary chapters with human summaries

**Map-Reduce Pipeline**
1. Text → overlapping 512-token chunks
2. Each chunk summarized by PEGASUS+LoRA → **Map**
3. All summaries merged into final output → **Reduce**
        """)
    with c2:
        st.markdown("""
**Fine-tuning config**

| Param | Value |
|---|---|
| Base model | `google/pegasus-xsum` |
| LoRA rank | r=16, α=64 |
| Epochs | 3 (early stopping) |
| ROUGE-1 (v1) | 10.54% |
| ROUGE-1 (v2) | **31.44%** |
| Gain | **+198%** |
        """)