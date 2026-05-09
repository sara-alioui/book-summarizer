"""
BookSum Demo — Interface Gradio
Personne A : Pipeline Map-Reduce avec PEGASUS + LoRA
"""

import gradio as gr
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ─────────────────────────────────────────────
#  CHEMINS
# ─────────────────────────────────────────────
MODEL_NAME = "google/pegasus-xsum"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "notebooks", "model_finetuned")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "notebooks", "donnees_propres.json")

# ─────────────────────────────────────────────
#  CHARGEMENT DU MODELE
# ─────────────────────────────────────────────
print("🔄 Chargement du modèle...")
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model      = PeftModel.from_pretrained(base_model, MODEL_DIR)
tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR)
model      = model.to(device)
model.eval()
print(f"✅ Modèle chargé sur {device.upper()}")

# ─────────────────────────────────────────────
#  EXEMPLES
# ─────────────────────────────────────────────
with open(DATA_PATH, "r", encoding="utf-8") as f:
    donnees = json.load(f)

EXAMPLES = [donnees[i]["chunks"][0] for i in range(3)]

# ─────────────────────────────────────────────
#  PIPELINE MAP-REDUCE
# ─────────────────────────────────────────────
def chunker(text: str, chunk_size: int = 512, overlap: int = 50):
    words = text.split()
    step  = max(1, chunk_size - overlap)
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), step) if words[i:i+chunk_size]]


def resumer_chunk(text: str, max_input: int = 512, max_output: int = 128) -> str:
    inputs = tokenizer(text, max_length=max_input, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_pdf_text(pdf_path: str) -> str:
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        return "❌ PyMuPDF non installé. Lancez : pip install pymupdf"
    except Exception as e:
        return f"❌ Erreur lecture PDF : {e}"


def pipeline_mapreduce(text: str, pdf_file, chunk_size: int, overlap: int, max_output: int):
    if pdf_file is not None:
        extracted = extract_pdf_text(pdf_file)
        if extracted.startswith("❌"):
            return extracted, extracted, ""
        text = extracted

    if not text or not text.strip():
        return "❌ Texte vide. Collez du texte ou uploadez un PDF.", "", ""

    chunks = chunker(text, int(chunk_size), int(overlap))
    log    = f"📄 {len(chunks)} chunk(s) · {int(chunk_size)} tokens · overlap {int(overlap)}\n\n"

    intermediaires = []
    for i, chunk in enumerate(chunks):
        resume = resumer_chunk(chunk, max_input=512, max_output=int(max_output))
        intermediaires.append(resume)
        log += f"  ✅ Chunk {i+1}/{len(chunks)} → {len(resume.split())} mots\n"

    log += f"\n🔀 Reduce en cours...\n"
    resume_final = resumer_chunk(" ".join(intermediaires), max_input=512, max_output=150)
    log += f"✅ Résumé final : {len(resume_final.split())} mots"

    nb_in  = len(text.split())
    nb_out = len(resume_final.split())
    ratio  = round(nb_in / max(nb_out, 1), 1)
    stats  = (
        f"**📥 Entrée:** {nb_in} mots &nbsp;·&nbsp; "
        f"**📤 Sortie:** {nb_out} mots &nbsp;·&nbsp; "
        f"**🗜 Compression:** {ratio}× &nbsp;·&nbsp; "
        f"**🔪 Chunks:** {len(chunks)} &nbsp;·&nbsp; "
        f"**⚙️ Device:** {device.upper()}"
    )

    return resume_final, log, stats


# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #f0f4ff !important;
}

.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 28px 20px !important;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #4facfe 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
}
.hero-title {
    font-size: 28px;
    font-weight: 700;
    color: white;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.15);
}
.hero-sub {
    font-size: 15px;
    color: rgba(255,255,255,0.88);
    margin-bottom: 18px;
}
.hero-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
    background: rgba(255,255,255,0.22);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: white;
}

/* ── STAT CARDS ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 18px 20px;
    border: 1px solid #e8eaf6;
    box-shadow: 0 2px 12px rgba(102,126,234,0.07);
}
.stat-icon { font-size: 20px; margin-bottom: 8px; }
.stat-val {
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 3px;
}
.stat-lbl { font-size: 12px; color: #8892b0; font-weight: 500; }

/* ── LABELS ── */
label, .label-wrap span {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #4a4a8a !important;
}

/* ── TEXTAREAS ── */
textarea {
    border: 2px solid #e8eaf6 !important;
    border-radius: 12px !important;
    font-size: 13.5px !important;
    background: #fafbff !important;
    color: #1a1a3e !important;
    transition: border 0.2s !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 4px rgba(102,126,234,0.1) !important;
    outline: none !important;
}

/* ── SLIDERS ── */
input[type=range] { accent-color: #667eea; }

/* ── BUTTONS ── */
button.primary {
    background: linear-gradient(135deg, #667eea, #f093fb) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    transition: all 0.2s !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.5) !important;
}
button.secondary {
    background: white !important;
    border: 2px solid #e8eaf6 !important;
    border-radius: 10px !important;
    color: #4a4a8a !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
button.secondary:hover {
    border-color: #667eea !important;
    color: #667eea !important;
    background: #f0f4ff !important;
}

/* ── OUTPUT ── */
.output-box textarea {
    background: linear-gradient(135deg, #fdf0ff, #f0f8ff) !important;
    border: 2px solid #d4b8f0 !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
    color: #1a1a3e !important;
}

/* ── STATS MD ── */
.stats-md p {
    font-size: 13px !important;
    color: #5a5a9a !important;
    background: #f0f4ff;
    padding: 10px 16px;
    border-radius: 10px;
    border-left: 3px solid #667eea;
    margin-top: 10px !important;
}

footer { display: none !important; }
"""

# ─────────────────────────────────────────────
#  INTERFACE
# ─────────────────────────────────────────────
with gr.Blocks(title="📚 BookSum Demo") as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-title">📚 BookSum — Automatic Book Summarization</div>
        <div class="hero-sub">Deep Learning Project · PEGASUS + LoRA · Map-Reduce Pipeline</div>
        <div class="hero-pills">
            <span class="pill">✨ PEGASUS-xsum</span>
            <span class="pill">🧬 LoRA Fine-tuned</span>
            <span class="pill">📊 ROUGE-1 0.184</span>
            <span class="pill">📚 BookSum Dataset</span>
            <span class="pill">🗂 Map-Reduce</span>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-icon">🤖</div>
            <div class="stat-val" style="font-size:17px;margin-top:2px;">PEGASUS</div>
            <div class="stat-lbl">Base model + LoRA</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🏆</div>
            <div class="stat-val">0.184</div>
            <div class="stat-lbl">Best ROUGE-1 (epoch 2)</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📖</div>
            <div class="stat-val">100</div>
            <div class="stat-lbl">BookSum chapters</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🔀</div>
            <div class="stat-val">11.1</div>
            <div class="stat-lbl">Avg chunks / book</div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="📝 Paste a book chapter",
                placeholder="Paste your text here, or upload a PDF below...",
                lines=9,
            )
        with gr.Column(scale=1):
            gr.HTML("<div style='font-size:12px;font-weight:700;color:#e91e8c;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;'>📌 Quick examples</div>")
            ex1 = gr.Button("📖 Chapter 1", variant="secondary", size="sm")
            ex2 = gr.Button("📖 Chapter 2", variant="secondary", size="sm")
            ex3 = gr.Button("📖 Chapter 3", variant="secondary", size="sm")

    pdf_input = gr.File(
        label="📄 Or upload a PDF (overrides text above)",
        file_types=[".pdf"],
        type="filepath",
    )

    with gr.Row():
        chunk_size = gr.Slider(128, 512, step=64, value=512, label="🔪 Chunk size (tokens)")
        overlap    = gr.Slider(0,   100, step=10,  value=50,  label="🔗 Overlap (tokens)")
        max_output = gr.Slider(64,  256, step=32,  value=128, label="📏 Max output length")

    with gr.Row():
        clear_btn = gr.ClearButton([input_text, pdf_input], value="🗑 Clear", variant="secondary")
        run_btn   = gr.Button("✨ Summarize", variant="primary", scale=3)

    gr.HTML("<div style='margin-top:8px'></div>")

    summary_out = gr.Textbox(
        label="✨ Final Summary (Map-Reduce)",
        lines=7,
        interactive=False,
        elem_classes="output-box",
    )
    stats_out = gr.Markdown(elem_classes="stats-md")

    with gr.Accordion("🔍 Pipeline log (step by step)", open=False):
        log_out = gr.Textbox(label="", lines=10, interactive=False)

    with gr.Accordion("ℹ️ About this project", open=False):
        gr.Markdown("""
### 🎓 BookSum — Deep Learning Project

**Team:** Personne A (preprocessing + pipeline) · Personne B (model selection + fine-tuning)

**Dataset:** [BookSum](https://huggingface.co/datasets/kmfoda/booksum) — 100 literary book chapters with human summaries

---

#### 🔀 Map-Reduce Pipeline *(Personne A)*
1. Text split into overlapping 512-token chunks (overlap = 50)
2. Each chunk summarized independently by PEGASUS+LoRA → **Map**
3. All summaries merged into one final output → **Reduce**

#### 🧬 Fine-tuning *(Personne B)*
| Config | Value |
|---|---|
| Base model | `google/pegasus-xsum` (568M params) |
| Trainable (LoRA) | ~4M params (r=8, alpha=32) |
| Tokenizer | PEGASUS BPE, vocab 96k, OOV 0% |
| Training | 3 epochs, best at epoch 2 |
| Best ROUGE-1 | **0.184** |

#### 📓 Notebooks
`01_preprocessing` · `02_B1_analyse_exploratoire` · `02_B2_choix_tokenizer` · `02_dataloader` · `03_B3_Selection_Modele` · `04_B4_finetuning_lora` · `03_A3_pipeline_mapreduce`
        """)

    # ── EVENTS ──
    ex1.click(fn=lambda: EXAMPLES[0], outputs=input_text)
    ex2.click(fn=lambda: EXAMPLES[1], outputs=input_text)
    ex3.click(fn=lambda: EXAMPLES[2], outputs=input_text)

    run_btn.click(
        fn=pipeline_mapreduce,
        inputs=[input_text, pdf_input, chunk_size, overlap, max_output],
        outputs=[summary_out, log_out, stats_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
    )