"""
Streamlit-App für das HTR-Modell.

Starten mit:
  streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.dataset import NUM_CLASSES, SEQ2SEQ_VOCAB, ALPHABET
from src.model import build_model, build_seq2seq_model
from src.transforms import get_val_transforms, ensure_training_polarity
from utils.ctc_decoder import greedy_decode, beam_search_decode, lm_decode
from utils.seq2seq_decoder import seq2seq_greedy_decode, seq2seq_beam_decode


# ── Seitenkonfiguration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="HTR – Handschrifterkennung",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Hintergrund ── */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    min-height: 100vh;
  }

  /* ── Hero-Banner ── */
  .hero-banner {
    background: linear-gradient(120deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
    animation: shimmer 6s infinite linear;
  }
  @keyframes shimmer {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
    letter-spacing: -0.5px;
  }
  .hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin: 0.5rem 0 0;
    font-weight: 400;
  }

  /* ── Cards ── */
  .card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(102,126,234,0.25);
  }
  .card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
  }

  /* ── Ergebnis-Box ── */
  .result-box {
    background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    font-size: 1.6rem;
    font-weight: 500;
    color: #f1f5f9;
    letter-spacing: 0.06em;
    line-height: 1.6;
    word-break: break-all;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.1), 0 4px 24px rgba(102,126,234,0.2);
  }

  /* ── Stat-Chips ── */
  .stat-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
  }
  .stat-chip {
    background: rgba(102,126,234,0.2);
    border: 1px solid rgba(102,126,234,0.35);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #c4b5fd;
  }

  /* ── Status-Badge ── */
  .badge-ok {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16,185,129,0.18);
    border: 1px solid rgba(16,185,129,0.4);
    color: #6ee7b7;
    border-radius: 50px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .badge-warn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(245,158,11,0.18);
    border: 1px solid rgba(245,158,11,0.4);
    color: #fcd34d;
    border-radius: 50px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #1a1a2e 100%) !important;
    border-right: 1px solid rgba(167,139,250,0.2);
  }
  [data-testid="stSidebar"] .stTextInput label,
  [data-testid="stSidebar"] .stNumberInput label,
  [data-testid="stSidebar"] .stRadio label {
    color: #c4b5fd !important;
    font-weight: 500;
  }

  /* ── Sidebar-Header ── */
  .sidebar-logo {
    text-align: center;
    padding: 1rem 0 1.5rem;
  }
  .sidebar-logo .icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.4rem;
  }
  .sidebar-logo .name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2d9f3;
    letter-spacing: 0.5px;
  }
  .sidebar-logo .version {
    font-size: 0.75rem;
    color: #7c6faa;
  }

  /* ── Divider ── */
  .fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.5), transparent);
    margin: 1.5rem 0;
    border: none;
  }

  /* ── Section-Label ── */
  .section-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #7c6faa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.75rem;
  }

  /* ── Info-Row in Sidebar ── */
  .info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .info-label { color: #7c6faa; font-size: 0.82rem; }
  .info-value { color: #e2d9f3; font-size: 0.85rem; font-weight: 600; }

  /* ── Upload-Zone ── */
  [data-testid="stFileUploader"] {
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 14px !important;
    background: rgba(102,126,234,0.05) !important;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.8) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.5) !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(102,126,234,0.65) !important;
  }

  /* ── Allgemeine Text-Farben ── */
  h1, h2, h3, h4, p, label, .stMarkdown {
    color: #e2d9f3 !important;
  }
  .stSubheader { color: #a78bfa !important; }

  /* ── Metrics ── */
  [data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
  }
  [data-testid="stMetricValue"] { color: #c4b5fd !important; font-weight: 700 !important; }
  [data-testid="stMetricLabel"] { color: #7c6faa !important; }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(167,139,250,0.2) !important;
    border-radius: 12px !important;
  }

  /* ── Spinner-Text ── */
  .stSpinner > div { color: #a78bfa !important; }

  /* ── Code-Block ── */
  .stCode { border-radius: 10px !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #1a1a2e; }
  ::-webkit-scrollbar-thumb { background: #4c3d7a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Modell-Caching ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(checkpoint_path: str, arch: str, img_height: int, img_width: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint  = torch.load(checkpoint_path, map_location=device)
        # Automatisch aus Gewichts-Shape ableiten (funktioniert auch für alte Checkpoints)
        sd = checkpoint["model_state_dict"]
        lstm_hidden = (sd["rnn.weight_hh_l0"].shape[0] // 4
                       if "rnn.weight_hh_l0" in sd
                       else checkpoint.get("lstm_hidden", 256))

        # Config aus dem Checkpoint übernehmen (überschreibt UI-Werte) – kein Mismatch-Risiko
        img_height = checkpoint.get("img_height", img_height)
        img_width  = checkpoint.get("img_width",  img_width)
        deslant    = checkpoint.get("deslant",    False)

        if arch == "seq2seq":
            model = build_seq2seq_model(img_height=img_height, vocab_size=SEQ2SEQ_VOCAB,
                                        lstm_hidden=lstm_hidden)
        else:
            model = build_model(img_height=img_height, num_classes=NUM_CLASSES,
                                lstm_hidden=lstm_hidden)

        model.load_state_dict(checkpoint["model_state_dict"])
        epoch   = checkpoint.get("epoch", "?")
        val_cer = checkpoint.get("val_cer", None)
        info    = {"epoch": epoch, "val_cer": val_cer, "loaded": True,
                   "arch": arch, "lstm_hidden": lstm_hidden,
                   "img_height": img_height, "img_width": img_width, "deslant": deslant}
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(f"Checkpoint nicht gefunden: {e}\nModell hat zufällige Gewichte.")
        lstm_hidden = 256
        deslant     = False
        if arch == "seq2seq":
            model = build_seq2seq_model(img_height=img_height, vocab_size=SEQ2SEQ_VOCAB)
        else:
            model = build_model(img_height=img_height, num_classes=NUM_CLASSES)
        info = {"epoch": 0, "val_cer": None, "loaded": False,
                "arch": arch, "lstm_hidden": lstm_hidden,
                "img_height": img_height, "img_width": img_width, "deslant": deslant}

    model.to(device)
    model.eval()
    transform = get_val_transforms(img_height, img_width, deslant=deslant)
    return model, transform, device, info


def predict_image(image: Image.Image, model, transform, device, arch: str = "crnn", decoder: str = "greedy"):
    img_gray = ensure_training_polarity(image)   # auf Trainings-Polarität bringen (sonst Kauderwelsch)
    preprocessed_pil = img_gray.resize((256, 32), Image.LANCZOS)
    tensor = transform(img_gray).unsqueeze(0).to(device)

    if arch == "seq2seq":
        texts, confs = seq2seq_beam_decode(model, tensor, beam_width=5, return_confidence=True)
        text       = texts[0]
        confidence = confs[0] * 100      # mittlere Token-Wahrscheinlichkeit der besten Hypothese
        greedy_txt = seq2seq_greedy_decode(model, tensor)[0]
        candidates = [("Seq2Seq Beam", text)]
        if greedy_txt != text:
            candidates.append(("Seq2Seq Greedy", greedy_txt))
        return text, confidence, preprocessed_pil, candidates

    with torch.no_grad():
        log_probs = model(tensor)

    confidence = float(torch.exp(log_probs).max(dim=2).values.mean()) * 100

    if decoder == "Beam Search":
        text = beam_search_decode(log_probs, beam_width=10)[0]
    elif decoder == "LM Beam Search":
        text = lm_decode(log_probs, beam_width=25)[0]
    else:
        text = greedy_decode(log_probs)[0]

    all_results = [
        ("Greedy",         greedy_decode(log_probs)[0]),
        ("Beam Search",    beam_search_decode(log_probs, beam_width=10)[0]),
        ("LM Beam Search", lm_decode(log_probs, beam_width=25)[0]),
    ]
    seen: set = set()
    candidates = []
    for label, result in all_results:
        if result not in seen:
            seen.add(result)
            candidates.append((label, result))

    return text, confidence, preprocessed_pil, candidates


# ── Hauptlayout ───────────────────────────────────────────────────────────────

def main() -> None:

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
          <span class="icon">✍️</span>
          <div class="name">HTR System</div>
          <div class="version">v1.0 · CRNN + CTC</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Modell-Konfiguration</div>', unsafe_allow_html=True)

        arch = st.radio(
            "Architektur",
            ["CRNN (CTC)", "Seq2Seq (Transformer)"],
            index=0,
            label_visibility="visible",
        )
        arch_key = "seq2seq" if arch == "Seq2Seq (Transformer)" else "crnn"

        default_ckpt = (
            "outputs/checkpoints/best_seq2seq.pt"
            if arch_key == "seq2seq"
            else "outputs/checkpoints/best_model.pt"
        )
        checkpoint_path = st.text_input("Checkpoint-Pfad", value=default_ckpt)
        img_height = st.number_input("Bild-Höhe (px)", min_value=16, max_value=64, value=32)
        img_width  = st.number_input("Bild-Breite (px)", min_value=64, max_value=512, value=256)

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Decoder</div>', unsafe_allow_html=True)

        if arch_key == "seq2seq":
            decoder = "Seq2Seq Beam"
            st.markdown('<span style="color:#7c6faa;font-size:0.85rem;">Seq2Seq nutzt autoregressives Beam Search (beam_width=5)</span>', unsafe_allow_html=True)
        else:
            decoder = st.radio(
                "Decoder-Methode",
                ["Greedy", "Beam Search", "LM Beam Search"],
                index=2,
                label_visibility="collapsed",
            )

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Modell-Status</div>', unsafe_allow_html=True)

        model, transform, device, info = load_model(checkpoint_path, arch_key, img_height, img_width)
        n_params = sum(p.numel() for p in model.parameters())

        if info["loaded"]:
            st.markdown('<span class="badge-ok">● Modell geladen</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warn">⚠ Zufällige Gewichte</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1rem;">
          <div class="info-row">
            <span class="info-label">Gerät</span>
            <span class="info-value">{device}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Architektur</span>
            <span class="info-value">{arch}</span>
          </div>
          <div class="info-row">
            <span class="info-label">LSTM Hidden</span>
            <span class="info-value">{info.get("lstm_hidden", 256)}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Bildgröße</span>
            <span class="info-value">{info.get("img_height", img_height)}×{info.get("img_width", img_width)}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Deslant</span>
            <span class="info-value">{"an" if info.get("deslant") else "aus"}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Klassen</span>
            <span class="info-value">{NUM_CLASSES if arch_key == "crnn" else SEQ2SEQ_VOCAB}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Parameter</span>
            <span class="info-value">{n_params:,}</span>
          </div>
          {"" if not info["loaded"] else f'''
          <div class="info-row">
            <span class="info-label">Epochen</span>
            <span class="info-value">{info["epoch"]}</span>
          </div>
          ''' + (f'''
          <div class="info-row">
            <span class="info-label">Val CER</span>
            <span class="info-value">{info["val_cer"]:.4f}</span>
          </div>
          ''' if info["val_cer"] is not None else "")}
        </div>
        """, unsafe_allow_html=True)

    # ── Hero-Banner ──
    st.markdown("""
    <div class="hero-banner">
      <h1 class="hero-title">Handwritten Text Recognition</h1>
      <p class="hero-subtitle">CNN + BiLSTM + CTC · CNN + BiLSTM + Transformer · Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Hauptbereich ──
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Eingabe</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "PNG, JPG, BMP oder TIFF hochladen",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            label_visibility="collapsed",
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Eingabebild", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if "prediction" in st.session_state:
                st.markdown('<div class="section-label" style="margin-top:1rem; font-size:0.7rem;">Was das Modell sieht (32×256 px)</div>', unsafe_allow_html=True)
                st.image(st.session_state["prediction"]["preprocessed"], use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Handschrift erkennen →", type="primary", use_container_width=True):
                with st.spinner("Analysiere Handschrift …"):
                    text, confidence, preprocessed_pil, candidates = predict_image(image, model, transform, device, arch_key, decoder)
                st.session_state["prediction"] = {
                    "text": text,
                    "confidence": confidence,
                    "preprocessed": preprocessed_pil,
                    "candidates": candidates,
                }
                st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem; color: #7c6faa;">
              <div style="font-size:3.5rem; margin-bottom:0.75rem;">🖼️</div>
              <div style="font-size:0.95rem; font-weight:500;">Bild hierher ziehen oder klicken</div>
              <div style="font-size:0.8rem; margin-top:0.3rem; opacity:0.7;">PNG · JPG · BMP · TIFF</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Erkannter Text</div>', unsafe_allow_html=True)

        if "prediction" in st.session_state:
            pred   = st.session_state["prediction"]
            result = pred["text"]
            conf   = pred["confidence"]
            candidates = pred["candidates"]

            empty_placeholder = "<em style='opacity:.5'>— leer —</em>"
            st.markdown(f'<div class="result-box">{result if result else empty_placeholder}</div>', unsafe_allow_html=True)

            chars = len(result)
            words = len(result.split()) if result.strip() else 0
            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-chip">{chars} Zeichen</div>
              <div class="stat-chip">{words} Wörter</div>
              <div class="stat-chip">Konfidenz: {conf:.0f}%</div>
              <div class="stat-chip">Decoder: {decoder}</div>
            </div>
            """, unsafe_allow_html=True)

            if len(candidates) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Alternative Kandidaten"):
                    for label, cand in candidates:
                        marker = "✓" if cand == result else "○"
                        st.markdown(f"**{marker} {label}:** `{cand}`")

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "Ergebnis herunterladen",
                data=result,
                file_name="erkannter_text.txt",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem; color: #7c6faa;">
              <div style="font-size:3.5rem; margin-bottom:0.75rem;">📝</div>
              <div style="font-size:0.95rem; font-weight:500;">Noch kein Ergebnis</div>
              <div style="font-size:0.8rem; margin-top:0.3rem; opacity:0.7;">Lade ein Bild hoch und starte die Erkennung</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Metriken-Leiste ──
    if info["loaded"] and info["val_cer"] is not None:
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Char Accuracy", f"{(1 - info['val_cer']) * 100:.1f}%")
        m2.metric("Val CER", f"{info['val_cer'] * 100:.2f}%")
        m3.metric("Epochen", info["epoch"])
        m4.metric("Parameter", f"{n_params/1e6:.1f} M")
        m5.metric("Decoder", decoder)

    # ── Trainingsverlauf ──
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Trainingsverlauf</div>', unsafe_allow_html=True)

    if arch_key == "seq2seq":
        history_path = Path("outputs/logs/history_seq2seq.json")
        curves_path  = Path("outputs/logs/training_curves_seq2seq.png")
    else:
        history_path = Path("outputs/logs/history.json")
        curves_path  = Path("outputs/logs/training_curves.png")

    if curves_path.exists():
        st.image(str(curves_path), caption="Loss & CER", use_container_width=True)
    elif history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.rcParams.update({
            "figure.facecolor": "none",
            "axes.facecolor":   "#1e1b4b",
            "axes.edgecolor":   "#4c3d7a",
            "axes.labelcolor":  "#c4b5fd",
            "xtick.color":      "#7c6faa",
            "ytick.color":      "#7c6faa",
            "text.color":       "#e2d9f3",
            "grid.color":       "#2d2660",
            "legend.facecolor": "#1e1b4b",
            "legend.edgecolor": "#4c3d7a",
        })

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_alpha(0)

        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label="Train Loss", color="#818cf8", lw=2)
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label="Val Loss", color="#f472b6", lw=2, ls="--")
        axes[0].set_title("Loss", fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if "train_cer" in history:
            axes[1].plot(history["train_cer"], label="Train CER", color="#34d399", lw=2)
        if "val_cer" in history:
            axes[1].plot(history["val_cer"], label="Val CER", color="#fb923c", lw=2, ls="--")
        axes[1].set_title("CER", fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        st.pyplot(fig)
    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:#7c6faa; background:rgba(255,255,255,0.03);
             border-radius:14px; border:1px dashed rgba(124,111,170,0.3);">
          <div style="font-size:2rem; margin-bottom:0.5rem;">📊</div>
          Noch keine Trainingsdaten vorhanden.<br>
          <span style="font-size:0.82rem; opacity:0.7;">Starte zuerst das Training mit <code>python -m training.train</code></span>
        </div>
        """, unsafe_allow_html=True)

    # ── Alphabet-Übersicht ──
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(f"Unterstützter Zeichensatz · {NUM_CLASSES} Klassen"):
        st.code(ALPHABET[1:], language=None)


if __name__ == "__main__":
    main()
