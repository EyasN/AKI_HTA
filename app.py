"""
Streamlit-App für das HTR-Modell.

Starten mit:
  streamlit run app.py

Features:
  - Bild hochladen oder Kamera-Snapshot (in Streamlit Cloud)
  - Handschrift erkennen (Greedy oder Beam Search)
  - Trainingskurven anzeigen (wenn vorhanden)
  - Modell-Infos
"""

import json
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
from PIL import Image

from src.dataset import NUM_CLASSES, ALPHABET
from src.model import build_model
from src.transforms import get_val_transforms
from utils.ctc_decoder import greedy_decode, beam_search_decode


# ── Seitenkonfiguration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="HTR – Handschrifterkennung",
    page_icon="✍️",
    layout="wide",
)


# ── Modell-Caching ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(checkpoint_path: str, img_height: int, img_width: int):
    """
    Lädt das Modell einmalig und cached es für die gesamte Session.
    @st.cache_resource verhindert erneutes Laden bei jedem UI-Update.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(img_height=img_height, num_classes=NUM_CLASSES)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch   = checkpoint.get("epoch", "?")
        val_cer = checkpoint.get("val_cer", None)
        info    = {"epoch": epoch, "val_cer": val_cer, "loaded": True}
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(f"Checkpoint nicht gefunden: {e}\nModell hat zufällige Gewichte.")
        info = {"epoch": 0, "val_cer": None, "loaded": False}

    model.to(device)
    model.eval()
    transform = get_val_transforms(img_height, img_width)
    return model, transform, device, info


def predict_image(
    image: Image.Image,
    model,
    transform,
    device: torch.device,
    decoder: str = "greedy",
) -> str:
    """Führt Inferenz auf einem PIL-Bild durch und gibt den erkannten Text zurück."""
    img_gray = image.convert("L")
    tensor   = transform(img_gray).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(tensor)

    if decoder == "Beam Search":
        return beam_search_decode(log_probs, beam_width=10)[0]
    return greedy_decode(log_probs)[0]


# ── Hauptlayout ───────────────────────────────────────────────────────────────

def main() -> None:
    st.title("✍️ Handwritten Text Recognition (HTR)")
    st.markdown(
        "**CRNN-Modell** (CNN + BiLSTM + CTC-Loss) zur Erkennung handgeschriebener Texte. "
        "Lade ein Bild mit Handschrift hoch und das Modell gibt den erkannten Text aus."
    )

    # ── Sidebar: Einstellungen ──
    with st.sidebar:
        st.header("⚙️ Einstellungen")

        checkpoint_path = st.text_input(
            "Checkpoint-Pfad",
            value="outputs/checkpoints/best_model.pt",
            help="Pfad zur gespeicherten Modelldatei (.pt)",
        )
        img_height = st.number_input("Bild-Höhe (px)", min_value=16, max_value=64, value=32)
        img_width  = st.number_input("Bild-Breite (px)", min_value=64, max_value=512, value=128)
        decoder    = st.radio("Decoder", ["Greedy", "Beam Search"], index=0)

        st.markdown("---")
        st.subheader("📊 Modell-Info")

        model, transform, device, info = load_model(checkpoint_path, img_height, img_width)
        st.write(f"**Gerät:** {device}")
        st.write(f"**Klassen:** {NUM_CLASSES}")
        if info["loaded"]:
            st.write(f"**Trainierte Epochen:** {info['epoch']}")
            if info["val_cer"] is not None:
                st.write(f"**Val CER:** {info['val_cer']:.4f}")
        else:
            st.error("Kein trainiertes Modell geladen.")

        n_params = sum(p.numel() for p in model.parameters())
        st.write(f"**Parameter:** {n_params:,}")

    # ── Hauptbereich ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Bild hochladen")
        uploaded = st.file_uploader(
            "PNG, JPG, BMP oder TIFF wählen",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Eingabebild", use_column_width=True)

            if st.button("🔍 Erkennen", type="primary"):
                with st.spinner("Handschrift wird erkannt …"):
                    text = predict_image(image, model, transform, device, decoder)
                st.session_state["result"] = text

    with col2:
        st.subheader("📝 Erkannter Text")

        if "result" in st.session_state:
            result = st.session_state["result"]
            st.success(f"**Ergebnis:**")
            st.markdown(
                f'<div style="background:#f0f2f6;padding:16px;border-radius:8px;'
                f'font-size:1.4rem;letter-spacing:0.05em;">{result}</div>',
                unsafe_allow_html=True,
            )
            st.write(f"**Zeichen:** {len(result)}  |  **Wörter:** {len(result.split())}")
            st.download_button("💾 Text herunterladen", data=result, file_name="erkannter_text.txt")
        else:
            st.info("Lade ein Bild hoch und klicke auf 'Erkennen'.")

    # ── Trainingskurven ──
    st.markdown("---")
    st.subheader("📈 Trainingsverlauf")

    history_path = Path("outputs/logs/history.json")
    curves_path  = Path("outputs/logs/training_curves.png")

    if curves_path.exists():
        st.image(str(curves_path), caption="Loss & CER Verlauf", use_column_width=True)
    elif history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        import pandas as pd
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3")
        if "val_loss" in history:
            axes[0].plot(history["val_loss"],   label="Val Loss",   color="#F44336", ls="--")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        if "train_cer" in history:
            axes[1].plot(history["train_cer"], label="Train CER", color="#4CAF50")
        if "val_cer" in history:
            axes[1].plot(history["val_cer"],   label="Val CER",   color="#FF9800", ls="--")
        axes[1].set_title("CER"); axes[1].legend(); axes[1].grid(alpha=0.3)

        st.pyplot(fig)
    else:
        st.info("Keine Trainingsdaten gefunden. Starte zuerst das Training.")

    # ── Alphabet-Übersicht ──
    with st.expander("🔤 Unterstützter Zeichensatz"):
        st.write(f"**{NUM_CLASSES} Klassen** (inkl. CTC-Blank)")
        st.code(ALPHABET[1:], language=None)


if __name__ == "__main__":
    main()
