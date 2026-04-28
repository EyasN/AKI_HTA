# HTR – Handwritten Text Recognition mit Deep Learning

Ein vollständiges Hochschulprojekt zur Erkennung handgeschriebener Texte mit einem **CRNN-Modell** (Convolutional Recurrent Neural Network) in PyTorch.

---

## Warum CNN + BiLSTM + CTC?

Diese Kombination ist der Industriestandard für Handschrifterkennung:

| Komponente | Aufgabe | Warum? |
|------------|---------|--------|
| **CNN** | Feature-Extraktion | Erkennt lokale Muster (Striche, Kurven) positions-unabhängig |
| **BiLSTM** | Sequenz-Modellierung | Versteht Kontext: ein Buchstabe hängt von seinen Nachbarn ab |
| **CTC-Loss** | Training ohne Alignment | Lernt Zeichen-Positionen automatisch, ohne Pixel-genaue Annotationen |

```
Bild (32×128)  →  CNN (Feature-Maps)  →  BiLSTM (Kontext)  →  Linear  →  CTC-Decode  →  "hello"
```

---

## Projektstruktur

```
AKI_Project/
├── data/
│   ├── raw/                  ← IAM-Datensatz (nach Download)
│   ├── processed/            ← Vorverarbeitete Daten
│   └── sample_images/        ← Beispielbilder zum Testen
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  ← Interaktive Analyse
│
├── src/
│   ├── model.py              ← CRNN-Architektur (CNN + BiLSTM)
│   ├── dataset.py            ← IAM- und Synthetik-Datensatz
│   └── transforms.py         ← Bildvorverarbeitung & Augmentation
│
├── training/
│   └── train.py              ← Training-Pipeline mit Early Stopping
│
├── evaluation/
│   └── evaluate.py           ← CER & WER Evaluation
│
├── utils/
│   ├── ctc_decoder.py        ← Greedy & Beam-Search Decoder
│   └── visualization.py      ← Trainingskurven & Beispielplots
│
├── outputs/
│   ├── checkpoints/          ← Gespeicherte Modelle (.pt)
│   └── logs/                 ← Trainingskurven, TensorBoard
│
├── predict.py                ← Inferenz für einzelne Bilder
├── app.py                    ← Streamlit Web-UI
└── requirements.txt
```

---

## Installation

### 1. Umgebung erstellen (empfohlen: conda oder venv)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Für GPU-Unterstützung (optional aber empfohlen)

Besuche https://pytorch.org/get-started/locally/ und wähle die passende CUDA-Version:

```bash
# Beispiel für CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Datensatz-Setup

### Option A: Synthetischer Datensatz (sofort nutzbar, kein Download)

Kein Setup nötig! Der `SyntheticHTRDataset` generiert Bilder automatisch.
Ideal zum Testen der Pipeline und für erste Experimente.

```bash
python -m training.train --dataset synthetic --epochs 30
```

### Option B: IAM Handwriting Dataset (empfohlen für echte Handschrift)

1. Kostenlose Registrierung unter: https://fki.inf.unibe.ch/databases/iam-handwriting-database
2. Herunterladen:
   - `words.tgz` → entpacken nach `data/raw/words/`
   - `ascii.tgz` → entpacken nach `data/raw/ascii/`

```
data/raw/
├── words/
│   ├── a01/
│   │   └── a01-000u/
│   │       ├── a01-000u-00-00.png
│   │       └── ...
│   └── ...
└── ascii/
    └── words.txt
```

3. Training starten:

```bash
python -m training.train --dataset iam --data-dir data/raw --epochs 100
```

---

## Training starten

```bash
# Mit synthetischen Daten (schneller Test)
python -m training.train \
    --dataset synthetic \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001

# Mit IAM-Datensatz (empfohlen)
python -m training.train \
    --dataset iam \
    --data-dir data/raw \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --patience 15

# Training fortsetzen
python -m training.train \
    --dataset iam \
    --resume outputs/checkpoints/best_model.pt \
    --epochs 50
```

### TensorBoard (Echtzeit-Monitoring)

```bash
tensorboard --logdir outputs/logs/
# Browser: http://localhost:6006
```

---

## Evaluation starten

```bash
python -m evaluation.evaluate \
    --checkpoint outputs/checkpoints/best_model.pt \
    --dataset synthetic

# Ausgabe:
# =============================================
#   Evaluation Ergebnis
# =============================================
#   Character Error Rate (CER): 0.0523  (5.23%)
#   Word Error Rate     (WER): 0.1200  (12.00%)
#   Samples evaluiert:          750
# =============================================
```

**Interpretation:**
- **CER < 5%**: Ausgezeichnet
- **CER 5–15%**: Gut (für Hochschulprojekt sehr solide)
- **WER < 20%**: Akzeptabel

---

## Beispielvorhersage

```bash
# Einzelbild
python predict.py \
    --image data/sample_images/example.png \
    --checkpoint outputs/checkpoints/best_model.pt

# Ordner mit Bildern
python predict.py \
    --folder data/sample_images/ \
    --checkpoint outputs/checkpoints/best_model.pt \
    --decoder beam    # Beam-Search für bessere Ergebnisse
```

---

## Web-UI (Streamlit)

```bash
streamlit run app.py
```

Browser öffnet sich automatisch unter `http://localhost:8501`.

Features:
- Bild hochladen und Handschrift erkennen
- Greedy oder Beam-Search Decoder wählen
- Trainingskurven anzeigen
- Erkannten Text herunterladen

---

## Jupyter Notebook

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

Das Notebook zeigt:
- Datensatz und Augmentation visualisieren
- Modellarchitektur erkunden
- CTC-Loss verstehen
- Trainingskurven anzeigen

---

## Architektur im Detail

```
Eingabebild: (Batch=B, Channels=1, Height=32, Width=128)

CNN Feature-Extraktor:
  Conv(1→64)   + BN + ReLU + Pool(2×2) → (B, 64,  16, 64)
  Conv(64→128) + BN + ReLU + Pool(2×2) → (B, 128,  8, 32)
  Conv(128→256)+ BN + ReLU             → (B, 256,  8, 32)
  Conv(256→256)+ BN + ReLU + Pool(2×1) → (B, 256,  4, 32)
  Conv(256→512)+ BN + ReLU             → (B, 512,  4, 32)
  Conv(512→512)+ BN + ReLU + Pool(2×1) → (B, 512,  2, 32)
  Conv(512→512)+ BN + ReLU             → (B, 512,  2, 32)

Reshape: (B, 512, 2, W') → (W', B, 1024)  ← W'=32 Zeitschritte

BiLSTM (2 Schichten, hidden=256):
  (32, B, 1024) → (32, B, 512)  ← 256×2 bidirektional

Linear + Log-Softmax:
  (32, B, 512) → (32, B, 96)   ← 96 Klassen (95 ASCII + Blank)

CTC-Decode:
  (32, B, 96) → ["hello", "world", ...]
```

---

## Hyperparameter

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `img_height` | 32 | Bildhöhe in Pixeln |
| `img_width` | 128 | Bildbreite in Pixeln |
| `lstm_hidden` | 256 | LSTM-Einheiten pro Richtung |
| `batch_size` | 32 | Bilder pro Trainingsschritt |
| `lr` | 1e-3 | Lernrate (Adam) |
| `patience` | 10 | Early-Stopping-Epochen |
| `epochs` | 30 | Max. Trainingsepochen |

---

## Erweiterungsmöglichkeiten

- **Sprachmodell-Integration**: Beam-Search + N-Gram-Modell für bessere Dekodierung
- **Attention-Mechanismus**: Transformer statt BiLSTM (moderne HTR-Systeme)
- **Datensatz-Erweiterung**: RIMES, CVL, GW-Dataset (alle frei verfügbar)
- **Transfer Learning**: Vortrainierte CNN-Backbone (ResNet, EfficientNet)
- **Zeilenerkennung**: Textzeilen im Bild detektieren vor HTR (CRAFT, DB-Net)

---

## Referenzen

- **CRNN Paper**: Shi et al. (2016) – *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*
- **CTC Loss**: Graves et al. (2006) – *Connectionist Temporal Classification*
- **IAM Dataset**: Marti & Bunke (2002) – *The IAM-database: an English sentence database*
- **PyTorch CTC**: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

---

## Lizenz

MIT License – Frei für akademische und kommerzielle Nutzung.
