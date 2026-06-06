# HTR – Handwritten Text Recognition

Dieses Projekt erkennt handgeschriebene Wörter automatisch mithilfe neuronaler Netze. Ein Bild mit einem handgeschriebenen Wort wird eingegeben, das Modell gibt den erkannten Text aus.

---

## Was das Projekt macht

Das Modell nimmt ein Bild eines handgeschriebenen Wortes und gibt den erkannten Text zurück. Jedes Eingabebild wird automatisch auf **32×128 Pixel** skaliert — egal wie groß das Original ist. Trainiert wurde es auf dem **IAM Handwriting Dataset** – einer Sammlung von tausenden echten Handschriftproben.

**Beispiel:**
```
Eingabe:  [Bild von "hello"]
Ausgabe:  "hello"
```

---

## Architekturen

Das Projekt implementiert **zwei Modell-Architekturen**, die beide trainiert und verglichen werden können:

### Architektur 1 – CRNN (CTC)

```
Bild (32×128) → CNN → BiLSTM → Linear → CTC-Decoder → Text
```

Das klassische Modell. Der CTC-Decoder (Connectionist Temporal Classification) wandelt die Ausgabe-Wahrscheinlichkeiten in Text um, ohne dass eine pixelgenaue Ausrichtung zwischen Bild und Label benötigt wird.

**Ziel-Accuracy:** ~81–87% Zeichengenauigkeit

### Architektur 2 – CNN + BiLSTM + Transformer Decoder (Seq2Seq)

```
Bild (32×128) → CNN → BiLSTM → Transformer Decoder → Text
```

Neuere Architektur mit autoregressivem Decoder. Statt CTC generiert ein Transformer-Decoder die Ausgabe Zeichen für Zeichen und berücksichtigt dabei alle bisher generierten Zeichen (eingebautes Sprachwissen durch Self-Attention). Training mit **Teacher Forcing**, Inferenz autoregressiv.

**Ziel-Accuracy:** ~88–93% Zeichengenauigkeit

---

## Architektur-Details

### CNN – Feature-Extraktor (identisch in beiden Architekturen)

7 ConvBlocks (Conv2d → BatchNorm → ReLU → MaxPool):

| Block | Kanäle | Ausgabe (H×W) |
|-------|--------|---------------|
| 1 | 1 → 64 | 16×64 |
| 2 | 64 → 128 | 8×32 |
| 3 | 128 → 256 | 8×32 |
| 4 | 256 → 256 | 4×32 |
| 5 | 256 → 512 | 4×32 |
| 6 | 512 → 512 | 2×32 |
| 7 | 512 → 512 | 2×32 |

Nach dem CNN liegen 32 Feature-Spalten vor, jede mit 1024 Werten (512 Kanäle × 2 Höhe).

### BiLSTM Encoder (identisch in beiden Architekturen)

- 2 gestapelte bidirektionale LSTM-Schichten
- 256 Units pro Richtung → 512 Ausgabe-Features
- Liest die 32 Feature-Spalten als Zeitsequenz vorwärts und rückwärts
- Dropout 0.3 zwischen den Schichten

### CTC-Decoder (nur CRNN)

- Lineare Projektion: 512 → 96 Klassen
- Log-Softmax Ausgabe für `nn.CTCLoss`
- 3 Decoder-Varianten: Greedy, Beam Search, LM Beam Search

### Transformer Decoder (nur Seq2Seq)

- Encoder-Projektion: 512 → 256 (d_model) + Positional Encoding
- Embedding-Schicht: 98 Token → 256 (d_model)
- 3 Transformer-Decoder-Schichten, 8 Attention-Heads
- Feed-Forward-Dimension: 1024 (d_model × 4)
- Autoregressive Ausgabe: ein Token pro Schritt bis EOS
- Sondertokens: PAD=0, BOS=96, EOS=97 (Vocab-Größe: 98)

### Gegenüberstellung

| Eigenschaft | CRNN (CTC) | Seq2Seq (Transformer) |
|-------------|-----------|----------------------|
| Loss | CTCLoss | CrossEntropyLoss |
| Ausgabe | Parallel (alle Zeitschritte) | Autoregressiv (ein Token nach dem anderen) |
| Sprachwissen | Kein (reiner Bildklassifizierer) | Eingebaut (Self-Attention über bisherige Tokens) |
| Training | Schneller | Langsamer (Teacher Forcing) |
| Inferenz | Sehr schnell | Langsamer (schleife über Tokenanzahl) |
| Accuracy | ~81–87% | ~88–93% |

---

## Datensatz

**IAM Handwriting Dataset** – einzelne Wort-Bilder mit zugehörigen Labels. Heruntergeladen von Kaggle (`nibinv23/iam-handwriting-word-database`).

- ~115.000 Wort-Bilder
- Echte Handschrift von verschiedenen Autoren
- Aufteilung: **80% Training / 10% Validierung / 10% Test** (zufälliger Split, Seed 42)

Der Datensatz liegt unter `data/raw/` und ist **nicht** im Git-Repository enthalten (zu groß).

---

## Projektstruktur

```
AKI_HTA/
├── src/
│   ├── model.py          ← CRNN + CRNN_Seq2Seq Architekturen
│   ├── dataset.py        ← IAM-Datensatz, synthetische Daten, BOS/EOS/PAD tokens
│   └── transforms.py     ← Bildvorverarbeitung und Augmentation
│
├── training/
│   └── train.py          ← Trainer (CTC) + Seq2SeqTrainer (Teacher Forcing), --arch Flag
│
├── evaluation/
│   └── evaluate.py       ← CER und WER berechnen
│
├── utils/
│   ├── ctc_decoder.py    ← Greedy, Beam Search, LM Beam Search (GPU-optimiert)
│   ├── seq2seq_decoder.py← Autoregressive Greedy Decode für Seq2Seq
│   └── visualization.py  ← Trainingskurven und Beispielplots
│
├── outputs/
│   ├── checkpoints/      ← best_model.pt (CRNN), best_seq2seq.pt (Seq2Seq)
│   └── logs/             ← Trainingskurven, TensorBoard-Logs, history.json
│
├── predict.py            ← Vorhersage: --arch crnn oder --arch seq2seq
├── app.py                ← Web-Demo (Streamlit) mit Architektur-Auswahl
├── ANLEITUNG.md          ← Schritt-für-Schritt Workflow
└── requirements.txt      ← Python-Abhängigkeiten
```

---

## Warum 32×128 Pixel?

Diese Größe kommt aus dem originalen CRNN-Paper (Shi et al., 2016) und ist der Standard für wortbasierte HTR:

- **32px Höhe** — reicht aus um alle Striche und Kurven eines Buchstabens zu erfassen. Mehr Pixel würden kaum mehr Information liefern, aber die Trainingszeit stark erhöhen.
- **128px Breite** — passt für die meisten Wörter. Längere Wörter werden gestaucht, kürzere aufgefüllt.

---

## Benutzung

### Voraussetzungen
- Python 3.11
- NVIDIA GPU empfohlen (RTX 3070 oder ähnlich)
- venv aktivieren: `.venv\Scripts\activate`

### Training

```powershell
# CRNN (CTC) – Standard
python -m training.train --dataset iam --data-dir data/raw --epochs 50 --batch-size 64

# Seq2Seq (Transformer Decoder) – höhere Genauigkeit
python -m training.train --arch seq2seq --dataset iam --data-dir data/raw --epochs 50 --batch-size 64 --lr 5e-4

# Training fortsetzen
python -m training.train --arch seq2seq --dataset iam --data-dir data/raw --epochs 50 --resume outputs/checkpoints/best_seq2seq.pt
```

### Vorhersage – Kommandozeile

```powershell
# CRNN mit LM Beam Search (Standard)
python predict.py --image mein_bild.png --checkpoint outputs/checkpoints/best_model.pt

# Seq2Seq (Transformer Decoder)
python predict.py --image mein_bild.png --checkpoint outputs/checkpoints/best_seq2seq.pt --arch seq2seq

# Ordner mit mehreren Bildern
python predict.py --folder data/sample_images/ --checkpoint outputs/checkpoints/best_model.pt
```

Das Bild sollte ein **einzelnes handgeschriebenes Wort** auf hellem Hintergrund zeigen.

### Web-Demo

```powershell
streamlit run app.py
```

Browser öffnet sich unter `http://localhost:8501`. In der Sidebar kann man zwischen CRNN und Seq2Seq wählen — der passende Checkpoint wird automatisch vorgeschlagen.

### Evaluation

```powershell
python -m evaluation.evaluate --checkpoint outputs/checkpoints/best_model.pt --dataset iam --data-dir data/raw
```

Gibt **CER** (Character Error Rate), **WER** (Word Error Rate) und **Char Accuracy** aus.

---

## Metriken

| Metrik | Bedeutung | Gut | Akzeptabel |
|--------|-----------|-----|------------|
| **CER** | % falsch erkannte Zeichen | < 5% | < 15% |
| **WER** | % falsch erkannte Wörter | < 10% | < 20% |
| **Char Accuracy** | 1 − CER (in %) | > 95% | > 85% |

---

## Decoder

### CRNN (CTC) – 3 Decoder-Varianten

Das Modell gibt Wahrscheinlichkeiten aus – der Decoder wandelt diese in Text um.

| Decoder | Geschwindigkeit | Qualität | Beschreibung |
|---------|----------------|----------|-------------|
| **Greedy** | Sehr schnell | Basis | Nimmt bei jedem Zeitschritt das wahrscheinlichste Zeichen |
| **Beam Search** | Mittel | Besser | Verfolgt mehrere Hypothesen gleichzeitig (`torch.topk` auf GPU) |
| **LM Beam Search** | Schnell | Beste (ohne externes LM) | Breiterer Beam (15), GPU-optimiert, kein externes Sprachmodell nötig |

Alle drei Decoder laufen GPU-optimiert: Die Wahrscheinlichkeiten werden mit `torch.topk` direkt auf der GPU berechnet, nur ein einziger GPU→CPU Transfer pro Batch.

### Seq2Seq (Transformer Decoder) – autoregressiv

Der Seq2Seq-Decoder benötigt keine separaten CTC-Decoder-Varianten. Er generiert Zeichen direkt autoregressiv:

1. Start mit `BOS`-Token (Index 96)
2. Transformer Decoder berechnet das nächste wahrscheinlichste Zeichen
3. Dieses Zeichen wird als nächste Eingabe verwendet
4. Wiederholen bis `EOS`-Token (Index 97) oder maximale Länge

Der Decoder "sieht" alle bereits generierten Zeichen durch Self-Attention — das entspricht einem eingebauten Sprachmodell.

---

## GPU-Optimierungen

- **Mixed Precision (AMP):** Float16 auf Tensor Cores des RTX 3070 (~2× Speedup)
- **cudnn.benchmark:** Findet automatisch den schnellsten CUDA-Algorithmus für die feste Eingabegröße
- **Gradient Clipping:** Max-Norm 5.0 (verhindert explodierende Gradienten beim Transformer)
- **GradScaler:** Verhindert Underflow bei Float16

---

## Training live beobachten

```powershell
tensorboard --logdir outputs/logs/
```

Browser: `http://localhost:6006` – zeigt Loss und CER in Echtzeit für beide Architekturen.

---

## Wichtige Hinweise

- Das Modell erkennt **einzelne Wörter**, keine ganzen Sätze
- Modelldateien (`.pt`) sind zu groß für Git – lokal aufbewahren
- Bei Trainingsunterbrechung immer `--resume` verwenden um Fortschritt nicht zu verlieren
- Seq2Seq braucht mehr Epochen bis es konvergiert (Patience 15 statt 10 empfohlen)
