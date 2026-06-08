# HTR – Handwritten Text Recognition

Dieses Projekt erkennt handgeschriebene Wörter automatisch mithilfe neuronaler Netze. Ein Bild mit einem handgeschriebenen Wort wird eingegeben, das Modell gibt den erkannten Text aus.

---

## Was das Projekt macht

Das Modell nimmt ein Bild eines handgeschriebenen Wortes und gibt den erkannten Text zurück. Jedes Eingabebild wird automatisch auf **32×256 Pixel** skaliert — egal wie groß das Original ist. Trainiert wurde es auf dem **IAM Handwriting Dataset** – einer Sammlung von tausenden echten Handschriftproben.

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
Bild (32×256) → CNN → BiLSTM → Linear → CTC-Decoder → Text
```

Das klassische Modell. Der CTC-Decoder (Connectionist Temporal Classification) wandelt die Ausgabe-Wahrscheinlichkeiten in Text um, ohne dass eine pixelgenaue Ausrichtung zwischen Bild und Label benötigt wird.

**Ziel-Accuracy:** ~81–87% Zeichengenauigkeit

### Architektur 2 – CNN + BiLSTM + Transformer Decoder (Seq2Seq)

```
Bild (32×256) → CNN → BiLSTM → Transformer Decoder → Text
```

Neuere Architektur mit autoregressivem Decoder. Statt CTC generiert ein Transformer-Decoder die Ausgabe Zeichen für Zeichen und berücksichtigt dabei alle bisher generierten Zeichen (eingebautes Sprachwissen durch Self-Attention). Training mit **Teacher Forcing**, Inferenz autoregressiv.

**Ziel-Accuracy:** ~88–93% Zeichengenauigkeit

---

## Architektur-Details

### CNN – Feature-Extraktor (identisch in beiden Architekturen)

7 ConvBlocks (Conv2d → BatchNorm → ReLU → MaxPool):

| Block | Kanäle | Ausgabe (H×W) |
|-------|--------|---------------|
| 1 | 1 → 64 | 16×128 |
| 2 | 64 → 128 | 8×64 |
| 3 | 128 → 256 | 8×64 |
| 4 | 256 → 256 | 4×64 |
| 5 | 256 → 512 | 4×64 |
| 6 | 512 → 512 | 2×64 |
| 7 | 512 → 512 | 2×64 |

Nach dem CNN liegen 64 Feature-Spalten vor, jede mit 1024 Werten (512 Kanäle × 2 Höhe). Doppelt so viele Spalten wie bei 128px Breite — entscheidend für CTC (mehr Zeitschritte) und weniger Beschneidung langer Wörter.

### BiLSTM Encoder (identisch in beiden Architekturen)

- 2 gestapelte bidirektionale LSTM-Schichten
- 256 Units pro Richtung → 512 Ausgabe-Features
- Liest die 64 Feature-Spalten als Zeitsequenz vorwärts und rückwärts
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
- Training mit **Label Smoothing 0.1** (`CrossEntropyLoss`): verhindert Übervertrauen und verbessert die Generalisierung — Standard bei Transformer-Decodern

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
│   └── transforms.py     ← Vorverarbeitung: Polarität, Deslanting, Augmentation
│
├── training/
│   └── train.py          ← Trainer (CTC) + Seq2SeqTrainer (Teacher Forcing), --arch Flag
│
├── evaluation/
│   └── evaluate.py       ← CER und WER berechnen
│
├── utils/
│   ├── ctc_decoder.py    ← Greedy, Beam Search, LM Beam Search (GPU-optimiert)
│   ├── seq2seq_decoder.py← Autoregressive Greedy + Beam Search Decode für Seq2Seq
│   └── visualization.py  ← Trainingskurven und Beispielplots
│
├── outputs/
│   ├── checkpoints/      ← best_model.pt (CRNN), best_seq2seq.pt (Seq2Seq), *_BASELINE_* (Backup)
│   └── logs/             ← Trainingskurven, TensorBoard-Logs, history.json
│
├── tools/
│   └── compare_heights.py← Diagnose: Bilddetail bei Höhe 32 vs 48 vs 64 vergleichen
├── predict.py            ← Vorhersage: --arch crnn oder --arch seq2seq
├── app.py                ← Web-Demo (Streamlit) mit Architektur-Auswahl
├── ANLEITUNG.md          ← Schritt-für-Schritt Workflow
└── requirements.txt      ← Python-Abhängigkeiten
```

---

## Warum 32×256 Pixel?

- **32px Höhe** — reicht aus um alle Striche und Kurven eines Buchstabens zu erfassen. Mehr Pixel (z.B. 64) würden kaum mehr Information liefern, aber die Trainingszeit stark erhöhen **und die Modell-Gewichte ändern** (die LSTM-Eingabedimension hängt an der Höhe → kein Weitertrainieren bestehender Checkpoints möglich).
- **256px Breite** — ursprünglich 128px (CRNN-Paper, Shi et al. 2016), hier auf 256px erhöht. Begründung:
  - **Mehr Zeitschritte für CTC.** Nach dem CNN ergeben 256px → 64 Feature-Spalten (statt 32). CTC benötigt mindestens so viele Zeitschritte wie Zeichen (mit Blanks ~2× so viele) — 32 Spalten waren bei langen Wörtern eine harte Obergrenze.
  - **Weniger Beschneidung.** Bei 128px wurden lange Wörter rechts abgeschnitten (`ResizeToHeight`), während das Label vollständig blieb → unlernbare Zeichen. 256px fasst die meisten IAM-Wörter ohne Verlust.
  - **Die Breite ändert keine Gewichts-Shapes** (nur die Höhe tut das) — bestehende Checkpoints lassen sich mit `--img-width 256` ohne Neustart weitertrainieren.

> **Wichtig:** Training und Inferenz (`predict.py`, `app.py`) müssen dieselbe Breite verwenden, sonst sieht das Modell bei der Vorhersage eine andere Stauchung als beim Training. Da Bildgröße und Deslant im Checkpoint gespeichert werden, geschieht das automatisch.

> **Experiment 48×384 + Deslanting:** `train_loop.ps1` startet einen frischen Lauf mit Höhe 48, Breite 384 und Deslanting (Schräglagen-Korrektur), um die letzten Prozentpunkte zu holen. Da die Höhe die Gewichts-Shapes ändert, ist das ein Neustart von 0; das 32×256-Modell bleibt als Backup erhalten.

---

## Bild-Vorverarbeitung (Polarität & Deslanting)

### Polarität – warum Uploads sonst scheitern
Das IAM-Training **invertiert** die Bilder (`ImageOps.invert`): das Modell lernt also **helle Schrift auf dunklem Grund**. Ein normales Foto/Scan ist aber **dunkle Schrift auf hellem Grund** — die umgekehrte Polarität. Ohne Korrektur sieht das Modell bei Uploads das „Negativ" und liefert Kauderwelsch, obwohl es auf IAM exzellent abschneidet.

`ensure_training_polarity()` ([transforms.py](src/transforms.py)) löst das: Es erkennt am Median-Pixel, ob der Hintergrund hell ist, und invertiert dann auf die Trainings-Polarität. Angewendet in `predict.py` und `app.py` (die Bilder direkt laden). `evaluate.py` braucht es **nicht** — dort kommen die Bilder über `IAMDataset`, das bereits invertiert.

→ **Für Uploads gilt:** schwarze Schrift auf hellem Grund, einzelnes Wort, eng zugeschnitten. Die Korrektur passiert automatisch.

### Deslanting (`--deslant`)
Optionale Schräglagen-Korrektur (Vinciarelli-Luettin-Heuristik): Das Bild wird über mehrere Scherwinkel getestet; gewählt wird der Winkel, der die Tinte am stärksten in senkrechte Spalten konzentriert. Das richtet kursive Schrift auf und reduziert Form-Verwechslungen (o/c, B/s, t/r). Wird im Checkpoint vermerkt und muss bei Training und Inferenz identisch sein (geschieht automatisch über die Checkpoint-Metadaten).

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

# Seq2Seq (Transformer Decoder) – Beam Search (Standard)
python predict.py --image mein_bild.png --checkpoint outputs/checkpoints/best_seq2seq.pt --arch seq2seq

# Seq2Seq – Greedy statt Beam
python predict.py --image mein_bild.png --checkpoint outputs/checkpoints/best_seq2seq.pt --arch seq2seq --decoder greedy

# Ordner mit mehreren Bildern
python predict.py --folder data/sample_images/ --checkpoint outputs/checkpoints/best_model.pt
```

Das Bild sollte ein **einzelnes handgeschriebenes Wort**, **dunkle Schrift auf hellem Grund**, eng zugeschnitten zeigen. Polarität, Bildgröße und Deslant werden automatisch aus dem Checkpoint bzw. per Hintergrund-Erkennung gesetzt — keine Flags nötig.

### Web-Demo

```powershell
streamlit run app.py
```

Browser öffnet sich unter `http://localhost:8501`. In der Sidebar kann man zwischen CRNN und Seq2Seq wählen — der passende Checkpoint wird automatisch vorgeschlagen.

### Evaluation

```powershell
# Test-Set (ungesehene Samples), passend zum random-Training
python -m evaluation.evaluate --checkpoint outputs/checkpoints/best_model.pt --dataset iam --data-dir data/raw

# Seq2Seq mit Beam Search
python -m evaluation.evaluate --checkpoint outputs/checkpoints/best_seq2seq.pt --arch seq2seq --dataset iam --data-dir data/raw
```

Gibt **CER** (Character Error Rate), **WER** (Word Error Rate) und **Char Accuracy** aus.

Wichtige Flags: `--arch crnn|seq2seq`, `--split test|val`, `--split-mode random|author`.
**Standard ist `--split test --split-mode random`** — der `--split-mode` muss zum Training passen. Da standardmäßig mit `random` trainiert wird (alle Schreiber gesehen), misst der `test`-Split ungesehene **Samples**, aber keine ungesehenen **Schreiber**. Ein echter neue-Schreiber-Test (`--split-mode author`) ist nur aussagekräftig, wenn auch mit `author` trainiert wurde.

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

Der Seq2Seq-Decoder generiert Zeichen autoregressiv:

1. Start mit `BOS`-Token (Index 96)
2. Transformer Decoder berechnet das nächste Zeichen
3. Dieses Zeichen wird als nächste Eingabe verwendet
4. Wiederholen bis `EOS`-Token (Index 97) oder maximale Länge

Der Decoder "sieht" alle bereits generierten Zeichen durch Self-Attention — das entspricht einem eingebauten Sprachmodell.

Es gibt zwei Decode-Strategien:

| Decoder | Geschwindigkeit | Qualität | Beschreibung |
|---------|----------------|----------|-------------|
| **Greedy** | Schnell | Basis | Nimmt bei jedem Schritt das wahrscheinlichste Token |
| **Beam Search** *(Standard)* | Langsamer | Besser | Verfolgt `beam_width=5` Hypothesen parallel und wählt am Ende die global wahrscheinlichste Sequenz; Längen-Normalisierung (Penalty 0.6) gegen die Bevorzugung kurzer Ausgaben |

Beam Search korrigiert frühe Fehlentscheidungen, die Greedy nicht mehr revidieren kann, und bringt typischerweise einige Prozent CER. In `predict.py` und der Web-Demo ist Beam Search der Standard; Greedy ist über `--decoder greedy` weiterhin wählbar.

---

## Training-Details

- **Optimizer: AdamW** – entkoppelter Weight Decay (`1e-4`), korrekte L2-Regularisierung → bessere Generalisierung als das frühere `Adam`
- **LR-Warmup:** linearer Anstieg über die ersten `--warmup-steps` (Standard 500) Schritte, stabilisiert den instabilen Transformer-Start. Nur bei Neustart, nicht beim Resume (dort ist die LR bereits eingependelt)
- **Scheduler:** `ReduceLROnPlateau` halbiert die LR bei stagnierender Val-Loss
- **Label Smoothing 0.1** (nur Seq2Seq): gegen Übervertrauen
- **EMA (Exponential Moving Average):** ein gleitender Mittelwert der Gewichte (Decay 0.999, `--no-ema` schaltet ab). Für Validierung **und** den gespeicherten Checkpoint werden die geglätteten EMA-Gewichte genutzt — meist etwas besser als die zuletzt trainierten. `best_seq2seq.pt`/`best_model.pt` enthalten also die EMA-Gewichte.
- **Train-CER:** wird über mehrere Batches gemittelt (verlässlicher als 1 Batch); **Val-CER** läuft mit Greedy über den ganzen Val-Satz (schnell, rankt Epochen wie Beam). Beam wird gezielt fürs finale Eval/Inferenz genutzt.
- **Selbst-konfigurierende Checkpoints:** `img_height`, `img_width` und `deslant` werden im Checkpoint gespeichert. `predict.py`, `app.py` und `evaluate.py` lesen sie automatisch — kein manuelles Setzen von Bildgröße/Deslant nötig.

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
- **Upload-Bilder:** dunkle Schrift auf hellem Grund, einzelnes Wort, eng zugeschnitten — die Polarität wird automatisch korrigiert (`ensure_training_polarity`), sonst kommt Kauderwelsch
- Modelldateien (`.pt`) sind zu groß für Git – lokal aufbewahren
- Bei Trainingsunterbrechung immer `--resume` verwenden um Fortschritt nicht zu verlieren
- Seq2Seq braucht mehr Epochen bis es konvergiert (Patience 15 statt 10 empfohlen)
- Beim `--resume` wird die **gespeicherte Lernrate beibehalten** (ggf. bereits abgesenkt), damit das Modell sauber auskonvergiert. Mit `--reset-lr` wird sie stattdessen auf `--lr` zurückgesetzt — nützlich beim ersten Resume oder wenn die LR durch viele Neustarts zu klein geworden ist.
- Für maximale Genauigkeit besser **ein langer Lauf** (hohe `--epochs` + `--patience`) als viele kurze Neustart-Runden: so verwaltet ein einziger Scheduler die Lernrate durchgehend.
