# HTR – Handwritten Text Recognition

Dieses Projekt erkennt handgeschriebene Wörter automatisch mithilfe eines neuronalen Netzes. Ein Bild mit einem handgeschriebenen Wort wird eingegeben, das Modell gibt den erkannten Text aus.

---

## Was das Projekt macht

Das Modell nimmt ein Bild eines handgeschriebenen Wortes und gibt den erkannten Text zurück. Jedes Eingabebild wird automatisch auf **32×128 Pixel** skaliert — egal wie groß das Original ist. Trainiert wurde es auf dem **IAM Handwriting Dataset** – einer Sammlung von tausenden echten Handschriftproben.

**Beispiel:**
```
Eingabe:  [Bild von "hello"]
Ausgabe:  "hello"
```

---

## Wie es funktioniert – die Architektur

Das Modell heißt **CRNN** (Convolutional Recurrent Neural Network) und besteht aus drei Teilen:

```
Bild (32×128) → CNN → BiLSTM → Linear → CTC-Decoder → Text
```

### 1. CNN – Feature-Extraktor
Erkennt visuelle Muster im Bild: Striche, Kurven, Kanten. Gibt Feature-Maps zurück die beschreiben "was wo im Bild steht".

### 2. BiLSTM – Kontext verstehen
Liest die Feature-Maps als Sequenz von links nach rechts **und** von rechts nach links. So versteht es den Kontext: ein Buchstabe hängt von seinen Nachbarn ab (z.B. "th" in "the").

### 3. CTC-Decoder
Wandelt die Modellausgabe in Text um – ohne dass das Modell wissen muss wo genau jeder Buchstabe im Bild ist.

| Komponente | Aufgabe |
|-----------|---------|
| CNN (7 Schichten) | Visuelle Features extrahieren |
| BiLSTM (2 Schichten, 256 Units) | Buchstabenkontext modellieren |
| Linear + Log-Softmax | Klassen-Wahrscheinlichkeiten |
| CTC-Decoder | Wahrscheinlichkeiten → Text |

---

## Datensatz

**IAM Handwriting Dataset** – einzelne Wort-Bilder mit zugehörigen Labels. Heruntergeladen von Kaggle (`nibinv23/iam-handwriting-word-database`).

- ~115.000 Wort-Bilder
- Echte Handschrift von verschiedenen Autoren
- Aufteilung: 80% Training / 10% Validierung / 10% Test

Der Datensatz liegt unter `data/raw/` und ist **nicht** im Git-Repository enthalten (zu groß).

---

## Projektstruktur

```
AKI_HTA/
├── src/
│   ├── model.py          ← CRNN-Architektur (CNN + BiLSTM)
│   ├── dataset.py        ← IAM-Datensatz laden + synthetische Daten
│   └── transforms.py     ← Bildvorverarbeitung und Augmentation
│
├── training/
│   └── train.py          ← Trainingsschleife mit Early Stopping
│
├── evaluation/
│   └── evaluate.py       ← CER und WER berechnen
│
├── utils/
│   ├── ctc_decoder.py    ← Greedy- und Beam-Search-Decoder
│   └── visualization.py  ← Trainingskurven und Beispielplots
│
├── outputs/
│   ├── checkpoints/      ← Gespeicherte Modelle (.pt) – nicht in Git
│   └── logs/             ← Trainingskurven, TensorBoard-Logs
│
├── predict.py            ← Vorhersage über Kommandozeile
├── app.py                ← Web-Demo mit Streamlit
├── ANLEITUNG.md          ← Schritt-für-Schritt Workflow
└── requirements.txt      ← Python-Abhängigkeiten
```

---

## Warum 32×128 Pixel?

Diese Größe kommt aus dem originalen CRNN-Paper (Shi et al., 2016) und ist der Standard für wortbasierte HTR:

- **32px Höhe** — reicht aus um alle Striche und Kurven eines Buchstabens zu erfassen. Mehr Pixel würden kaum mehr Information liefern, aber die Trainingszeit stark erhöhen.
- **128px Breite** — passt für die meisten Wörter. Längere Wörter werden gestaucht, kürzere aufgefüllt.

Eingabebilder können beliebig groß sein — das Modell skaliert sie automatisch beim Laden.

---

## Benutzung

### Voraussetzungen
- Python 3.11
- NVIDIA GPU empfohlen (RTX 3070 oder ähnlich)
- venv aktivieren: `.venv\Scripts\activate`

### Training

```powershell
# Neu trainieren
python -m training.train --dataset iam --data-dir data/raw --epochs 50 --batch-size 64 --patience 15

# Training fortsetzen
python -m training.train --dataset iam --data-dir data/raw --epochs 50 --resume outputs/checkpoints/best_model.pt --patience 15
```

### Vorhersage – Kommandozeile

```powershell
# Einzelnes Bild
python predict.py --image mein_bild.png --checkpoint outputs/checkpoints/best_model.pt

# Mit Beam-Search (langsamer aber genauer)
python predict.py --image mein_bild.png --checkpoint outputs/checkpoints/best_model.pt --decoder beam
```

Das Bild sollte ein **einzelnes handgeschriebenes Wort** auf hellem Hintergrund zeigen.

### Web-Demo

```powershell
streamlit run app.py
```

Browser öffnet sich unter `http://localhost:8501`. Dort kann man ein Bild hochladen und das Ergebnis direkt sehen.

### Evaluation

```powershell
python -m evaluation.evaluate --checkpoint outputs/checkpoints/best_model.pt --dataset iam --data-dir data/raw
```

Gibt **CER** (Character Error Rate) und **WER** (Word Error Rate) aus.

---

## Metriken

| Metrik | Bedeutung | Gut | Akzeptabel |
|--------|-----------|-----|------------|
| **CER** | % falsch erkannte Zeichen | < 5% | < 15% |
| **WER** | % falsch erkannte Wörter | < 10% | < 20% |

---

## Decoder

Das Modell gibt Wahrscheinlichkeiten aus – der Decoder wandelt diese in Text um.

| Decoder | Geschwindigkeit | Qualität | Beschreibung |
|---------|----------------|----------|-------------|
| **Greedy** | Schnell | Basis | Nimmt bei jedem Zeitschritt das wahrscheinlichste Zeichen |
| **Beam Search** | Mittel | Besser | Verfolgt mehrere Hypothesen gleichzeitig |
| **LM Beam Search** | Langsam | Am besten | Beam Search + Sprachmodell (pyctcdecode) |

Der **LM Beam Search** Decoder (Language Model Beam Search) ist der Standard in der App und bei Evaluation. Er nutzt die Library `pyctcdecode` und kann optional mit einem vortrainierten KenLM-Sprachmodell (.arpa-Datei) erweitert werden — dann verbessert er die WER deutlich, weil er weiß welche Wortfolgen auf Englisch wahrscheinlich sind.

Ohne KenLM-Datei verhält er sich wie ein verbesserter Beam Search. Mit einer KenLM-Datei:
```powershell
python predict.py --image bild.png --checkpoint outputs/checkpoints/best_model.pt --decoder lm
```

---

## Training live beobachten

```powershell
tensorboard --logdir outputs/logs/
```

Browser: `http://localhost:6006` – zeigt Loss und CER in Echtzeit.

---

## Wichtige Hinweise

- Das Modell erkennt **einzelne Wörter**, keine ganzen Sätze
- Modelldateien (`.pt`) sind zu groß für Git – lokal aufbewahren oder über OneDrive teilen
- Bei Trainingsunterbrechung immer `--resume` verwenden um Fortschritt nicht zu verlieren
