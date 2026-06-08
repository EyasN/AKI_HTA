# Handschrifterkennung (HTR) – Projektpräsentation

> Bild eines handgeschriebenen Wortes → erkannter Text.
> Zwei Deep-Learning-Architekturen, trainiert auf dem IAM Handwriting Dataset.

---

## 1. Worum geht es? (Problemstellung)

- **Ziel:** Ein Foto/Scan eines handgeschriebenen **Wortes** rein → der **Text** raus.
- **Warum schwierig?** Jede Handschrift ist anders; keine feste Ausrichtung zwischen Pixeln und Buchstaben; variable Wortlängen.
- **Anwendung:** Digitalisierung von Formularen, Notizen, historischen Dokumenten.

**Sprechpunkt:** Das ist klassische „Sequence-to-Sequence"-Aufgabe — Bild (2D) → Zeichenfolge (1D), ohne zu wissen, wo welcher Buchstabe genau sitzt.

---

## 2. Die Daten – IAM Handwriting Dataset

- Echte Handschrift vieler verschiedener Schreiber, einzelne Wort-Bilder + Labels.
- Nach Filterung (gültige Zeichen, max. 32 Zeichen, Bild vorhanden) nutzbar:
  - **Training: 30.644 · Validierung: 3.830 · Test: 3.831** Wörter
- Aufteilung **80 / 10 / 10** (zufällig, fester Seed 42 → reproduzierbar).

**Wichtige Ehrlichkeit (Folie zum Hervorheben):**
- `random`-Split = **dieselben Schreiber** in Train & Test → misst „ungesehene **Samples**".
- Ein echter „neue-**Schreiber**"-Test (writer-disjunkt) wäre strenger — dafür bräuchte man die offizielle IAM-Aufteilung. Unsere Zahlen sind also **In-Domain**-Genauigkeit.

---

## 3. Vorverarbeitung (Pipeline)

```
Bild → Graustufen → Polarität normalisieren → (Deslanting) → auf Höhe skalieren → Augmentation* → Tensor [-1,1]
```
\*Augmentation nur beim Training (Rotation, Scherung, Perspektive, Helligkeit, Blur, Rauschen, Stiftstärke).

- **Größe:** auf **32×256 px** (Höhe fix, Breite proportional, Rest gepaddet).
- **Polaritäts-Normalisierung** (entscheidend, s. Folie 9).
- **Deslanting** (optional): richtet kursive Schrift senkrecht.

---

## 4. Architektur 1 – CRNN (CTC)

```
Bild (32×256) → CNN → BiLSTM → Linear → CTC-Loss → Text
```

- **CNN:** 7 Conv-Blöcke extrahieren visuelle Merkmale (Striche, Kurven). Reduziert das Bild auf eine Folge von **Feature-Spalten**.
- **BiLSTM:** liest diese Spalten als Sequenz vor- und rückwärts → Kontext („th", „ing"…).
- **CTC-Loss:** löst das Ausrichtungsproblem — lernt selbst, welche Spalte zu welchem Buchstaben gehört, ohne Pixel-genaue Annotation.

**Sprechpunkt:** CTC ist genial für „kein Alignment vorhanden", hat aber **kein Sprachwissen** — es klassifiziert nur pro Zeitschritt.

---

## 5. Architektur 2 – Seq2Seq (CNN + BiLSTM + Transformer-Decoder)

```
Bild (32×256) → CNN → BiLSTM (Encoder) → Transformer-Decoder → Text
```

- Gleicher CNN+BiLSTM-Encoder, aber statt CTC ein **autoregressiver Transformer-Decoder**.
- Generiert das Wort **Zeichen für Zeichen** und „sieht" dabei alle bisher erzeugten Zeichen (Self-Attention) → **eingebautes Sprachmodell**.
- Training mit **Teacher Forcing**, Inferenz autoregressiv (BOS → … → EOS).
- ~**22,9 Mio. Parameter** (LSTM-Hidden 512, d_model 256, 3 Decoder-Layer, 8 Heads).

**Sprechpunkt:** Der Vorteil gegenüber CTC: Der Decoder kennt Sprachkontext. Das ist auch der Grund, warum ein **externer** LM hier wenig bringt — der LM steckt schon im Modell.

| | CRNN (CTC) | Seq2Seq (Transformer) |
|---|---|---|
| Sprachwissen | keins | eingebaut |
| Inferenz | sehr schnell (parallel) | langsamer (Token für Token) |
| Genauigkeit (unser Ergebnis) | ~81–87 % | **93,9 %** |

---

## 6. Training – die wichtigsten Bausteine

- **Loss:** CTC (CRNN) bzw. CrossEntropy + **Label Smoothing 0.1** (Seq2Seq).
- **Optimizer: AdamW** (entkoppelter Weight Decay → bessere Generalisierung).
- **LR-Warmup** (500 Schritte) → stabiler Transformer-Start; danach `ReduceLROnPlateau`.
- **EMA (Exponential Moving Average)** der Gewichte → geglättete, meist bessere Gewichte fürs finale Modell.
- **Mixed Precision (AMP)** auf der RTX 3070 → ~2× schneller.
- **Early Stopping** + Checkpointing des besten Modells.

**Sprechpunkt:** Jede dieser Maßnahmen ist ein kleiner, gezielter Hebel — zusammen haben sie uns von ~88 % auf ~94 % gebracht.

---

## 7. Decoding – Greedy vs. Beam Search

- **Greedy:** nimmt pro Schritt das wahrscheinlichste Zeichen — schnell, aber kann frühe Fehler nicht korrigieren.
- **Beam Search:** verfolgt mehrere Hypothesen parallel (`beam_width=5`) und wählt die global wahrscheinlichste Sequenz; **Längen-Penalty** gegen zu kurze Ausgaben.

**Ergebnis (Test-Set):**

| Decoder | Char Accuracy | CER |
|---|---|---|
| Greedy | 90,48 % | 9,52 % |
| **Beam Search** | **92,98 %** | **7,02 %** |

→ Beam bringt real **+1,7 %**. Standard in App & Vorhersage.

---

## 8. Ergebnisse

**Finales Modell: Seq2Seq, 48×384 + Deslanting, Beam Search (Test-Set):**

| Metrik | Wert |
|---|---|
| **Char Accuracy** | **93,91 %** |
| **CER** | **6,09 %** |
| **Word Accuracy** | 85,59 % |
| **WER** | 14,41 % |

**Fortschritt durch die Optimierungen:**

| Stand | Char Accuracy | CER |
|---|---|---|
| Ausgangspunkt | ~88 % | ~12 % |
| 32×256 (optimiert) | 92,98 % | 7,02 % |
| **48×384 + Deslanting** | **93,91 %** | **6,09 %** |

- **Eigene Fotos (andere Handschrift, Handykamera): ~90 %** — zeigt echte Generalisierung über IAM hinaus.

**Typische Restfehler:** `Williams → Williame`, `coordinate → cordinate`, `enemy → ememy` — Eigennamen, Doppelbuchstaben, mehrdeutige Zeichen.

**Sprechpunkt:** Die übrigen ~6 % sind der schwierige „Schwanz" — seltene Wörter, Eigennamen, mehrdeutige Buchstaben. 95 % auf IAM ist Forschungsniveau; mit ~94 % sind wir nah dran.

---

## 9. Der entscheidende Bug – Polaritäts-Mismatch ⭐

> **Die beste Story der Präsentation.**

- **Symptom:** Modell erreicht 93 % im Test — aber in der Web-App kam bei Uploads **Kauderwelsch**.
- **Ursache:** Beim IAM-Training werden die Bilder **invertiert** (helle Schrift auf dunklem Grund). Die App lud Uploads aber **nicht-invertiert** (dunkle Schrift auf hellem Grund) → das Modell sah das **Negativ**.

| Label | App (ohne Fix) | mit Fix |
|---|---|---|
| and | learities ❌ | **and** ✓ |
| conference | appose ❌ | **conference** ✓ |
| Williams | MONGERALD ❌ | **Williams** ✓ |

- **Fix:** `ensure_training_polarity()` erkennt den Hintergrund und dreht das Bild auf die Trainings-Polarität → **9/10 wieder korrekt**.

**Lehre:** Inferenz muss die Trainings-Vorverarbeitung **exakt** reproduzieren. Das Modell war nie schlecht — es wurde nur falsch gefüttert.

---

## 10. Optimierungs-Reise (was wir probiert haben & warum)

| Maßnahme | Effekt |
|---|---|
| Breite 128 → **256** | Mehr Zeitschritte für CTC, lange Wörter nicht mehr abgeschnitten |
| Adam → **AdamW** + **Warmup** | Stabileres, besseres Training |
| **Label Smoothing** + **EMA** | Bessere Generalisierung, geglättete Gewichte |
| **Beam Search** statt Greedy | +1,7 % Genauigkeit |
| LR-Management (kein Reset-Loop) | Sauberes Auskonvergieren |
| **48×384 + Deslanting** | CER 7,0 → 6,1 % (−13 % relativ) + Overfitting fast eliminiert |
| **Polaritäts-Fix** | App/Uploads funktionieren überhaupt erst |

**Spannende Erkenntnis (datengestützt):** „Ist das Bild mit 32px zu klein/verschwommen?" → **Nein.** Bei 32px war der **Train-CER ~2 %**, der **Val-CER ~7 %** — die große Lücke = **Overfitting/Generalisierung**, nicht Auflösung. Deslanting + 48px haben dann genau diese **Lücke von ~5 % auf ~1,5 % verkleinert** und die CER gesenkt. Die Hypothese „mehr Pixel" wurde also erst widerlegt und dann gezielt der richtige Hebel (Generalisierung) gezogen.

---

## 11. Live-Demo (Streamlit)

```powershell
streamlit run app.py
```

- Architektur (CRNN/Seq2Seq) + Modell wählbar.
- Bild hochladen → erkannter Text + Konfidenz + Vorschau „Was das Modell sieht".
- **Gutes Eingabebild:** einzelnes Wort, dunkle Schrift auf hellem Grund, eng zugeschnitten.

**Sprechpunkt:** Polarität, Bildgröße und Deslant werden automatisch aus dem Checkpoint übernommen — der Nutzer muss nichts einstellen.

---

## 12. Grenzen & Ausblick

- **Nur einzelne Wörter**, keine ganzen Zeilen/Sätze.
- **Ehrliche Evaluation:** für eine echte „neue-Schreiber"-Zahl bräuchte man den offiziellen writer-disjunkten Split.
- **Foto-Robustheit:** automatische Binarisierung/Zuschnitt für schlechte Handyfotos würde die ~90 % auf eigenen Bildern weiter anheben.
- **Richtung 95 %:** der schwierige Schwanz (Eigennamen, seltene Wörter) bräuchte mehr/andere Daten oder Synthetic-Pretraining.

---

## 13. Fazit

- Funktionierendes End-to-End-System: **Bild rein → Wort raus**, **93,9 % auf IAM**, ~90 % auf eigenen Fotos.
- Zwei Architekturen implementiert und verglichen (CTC vs. Transformer-Decoder).
- Systematische Optimierung: von ~88 % auf 93,9 % über viele gezielte Hebel (Beam, AdamW, EMA, Deslanting …).
- Wichtigste Lektion: **Der Teufel steckt in der Vorverarbeitung** — ein Polaritäts-Detail entschied über Erfolg/Misserfolg.

---

### Technischer Steckbrief (Backup-Folie)

- **Framework:** PyTorch · **GPU:** NVIDIA RTX 3070 · **Python 3.11**
- **Eingabe:** Graustufenbild, finales Modell **48×384** (Basismodell 32×256) · **Vokabular:** 96 Zeichen (CTC) / 98 (Seq2Seq, mit BOS/EOS/PAD)
- **Seq2Seq:** ~22,9 Mio. Parameter · **Metriken:** CER (Zeichen), WER (Wörter)
- **Decoder:** Greedy / Beam Search (Seq2Seq) · Greedy / Beam / LM-Beam (CTC)
- **Vorverarbeitung:** Polaritäts-Normalisierung + Deslanting; Config self-describing im Checkpoint
