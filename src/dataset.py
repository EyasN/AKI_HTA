"""
Dataset-Loader für HTR.

Unterstützte Quellen:
  1. IAM Handwriting Dataset  (benötigt manuelle Registrierung auf fki.inf.unibe.ch)
  2. SyntheticHTRDataset      (generiert Trainingsbilder on-the-fly mit PIL – sofort nutzbar)

Das SyntheticDataset erlaubt sofortiges Testen der gesamten Pipeline, ohne
externen Datensatz herunterladen zu müssen. Für ernsthafte Forschung wird IAM empfohlen.

Zeichensatz (ALPHABET):
  Alle druckbaren ASCII-Zeichen (95 Zeichen) + CTC-Blank-Token am Index 0.
  ALPHABET[0] = '-' (Blank), ALPHABET[1:] = Zeichen 32-126
"""

import os
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.transforms import get_train_transforms, get_val_transforms


# ── Zeichensatz ───────────────────────────────────────────────────────────────
BLANK_TOKEN = "-"
ALPHABET = BLANK_TOKEN + string.printable[:95]   # 96 Klassen insgesamt
CHAR2IDX: Dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}
IDX2CHAR: Dict[int, str] = {i: c for c, i in CHAR2IDX.items()}

NUM_CLASSES = len(ALPHABET)   # 96

# ── Seq2Seq Sondertokens ──────────────────────────────────────────────────────
PAD_IDX       = 0               # Blank-Index als Padding wiederverwenden
BOS_IDX       = NUM_CLASSES     # 96  – Beginn der Sequenz
EOS_IDX       = NUM_CLASSES + 1 # 97  – Ende der Sequenz
SEQ2SEQ_VOCAB = NUM_CLASSES + 2 # 98  – Gesamtvokabular für Seq2Seq


def encode_label(text: str) -> torch.Tensor:
    """Wandelt einen String in einen Tensor aus Klassen-Indizes um."""
    return torch.tensor([CHAR2IDX[c] for c in text if c in CHAR2IDX], dtype=torch.long)


def decode_label(indices: List[int]) -> str:
    """Wandelt eine Liste von Indizes zurück in einen String."""
    return "".join(IDX2CHAR.get(i, "") for i in indices)


# ── IAM Dataset ───────────────────────────────────────────────────────────────

class IAMDataset(Dataset):
    """
    Loader für das IAM Handwriting Dataset.

    Ordnerstruktur (nach dem Entpacken):
        data/raw/
        ├── words/          ← Bilddateien  (z.B. a01/a01-000u/a01-000u-00-00.png)
        └── ascii/
            └── words.txt   ← Annotationen

    Registrierung & Download:
        https://fki.inf.unibe.ch/databases/iam-handwriting-database
        Kostenlos für Forschung nach Registrierung.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",    # "train", "val", "test"
        img_height: int = 32,
        img_width: int = 128,
        max_label_len: int = 32,
        split_mode: str = "random",  # "random" (80/10/10) oder "author" (offizielle IAM-Aufteilung)
        deslant: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_label_len = max_label_len
        self.img_height = img_height
        self.img_width = img_width
        self.split_mode = split_mode

        self.transform = (
            get_train_transforms(img_height, img_width, deslant=deslant)
            if split == "train"
            else get_val_transforms(img_height, img_width, deslant=deslant)
        )

        self.samples: List[Tuple[Path, str]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        # Support both original IAM layout (ascii/words.txt) and Kaggle layout (words.txt)
        for candidate in [
            self.root_dir / "ascii" / "words.txt",
            self.root_dir / "words.txt",
            self.root_dir / "iam_words" / "words.txt",
        ]:
            if candidate.exists():
                words_file = candidate
                break
        else:
            raise FileNotFoundError(
                f"words.txt nicht gefunden unter {self.root_dir}.\n"
                "Bitte IAM-Datensatz herunterladen: "
                "https://fki.inf.unibe.ch/databases/iam-handwriting-database"
            )

        # Alle gültigen Samples laden
        all_samples: List[Tuple[Path, str]] = []

        # Für author-Split: Autoren-IDs filtern
        TRAIN_AUTHORS = {"a01", "a02", "d01", "d04", "e01", "f07", "g01", "g07"}
        VAL_AUTHORS   = {"a03", "b01", "b02", "c02", "d06"}
        TEST_AUTHORS  = {"a04", "a05", "a06", "b04", "c01", "d02", "d03"}
        author_split_map = {"train": TRAIN_AUTHORS, "val": VAL_AUTHORS, "test": TEST_AUTHORS}

        with open(words_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split(" ")
                if len(parts) < 9 or parts[1] != "ok":
                    continue

                word_id = parts[0]
                author  = word_id.split("-")[0]

                # Bei author-Split: nur erlaubte Autoren laden
                if self.split_mode == "author" and author not in author_split_map[self.split]:
                    continue

                label = parts[-1]
                if len(label) > self.max_label_len:
                    continue
                if not all(c in CHAR2IDX for c in label):
                    continue

                parts_id = word_id.split("-")
                rel = Path("words") / parts_id[0] / f"{parts_id[0]}-{parts_id[1]}" / f"{word_id}.png"
                img_path = words_file.parent / rel
                if img_path.exists():
                    all_samples.append((img_path, label))

        if self.split_mode == "random":
            # Reproduzierbarer 80/10/10-Split über alle verfügbaren Daten
            import random as _rnd
            rng = _rnd.Random(42)
            rng.shuffle(all_samples)
            n = len(all_samples)
            splits = {
                "train": all_samples[:int(n * 0.80)],
                "val":   all_samples[int(n * 0.80):int(n * 0.90)],
                "test":  all_samples[int(n * 0.90):],
            }
            self.samples = splits[self.split]
        else:
            self.samples = all_samples

        print(f"IAM [{self.split}] ({self.split_mode}): {len(self.samples)} Samples geladen.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        img = ImageOps.invert(img)    # IAM: weißer Hintergrund → invertieren

        image_tensor = self.transform(img)
        label_tensor = encode_label(label)
        return image_tensor, label_tensor, len(label_tensor)


# ── Synthetischer Datensatz (sofort nutzbar, kein Download nötig) ─────────────

class SyntheticHTRDataset(Dataset):
    """
    Generiert handschriftähnliche Bilder on-the-fly mit PIL.

    Nützlich für:
      - Schnelles Testen der gesamten Pipeline
      - Überprüfen von Modellarchitektur und Training-Loop
      - Demo ohne externe Daten

    Hinweis: Synthetische Daten liefern schlechtere Generalisierung als echte
    Handschrift. Für das finale Modell IAM oder ähnliche Datensätze verwenden.
    """

    FONT_STYLES = [
        "DejaVuSans",
        "DejaVuSans-Bold",
        "DejaVuSans-Oblique",
    ]

    WORD_LIST = (
        "hello world python deep learning neural network handwriting "
        "recognition convolutional recurrent training evaluation "
        "model accuracy loss epoch batch gradient descent optimizer "
        "the quick brown fox jumps over the lazy dog "
        "image text sequence character word sentence "
        "data science artificial intelligence machine "
        "university project research experiment result"
    ).split()

    def __init__(
        self,
        size: int = 5000,
        img_height: int = 32,
        img_width: int = 128,
        split: str = "train",
        seed: int = 42,
        deslant: bool = False,
    ) -> None:
        self.size = size
        self.img_height = img_height
        self.img_width = img_width
        self.split = split

        self.transform = (
            get_train_transforms(img_height, img_width, deslant=deslant)
            if split == "train"
            else get_val_transforms(img_height, img_width, deslant=deslant)
        )

        # Reproduzierbare Zufallstexte generieren
        rng = random.Random(seed)
        self.labels: List[str] = []
        for _ in range(size):
            n_words = rng.randint(1, 3)
            phrase = " ".join(rng.choice(self.WORD_LIST) for _ in range(n_words))
            self.labels.append(phrase[:32])   # max 32 Zeichen

        print(f"SyntheticHTR [{split}]: {size} Samples.")

    def _render_text_image(self, text: str) -> Image.Image:
        """Rendert Text als Bild mit zufälliger Schriftgröße und Versatz."""
        img = Image.new("L", (self.img_width * 3, self.img_height * 3), color=255)
        draw = ImageDraw.Draw(img)

        font_size = random.randint(24, 36)
        try:
            font = ImageFont.truetype(
                f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
            font = ImageFont.load_default()

        # Zufälliger Versatz für Variation
        x_off = random.randint(5, 20)
        y_off = random.randint(5, 20)
        draw.text((x_off, y_off), text, fill=0, font=font)

        return img

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        label = self.labels[idx]
        img = self._render_text_image(label)
        image_tensor = self.transform(img)
        label_tensor = encode_label(label)
        return image_tensor, label_tensor, len(label_tensor)


# ── DataLoader-Hilfsfunktion ──────────────────────────────────────────────────

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stapelt Bilder und Labels unterschiedlicher Länge zu einem Batch.
    Labels werden für CTCLoss als flacher Tensor übergeben.
    """
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths


def seq2seq_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate für Seq2Seq: fügt BOS/EOS hinzu und paddet auf gleiche Länge.
    Gibt (images, padded_labels) zurück – kein flacher Label-Tensor wie bei CTC.
    """
    images, labels, _ = zip(*batch)
    images = torch.stack(images)
    seq_labels = [
        torch.cat([torch.tensor([BOS_IDX]), lbl, torch.tensor([EOS_IDX])])
        for lbl in labels
    ]
    max_len = max(l.size(0) for l in seq_labels)
    padded = torch.full((len(seq_labels), max_len), PAD_IDX, dtype=torch.long)
    for i, lbl in enumerate(seq_labels):
        padded[i, :lbl.size(0)] = lbl
    return images, padded


def get_dataloaders(
    dataset_type: str = "synthetic",
    data_dir: str = "data/raw",
    batch_size: int = 32,
    img_height: int = 32,
    img_width: int = 128,
    num_workers: int = 4,
    synthetic_size: int = 5000,
    deslant: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Erstellt DataLoader für Training und Validierung.

    Args:
        dataset_type:    "synthetic" oder "iam"
        data_dir:        Pfad zum IAM-Datensatz (nur bei dataset_type="iam")
        batch_size:      Batch-Größe
        img_height:      Bildhöhe (Standard: 32)
        img_width:       Bildbreite (Standard: 128)
        num_workers:     Parallel-Worker für Datenladen
        synthetic_size:  Anzahl synthetischer Samples

    Returns:
        (train_loader, val_loader)
    """
    if dataset_type == "iam":
        train_ds = IAMDataset(data_dir, split="train", img_height=img_height, img_width=img_width, split_mode="random", deslant=deslant)
        val_ds   = IAMDataset(data_dir, split="val",   img_height=img_height, img_width=img_width, split_mode="random", deslant=deslant)
    else:
        train_size = int(synthetic_size * 0.85)
        val_size   = synthetic_size - train_size
        train_ds = SyntheticHTRDataset(train_size, img_height, img_width, split="train", deslant=deslant)
        val_ds   = SyntheticHTRDataset(val_size,   img_height, img_width, split="val", seed=99, deslant=deslant)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_seq2seq_dataloaders(
    dataset_type: str = "iam",
    data_dir: str = "data/raw",
    batch_size: int = 32,
    img_height: int = 32,
    img_width: int = 128,
    num_workers: int = 4,
    deslant: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """DataLoader für Seq2Seq-Training (BOS/EOS/PAD statt CTC-Format)."""
    if dataset_type == "iam":
        train_ds = IAMDataset(data_dir, split="train", img_height=img_height, img_width=img_width, split_mode="random", deslant=deslant)
        val_ds   = IAMDataset(data_dir, split="val",   img_height=img_height, img_width=img_width, split_mode="random", deslant=deslant)
    else:
        train_ds = SyntheticHTRDataset(4250, img_height, img_width, split="train", deslant=deslant)
        val_ds   = SyntheticHTRDataset(750,  img_height, img_width, split="val", seed=99, deslant=deslant)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=seq2seq_collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=seq2seq_collate_fn, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
