"""
Bildvorverarbeitung und Datenaugmentation für HTR.

Pipeline:
  1. Grayscale-Konvertierung
  2. Resize auf Zielhöhe (Breite bleibt proportional, wird dann gepaddet/gecroppt)
  3. Normalisierung [0, 255] → [-1, 1]
  4. Augmentation (nur Training): Rotation, Rauschen, Blur, Helligkeit
"""

import random
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torchvision import transforms as T


class ResizeToHeight:
    """
    Skaliert das Bild so, dass die Höhe auf `target_height` kommt.
    Die Breite wird proportional mitgeändert und dann auf `target_width` gepaddet/gecroppt.
    So bleiben Seitenverhältnisse erhalten – wichtig für Handschrift.
    """

    def __init__(self, target_height: int, target_width: int) -> None:
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        # Neue Breite proportional berechnen
        new_w = int(w * self.target_height / h)
        img = img.resize((new_w, self.target_height), Image.LANCZOS)

        # Breite anpassen: kürzer → rechts mit Weiß padden, breiter → rechts croppen
        current_w = img.size[0]
        if current_w < self.target_width:
            padded = Image.new("L", (self.target_width, self.target_height), color=255)
            padded.paste(img, (0, 0))
            img = padded
        else:
            img = img.crop((0, 0, self.target_width, self.target_height))

        return img


def ensure_training_polarity(img: Image.Image) -> Image.Image:
    """
    Bringt ein Eingabebild auf die Trainings-Polarität: **helle Schrift auf dunklem Grund**
    (so wie das IAM-Training, das die Bilder via ImageOps.invert umdreht).

    Hintergrund wird am Median-Pixel erkannt: ist der Hintergrund hell (normales Foto/Scan
    mit dunkler Tinte auf hellem Papier), wird invertiert. Ist er bereits dunkel, bleibt es.

    WICHTIG: Ohne diesen Schritt sieht das Modell bei Uploads die vertauschte Polarität und
    liefert Kauderwelsch – obwohl es auf IAM exzellent abschneidet.
    """
    gray = img.convert("L")
    arr = np.asarray(gray)
    if np.median(arr) > 127:          # heller Hintergrund → invertieren
        return ImageOps.invert(gray)
    return gray


def deslant_image(img: Image.Image, max_shear: float = 0.4, steps: int = 17) -> Image.Image:
    """
    Deslanting: korrigiert die Schräglage (Kursivität) eines Handschriftbildes.

    Klassische HTR-Vorverarbeitung (Vinciarelli & Luettin): Das Bild wird über eine
    Reihe von Scherwinkeln geschert; gewählt wird der Winkel, der die Tinte am stärksten
    in wenige Spalten konzentriert (Summe der quadrierten Spaltensummen maximal). Dann
    stehen die senkrechten Striche tatsächlich senkrecht – das reduziert Form-Verwechslungen
    (o/c, B/s, t/r) und verkleinert die Variation, über die das Modell generalisieren muss.

    Polaritäts-robust: funktioniert für dunkle Schrift auf hellem Grund und umgekehrt.
    """
    arr = np.asarray(img.convert("L"))
    h, w = arr.shape
    if h < 3 or w < 3:
        return img

    # Binarisieren (Otsu); Tinte = Minderheitsklasse als 255
    _, binimg = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binimg.mean() > 127:
        binimg = 255 - binimg

    best_score, best_shear = -1.0, 0.0
    for shear in np.linspace(-max_shear, max_shear, steps):
        tx = -shear * h / 2.0
        M = np.float32([[1, shear, tx], [0, 1, 0]])
        sheared = cv2.warpAffine(binimg, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        colsum = sheared.sum(axis=0, dtype=np.float64) / 255.0
        score = float((colsum * colsum).sum())   # konzentrierte Spalten → höherer Score
        if score > best_score:
            best_score, best_shear = score, shear

    if abs(best_shear) < 1e-3:
        return img

    bg = int(np.median(arr))   # Hintergrundfarbe schätzen (für die Ränder)
    tx = -best_shear * h / 2.0
    M = np.float32([[1, best_shear, tx], [0, 1, 0]])
    out = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=bg)
    return Image.fromarray(out)


class Deslant:
    """Deslanting als Transform-Schritt (deterministisch, für Training UND Inferenz identisch)."""

    def __init__(self, max_shear: float = 0.4, steps: int = 17) -> None:
        self.max_shear = max_shear
        self.steps = steps

    def __call__(self, img: Image.Image) -> Image.Image:
        return deslant_image(img, self.max_shear, self.steps)


class AddGaussianNoise:
    """Fügt Gaußsches Rauschen zum Bild hinzu (Augmentation)."""

    def __init__(self, mean: float = 0.0, std: float = 0.05) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class RandomDilateErode:
    """
    Simuliert Stiftvarianz: Dilatation = dickerer Strich, Erosion = dünner.
    Nur auf Numpy-Arrays angewendet, vor der Tensor-Konvertierung.
    """

    def __init__(self, prob: float = 0.3) -> None:
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        arr = np.array(img)
        kernel = np.ones((2, 2), np.uint8)
        if random.random() > 0.5:
            arr = cv2.dilate(arr, kernel, iterations=1)
        else:
            arr = cv2.erode(arr, kernel, iterations=1)
        return Image.fromarray(arr)


def get_train_transforms(img_height: int = 32, img_width: int = 128, deslant: bool = False) -> T.Compose:
    """
    Augmentation-Pipeline für das Training.
    Reihenfolge: (Deslant) → geometrische Ops → Textur-Ops → Tensor-Konvertierung → Normalisierung
    """
    steps: list = [T.Grayscale(num_output_channels=1)]
    if deslant:
        steps.append(Deslant())               # Schräglage VOR Resize korrigieren
    steps += [
        ResizeToHeight(img_height, img_width),
        T.RandomRotation(degrees=5, fill=255),                        # Neigung bis 5°
        T.RandomAffine(degrees=0, shear=8, fill=255),                 # stärkere Scherung
        T.RandomPerspective(distortion_scale=0.15, p=0.3, fill=255),  # Perspektivverzerrung
        T.ColorJitter(brightness=0.4, contrast=0.4),                  # Helligkeit/Kontrast
        RandomDilateErode(prob=0.4),                                   # Stiftstärke variieren
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),  # Unschärfe
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
        AddGaussianNoise(std=0.03),                                    # etwas mehr Rauschen
    ]
    return T.Compose(steps)


def get_val_transforms(img_height: int = 32, img_width: int = 128, deslant: bool = False) -> T.Compose:
    """
    Validierungs-/Inferenz-Pipeline: nur deterministisches Preprocessing.
    Keine Augmentation, damit Evaluation reproduzierbar ist.
    Deslant muss identisch zum Training gesetzt sein!
    """
    steps: list = [T.Grayscale(num_output_channels=1)]
    if deslant:
        steps.append(Deslant())
    steps += [
        ResizeToHeight(img_height, img_width),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
    return T.Compose(steps)
