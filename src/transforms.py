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


def get_train_transforms(img_height: int = 32, img_width: int = 128) -> T.Compose:
    """
    Augmentation-Pipeline für das Training.
    Reihenfolge: geometrische Ops → Textur-Ops → Tensor-Konvertierung → Normalisierung
    """
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        ResizeToHeight(img_height, img_width),
        T.RandomRotation(degrees=3, fill=255),          # leichte Neigung
        T.RandomAffine(degrees=0, shear=5, fill=255),   # Scherung
        T.ColorJitter(brightness=0.3, contrast=0.3),    # Helligkeit/Kontrast
        RandomDilateErode(prob=0.3),                    # Stiftstärke variieren
        T.ToTensor(),                                    # [0,255] → [0,1]
        T.Normalize(mean=[0.5], std=[0.5]),              # [0,1] → [-1,1]
        AddGaussianNoise(std=0.02),                      # leichtes Rauschen
    ])


def get_val_transforms(img_height: int = 32, img_width: int = 128) -> T.Compose:
    """
    Validierungs-/Inferenz-Pipeline: nur deterministisches Preprocessing.
    Keine Augmentation, damit Evaluation reproduzierbar ist.
    """
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        ResizeToHeight(img_height, img_width),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
