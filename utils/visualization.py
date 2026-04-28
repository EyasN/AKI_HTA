"""
Visualisierung: Trainingskurven, Beispielvorhersagen und Feature-Maps.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # kein Display nötig (funktioniert auf Servern)
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = "outputs/logs/training_curves.png",
) -> None:
    """
    Zeichnet Loss und CER-Kurven für Training und Validierung.

    Args:
        history:   Dict mit Schlüsseln "train_loss", "val_loss", "train_cer", "val_cer"
        save_path: Speicherpfad für das PNG
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HTR Training – Verlauf", fontsize=14, fontweight="bold")

    # ── Loss ──
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss", color="#2196F3")
    if "val_loss" in history:
        ax.plot(history["val_loss"],   label="Val Loss",   color="#F44336", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── CER ──
    ax = axes[1]
    if "train_cer" in history:
        ax.plot(history["train_cer"], label="Train CER", color="#4CAF50")
    if "val_cer" in history:
        ax.plot(history["val_cer"],   label="Val CER",   color="#FF9800", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CER (niedriger = besser)")
    ax.set_title("Character Error Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trainingskurven gespeichert: {save_path}")


def show_predictions(
    images: torch.Tensor,
    predictions: List[str],
    ground_truths: List[str],
    n: int = 8,
    save_path: str = "outputs/logs/predictions.png",
) -> None:
    """
    Zeigt n Beispielbilder mit Vorhersage und Ground Truth nebeneinander.

    Args:
        images:       (Batch, 1, H, W) normalisierter Tensor
        predictions:  Dekodierte Vorhersagen
        ground_truths: Echte Labels
        n:            Anzahl der anzuzeigenden Beispiele
        save_path:    Speicherpfad
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    n = min(n, len(images))

    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()
        img = (img * 0.5 + 0.5)   # Denormalisierung [-1,1] → [0,1]

        axes[i].imshow(img, cmap="gray", aspect="auto")
        axes[i].axis("off")

        pred = predictions[i]
        gt   = ground_truths[i]
        color = "#4CAF50" if pred == gt else "#F44336"
        axes[i].set_title(
            f"GT:   {gt}\nPred: {pred}",
            fontsize=7,
            color=color,
            loc="left",
        )

    plt.suptitle("Beispielvorhersagen (grün=korrekt, rot=falsch)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Vorhersagen gespeichert: {save_path}")
