"""
CRNN – Convolutional Recurrent Neural Network für Handwritten Text Recognition
==============================================================================

Architektur-Überblick:

  Eingabebild  →  [CNN]  →  [BiLSTM]  →  [Linear]  →  CTC-Loss
   (1×32×128)     Feature-   Sequenz-     Zeichenklassen-
                  Maps       Kontext      Wahrscheinlichkeiten

Warum CNN?
  Handschrift enthält lokale visuelle Muster: Striche, Kurven, Serifenenden.
  Convolutional Layers extrahieren diese Merkmale positions-unabhängig und
  hierarchisch (Kanten → Teile → Zeichen). Pooling reduziert die räumliche
  Dimension, erhält aber die wichtigsten Features.

Warum BiLSTM?
  Nach dem CNN liegen Feature-Spalten (Zeitschritte) vor – eine Sequenz.
  LSTMs modellieren zeitliche Abhängigkeiten: der Kontext eines Buchstabens
  hängt von vorherigen UND nachfolgenden Zeichen ab (z.B. "th" in "the").
  Bidirektional = liest die Sequenz vorwärts und rückwärts → besserer Kontext.

Warum CTC-Loss?
  Handschriftbilder haben keine pixelgenaue Ausrichtung zwischen Bild und Label.
  CTC (Connectionist Temporal Classification) lernt diese Ausrichtung automatisch:
  Es summiert alle möglichen Pfade, die zum gleichen Label führen, und ist daher
  ideal für variable Textlängen ohne Segmentierungsannotationen.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """Ein einzelner CNN-Block: Conv2d → BatchNorm → ReLU → (optional) MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool: bool = True,
        pool_kernel: Tuple[int, int] = (2, 2),
        pool_stride: Tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CRNN(nn.Module):
    """
    CRNN-Modell für Handwritten Text Recognition.

    Input:  (Batch, 1, img_height, img_width)  – Graustufenbild
    Output: (Sequenzlänge, Batch, num_classes)  – für CTC-Loss
    """

    def __init__(self, img_height: int, num_classes: int, lstm_hidden: int = 256) -> None:
        """
        Args:
            img_height:   Höhe des normalisierten Eingabebildes (z.B. 32)
            num_classes:  Anzahl der Ausgabeklassen (Alphabet + 1 für CTC-Blank)
            lstm_hidden:  Versteckte Einheiten pro LSTM-Richtung
        """
        super().__init__()

        # ── CNN Feature-Extraktor ─────────────────────────────────────────────
        # Jeder Block halbiert (bei pool=True) Höhe und Breite.
        # Nach 4 Pooling-Operationen: Höhe 32 → 2, Breite 128 → 8
        self.cnn = nn.Sequential(
            ConvBlock(1,   64,  pool=True,  pool_kernel=(2, 2)),   # 32×128 → 16×64
            ConvBlock(64,  128, pool=True,  pool_kernel=(2, 2)),   # 16×64  →  8×32
            ConvBlock(128, 256, pool=False),                        # 8×32   →  8×32
            ConvBlock(256, 256, pool=True,  pool_kernel=(2, 1)),   # 8×32   →  4×32
            ConvBlock(256, 512, pool=False),                        # 4×32   →  4×32
            ConvBlock(512, 512, pool=True,  pool_kernel=(2, 1)),   # 4×32   →  2×32
            ConvBlock(512, 512, pool=False),                        # 2×32   →  2×32
        )

        # Berechne CNN-Ausgabehöhe → wird mit den Kanälen zusammengefaltet
        cnn_output_height = img_height // 16   # nach 4× Pooling in H-Richtung
        rnn_input_size = 512 * cnn_output_height

        # ── Rekurrenter Teil (BiLSTM) ─────────────────────────────────────────
        # 2 gestapelte BiLSTM-Schichten für tieferes kontextuelles Verständnis
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=False,   # Erwartet (Seq, Batch, Features) – passt zu CTC
            dropout=0.3,
        )

        # ── Ausgabe-Projektion ────────────────────────────────────────────────
        # lstm_hidden * 2: bidirektional → doppelte versteckte Dimension
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Eingabetensor (Batch, 1, H, W)

        Returns:
            log_probs: (SeqLen, Batch, num_classes) – Log-Softmax Ausgabe für CTC
        """
        # CNN: (B, 1, H, W) → (B, C, H', W')
        features = self.cnn(x)

        # Reshape für RNN: (B, C, H', W') → (W', B, C*H')
        batch_size, channels, height, width = features.size()
        features = features.permute(3, 0, 1, 2)          # (W', B, C, H')
        features = features.contiguous().view(width, batch_size, channels * height)

        # BiLSTM: (SeqLen, B, Features) → (SeqLen, B, lstm_hidden*2)
        rnn_out, _ = self.rnn(features)

        # Lineare Projektion: (SeqLen, B, lstm_hidden*2) → (SeqLen, B, num_classes)
        logits = self.fc(rnn_out)

        # Log-Softmax wird von nn.CTCLoss erwartet (reduction='mean')
        return torch.log_softmax(logits, dim=2)


def build_model(img_height: int, num_classes: int, lstm_hidden: int = 256) -> CRNN:
    """Factory-Funktion: erstellt und gibt ein CRNN-Modell zurück."""
    model = CRNN(img_height=img_height, num_classes=num_classes, lstm_hidden=lstm_hidden)
    return model


if __name__ == "__main__":
    # Schneller Sanity-Check
    img_h, img_w = 32, 128
    n_classes = 80   # Alphabet + Blank
    model = build_model(img_h, n_classes)
    dummy = torch.randn(4, 1, img_h, img_w)   # Batch von 4 Bildern
    out = model(dummy)
    print(f"Eingabe:  {dummy.shape}")
    print(f"Ausgabe:  {out.shape}")  # Erwartet: (seq_len, 4, 80)
    print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")
