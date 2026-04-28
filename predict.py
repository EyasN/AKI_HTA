"""
predict.py – Inferenz-Script für das trainierte CRNN-Modell.

Verwendung:
  # Einzelbild
  python predict.py --image data/sample_images/example.png --checkpoint outputs/checkpoints/best_model.pt

  # Ordner mit mehreren Bildern
  python predict.py --folder data/sample_images/ --checkpoint outputs/checkpoints/best_model.pt

  # Mit Beam-Search-Decoding (langsamer, oft besser)
  python predict.py --image bild.png --checkpoint best_model.pt --decoder beam
"""

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

from src.dataset import NUM_CLASSES
from src.model import build_model
from src.transforms import get_val_transforms
from utils.ctc_decoder import greedy_decode, beam_search_decode


class HTRPredictor:
    """
    Kapselt das geladene Modell und bietet eine einfache predict()-Schnittstelle.
    Einmalige Initialisierung – dann wiederholtes Vorhersagen ohne erneutes Laden.
    """

    def __init__(
        self,
        checkpoint_path: str,
        img_height: int = 32,
        img_width:  int = 128,
        decoder:    str = "greedy",
        device:     str = "auto",
    ) -> None:
        """
        Args:
            checkpoint_path: Pfad zum gespeicherten Modell-Checkpoint (.pt)
            img_height:      Bildhöhe (muss mit Training übereinstimmen)
            img_width:       Bildbreite
            decoder:         "greedy" oder "beam"
            device:          "auto", "cpu" oder "cuda"
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.decoder   = decoder
        self.transform = get_val_transforms(img_height, img_width)

        # Modell laden
        self.model = build_model(img_height=img_height, num_classes=NUM_CLASSES)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        epoch  = checkpoint.get("epoch", "?")
        val_cer = checkpoint.get("val_cer", "?")
        print(f"Modell geladen: Epoch {epoch}, Val-CER: {val_cer}")
        print(f"Gerät: {self.device} | Decoder: {decoder}")

    def predict(self, image_path: str) -> str:
        """
        Erkennt Handschrift in einem einzelnen Bild.

        Args:
            image_path: Pfad zum Eingabebild (PNG, JPG, etc.)

        Returns:
            Erkannter Text als String
        """
        img = Image.open(image_path).convert("L")
        tensor = self.transform(img).unsqueeze(0).to(self.device)   # (1, 1, H, W)

        with torch.no_grad():
            log_probs = self.model(tensor)   # (SeqLen, 1, NumClasses)

        if self.decoder == "beam":
            results = beam_search_decode(log_probs, beam_width=10)
        else:
            results = greedy_decode(log_probs)

        return results[0]

    def predict_batch(self, image_paths: List[str]) -> List[str]:
        """Erkennt Handschrift in mehreren Bildern gleichzeitig (effizienter als einzeln)."""
        tensors = []
        for path in image_paths:
            img = Image.open(path).convert("L")
            tensors.append(self.transform(img))

        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            log_probs = self.model(batch)

        if self.decoder == "beam":
            return beam_search_decode(log_probs, beam_width=10)
        return greedy_decode(log_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="HTR Inference – Handschrift erkennen")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Pfad zu einem einzelnen Bild")
    group.add_argument("--folder", help="Pfad zu einem Ordner mit Bildern")

    parser.add_argument("--checkpoint", required=True, help="Checkpoint-Datei (.pt)")
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width",  type=int, default=128)
    parser.add_argument("--decoder",    default="greedy", choices=["greedy", "beam"])
    args = parser.parse_args()

    predictor = HTRPredictor(
        checkpoint_path=args.checkpoint,
        img_height=args.img_height,
        img_width=args.img_width,
        decoder=args.decoder,
    )

    if args.image:
        text = predictor.predict(args.image)
        print(f"\nBild:     {args.image}")
        print(f"Ergebnis: {text}")

    elif args.folder:
        folder = Path(args.folder)
        images = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if not images:
            print(f"Keine Bilder in {folder} gefunden.")
            return

        print(f"\nVerarbeite {len(images)} Bilder aus: {folder}\n")
        results = predictor.predict_batch([str(p) for p in images])

        print(f"{'Datei':<40} | Erkannter Text")
        print("-" * 70)
        for img_path, text in zip(images, results):
            print(f"{img_path.name:<40} | {text}")


if __name__ == "__main__":
    main()
