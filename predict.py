"""
predict.py – Inferenz-Script für HTR-Modelle.

Verwendung:
  # CRNN (CTC)
  python predict.py --image bild.png --checkpoint outputs/checkpoints/best_model.pt

  # Seq2Seq (Transformer Decoder)
  python predict.py --image bild.png --checkpoint outputs/checkpoints/best_seq2seq.pt --arch seq2seq

  # Ordner mit mehreren Bildern
  python predict.py --folder data/sample_images/ --checkpoint outputs/checkpoints/best_model.pt
"""

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

from src.dataset import NUM_CLASSES, SEQ2SEQ_VOCAB
from src.model import build_model, build_seq2seq_model
from src.transforms import get_val_transforms
from utils.ctc_decoder import greedy_decode, beam_search_decode, lm_decode
from utils.seq2seq_decoder import seq2seq_greedy_decode


class HTRPredictor:
    """
    Kapselt das geladene Modell und bietet eine einfache predict()-Schnittstelle.
    Unterstützt CRNN (CTC) und Seq2Seq (Transformer Decoder).
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch:       str = "crnn",
        img_height: int = 32,
        img_width:  int = 128,
        decoder:    str = "greedy",
        device:     str = "auto",
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.arch      = arch
        self.decoder   = decoder
        self.transform = get_val_transforms(img_height, img_width)

        checkpoint  = torch.load(checkpoint_path, map_location=self.device)
        sd          = checkpoint["model_state_dict"]
        lstm_hidden = (sd["rnn.weight_hh_l0"].shape[0] // 4
                       if "rnn.weight_hh_l0" in sd
                       else checkpoint.get("lstm_hidden", 256))

        if arch == "seq2seq":
            self.model = build_seq2seq_model(img_height=img_height, vocab_size=SEQ2SEQ_VOCAB,
                                             lstm_hidden=lstm_hidden)
        else:
            self.model = build_model(img_height=img_height, num_classes=NUM_CLASSES,
                                     lstm_hidden=lstm_hidden)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        epoch   = checkpoint.get("epoch", "?")
        val_cer = checkpoint.get("val_cer", "?")
        print(f"Modell geladen ({arch}): Epoch {epoch}, Val-CER: {val_cer}")
        print(f"Gerät: {self.device} | Decoder: {decoder}")

    def predict(self, image_path: str) -> str:
        img    = Image.open(image_path).convert("L")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        if self.arch == "seq2seq":
            return seq2seq_greedy_decode(self.model, tensor)[0]

        with torch.no_grad():
            log_probs = self.model(tensor)

        if self.decoder == "beam":
            return beam_search_decode(log_probs, beam_width=10)[0]
        if self.decoder == "lm":
            return lm_decode(log_probs, beam_width=25)[0]
        return greedy_decode(log_probs)[0]

    def predict_batch(self, image_paths: List[str]) -> List[str]:
        tensors = [self.transform(Image.open(p).convert("L")) for p in image_paths]
        batch   = torch.stack(tensors, dim=0).to(self.device)

        if self.arch == "seq2seq":
            return seq2seq_greedy_decode(self.model, batch)

        with torch.no_grad():
            log_probs = self.model(batch)

        if self.decoder == "beam":
            return beam_search_decode(log_probs, beam_width=10)
        if self.decoder == "lm":
            return lm_decode(log_probs, beam_width=25)
        return greedy_decode(log_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="HTR Inference – Handschrift erkennen")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Pfad zu einem einzelnen Bild")
    group.add_argument("--folder", help="Pfad zu einem Ordner mit Bildern")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch",       default="crnn",  choices=["crnn", "seq2seq"])
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width",  type=int, default=128)
    parser.add_argument("--decoder",    default="lm", choices=["greedy", "beam", "lm"],
                        help="Decoder für CRNN; ignoriert bei --arch seq2seq")
    args = parser.parse_args()

    predictor = HTRPredictor(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        img_height=args.img_height,
        img_width=args.img_width,
        decoder=args.decoder,
    )

    if args.image:
        text = predictor.predict(args.image)
        print(f"\nBild:     {args.image}")
        print(f"Ergebnis: {text}")
    else:
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
