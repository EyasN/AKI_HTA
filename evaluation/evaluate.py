"""
Evaluation: Character Error Rate (CER) und Word Error Rate (WER).

CER = Levenshtein-Distanz auf Zeichenebene / Länge der Referenz
WER = Levenshtein-Distanz auf Wortebene   / Anzahl Wörter der Referenz

Beide Metriken liegen idealerweise bei 0 (perfekte Erkennung).
Ein CER von 0.05 bedeutet: 5% der Zeichen sind falsch.
"""

from typing import List, Tuple

import torch
from jiwer import cer as jiwer_cer, wer as jiwer_wer
from tqdm import tqdm

from src.dataset import ALPHABET, NUM_CLASSES, decode_label
from src.model import build_model
from utils.ctc_decoder import greedy_decode, beam_search_decode, lm_decode
from utils.visualization import show_predictions


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Character Error Rate (CER) über eine Liste von Vorhersage/Referenz-Paaren.

    Formel: CER = (Substitutionen + Einfügungen + Löschungen) / Anzahl Referenzzeichen

    Args:
        predictions: Liste der Modell-Vorhersagen
        references:  Liste der echten Labels

    Returns:
        Durchschnittliche CER (0.0 = perfekt, >1.0 = sehr schlecht)
    """
    if not predictions or not references:
        return 0.0

    # jiwer erwartet mindestens ein Zeichen pro String
    safe_preds = [p if p else " " for p in predictions]
    safe_refs  = [r if r else " " for r in references]

    return float(jiwer_cer(safe_refs, safe_preds))


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Word Error Rate (WER) über eine Liste von Vorhersage/Referenz-Paaren.

    Args:
        predictions: Liste der Modell-Vorhersagen
        references:  Liste der echten Labels

    Returns:
        Durchschnittliche WER (0.0 = perfekt)
    """
    if not predictions or not references:
        return 0.0

    safe_preds = [p if p else " " for p in predictions]
    safe_refs  = [r if r else " " for r in references]

    return float(jiwer_wer(safe_refs, safe_preds))


def decode_batch_labels(
    flat_labels: torch.Tensor,
    label_lengths: torch.Tensor,
) -> List[str]:
    """
    Wandelt den flachen Label-Tensor (CTCLoss-Format) zurück in Strings.

    Args:
        flat_labels:    1D-Tensor, alle Labels hintereinander
        label_lengths:  Länge jedes Labels im Batch

    Returns:
        Liste von Strings
    """
    texts = []
    offset = 0
    for length in label_lengths.tolist():
        indices = flat_labels[offset: offset + length].tolist()
        texts.append(decode_label(indices))
        offset += length
    return texts


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_examples: bool = True,
    examples_path: str = "outputs/logs/eval_predictions.png",
    decoder: str = "greedy",
) -> Tuple[float, float]:
    """
    Vollständige Evaluation: CER und WER über einen kompletten Datensatz.

    Args:
        model:         Trainiertes CRNN-Modell
        dataloader:    DataLoader für den Evaluationsdatensatz
        device:        torch.device ("cuda" oder "cpu")
        save_examples: Ob Beispielvorhersagen gespeichert werden sollen
        examples_path: Speicherpfad für Beispielplot

    Returns:
        (cer, wer) – Character Error Rate und Word Error Rate
    """
    model.eval()

    all_preds:  List[str] = []
    all_labels: List[str] = []
    example_images = None
    example_preds:  List[str] = []
    example_labels: List[str] = []

    with torch.no_grad():
        for batch_idx, (images, labels, label_lengths) in enumerate(tqdm(dataloader, desc="Evaluation")):
            images = images.to(device)
            log_probs = model(images)

            if decoder == "beam":
                preds = beam_search_decode(log_probs)
            elif decoder == "lm":
                preds = lm_decode(log_probs, beam_width=25)
            else:
                preds = greedy_decode(log_probs)
            truths = decode_batch_labels(labels, label_lengths)

            all_preds.extend(preds)
            all_labels.extend(truths)

            # Erste Batch für visuelle Inspektion merken
            if save_examples and example_images is None:
                example_images = images.cpu()
                example_preds  = preds
                example_labels = truths

    cer = compute_cer(all_preds, all_labels)
    wer = compute_wer(all_preds, all_labels)

    print(f"\n{'='*45}")
    print(f"  Evaluation Ergebnis")
    print(f"{'='*45}")
    print(f"  Character Error Rate (CER): {cer:.4f}  ({cer*100:.2f}%)")
    print(f"  Word Error Rate     (WER): {wer:.4f}  ({wer*100:.2f}%)")
    print(f"  Samples evaluiert:          {len(all_preds)}")
    print(f"{'='*45}\n")

    # Beispiele: erste 5 korrekte und 5 falsche Vorhersagen anzeigen
    print("Beispiele:")
    correct   = [(p, l) for p, l in zip(all_preds, all_labels) if p == l][:3]
    incorrect = [(p, l) for p, l in zip(all_preds, all_labels) if p != l][:5]
    for p, l in correct:
        print(f"  ✓  GT: '{l}'  →  Pred: '{p}'")
    for p, l in incorrect:
        print(f"  ✗  GT: '{l}'  →  Pred: '{p}'")

    if save_examples and example_images is not None:
        show_predictions(example_images, example_preds, example_labels, save_path=examples_path)

    return cer, wer


# ── Standalone-Aufruf ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.dataset import get_dataloaders

    parser = argparse.ArgumentParser(description="HTR Evaluation")
    parser.add_argument("--checkpoint", required=True,  help="Pfad zum Modell-Checkpoint (.pt)")
    parser.add_argument("--dataset",    default="synthetic", choices=["synthetic", "iam"])
    parser.add_argument("--data-dir",   default="data/raw")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width",  type=int, default=128)
    parser.add_argument("--decoder",    default="lm", choices=["greedy", "beam", "lm"],
                        help="Decoder-Typ: greedy (schnell), beam (besser), lm (best, pyctcdecode)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_dataloaders(
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_height=args.img_height,
        img_width=args.img_width,
    )

    model = build_model(img_height=args.img_height, num_classes=NUM_CLASSES)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    evaluate_model(model, val_loader, device, decoder=args.decoder)
