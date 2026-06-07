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
                preds = lm_decode(log_probs)
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

    char_acc = (1 - cer) * 100
    word_acc = (1 - wer) * 100

    print(f"\n{'='*45}")
    print(f"  Evaluation Ergebnis")
    print(f"{'='*45}")
    print(f"  Character Accuracy:         {char_acc:.2f}%")
    print(f"  Word Accuracy:              {word_acc:.2f}%")
    print(f"  Character Error Rate (CER): {cer*100:.2f}%")
    print(f"  Word Error Rate     (WER):  {wer*100:.2f}%")
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


def evaluate_seq2seq_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    decoder: str = "beam",
) -> Tuple[float, float]:
    """
    Evaluation für das Seq2Seq-Modell (autoregressiver Transformer-Decoder).

    Args:
        model:      Trainiertes CRNN_Seq2Seq-Modell
        dataloader: DataLoader im Seq2Seq-Format (images, padded_labels)
        device:     torch.device
        decoder:    "beam" (Standard) oder "greedy"

    Returns:
        (cer, wer)
    """
    from utils.seq2seq_decoder import (
        seq2seq_beam_decode, seq2seq_greedy_decode, decode_seq2seq_labels,
    )

    model.eval()
    all_preds:  List[str] = []
    all_labels: List[str] = []

    with torch.no_grad():
        for images, padded in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            if decoder == "greedy":
                preds = seq2seq_greedy_decode(model, images)
            else:
                preds = seq2seq_beam_decode(model, images, beam_width=5)
            all_preds.extend(preds)
            all_labels.extend(decode_seq2seq_labels(padded))

    cer = compute_cer(all_preds, all_labels)
    wer = compute_wer(all_preds, all_labels)

    print(f"\n{'='*45}")
    print(f"  Evaluation Ergebnis (Seq2Seq, {decoder})")
    print(f"{'='*45}")
    print(f"  Character Accuracy:         {(1-cer)*100:.2f}%")
    print(f"  Word Accuracy:              {(1-wer)*100:.2f}%")
    print(f"  Character Error Rate (CER): {cer*100:.2f}%")
    print(f"  Word Error Rate     (WER):  {wer*100:.2f}%")
    print(f"  Samples evaluiert:          {len(all_preds)}")
    print(f"{'='*45}\n")

    print("Beispiele:")
    correct   = [(p, l) for p, l in zip(all_preds, all_labels) if p == l][:3]
    incorrect = [(p, l) for p, l in zip(all_preds, all_labels) if p != l][:5]
    for p, l in correct:
        print(f"  ✓  GT: '{l}'  →  Pred: '{p}'")
    for p, l in incorrect:
        print(f"  ✗  GT: '{l}'  →  Pred: '{p}'")

    return cer, wer


# ── Standalone-Aufruf ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from src.dataset import (
        IAMDataset, collate_fn, seq2seq_collate_fn, SEQ2SEQ_VOCAB,
        get_dataloaders, get_seq2seq_dataloaders,
    )
    from src.model import build_seq2seq_model

    parser = argparse.ArgumentParser(description="HTR Evaluation")
    parser.add_argument("--checkpoint", required=True,  help="Pfad zum Modell-Checkpoint (.pt)")
    parser.add_argument("--arch",       default="crnn", choices=["crnn", "seq2seq"])
    parser.add_argument("--dataset",    default="iam", choices=["synthetic", "iam"])
    parser.add_argument("--data-dir",   default="data/raw")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width",  type=int, default=256)
    parser.add_argument("--split",      default="test", choices=["val", "test"],
                        help="Welcher Datensplit evaluiert wird (Standard: test = ungesehene Samples)")
    parser.add_argument("--split-mode", default="random", choices=["random", "author"],
                        help="MUSS zum Training passen! random (Standard) = Held-out-Samples derselben "
                             "Schreiber. author nur sinnvoll, wenn auch mit --split-mode author trainiert wurde.")
    parser.add_argument("--decoder",    default="lm", choices=["greedy", "beam", "lm"],
                        help="CRNN: greedy/beam/lm. Seq2Seq: greedy oder beam (sonst beam).")
    parser.add_argument("--deslant",    action="store_true",
                        help="Deslanting (Fallback, falls nicht im Checkpoint vermerkt).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint laden + lstm_hidden aus den Gewichten ableiten (wie in predict.py)
    checkpoint  = torch.load(args.checkpoint, map_location=device)
    sd          = checkpoint["model_state_dict"]
    lstm_hidden = (sd["rnn.weight_hh_l0"].shape[0] // 4
                   if "rnn.weight_hh_l0" in sd
                   else checkpoint.get("lstm_hidden", 256))

    # Config aus dem Checkpoint übernehmen, falls vorhanden (sonst CLI-Werte)
    img_height = checkpoint.get("img_height", args.img_height)
    img_width  = checkpoint.get("img_width",  args.img_width)
    deslant    = checkpoint.get("deslant",    args.deslant)
    print(f"Config: {img_height}x{img_width} | Deslant: {deslant} | Split: {args.split}/{args.split_mode}")

    # Passenden Dataloader für den gewählten Split bauen
    if args.dataset == "iam":
        is_seq2seq = args.arch == "seq2seq"
        ds = IAMDataset(args.data_dir, split=args.split, img_height=img_height,
                        img_width=img_width, split_mode=args.split_mode, deslant=deslant)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=(seq2seq_collate_fn if is_seq2seq else collate_fn),
        )
    else:
        # Synthetische Daten kennen keinen test/author-Split → Val-Loader
        if args.arch == "seq2seq":
            _, loader = get_seq2seq_dataloaders(dataset_type="synthetic",
                                                img_height=img_height, img_width=img_width, deslant=deslant)
        else:
            _, loader = get_dataloaders(dataset_type="synthetic",
                                        img_height=img_height, img_width=img_width, deslant=deslant)

    if args.arch == "seq2seq":
        model = build_seq2seq_model(img_height=img_height, vocab_size=SEQ2SEQ_VOCAB,
                                    lstm_hidden=lstm_hidden)
        model.load_state_dict(sd)
        model.to(device)
        dec = "greedy" if args.decoder == "greedy" else "beam"
        evaluate_seq2seq_model(model, loader, device, decoder=dec)
    else:
        model = build_model(img_height=img_height, num_classes=NUM_CLASSES,
                            lstm_hidden=lstm_hidden)
        model.load_state_dict(sd)
        model.to(device)
        evaluate_model(model, loader, device, decoder=args.decoder)
