"""
CTC-Decoder: wandelt Modellausgaben (Log-Wahrscheinlichkeiten) in lesbaren Text um.

Zwei Varianten:
  - Greedy Decoder:      schnell, gut für Training und schnelle Evaluation
  - Beam Search Decoder: langsamer, aber messbar besser bei verrauschten Ausgaben
"""

from typing import List
import numpy as np
import torch

from src.dataset import IDX2CHAR, BLANK_TOKEN

BLANK_IDX = 0


def greedy_decode(log_probs: torch.Tensor) -> List[str]:
    """
    Greedy CTC-Decoding: nimmt bei jedem Zeitschritt das wahrscheinlichste Zeichen,
    entfernt dann Wiederholungen und Blank-Token.

    Args:
        log_probs: (SeqLen, Batch, NumClasses) – Modellausgabe (Log-Softmax)

    Returns:
        Liste von dekodierten Strings, eine pro Batch-Element
    """
    predictions = torch.argmax(log_probs, dim=2).permute(1, 0)   # (Batch, SeqLen)

    results: List[str] = []
    for seq in predictions:
        chars: List[str] = []
        prev = None
        for idx in seq.tolist():
            char = IDX2CHAR.get(idx, "")
            if char != BLANK_TOKEN and char != prev:
                chars.append(char)
            prev = char
        results.append("".join(chars))

    return results


def beam_search_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
) -> List[str]:
    """
    Beam Search CTC-Decoding (ohne Sprachmodell).

    Für jedes Batch-Element werden `beam_width` Hypothesen gleichzeitig verfolgt.
    Nutzt numpy für schnellere Verarbeitung.

    Args:
        log_probs:   (SeqLen, Batch, NumClasses)
        beam_width:  Anzahl der Hypothesen im Beam (höher = besser aber langsamer)

    Returns:
        Liste von dekodiertem Text, eine Zeichenkette pro Batch-Element
    """
    seq_len, batch_size, num_classes = log_probs.shape
    lp = log_probs.detach().cpu().numpy()  # (SeqLen, Batch, NumClasses)

    results: List[str] = []

    for b in range(batch_size):
        # Beam: dict von (prefix_tuple, last_char_idx) → log_score
        # last_char_idx = -1 bedeutet: letztes ausgegebenes Zeichen war Blank
        beams: dict = {((), -1): 0.0}

        for t in range(seq_len):
            lp_t = lp[t, b]  # (NumClasses,)

            # Nur Top-k Klassen berücksichtigen für Effizienz
            top_k = int(min(beam_width * 3, num_classes))
            top_indices = np.argpartition(lp_t, -top_k)[-top_k:]

            new_beams: dict = {}
            for (prefix, last_c), score in beams.items():
                for c in top_indices:
                    lp_c = lp_t[c]
                    new_score = score + lp_c

                    if c == BLANK_IDX:
                        key = (prefix, -1)
                    elif c == last_c:
                        # Gleiche Wiederholung ohne zwischenzeitliches Blank → ignorieren
                        key = (prefix, c)
                    else:
                        key = (prefix + (int(c),), int(c))

                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

            # Top beam_width Hypothesen behalten
            sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
            beams = dict(sorted_beams[:beam_width])

        if beams:
            best_prefix = max(beams.items(), key=lambda x: x[1])[0][0]
            results.append("".join(IDX2CHAR.get(c, "") for c in best_prefix))
        else:
            results.append("")

    return results
