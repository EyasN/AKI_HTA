"""
CTC-Decoder: wandelt Modellausgaben (Log-Wahrscheinlichkeiten) in lesbaren Text um.

Zwei Varianten:
  - Greedy Decoder:      schnell, ausreichend für Training und Evaluation
  - Beam Search Decoder: langsamer, aber besser bei verrauschten Ausgaben
                         (via torch.nn.CTCLoss-kompatiblem Format)
"""

from typing import List
import torch

from src.dataset import IDX2CHAR, BLANK_TOKEN


def greedy_decode(log_probs: torch.Tensor) -> List[str]:
    """
    Greedy CTC-Decoding: nimmt bei jedem Zeitschritt das wahrscheinlichste Zeichen,
    entfernt dann Wiederholungen und Blank-Token.

    Args:
        log_probs: (SeqLen, Batch, NumClasses) – Modellausgabe (Log-Softmax)

    Returns:
        Liste von dekodierten Strings, eine pro Batch-Element
    """
    # Argmax über Klassen-Dimension → (SeqLen, Batch)
    predictions = torch.argmax(log_probs, dim=2).permute(1, 0)   # (Batch, SeqLen)

    results: List[str] = []
    for seq in predictions:
        chars: List[str] = []
        prev = None
        for idx in seq.tolist():
            char = IDX2CHAR.get(idx, "")
            # CTC-Regel: Blank ignorieren, Duplikate zusammenführen
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
    Beam Search CTC-Decoding (vereinfachte Version ohne Sprachmodell).

    Für jedes Batch-Element werden `beam_width` Hypothesen gleichzeitig verfolgt.
    Am Ende wird die Hypothese mit dem höchsten Score zurückgegeben.

    Hinweis: Für Produktionseinsatz empfiehlt sich die ctcdecode-Bibliothek
    (https://github.com/parlance/ctcdecode) mit Sprachmodell-Integration.

    Args:
        log_probs:   (SeqLen, Batch, NumClasses)
        beam_width:  Anzahl der Hypothesen im Beam

    Returns:
        Liste von dekodiertem Text, eine Zeichenkette pro Batch-Element
    """
    seq_len, batch_size, num_classes = log_probs.shape
    probs = log_probs.exp().cpu()   # → normale Wahrscheinlichkeiten

    results: List[str] = []
    for b in range(batch_size):
        # Beam: Liste von (score, sequence_as_list, last_char)
        beams = [(0.0, [], None)]

        for t in range(seq_len):
            new_beams: list = []
            for score, seq, last_char in beams:
                for c in range(num_classes):
                    char = IDX2CHAR.get(c, "")
                    p = probs[t, b, c].item()
                    if p < 1e-9:
                        continue
                    new_score = score + float(torch.log(torch.tensor(p)))

                    if char == BLANK_TOKEN:
                        # Blank: Sequenz unverändert, last_char zurücksetzen
                        new_beams.append((new_score, seq, None))
                    elif char == last_char:
                        # Gleiches Zeichen wie zuvor: nur möglich nach Blank
                        new_beams.append((new_score, seq, char))
                    else:
                        new_beams.append((new_score, seq + [char], char))

            # Top-k Beams behalten
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

        best_seq = beams[0][1] if beams else []
        results.append("".join(best_seq))

    return results
