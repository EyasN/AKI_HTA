"""
CTC-Decoder: wandelt Modellausgaben (Log-Wahrscheinlichkeiten) in lesbaren Text um.

Drei Varianten:
  - Greedy Decoder:      schnell, vollständig auf GPU
  - Beam Search Decoder: mittlere Qualität, CPU-Schleife über GPU-vorberechnete Top-k
  - LM Beam Search:      beste Qualität, größerer Beam auf GPU-vorberechneten Top-k
"""

from typing import List, Optional
import torch

from src.dataset import IDX2CHAR, BLANK_TOKEN

BLANK_IDX = 0


def greedy_decode(log_probs: torch.Tensor) -> List[str]:
    """
    Greedy CTC-Decoding: vollständig auf GPU via torch.argmax.

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


def _beam_decode_from_topk(
    top_idxs,   # numpy (SeqLen, top_k)
    top_vals,   # numpy (SeqLen, top_k)
    beam_width: int,
) -> str:
    """Kern-Beam-Search über vorberechnete Top-k Werte (CPU, numpy-frei)."""
    beams: dict = {((), -1): 0.0}

    for t in range(top_idxs.shape[0]):
        idxs = top_idxs[t]
        vals = top_vals[t]
        new_beams: dict = {}

        for (prefix, last_c), score in beams.items():
            for c, lp_c in zip(idxs, vals):
                c = int(c)
                new_score = score + float(lp_c)

                if c == BLANK_IDX:
                    key = (prefix, -1)
                elif c == last_c:
                    key = (prefix, c)
                else:
                    key = (prefix + (c,), c)

                if key not in new_beams or new_beams[key] < new_score:
                    new_beams[key] = new_score

        beams = dict(sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width])

    if not beams:
        return ""
    best_prefix = max(beams.items(), key=lambda x: x[1])[0][0]
    return "".join(IDX2CHAR.get(c, "") for c in best_prefix)


def beam_search_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
) -> List[str]:
    """
    Beam Search CTC-Decoding.

    Top-k Selektion wird einmalig auf GPU berechnet (torch.topk), dann ein
    einziger GPU→CPU Transfer. Die Beam-Schleife läuft auf CPU.

    Args:
        log_probs:   (SeqLen, Batch, NumClasses)
        beam_width:  Anzahl der Hypothesen im Beam

    Returns:
        Liste von dekodiertem Text
    """
    top_k = min(beam_width * 4, log_probs.shape[2])

    # Einmalige GPU-Berechnung aller Top-k Werte
    top_vals, top_idxs = torch.topk(log_probs, top_k, dim=2)   # (SeqLen, Batch, top_k)
    top_vals_np = top_vals.detach().cpu().numpy()
    top_idxs_np = top_idxs.detach().cpu().numpy()

    batch_size = log_probs.shape[1]
    return [
        _beam_decode_from_topk(top_idxs_np[:, b, :], top_vals_np[:, b, :], beam_width)
        for b in range(batch_size)
    ]


def lm_decode(
    log_probs: torch.Tensor,
    beam_width: int = 15,
    lm_path: Optional[str] = None,
) -> List[str]:
    """
    Hochwertiger Beam Search mit größerem Beam – vollständig in PyTorch/GPU.

    Alle Top-k Werte werden einmalig auf der GPU mit torch.topk berechnet,
    dann ein einziger GPU→CPU Transfer. Die Beam-Schleife läuft auf CPU.
    Kein externer Decoder nötig.

    Args:
        log_probs:  (SeqLen, Batch, NumClasses)
        beam_width: Anzahl der Hypothesen (Standard 15, höher = besser aber langsamer)
        lm_path:    Nicht genutzt – für Kompatibilität behalten

    Returns:
        Liste von dekodiertem Text
    """
    top_k = min(beam_width * 3, log_probs.shape[2])

    top_vals, top_idxs = torch.topk(log_probs, top_k, dim=2)   # GPU
    top_vals_np = top_vals.detach().cpu().numpy()               # ein Transfer
    top_idxs_np = top_idxs.detach().cpu().numpy()

    batch_size = log_probs.shape[1]
    return [
        _beam_decode_from_topk(top_idxs_np[:, b, :], top_vals_np[:, b, :], beam_width)
        for b in range(batch_size)
    ]
