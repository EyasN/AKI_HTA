"""
Autoregressive Decoder für das Seq2Seq-Modell (CNN + BiLSTM + Transformer Decoder).

Greedy Decode: nimmt bei jedem Schritt das wahrscheinlichste Token.
"""

from typing import List

import torch
import torch.nn as nn
from torch.amp import autocast

from src.dataset import BOS_IDX, EOS_IDX, PAD_IDX, IDX2CHAR


def seq2seq_greedy_decode(
    model: nn.Module,
    images: torch.Tensor,
    max_len: int = 32,
) -> List[str]:
    """
    Autoregressives Greedy Decoding für CRNN_Seq2Seq.

    Startet mit BOS-Token und generiert Schritt für Schritt das
    wahrscheinlichste Zeichen, bis EOS oder max_len erreicht ist.

    Args:
        model:   CRNN_Seq2Seq Modell (eval-Modus)
        images:  (Batch, 1, H, W) Eingabebilder
        max_len: Maximale Ausgabelänge

    Returns:
        Liste von dekodiertem Text, eine Zeichenkette pro Batch-Element
    """
    device     = images.device
    batch_size = images.size(0)
    use_amp    = device.type == "cuda"

    with torch.no_grad():
        with autocast(device.type, enabled=use_amp):
            memory = model.encode(images)                                    # (seq_len, B, d_model)
        generated = torch.full((1, batch_size), BOS_IDX, dtype=torch.long, device=device)
        finished  = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_len  = generated.size(0)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1)
            with autocast(device.type, enabled=use_amp):
                logits = model.decode(generated, memory, tgt_mask=tgt_mask) # (tgt_len, B, vocab)
            next_tok = logits[-1].argmax(dim=-1)                            # (B,)
            next_tok[finished] = PAD_IDX
            generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=0)
            finished |= (next_tok == EOS_IDX)
            if finished.all():
                break

    results: List[str] = []
    for b in range(batch_size):
        tokens = generated[1:, b].tolist()   # BOS überspringen
        text = ""
        for t in tokens:
            if t in (EOS_IDX, PAD_IDX):
                break
            text += IDX2CHAR.get(t, "")
        results.append(text)
    return results


def decode_seq2seq_labels(tgt_padded: torch.Tensor) -> List[str]:
    """
    Wandelt padded Label-Tensor (Batch, max_len) zurück in Strings.
    Überspringt BOS, stoppt bei EOS oder PAD.
    """
    results: List[str] = []
    for row in tgt_padded:
        text = ""
        for t in row.tolist():
            if t == BOS_IDX:
                continue
            if t in (EOS_IDX, PAD_IDX):
                break
            text += IDX2CHAR.get(t, "")
        results.append(text)
    return results
