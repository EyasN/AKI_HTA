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


def seq2seq_beam_decode(
    model: nn.Module,
    images: torch.Tensor,
    beam_width: int = 5,
    max_len: int = 32,
    length_penalty: float = 0.6,
    return_confidence: bool = False,
) -> "List[str] | tuple[List[str], List[float]]":
    """
    Autoregressives Beam-Search-Decoding für CRNN_Seq2Seq.

    Statt bei jedem Schritt nur das wahrscheinlichste Token zu nehmen (Greedy),
    werden die `beam_width` besten Teilsequenzen parallel weitergeführt. Das findet
    global wahrscheinlichere Sätze und korrigiert frühe Fehlentscheidungen – das
    bringt bei autoregressiven Decodern typischerweise einige Prozent CER.

    Length-Penalty (0.6): normalisiert den Score über die Sequenzlänge, damit
    kurze Sequenzen nicht systematisch bevorzugt werden (Wu et al., 2016).

    Args:
        model:             CRNN_Seq2Seq Modell (eval-Modus)
        images:            (Batch, 1, H, W) Eingabebilder
        beam_width:        Anzahl parallel verfolgter Hypothesen
        max_len:           Maximale Ausgabelänge
        length_penalty:    Exponent der Längen-Normalisierung (0 = aus)
        return_confidence: Wenn True, zusätzlich eine Confidence pro Sample
                           (mittlere Token-Wahrscheinlichkeit der besten Hypothese, 0–1)

    Returns:
        Liste von dekodiertem Text – oder (Texte, Confidences) wenn return_confidence
    """
    device  = images.device
    use_amp = device.type == "cuda"

    with torch.no_grad():
        with autocast(device.type, enabled=use_amp):
            memory_full = model.encode(images)          # (seq_len, B, d_model)

        batch_size = images.size(0)
        results:     List[str]   = []
        confidences: List[float] = []

        for b in range(batch_size):
            # Encoder-Ausgabe dieses Bildes auf beam_width Hypothesen kopieren
            memory = memory_full[:, b:b + 1, :].expand(-1, beam_width, -1).contiguous()

            seqs   = torch.full((1, beam_width), BOS_IDX, dtype=torch.long, device=device)  # (len, K)
            scores = torch.zeros(beam_width, device=device)
            scores[1:] = float("-inf")   # zu Beginn nur eine aktive Hypothese (keine Duplikate)

            # (tokens inkl. EOS, Rang-Score, Gesamt-LogProb, Länge)
            finished: List[tuple[List[int], float, float, int]] = []

            for _ in range(max_len):
                tgt_len  = seqs.size(0)
                tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1)
                with autocast(device.type, enabled=use_amp):
                    logits = model.decode(seqs, memory, tgt_mask=tgt_mask)   # (len, K, vocab)

                log_probs = torch.log_softmax(logits[-1].float(), dim=-1)    # (K, vocab)
                vocab     = log_probs.size(-1)

                # Score jeder Hypothese + jedes möglichen nächsten Tokens
                cand = scores.unsqueeze(1) + log_probs                       # (K, vocab)
                topv, topi = cand.view(-1).topk(beam_width)
                beam_idx = torch.div(topi, vocab, rounding_mode="floor")     # welche Hypothese
                tok_idx  = topi % vocab                                       # welches Token

                seqs   = torch.cat([seqs[:, beam_idx], tok_idx.unsqueeze(0)], dim=0)
                scores = topv.clone()

                # Hypothesen, die EOS erreichen, abschließen und deaktivieren
                for k in range(beam_width):
                    if tok_idx[k].item() == EOS_IDX:
                        tokens   = seqs[1:, k].tolist()                      # BOS überspringen
                        length   = max(len(tokens), 1)
                        total_lp = topv[k].item()                           # kumulierte Log-Prob inkl. EOS
                        rank     = total_lp / (length ** length_penalty)
                        finished.append((tokens, rank, total_lp, length))
                        scores[k] = float("-inf")

                if torch.isinf(scores).all():
                    break

            # Beste abgeschlossene Hypothese, sonst beste laufende
            if finished:
                tokens, _, total_lp, length = max(finished, key=lambda x: x[1])
            else:
                best_k   = int(scores.argmax().item())
                tokens   = seqs[1:, best_k].tolist()
                length   = max(len(tokens), 1)
                total_lp = float(scores[best_k].item())

            text = ""
            for t in tokens:
                if t in (EOS_IDX, PAD_IDX):
                    break
                text += IDX2CHAR.get(t, "")
            results.append(text)
            # Confidence = exp(mittlere Token-Log-Prob) ∈ (0, 1]
            confidences.append(float(torch.tensor(total_lp / length).exp().clamp(0, 1)))

    if return_confidence:
        return results, confidences
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
