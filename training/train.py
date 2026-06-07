"""
Training-Pipeline für das CRNN-Modell.

Ablauf pro Epoch:
  1. Vorwärtsdurchlauf: Bild → Modell → Log-Wahrscheinlichkeiten
  2. CTC-Loss berechnen
  3. Rückwärtsdurchlauf (Backpropagation)
  4. Optimierer-Schritt
  5. Validierung + Metriken
  6. Early Stopping prüfen
  7. Checkpoint speichern (wenn bestes Modell)

Starten mit:
  python -m training.train --dataset synthetic --epochs 30

Mit IAM-Datensatz:
  python -m training.train --dataset iam --data-dir data/raw --epochs 100
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import NUM_CLASSES, SEQ2SEQ_VOCAB, PAD_IDX, get_dataloaders, get_seq2seq_dataloaders
from src.model import build_model, build_seq2seq_model
from evaluation.evaluate import compute_cer, decode_batch_labels
from utils.ctc_decoder import greedy_decode
from utils.seq2seq_decoder import seq2seq_greedy_decode, decode_seq2seq_labels
from utils.visualization import plot_training_curves


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def apply_warmup_lr(optimizer, step: int, warmup_steps: int, base_lr: float) -> None:
    """
    Linearer Lernraten-Warmup über die ersten `warmup_steps` Optimizer-Schritte.

    Transformer-Decoder sind ganz am Anfang instabil (große, verrauschte Gradienten).
    Ein sanfter Anstieg von 0 → base_lr stabilisiert die ersten Updates. Nach dem
    Warmup übernimmt der reguläre Scheduler (ReduceLROnPlateau).
    """
    if warmup_steps and step <= warmup_steps:
        lr = base_lr * step / warmup_steps
        for pg in optimizer.param_groups:
            pg["lr"] = lr


class EMA:
    """
    Exponential Moving Average der Modellgewichte.

    Führt einen langsam nachgezogenen Mittelwert aller Float-Gewichte (und BN-Statistiken)
    mit. Für Eval und Inferenz ist dieser geglättete Mittelwert meist etwas besser als die
    zuletzt trainierten Gewichte – die letzten Updates rauschen, der Mittelwert nicht. Ein
    günstiger Genauigkeitsgewinn ohne zusätzliche Trainingszeit.

    Nutzung:
        ema = EMA(model, decay=0.999)
        ... nach jedem optimizer.step(): ema.update(model)
        # für Validierung/Speichern:
        backup = ema.store_and_apply(model)   # EMA-Gewichte ins Modell kopieren
        ... eval / save ...
        ema.restore(model, backup)            # Trainingsgewichte zurückholen
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay  = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def store_and_apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Sichert die aktuellen Gewichte und kopiert die EMA-Gewichte ins Modell."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        msd = model.state_dict()
        for k, v in backup.items():
            msd[k].copy_(v)


# ── Trainer-Klasse ────────────────────────────────────────────────────────────

class Trainer:
    """
    Kapselt die gesamte Trainingslogik: Optimizer, Scheduler, Checkpointing,
    Early Stopping und Logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        checkpoint_dir: str = "outputs/checkpoints",
        log_dir: str = "outputs/logs",
        device: torch.device = torch.device("cpu"),
        warmup_steps: int = 0,
        ema_decay: float = 0.0,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.patience     = patience

        # Warmup-Status (linearer LR-Anstieg über die ersten Schritte)
        self.base_lr      = lr
        self.warmup_steps = warmup_steps
        self.global_step  = 0

        # EMA der Gewichte (für stabilere Eval/Inferenz); ema_decay=0 → aus
        self.ema = EMA(self.model, ema_decay) if ema_decay and ema_decay > 0 else None

        # Konfigurations-Metadaten, die in den Checkpoint geschrieben werden
        # (img_height/img_width/deslant) – so kann Inferenz/Eval sich selbst konfigurieren.
        self.save_meta: Dict[str, object] = {}

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir        = log_dir

        # CTC-Loss: blank_label=0 (Index 0 = '-' in unserem Alphabet)
        self.criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # AdamW: entkoppelter Weight Decay (korrekte L2-Regularisierung, bessere Generalisierung)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Lernrate halbieren wenn Val-Loss über `patience//2` Epochen nicht sinkt
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=patience // 2
        )

        # Mixed Precision: GradScaler für float16 auf CUDA (Tensor Cores)
        self.use_amp = device.type == "cuda"
        self.scaler  = GradScaler("cuda" if self.use_amp else "cpu")

        self.writer = SummaryWriter(log_dir=log_dir)

        # Verlaufs-Tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_cer":  [], "val_cer": [],
        }
        self.best_val_cer   = float("inf")
        self.epochs_no_improve = 0

    # ── Haupt-Trainingsschleife ────────────────────────────────────────────────

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        """
        Trainiert das Modell für `epochs` Epochen mit Early Stopping.

        Returns:
            history: Dict mit Loss- und CER-Verläufen
        """
        print(f"\nTraining gestartet auf: {self.device}")
        print(f"Epochen: {epochs}  |  Early-Stopping-Patience: {self.patience}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_cer = self._train_epoch(epoch)

            # Validierung & Speichern auf den EMA-Gewichten (geglättet, meist besser)
            ema_backup = self.ema.store_and_apply(self.model) if self.ema else None
            val_loss,   val_cer   = self._val_epoch()

            elapsed = time.time() - t0

            # Verlauf speichern
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_cer"].append(train_cer)
            self.history["val_cer"].append(val_cer)

            # TensorBoard
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("CER",  {"train": train_cer,  "val": val_cer},  epoch)
            self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]["lr"], epoch)

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss:.4f}  Acc: {max(0, 1-train_cer)*100:.1f}%  CER: {train_cer:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {max(0, 1-val_cer)*100:.1f}%  CER: {val_cer:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            # Learning Rate anpassen
            self.scheduler.step(val_loss)

            # Bestes Modell speichern
            if val_cer < self.best_val_cer:
                self.best_val_cer = val_cer
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_cer, is_best=True)
                print(f"  → Neues bestes Modell gespeichert (CER: {val_cer:.4f})")
            else:
                self.epochs_no_improve += 1

            # Regelmäßiger Checkpoint alle 10 Epochen
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_cer, is_best=False)

            # EMA-Gewichte wurden gespeichert – Trainingsgewichte zurückholen
            if self.ema is not None:
                self.ema.restore(self.model, ema_backup)

            # Early Stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly Stopping nach {epoch} Epochen (keine Verbesserung seit {self.patience} Epochen).")
                break

        self.writer.close()
        self._save_history()
        plot_training_curves(self.history, save_path=f"{self.log_dir}/training_curves.png")
        return self.history

    # ── Epoch-Funktionen ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Trainings-Epoch: Vorwärts- + Rückwärtsdurchlauf für alle Batches."""
        self.model.train()
        total_loss = 0.0
        all_preds:  List[str] = []
        all_labels: List[str] = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train", leave=False)
        for images, labels, label_lengths in pbar:
            images        = images.to(self.device)
            labels        = labels.to(self.device)
            label_lengths = label_lengths.to(self.device)

            # Vorwärtsdurchlauf mit Mixed Precision (float16 auf Tensor Cores)
            self.optimizer.zero_grad()
            self.global_step += 1
            apply_warmup_lr(self.optimizer, self.global_step, self.warmup_steps, self.base_lr)
            with autocast("cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                log_probs = self.model(images)  # (SeqLen, Batch, NumClasses)

                seq_len       = log_probs.size(0)
                batch_size    = images.size(0)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)

                loss = self.criterion(log_probs, labels, input_lengths, label_lengths)

            # Rückwärtsdurchlauf mit GradScaler (verhindert Underflow bei float16)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += loss.item()

            # CER für diesen Batch (nur auf CPU)
            with torch.no_grad():
                preds  = greedy_decode(log_probs.detach())
                truths = decode_batch_labels(labels.cpu(), label_lengths.cpu())
                all_preds.extend(preds)
                all_labels.extend(truths)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        cer      = compute_cer(all_preds, all_labels)
        return avg_loss, cer

    def _val_epoch(self) -> Tuple[float, float]:
        """Validierungs-Epoch: nur Vorwärtsdurchlauf, kein Gradient."""
        self.model.eval()
        total_loss = 0.0
        all_preds:  List[str] = []
        all_labels: List[str] = []

        with torch.no_grad():
            for images, labels, label_lengths in tqdm(self.val_loader, desc="  Val", leave=False):
                images        = images.to(self.device)
                labels_dev    = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)

                log_probs = self.model(images)
                seq_len     = log_probs.size(0)
                batch_size  = images.size(0)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)

                loss = self.criterion(log_probs, labels_dev, input_lengths, label_lengths)
                total_loss += loss.item()

                preds  = greedy_decode(log_probs)
                truths = decode_batch_labels(labels, label_lengths.cpu())
                all_preds.extend(preds)
                all_labels.extend(truths)

        avg_loss = total_loss / len(self.val_loader)
        cer      = compute_cer(all_preds, all_labels)
        return avg_loss, cer

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_cer: float, is_best: bool) -> None:
        filename = "best_model.pt" if is_best else f"checkpoint_epoch{epoch:03d}.pt"
        path = Path(self.checkpoint_dir) / filename
        torch.save({
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_cer":             val_cer,
            "history":             self.history,
            **self.save_meta,
        }, path)

    def _save_history(self) -> None:
        path = Path(self.log_dir) / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Verlauf gespeichert: {path}")


# ── Seq2Seq Trainer ───────────────────────────────────────────────────────────

class Seq2SeqTrainer:
    """
    Trainingslogik für CNN + BiLSTM + Transformer Decoder.
    Nutzt Teacher Forcing und CrossEntropyLoss statt CTCLoss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        checkpoint_dir: str = "outputs/checkpoints",
        log_dir: str = "outputs/logs",
        device: torch.device = torch.device("cpu"),
        warmup_steps: int = 0,
        ema_decay: float = 0.0,
        train_cer_batches: int = 8,
    ) -> None:
        self.model        = model.to(device)
        self.device       = device
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.patience     = patience
        self.checkpoint_dir = checkpoint_dir
        self.log_dir        = log_dir

        # Warmup-Status (linearer LR-Anstieg über die ersten Schritte)
        self.base_lr      = lr
        self.warmup_steps = warmup_steps
        self.global_step  = 0

        # EMA der Gewichte (für stabilere Eval/Inferenz); ema_decay=0 → aus
        self.ema = EMA(self.model, ema_decay) if ema_decay and ema_decay > 0 else None

        # Train-CER über die ersten N Batches schätzen (Greedy; verlässlicher als 1 Batch).
        # Die Val-CER läuft dagegen über den ganzen Val-Satz (Greedy, schnell) – Greedy rankt
        # die Epochen praktisch identisch zu Beam, Beam nutzen wir gezielt fürs finale Eval.
        self.train_cer_batches = train_cer_batches

        # Konfigurations-Metadaten für den Checkpoint (img_height/img_width/deslant)
        self.save_meta: Dict[str, object] = {}

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Label Smoothing (0.1): verhindert Übervertrauen, verbessert Generalisierung
        # – Standard bei Transformer-Decodern, hilft beim Schritt Richtung höherer Genauigkeit.
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
        # AdamW: entkoppelter Weight Decay (korrekte L2-Regularisierung, bessere Generalisierung)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=patience // 2)
        self.writer    = SummaryWriter(log_dir=log_dir)

        self.use_amp = device.type == "cuda"
        self.scaler  = GradScaler("cuda" if self.use_amp else "cpu")

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_cer": [], "val_cer": [],
        }
        self.best_val_cer      = float("inf")
        self.epochs_no_improve = 0

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        print(f"\nSeq2Seq Training auf: {self.device}")
        print(f"Epochen: {epochs}  |  Patience: {self.patience}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_cer = self._train_epoch(epoch)

            # Validierung & Speichern auf den EMA-Gewichten (geglättet, meist besser)
            ema_backup = self.ema.store_and_apply(self.model) if self.ema else None
            val_loss,   val_cer   = self._val_epoch()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_cer"].append(train_cer)
            self.history["val_cer"].append(val_cer)

            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("CER",  {"train": train_cer,  "val": val_cer},  epoch)
            self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]["lr"], epoch)

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss:.4f}  Acc: {max(0, 1-train_cer)*100:.1f}%  CER: {train_cer:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {max(0, 1-val_cer)*100:.1f}%  CER: {val_cer:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            self.scheduler.step(val_loss)

            if val_cer < self.best_val_cer:
                self.best_val_cer = val_cer
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_cer, is_best=True)
                print(f"  → Bestes Seq2Seq-Modell gespeichert (CER: {val_cer:.4f})")
            else:
                self.epochs_no_improve += 1

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_cer, is_best=False)

            # EMA-Gewichte wurden gespeichert – Trainingsgewichte zurückholen
            if self.ema is not None:
                self.ema.restore(self.model, ema_backup)

            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly Stopping nach {epoch} Epochen.")
                break

        self.writer.close()
        self._save_history()
        plot_training_curves(self.history, save_path=f"{self.log_dir}/training_curves_seq2seq.png")
        return self.history

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        cer_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train", leave=False)
        for batch_idx, (images, tgt) in enumerate(pbar):
            images = images.to(self.device)
            tgt    = tgt.to(self.device).T          # (max_len, batch)

            tgt_input  = tgt[:-1]
            tgt_output = tgt[1:]
            tgt_len    = tgt_input.size(0)
            tgt_mask   = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=self.device), diagonal=1)
            pad_mask   = (tgt_input.T == PAD_IDX)

            self.optimizer.zero_grad()
            self.global_step += 1
            apply_warmup_lr(self.optimizer, self.global_step, self.warmup_steps, self.base_lr)
            with autocast("cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                logits = self.model(images, tgt_input, tgt_mask=tgt_mask,
                                    tgt_key_padding_mask=pad_mask)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1),
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += loss.item()
            if batch_idx < self.train_cer_batches:
                cer_samples.append((images, tgt))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Train-CER über eine Stichprobe mehrerer Batches (Greedy) – verlässlicher
        # als die alte Ein-Batch-Schätzung, daher kein Rauschen mehr in der Anzeige.
        train_cer = 0.0
        if cer_samples:
            preds:  List[str] = []
            truths: List[str] = []
            with torch.no_grad():
                for imgs, t in cer_samples:
                    with autocast("cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                        preds.extend(seq2seq_greedy_decode(self.model, imgs))
                    truths.extend(decode_seq2seq_labels(t.T))
            train_cer = compute_cer(preds, truths)

        return total_loss / len(self.train_loader), train_cer

    def _val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds:  List[str] = []
        all_labels: List[str] = []

        with torch.no_grad():
            for images, tgt in tqdm(self.val_loader, desc="  Val", leave=False):
                images = images.to(self.device)
                tgt    = tgt.to(self.device).T

                tgt_input  = tgt[:-1]
                tgt_output = tgt[1:]
                tgt_len    = tgt_input.size(0)
                tgt_mask   = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=self.device), diagonal=1)
                pad_mask   = (tgt_input.T == PAD_IDX)

                with autocast("cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                    logits = self.model(images, tgt_input, tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=pad_mask)
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                total_loss += loss.item()

                # CER mit Greedy über den ganzen Val-Satz – schnell, stabil und
                # rankt die Epochen praktisch identisch zu Beam. Beam nur fürs finale Eval.
                with autocast("cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                    preds = seq2seq_greedy_decode(self.model, images)
                all_preds.extend(preds)
                all_labels.extend(decode_seq2seq_labels(tgt.T))

        return total_loss / len(self.val_loader), compute_cer(all_preds, all_labels)

    def _save_checkpoint(self, epoch: int, val_cer: float, is_best: bool) -> None:
        filename = "best_seq2seq.pt" if is_best else f"seq2seq_epoch{epoch:03d}.pt"
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_cer":              val_cer,
            "history":              self.history,
            **self.save_meta,
            "arch":                 "seq2seq",
            "lstm_hidden":          self.model.rnn.hidden_size,
        }, Path(self.checkpoint_dir) / filename)

    def _save_history(self) -> None:
        path = Path(self.log_dir) / "history_seq2seq.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Verlauf gespeichert: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTR Training")
    parser.add_argument("--arch",         default="crnn",          choices=["crnn", "seq2seq"],
                        help="Modell-Architektur: crnn (CTC) oder seq2seq (Transformer Decoder)")
    parser.add_argument("--dataset",     default="synthetic",      choices=["synthetic", "iam"])
    parser.add_argument("--data-dir",    default="data/raw")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--img-height",  type=int, default=32)
    parser.add_argument("--img-width",   type=int, default=256)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--patience",    type=int, default=15)
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--resume",      default=None)
    parser.add_argument("--reset-lr",    action="store_true",
                        help="LR beim Resume auf --lr zurücksetzen. Standard: gespeicherte (ggf. abgesenkte) LR beibehalten.")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linearer LR-Warmup über die ersten N Optimizer-Schritte (nur bei Neustart, nicht beim Resume).")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                        help="EMA der Gewichte für Eval/Inferenz (--no-ema schaltet ab).")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA-Decay (näher an 1 = träger/glatter, z.B. 0.999).")
    parser.add_argument("--deslant", action="store_true",
                        help="Deslanting (Schräglagen-Korrektur) auf alle Bilder anwenden. "
                             "Muss bei Inferenz/Eval identisch gesetzt sein (wird im Checkpoint vermerkt).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gerät: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    if args.arch == "seq2seq":
        # ── Seq2Seq ──────────────────────────────────────────────────────────
        train_loader, val_loader = get_seq2seq_dataloaders(
            dataset_type=args.dataset, data_dir=args.data_dir,
            batch_size=args.batch_size, img_height=args.img_height,
            img_width=args.img_width, num_workers=args.workers,
            deslant=args.deslant,
        )
        model = build_seq2seq_model(
            img_height=args.img_height, vocab_size=SEQ2SEQ_VOCAB,
            lstm_hidden=args.lstm_hidden,
        )
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Seq2Seq Modell-Parameter: {total_params:,}")

        trainer = Seq2SeqTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            lr=args.lr, patience=args.patience, device=device,
            warmup_steps=args.warmup_steps,
            ema_decay=args.ema_decay if args.ema else 0.0,
        )
        if args.resume and Path(args.resume).exists():
            ck = torch.load(args.resume, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            trainer.optimizer.load_state_dict(ck["optimizer_state_dict"])
            trainer.best_val_cer = ck.get("val_cer", float("inf"))
            trainer.warmup_steps = 0   # kein Warmup beim Resume – LR ist bereits eingependelt
            if trainer.ema is not None:
                trainer.ema = EMA(model, args.ema_decay)   # EMA von geladenen Gewichten neu starten
            if args.reset_lr:
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = args.lr
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            print(f"Seq2Seq fortgesetzt: {args.resume}  (CER: {trainer.best_val_cer:.4f}, LR: {current_lr:.2e})")
        elif args.resume:
            print(f"Kein Checkpoint gefunden unter {args.resume} – starte neu.")

        trainer.save_meta = {"img_height": args.img_height, "img_width": args.img_width, "deslant": args.deslant}
        trainer.fit(epochs=args.epochs)
        print("\nTraining abgeschlossen!")
        print("Bestes Modell: outputs/checkpoints/best_seq2seq.pt")

    else:
        # ── CRNN (CTC) ───────────────────────────────────────────────────────
        train_loader, val_loader = get_dataloaders(
            dataset_type=args.dataset, data_dir=args.data_dir,
            batch_size=args.batch_size, img_height=args.img_height,
            img_width=args.img_width, num_workers=args.workers,
            deslant=args.deslant,
        )
        model = build_model(
            img_height=args.img_height, num_classes=NUM_CLASSES,
            lstm_hidden=args.lstm_hidden,
        )
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"CRNN Modell-Parameter: {total_params:,}")

        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            lr=args.lr, patience=args.patience, device=device,
            warmup_steps=args.warmup_steps,
            ema_decay=args.ema_decay if args.ema else 0.0,
        )
        if args.resume and Path(args.resume).exists():
            ck = torch.load(args.resume, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            trainer.optimizer.load_state_dict(ck["optimizer_state_dict"])
            trainer.best_val_cer = ck.get("val_cer", float("inf"))
            trainer.warmup_steps = 0   # kein Warmup beim Resume – LR ist bereits eingependelt
            if trainer.ema is not None:
                trainer.ema = EMA(model, args.ema_decay)   # EMA von geladenen Gewichten neu starten
            if args.reset_lr:
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = args.lr
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            print(f"Training fortgesetzt: {args.resume}  (CER: {trainer.best_val_cer:.4f}, LR: {current_lr:.2e})")
        elif args.resume:
            print(f"Kein Checkpoint gefunden unter {args.resume} – starte neu.")

        trainer.save_meta = {"img_height": args.img_height, "img_width": args.img_width, "deslant": args.deslant}
        trainer.fit(epochs=args.epochs)
        print("\nTraining abgeschlossen!")
        print("Bestes Modell: outputs/checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
