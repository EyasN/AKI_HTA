# Frischer Seq2Seq-Lauf: Hoehe 48, Breite 384, Deslanting (Schraeglagen-Korrektur)
# ------------------------------------------------------------------------------
# Wird von 0 trainiert: andere Bildgroesse (48px) ist NICHT resume-kompatibel mit
# dem 32px-Modell (LSTM-Eingabedimension haengt an der Hoehe). Deshalb kein --resume.
#
# Das bisherige 92,98%-Modell liegt sicher als Backup:
#   outputs/checkpoints/best_seq2seq_BASELINE_32x256_cer702.pt
# Dieser Lauf ueberschreibt best_seq2seq.pt mit dem neuen 48px+Deslant-Modell.
#
# EMA ist standardmaessig an, Warmup (500 Schritte) greift beim Neustart automatisch.
# Bildgroesse + Deslant werden im Checkpoint vermerkt -> Inferenz/Eval konfigurieren
# sich selbst. Stoppen mit Ctrl+C; erneut starten setzt NICHT fort (Neustart).

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Seq2Seq FRISCH - 48x384 + Deslanting" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backup des 32px-Modells: best_seq2seq_BASELINE_32x256_cer702.pt" -ForegroundColor DarkGray

python -m training.train `
    --arch seq2seq `
    --dataset iam `
    --data-dir data/raw `
    --epochs 300 `
    --batch-size 64 `
    --lstm-hidden 512 `
    --lr 3e-4 `
    --patience 40 `
    --img-height 48 `
    --img-width 384 `
    --deslant

Write-Host ""
Write-Host "Lauf beendet. Falls VRAM-Fehler (OOM): --batch-size 48 oder 32 probieren." -ForegroundColor Yellow
