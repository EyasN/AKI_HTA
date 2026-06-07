# Langer Einzellauf fuer CRNN (CTC)
# ---------------------------------
# Ein durchgehender Lauf statt vieler kurzer Neustart-Runden:
# So verwaltet EIN Scheduler die Lernrate konsistent und senkt sie bei
# Plateaus ab, bis das Modell sauber auskonvergiert. Early Stopping greift
# erst nach 40 Epochen ohne Verbesserung.
#
# Setzt automatisch beim besten Checkpoint fort (--resume) und behaelt die
# gespeicherte Lernrate bei. Stoppen mit Ctrl+C; einfach erneut starten,
# um weiterzutrainieren.

$checkpoint = "outputs/checkpoints/best_model.pt"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CRNN (CTC) – langer Lauf" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python -m training.train `
    --arch crnn `
    --dataset iam `
    --data-dir data/raw `
    --epochs 300 `
    --batch-size 128 `
    --lr 1e-3 `
    --patience 40 `
    --resume $checkpoint

Write-Host ""
Write-Host "Lauf beendet (Early Stopping oder Ctrl+C)." -ForegroundColor Yellow
Write-Host "Erneut starten zum Weitertrainieren – die gespeicherte LR wird beibehalten." -ForegroundColor Yellow
