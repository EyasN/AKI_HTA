# Overnight resume loop for Seq2Seq (48x384 + deslanting)
# Always resumes from best_seq2seq.pt - never starts from scratch.
# Each round: up to 300 epochs, early stopping after 40 epochs without improvement.
# If a round stops early, the loop starts the next one (resuming from the best).
# Stop anytime with Ctrl+C.

$ckpt = "outputs/checkpoints/best_seq2seq.pt"

if (-not (Test-Path $ckpt)) {
    Write-Host "ERROR: $ckpt not found - aborting (would otherwise train from scratch)." -ForegroundColor Red
    exit 1
}

$round = 1
while ($true) {
    Write-Host ""
    Write-Host "===== Round $round : resume 48x384 + deslant from $ckpt =====" -ForegroundColor Cyan
    python -m training.train --arch seq2seq --dataset iam --data-dir data/raw --epochs 300 --batch-size 64 --lstm-hidden 512 --lr 3e-4 --patience 40 --img-height 48 --img-width 384 --deslant --resume $ckpt
    Write-Host "Round $round finished. Next round in 5s (Ctrl+C to stop)." -ForegroundColor Yellow
    $round++
    Start-Sleep -Seconds 5
}
