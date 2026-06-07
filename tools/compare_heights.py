"""
Diagnose: Wie viel Detail verliert das Modell bei Höhe 32 vs. 48 vs. 64?

Rendert ein paar echte IAM-Wortbilder so, wie das Modell sie nach dem Resize
"sieht" (Höhe-skaliert, Seitenverhältnis erhalten), und stellt die Höhen
nebeneinander – jeweils auf eine gemeinsame Anzeigehöhe hochskaliert (Nearest,
damit man die echte Pixelschärfe sieht, nicht eine geglättete Version).

Aufruf:
    python -m tools.compare_heights
    python -m tools.compare_heights --n 6 --data-dir data/raw
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def height_resize(img: Image.Image, target_h: int) -> Image.Image:
    """Skaliert auf target_h, Seitenverhältnis erhalten (wie ResizeToHeight, ohne Padding)."""
    w, h = img.size
    new_w = max(1, int(w * target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--n", type=int, default=5, help="Anzahl Beispielbilder")
    ap.add_argument("--heights", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--display-h", type=int, default=96, help="Anzeigehöhe (Hochskalierung)")
    ap.add_argument("--out", default="outputs/logs/height_compare.png")
    args = ap.parse_args()

    # Ein paar echte IAM-Wortbilder einsammeln
    pngs = sorted(Path(args.data_dir).rglob("*.png"))
    pngs = [p for p in pngs if "words" in str(p).lower()][: args.n * 4]
    if not pngs:
        pngs = sorted(Path(args.data_dir).rglob("*.png"))[: args.n * 4]
    if not pngs:
        print(f"Keine PNGs unter {args.data_dir} gefunden.")
        return
    # gleichmäßig über die Liste streuen
    step = max(1, len(pngs) // args.n)
    samples = pngs[::step][: args.n]

    rows, cols = len(samples), len(args.heights)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 1.6))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for r, path in enumerate(samples):
        img = Image.open(path).convert("L")
        for c, h in enumerate(args.heights):
            small = height_resize(img, h)                       # so "sieht" es das Modell
            disp_w = max(1, int(small.size[0] * args.display_h / h))
            disp = small.resize((disp_w, args.display_h), Image.NEAREST)  # ohne Glättung hochziehen
            ax = axes[r, c]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Höhe {h}px", fontsize=12)
            if c == 0:
                ax.set_ylabel(path.name[:14], fontsize=7)

    fig.suptitle("Was das Modell bei verschiedenen Höhen sieht (Nearest-Upscale)", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Vergleichsbild gespeichert: {args.out}")
    print(f"Verwendete Beispiele: {[p.name for p in samples]}")


if __name__ == "__main__":
    main()
