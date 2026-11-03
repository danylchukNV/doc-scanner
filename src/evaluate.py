"""Оцінка точності пошуку документа на датасетах з розміткою кутів.

Метрики для кожного зображення:
    IoU         — Jaccard index: площа перетину / площа об'єднання знайденого
                  та еталонного чотирикутників (як у SmartDoc 2015, але в площині кадру)
    corner_err  — середня відстань між кутами, % від діагоналі кадру
    time_ms     — час пошуку кутів
Успіх: IoU ≥ --iou (за замовчуванням 0.9).
Для зображень без документа (розмітка None) успіх — якщо сканер нічого не знайшов.

Приклади:
    python src/evaluate.py                                         # синтетика
    python src/evaluate.py datasets/doccorner_sample               # ground_truth.json
    python src/evaluate.py "docs/datasets/All My Dataset" --limit 200
    python src/evaluate.py docs/datasets/SmartDoc/testDataset --frame-step 50 --limit 300
    python src/evaluate.py samples/synthetic --worst 5 --save-worst
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset  # noqa: E402
from scanner import ScanConfig, detect_corners, draw_corners, write_image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU двох опуклих чотирикутників (cv2.intersectConvexConvex)."""
    a = a.astype(np.float32).reshape(-1, 1, 2)
    b = b.astype(np.float32).reshape(-1, 1, 2)
    area_a, area_b = abs(cv2.contourArea(a)), abs(cv2.contourArea(b))
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def corner_error_pct(pred: np.ndarray, gt: np.ndarray, shape) -> float:
    diag = float(np.hypot(shape[0], shape[1]))
    return float(np.linalg.norm(pred - gt, axis=1).mean() / diag * 100)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Оцінка точності сканера на датасеті")
    p.add_argument("folder", nargs="?", default=str(ROOT / "samples" / "synthetic"))
    p.add_argument("--format", default="auto", choices=["auto", "gt", "mendeley", "smartdoc"])
    p.add_argument("--limit", type=int, default=0, help="випадкова вибірка N зображень")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-step", type=int, default=30, help="SmartDoc: брати кожен N-й кадр")
    p.add_argument("--iou", type=float, default=0.9, help="поріг IoU для успіху")
    p.add_argument("--worst", type=int, default=10, help="скільки найгірших випадків показати")
    p.add_argument("--save-worst", action="store_true",
                   help="зберегти картинки найгірших випадків (зелений — еталон, червоний — знайдене)")
    p.add_argument("-o", "--out", default=None, help="CSV-звіт (за замовч. output/eval_<папка>.csv)")
    # параметри алгоритму — ті самі, що в main.py, щоб порівнювати налаштування
    p.add_argument("--canny", nargs=2, type=int, metavar=("LOW", "HIGH"))
    p.add_argument("--sigma", type=float, default=0.33)
    p.add_argument("--blur", type=int, default=5)
    p.add_argument("--morph", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--min-area", type=float, default=0.10)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    folder = Path(a.folder)
    if not folder.exists():
        print(f"Не знайдено: {folder}")
        return 2
    cfg = ScanConfig(blur_ksize=a.blur, canny_low=a.canny[0] if a.canny else None,
                     canny_high=a.canny[1] if a.canny else None, canny_sigma=a.sigma,
                     morph_ksize=a.morph, approx_eps=a.eps, min_area_ratio=a.min_area)

    fmt, samples = load_dataset(folder, a.format, a.frame_step)
    total = len(samples)
    if a.limit and a.limit < total:
        samples = random.Random(a.seed).sample(samples, a.limit)
    print(f"Датасет: {folder} (формат {fmt}), зображень: {len(samples)} з {total}\n")

    rows = []
    for i, s in enumerate(samples, 1):
        try:
            img = s.load()
        except Exception as e:
            print(f"  [!] {s.name}: {e}")
            continue
        t0 = time.perf_counter()
        pred, method = detect_corners(img, cfg)
        ms = (time.perf_counter() - t0) * 1000

        if s.corners is None:
            row = dict(file=s.name, method=method, negative=True, iou="", corner_err_pct="",
                       success=pred is None, time_ms=round(ms, 1))
        elif pred is None:
            row = dict(file=s.name, method=method, negative=False, iou=0.0, corner_err_pct="",
                       success=False, time_ms=round(ms, 1))
        else:
            iou = polygon_iou(pred, s.corners)
            row = dict(file=s.name, method=method, negative=False, iou=round(iou, 4),
                       corner_err_pct=round(corner_error_pct(pred, s.corners, img.shape), 2),
                       success=iou >= a.iou, time_ms=round(ms, 1))
        row["_pred"], row["_gt"], row["_load"] = pred, s.corners, s.load
        rows.append(row)
        if i % 50 == 0:
            print(f"  ...{i}/{len(samples)}")

    if not rows:
        print("Немає оброблених зображень")
        return 1

    # ---------------- підсумок ----------------
    pos = [r for r in rows if not r["negative"]]
    neg = [r for r in rows if r["negative"]]
    print(f"{'Підсумок':=^60}")
    if pos:
        ious = np.array([r["iou"] for r in pos], dtype=float)
        found = [r for r in pos if r["_pred"] is not None]
        errs = np.array([r["corner_err_pct"] for r in found], dtype=float)
        print(f"З документом:            {len(pos)}")
        print(f"  знайдено щось:         {len(found)} ({len(found) / len(pos):.0%})")
        print(f"  успіх (IoU ≥ {a.iou}):    {sum(r['success'] for r in pos)} "
              f"({sum(r['success'] for r in pos) / len(pos):.1%})")
        print(f"  середній IoU:          {ious.mean():.3f}  (медіана {np.median(ious):.3f})")
        if len(errs):
            print(f"  похибка кутів:         {errs.mean():.2f}% діагоналі (медіана {np.median(errs):.2f}%)")
        methods: dict[str, int] = {}
        for r in pos:
            methods[r["method"]] = methods.get(r["method"], 0) + 1
        print(f"  методи:                {methods}")
    if neg:
        print(f"Без документа:           {len(neg)}, хибних спрацювань: "
              f"{sum(not r['success'] for r in neg)}")
    times = np.array([r["time_ms"] for r in rows])
    print(f"Час:                     {times.mean():.1f} мс/зобр. (макс {times.max():.0f})")

    # ---------------- найгірші ----------------
    worst = sorted(pos, key=lambda r: r["iou"])[:a.worst]
    if worst:
        print(f"\n{'Найгірші випадки':=^60}")
        for r in worst:
            print(f"  IoU {r['iou']:<6}  {r['method']:<15} {r['file']}")
    if a.save_worst and worst:
        wdir = ROOT / "output" / "eval_worst"
        for k, r in enumerate(worst, 1):
            img = r["_load"]()
            vis = draw_corners(img, r["_gt"], (0, 255, 0))
            if r["_pred"] is not None:
                vis = draw_corners(vis, r["_pred"], (0, 0, 255))
            safe = r["file"].replace("/", "_").replace("#", "_")
            write_image(wdir / f"{k:02d}_iou{r['iou']:.2f}_{Path(safe).stem}.jpg", vis)
        print(f"\nКартинки збережено в {wdir}")

    # ---------------- CSV ----------------
    out = Path(a.out) if a.out else ROOT / "output" / f"eval_{folder.name.replace(' ', '_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Звіт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
