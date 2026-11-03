"""Генератор синтетичних тестових зображень з відомими кутами документа.

Дозволяє перевірити алгоритм до появи реальних фото та виміряти точність
(відстань між знайденими й справжніми кутами).

Запуск:  python src/make_synthetic.py
Результат: samples/synthetic/*.jpg + ground_truth.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples" / "synthetic"
W, H = 1600, 1200
rng = np.random.default_rng(42)


def make_document(w=840, h=1188) -> np.ndarray:
    """Аркуш «A4» з заголовком і рядками тексту."""
    doc = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(doc, "DOCUMENT SCANNER TEST", (60, 110), cv2.FONT_HERSHEY_DUPLEX, 1.4, (20, 20, 20), 3)
    words = ["computer", "vision", "homography", "contour", "edge", "threshold",
             "perspective", "canny", "opencv", "image", "matrix", "pixel"]
    y = 190
    while y < h - 80:
        x = 60
        while x < w - 160:
            wd = str(rng.choice(words))
            cv2.putText(doc, wd, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            x += 30 + 16 * len(wd)
        y += 48
    return doc


def background(kind: str) -> np.ndarray:
    if kind == "white":
        base = np.full((H, W, 3), 225, np.uint8)
    else:  # темний «дерев'яний» стіл
        base = np.zeros((H, W, 3), np.uint8)
        base[:] = (40, 60, 90)
        stripes = (np.sin(np.linspace(0, 60, H))[:, None] * 12).astype(np.int16)
        base = np.clip(base.astype(np.int16) + stripes[..., None], 0, 255).astype(np.uint8)
    noise = rng.normal(0, 6, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def place(doc, bg, corners) -> np.ndarray:
    h, w = doc.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(corners))
    warped = cv2.warpPerspective(doc, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    return out


def add_shadow(img) -> np.ndarray:
    mask = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(mask, (1000, 400), (450, 700), 30, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 60)
    return (img * (1 - 0.55 * mask[..., None])).astype(np.uint8)


def low_light(img) -> np.ndarray:
    dark = (img * 0.35).astype(np.float32)
    return np.clip(dark + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)


def add_clutter(img) -> np.ndarray:
    out = img.copy()
    for _ in range(12):
        x, y = int(rng.integers(0, W - 200)), int(rng.integers(0, H - 200))
        color = tuple(int(c) for c in rng.integers(30, 255, 3))
        cv2.rectangle(out, (x, y), (x + int(rng.integers(40, 260)), y + int(rng.integers(20, 120))),
                      color, -1)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = make_document()
    dark, white = background("dark"), background("white")

    straight = [[400, 120], [1200, 120], [1200, 1080], [400, 1080]]
    angled = [[480, 180], [1150, 140], [1320, 1100], [300, 1060]]
    rotated = [[800, 80], [1350, 600], [800, 1120], [250, 600]]
    cut = [[500, -150], [1250, -100], [1300, 900], [450, 950]]

    cases = {
        "syn_01_straight": (place(doc, dark, straight), straight),
        "syn_02_angle": (place(doc, dark, angled), angled),
        "syn_03_rotated": (place(doc, dark, rotated), rotated),
        "syn_07_shadow": (add_shadow(place(doc, dark, angled)), angled),
        "syn_08_low_light": (low_light(place(doc, dark, angled)), angled),
        "syn_09_white_on_white": (place(doc, white, angled), angled),
        "syn_11_cut_off": (place(doc, dark, cut), cut),
        "syn_12_cluttered": (place(doc, add_clutter(dark), angled), angled),
    }

    # загнутий кут: «відрізаємо» трикутник у правому верхньому куті
    folded = place(doc, dark, angled)
    tri = np.array([[1150, 140], [1000, 150], [1190, 330]], np.int32)
    cv2.fillPoly(folded, [tri], (40, 60, 90))
    cv2.fillPoly(folded, [np.array([[1000, 150], [1190, 330], [1060, 300]], np.int32)], (200, 200, 200))
    cases["syn_10_folded_corner"] = (folded, angled)

    gt = {}
    for name, (img, corners) in sorted(cases.items()):
        cv2.imwrite(str(OUT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        gt[f"{name}.jpg"] = corners
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"Згенеровано {len(cases)} зображень у {OUT}")


if __name__ == "__main__":
    main()
