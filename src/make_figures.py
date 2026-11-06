"""Генерує ілюстрації для README у docs/img/readme/.

Запуск:  python src/make_figures.py
Потрібен matplotlib (pip install matplotlib). Усі картинки будуються з samples/examples,
тому повністю відтворювані.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import (ScanConfig, detect_segments, edge_maps, find_document_multi,  # noqa: E402
                     make_debug_montage, preprocess, read_image, scan)

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "samples" / "examples"
OUT = ROOT / "docs" / "img" / "readme"

# палітра (перевірена на розрізнення при дальтонізмі): синій, помаранчевий, бірюзовий
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
})


def rgb(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def poly(ax, q, color, lw=2.5, ls="-"):
    q = np.vstack([q, q[:1]])
    ax.plot(q[:, 0], q[:, 1], color=color, lw=lw, ls=ls)


def grid(images, titles, ncols, path, width=14, cmap=None):
    nrows = int(np.ceil(len(images) / ncols))
    h = max(im.shape[0] / im.shape[1] for im in images)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, width / ncols * h * nrows + 0.5 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, im, t in zip(np.ravel(axes), images, titles):
        ax.imshow(rgb(im), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t)
    fig.tight_layout()
    fig.savefig(path, dpi=90, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def hero():
    names = ["03_in_hand", "01_low_contrast_table", "02_card_angle"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), gridspec_kw={"height_ratios": [1, 1]})
    for col, n in enumerate(names):
        img = read_image(EX / f"{n}.jpg")
        r = scan(img)
        a = axes[0, col]
        a.imshow(rgb(img))
        poly(a, r.corners, C2, 3)
        a.set_title("фото → знайдений документ")
        a.axis("off")
        b = axes[1, col]
        b.imshow(r.output, cmap="gray")
        b.set_title("скан")
        b.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "hero.jpg", dpi=80, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def pipeline():
    img = read_image(EX / "04_color_flyer.jpg")
    r = scan(img)
    titles = {"1_original": "1. оригінал", "2_gray_blur": "2. сірий + розмиття Гауса",
              "3_edges": "3. карта країв переможця", "3b_candidates": "3б. усі кандидати",
              "4_contour": "4. знайдений документ", "5_warped": "5. корекція перспективи",
              "6_output": "6. адаптивна бінаризація"}
    st = [(k, v) for k, v in r.stages.items() if k != "1_original"]
    grid([v for _, v in st], [titles[k] for k, _ in st], 3, OUT / "pipeline.jpg")


def edge_maps_fig():
    img = read_image(EX / "01_low_contrast_table.jpg")
    cfg = ScanConfig()
    small, blurred, _ = preprocess(img, cfg)
    maps = edge_maps(small, cfg)
    q, method, _, _ = find_document_multi(small, cfg)
    shown = [("gray_auto", "gray_auto — класичний Canny\n(межа аркуша не виділилась)"),
             ("gray_low", "gray_low — низькі пороги 10/40\n(межа є — тут перемагає)"),
             ("bgr_low", "bgr_low — Canny на каналах B, G, R"),
             ("otsu_sat_inv", "otsu_sat_inv — Otsu за насиченістю")]
    ims = [small] + [255 - maps[k] for k, _ in shown]
    tts = ["фото (білий аркуш на світлому столі)"] + [t for _, t in shown]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.0))
    for ax, im, t in zip(axes, ims, tts):
        ax.imshow(rgb(im), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    poly(axes[0], q, C2, 2.5)
    fig.tight_layout()
    fig.savefig(OUT / "edge_maps.jpg", dpi=85, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def lines_fig():
    img = read_image(EX / "06_fills_frame.jpg")
    cfg = ScanConfig()
    small, _, _ = preprocess(img, cfg)
    segs = detect_segments(small, cfg)
    ang = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])
    feats = np.stack([np.cos(2 * ang), np.sin(2 * ang)], 1).astype(np.float32)
    cv2.setRNGSeed(0)
    _, lab, _ = cv2.kmeans(feats, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3),
                           3, cv2.KMEANS_PP_CENTERS)
    q, method, _, _ = find_document_multi(small, cfg)
    no_lines, _, _, _ = find_document_multi(small, ScanConfig(use_lines=False))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].imshow(rgb(small))
    axes[0].set_title("без прямих: документ торкається країв кадру,\nконтур не замкнений → не знайдено"
                      if no_lines is None else "без прямих")
    axes[1].imshow(rgb(small), alpha=0.45)
    for s, l in zip(segs, lab.ravel()):
        axes[1].plot([s[0], s[2]], [s[1], s[3]], color=C1 if l == 0 else C2, lw=2)
    axes[1].set_title("відрізки LSD, поділені k-means\nна два напрямки (синій / помаранчевий)")
    axes[2].imshow(rgb(small))
    poly(axes[2], q, C3, 3)
    axes[2].set_title(f"4-кутник з перетинів двох пар прямих\n({method})")
    for a in axes:
        a.axis("off")
        a.set_xlim(0, small.shape[1])
        a.set_ylim(small.shape[0], 0)
    fig.tight_layout()
    fig.savefig(OUT / "lines.jpg", dpi=85, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def candidates_fig():
    img = read_image(EX / "03_in_hand.jpg")
    cfg = ScanConfig()
    small, _, _ = preprocess(img, cfg)
    q, method, cands, _ = find_document_multi(small, cfg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].imshow(rgb(small))
    for c in cands:
        poly(axes[0], c, C1, 1.2)
    axes[0].set_title(f"усі кандидати після фільтра: {len(cands)}")
    axes[1].imshow(rgb(small))
    poly(axes[1], q, C2, 3)
    axes[1].set_title(f"переможець за оцінкою\nконтраст × опора × площа ({method})")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "candidates.jpg", dpi=85, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def binarization_fig():
    img = read_image(EX / "05_shadow_synthetic.jpg")
    r = scan(img, ScanConfig(mode="none"))
    gray = cv2.cvtColor(r.warped, cv2.COLOR_BGR2GRAY)
    _, glob = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    grid([r.warped, glob, adap],
         ["після корекції перспективи", "глобальний поріг (Otsu):\nтінь стала чорною",
          "адаптивний поріг (вікно 21, C = 10):\nтінь прибрана"], 3, OUT / "binarization.jpg", width=13)


def results_chart():
    data = {"simple": (20.2, 13.0), "multi v1": (78.9, 35.0), "multi v2 (фінальна)": (86.8, 42.7)}
    groups = ["SmartDoc 2015\n(341 кадр)", "Mendeley\n(300 фото)"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(groups))
    w = 0.26
    for i, ((name, vals), col) in enumerate(zip(data.items(), (C1, C2, C3))):
        bars = ax.bar(x + (i - 1) * (w + 0.01), vals, w, color=col, label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=10, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 100)
    ax.set_ylabel("успіх, % (IoU ≥ 0.9)")
    ax.set_title("Точність на тестовій частині датасетів", loc="left", fontsize=12)
    ax.grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=100, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (hero, pipeline, edge_maps_fig, lines_fig, candidates_fig, binarization_fig, results_chart):
        f()
        print("ok:", f.__name__)


if __name__ == "__main__":
    main()
