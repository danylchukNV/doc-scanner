"""Алгоритм сканування документа.

Конвеєр:
    1. preprocess            — зменшення, grayscale, розмиття Гауса
    2. detect_edges          — детектор Canny + морфологічне закриття
    3. find_document_corners — контури -> approxPolyDP -> 4 кути
    4. four_point_transform  — гомографія (корекція перспективи)
    5. enhance               — адаптивна бінаризація / CLAHE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Параметри
# --------------------------------------------------------------------------- #
@dataclass
class ScanConfig:
    work_height: int = 500          # висота зменшеного зображення для пошуку контуру
    blur_ksize: int = 5             # розмір ядра Гауса (непарний)
    canny_low: int | None = None    # None -> автопоріг за медіаною яскравості
    canny_high: int | None = None
    canny_sigma: float = 0.33       # ширина діапазону автопорогу
    morph_ksize: int = 5            # ядро морфологічного закриття (0 = вимкнено)
    approx_eps: float = 0.02        # epsilon approxPolyDP як частка периметра
    min_area_ratio: float = 0.10    # мінімальна площа документа від площі кадру
    mode: str = "bw"                # bw | color | none
    block_size: int = 21            # вікно adaptiveThreshold (непарне)
    thresh_c: int = 10              # константа, що віднімається від локального середнього
    clahe_clip: float = 2.0


@dataclass
class ScanResult:
    ok: bool
    method: str                                  # contour | approx_relaxed | min_area_rect | manual | none
    corners: np.ndarray | None                   # 4x2 у координатах ОРИГІНАЛУ (TL, TR, BR, BL)
    warped: np.ndarray | None                    # вирівняне кольорове зображення
    output: np.ndarray | None                    # після підсилення
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    message: str = ""


# --------------------------------------------------------------------------- #
# Ввід / вивід (через imdecode/imencode — працює з кирилицею в шляхах на Windows)
# --------------------------------------------------------------------------- #
def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не вдалося прочитати зображення: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"Не вдалося зберегти: {path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------------- #
# 1. Попередня обробка
# --------------------------------------------------------------------------- #
def preprocess(img: np.ndarray, cfg: ScanConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Повертає (зменшене BGR, розмите grayscale, коефіцієнт масштабу оригінал/зменшене)."""
    h = img.shape[0]
    ratio = h / cfg.work_height
    small = cv2.resize(img, (int(img.shape[1] / ratio), cfg.work_height),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k = cfg.blur_ksize | 1  # гарантуємо непарність
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    return small, blurred, ratio


# --------------------------------------------------------------------------- #
# 2. Пошук країв
# --------------------------------------------------------------------------- #
def auto_canny_thresholds(gray: np.ndarray, sigma: float) -> tuple[int, int]:
    """Пороги Canny навколо медіани яскравості: [(1-σ)·m, (1+σ)·m]."""
    m = float(np.median(gray))
    low = int(max(0, (1.0 - sigma) * m))
    high = int(min(255, (1.0 + sigma) * m))
    return low, high


def detect_edges(gray: np.ndarray, cfg: ScanConfig) -> tuple[np.ndarray, tuple[int, int]]:
    if cfg.canny_low is None or cfg.canny_high is None:
        low, high = auto_canny_thresholds(gray, cfg.canny_sigma)
    else:
        low, high = cfg.canny_low, cfg.canny_high
    edges = cv2.Canny(gray, low, high)
    if cfg.morph_ksize > 0:
        # Закриття = дилатація + ерозія: з'єднує розірвані фрагменти межі аркуша
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_ksize, cfg.morph_ksize))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges, (low, high)


# --------------------------------------------------------------------------- #
# 3. Пошук 4 кутів документа
# --------------------------------------------------------------------------- #
def order_points(pts: np.ndarray) -> np.ndarray:
    """Впорядковує 4 точки як TL, TR, BR, BL.

    TL має найменшу суму x+y, BR — найбільшу;
    TR має найменшу різницю y-x, BL — найбільшу.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_document_corners(edges: np.ndarray, cfg: ScanConfig) -> tuple[np.ndarray | None, str]:
    """Шукає найбільший опуклий 4-кутник. Повертає (кути у зменшених координатах, метод)."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "none"

    frame_area = edges.shape[0] * edges.shape[1]
    min_area = cfg.min_area_ratio * frame_area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        if cv2.contourArea(c) < min_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, cfg.approx_eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx), "contour"

    largest = contours[0]
    if cv2.contourArea(largest) < min_area:
        return None, "none"

    # Резерв 1: поступово збільшуємо epsilon для контуру та його опуклої оболонки.
    # Допомагає, коли до аркуша «прилипли» сторонні предмети або загнуто кут:
    # дрібні виступи відкидаються, лишаються 4 головні вершини.
    hull = cv2.convexHull(largest)
    for eps in np.arange(cfg.approx_eps + 0.01, 0.10, 0.01):
        for cand in (largest, hull):
            approx = cv2.approxPolyDP(cand, eps * cv2.arcLength(cand, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx) \
                    and cv2.contourArea(approx) >= min_area:
                return order_points(approx), "approx_relaxed"

    # Резерв 2: мінімальний повернутий прямокутник навколо найбільшого контуру
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return order_points(box), "min_area_rect"


# --------------------------------------------------------------------------- #
# 4. Корекція перспективи
# --------------------------------------------------------------------------- #
def four_point_transform(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    # Матриця гомографії 3x3 з 4 пар відповідних точок (8 ступенів свободи)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- #
# 5. Підсилення
# --------------------------------------------------------------------------- #
def enhance(warped: np.ndarray, cfg: ScanConfig) -> np.ndarray:
    if cfg.mode == "none":
        return warped
    if cfg.mode == "color":
        # CLAHE лише на каналі яскравості L (Lab), кольори не спотворюються
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    # bw: поріг для кожного пікселя = середнє (Гаусове) у вікні block_size мінус C
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, cfg.block_size | 1, cfg.thresh_c)


# --------------------------------------------------------------------------- #
# Повний конвеєр
# --------------------------------------------------------------------------- #
def draw_corners(img: np.ndarray, corners: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    vis = img.copy()
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    t = max(2, img.shape[0] // 250)
    cv2.polylines(vis, [pts], True, color, t)
    for (x, y), name in zip(corners.astype(int), ("TL", "TR", "BR", "BL")):
        cv2.circle(vis, (x, y), t * 3, (0, 0, 255), -1)
        cv2.putText(vis, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, t * 0.3, (0, 0, 255), t)
    return vis


def scan(img: np.ndarray, cfg: ScanConfig | None = None,
         manual_corners: np.ndarray | None = None) -> ScanResult:
    """Сканує документ. manual_corners — 4 точки в координатах оригіналу (ручний режим)."""
    cfg = cfg or ScanConfig()
    stages: dict[str, np.ndarray] = {"1_original": img}

    small, blurred, ratio = preprocess(img, cfg)
    stages["2_gray_blur"] = blurred
    edges, (lo, hi) = detect_edges(blurred, cfg)
    stages["3_edges"] = edges

    if manual_corners is not None:
        corners, method = order_points(manual_corners), "manual"
    else:
        small_corners, method = find_document_corners(edges, cfg)
        if small_corners is None:
            return ScanResult(False, "none", None, None, None, stages,
                              f"Документ не знайдено (Canny {lo}/{hi}). "
                              "Спробуйте ручний режим --manual або інші пороги.")
        corners = small_corners * ratio

    stages["4_contour"] = draw_corners(img, corners,
                                       (0, 255, 0) if method in ("contour", "manual") else (0, 165, 255))
    warped = four_point_transform(img, corners)
    stages["5_warped"] = warped
    output = enhance(warped, cfg)
    stages["6_output"] = output
    return ScanResult(True, method, corners, warped, output, stages,
                      f"OK ({method}, Canny {lo}/{hi})")


def make_debug_montage(stages: dict[str, np.ndarray], tile_h: int = 360) -> np.ndarray:
    """Склеює всі етапи в одне зображення для звіту / захисту."""
    tiles = []
    for name, im in stages.items():
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        scale = tile_h / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), tile_h))
        im = cv2.copyMakeBorder(im, 30, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.putText(im, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(im)
    # два ряди по три
    rows = [tiles[i:i + 3] for i in range(0, len(tiles), 3)]
    row_imgs = [np.hstack(r) for r in rows]
    w = max(r.shape[1] for r in row_imgs)
    row_imgs = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                                   value=(40, 40, 40)) for r in row_imgs]
    return np.vstack(row_imgs)
