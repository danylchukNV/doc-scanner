"""Алгоритм сканування документа.

Конвеєр:
    1. preprocess            — зменшення, grayscale, розмиття Гауса
    2. пошук країв           — Canny (+ морфологічне закриття)
    3. пошук 4 кутів         — контури -> approxPolyDP -> чотирикутник
    4. four_point_transform  — гомографія (корекція перспективи)
    5. enhance               — адаптивна бінаризація / CLAHE

Дві стратегії пошуку кутів (ScanConfig.strategy):
    simple — одна карта країв (Canny на grayscale з автопорогом), найбільший 4-кутник;
    multi  — кілька карт країв (grayscale, кожен канал BGR, різні пороги, Otsu),
             з усіх знайдених 4-кутників вибирається найкращий за оцінкою
             «контраст з фоном × опора на краї × площа^0.25». За замовчуванням.
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
    min_area_ratio: float = 0.05    # мінімальна площа документа від площі кадру
    strategy: str = "multi"         # multi | simple
    mode: str = "bw"                # bw | color | none
    block_size: int = 21            # вікно adaptiveThreshold (непарне)
    thresh_c: int = 10              # константа, що віднімається від локального середнього
    clahe_clip: float = 2.0


@dataclass
class ScanResult:
    ok: bool
    method: str                                  # simple: contour | approx_relaxed | min_area_rect; multi:<карта>; manual; none
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

    Точки сортуються за кутом навколо центру мас (за годинниковою стрілкою
    в координатах зображення), потім першою стає точка з найменшою x+y.
    Класичний варіант «мін/макс суми та різниці» дає дублікати, коли аркуш
    повернутий рівно на 45°, — цей підхід такої проблеми не має.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(angles)]          # -π..π: у системі з віссю y вниз — за годинниковою
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


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
# 3б. Стратегія multi: кілька карт країв + вибір найкращого кандидата
# --------------------------------------------------------------------------- #
def _kernel(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def edge_maps(small: np.ndarray, cfg: ScanConfig) -> dict[str, np.ndarray]:
    """Набір карт країв, кожна «бачить» межу аркуша в інших умовах.

    gray_auto  — Canny на яскравості з автопорогом (класичний варіант)
    gray_low   — Canny з низькими порогами: слабкий контраст (білий аркуш на світлому столі)
    bgr_high / bgr_low — Canny окремо на каналах B, G, R, об'єднання: межа, яка помітна
                 лише за кольором (білий папір на помаранчевому столі)
    otsu_*     — межі областей після бінаризації Otsu яскравості (gray) та насиченості (sat),
                 в обох полярностях: суцільний аркуш як «пляма»
    """
    k = cfg.blur_ksize | 1
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (k, k), 0)
    maps: dict[str, np.ndarray] = {}
    lo, hi = auto_canny_thresholds(gray, cfg.canny_sigma)
    maps["gray_auto"] = cv2.Canny(gray, lo, hi)
    maps["gray_low"] = cv2.Canny(gray, 10, 40)
    channels = [cv2.GaussianBlur(c, (k, k), 0) for c in cv2.split(small)]
    for name, (lo, hi) in (("bgr_high", (40, 120)), ("bgr_low", (20, 60))):
        e = np.zeros_like(gray)
        for c in channels:
            e |= cv2.Canny(c, lo, hi)
        maps[name] = e
    sat = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)
    gray7 = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    for name, c in (("gray", gray7), ("sat", sat)):
        _, b = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for suffix, bb in (("", b), ("_inv", 255 - b)):
            bb = cv2.morphologyEx(bb, cv2.MORPH_OPEN, _kernel(7))  # прибрати дрібні плями
            maps[f"otsu_{name}{suffix}"] = cv2.Canny(bb, 50, 150)
    return maps


def find_quads(edges: np.ndarray, cfg: ScanConfig, max_contours: int = 15) -> list[np.ndarray]:
    """Усі опуклі 4-кутники серед найбільших контурів карти країв."""
    if cfg.morph_ksize > 0:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _kernel(cfg.morph_ksize))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cfg.min_area_ratio * edges.shape[0] * edges.shape[1]
    quads = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]:
        if cv2.contourArea(c) < min_area:
            break
        for cand in (c, cv2.convexHull(c)):
            approx = cv2.approxPolyDP(cand, cfg.approx_eps * cv2.arcLength(cand, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx) \
                    and cv2.contourArea(approx) >= min_area:
                quads.append(approx.reshape(4, 2).astype(np.float32))
                break
    return quads


def score_quad(small: np.ndarray, lab: np.ndarray, quad: np.ndarray, edges: np.ndarray) -> float:
    """Оцінка кандидата = контраст × опора на краї × площа^0.25.

    контраст  — відстань між середнім кольором (Lab) усередині та у смузі навколо;
    опора     — частка периметра 4-кутника, що лежить на знайдених краях;
    площа     — частка кадру (у степені 0.25 — слабка перевага більших кандидатів).
    """
    h, w = small.shape[:2]
    inside = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
    ring = cv2.dilate(inside, _kernel(15)) - inside
    if not inside.any() or not ring.any():
        return 0.0
    contrast = float(np.linalg.norm(lab[inside > 0].mean(0) - lab[ring > 0].mean(0)))
    border = np.zeros((h, w), np.uint8)
    cv2.polylines(border, [quad.astype(np.int32)], True, 255, 3)
    support = float((cv2.dilate(edges, _kernel(3))[border > 0] > 0).mean())
    area = cv2.contourArea(quad) / (h * w)
    return contrast * support * area ** 0.25


def find_document_multi(small: np.ndarray, cfg: ScanConfig
                        ) -> tuple[np.ndarray | None, str, list[np.ndarray], np.ndarray | None]:
    """Повертає (кути, метод, усі кандидати, карта країв переможця)."""
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    best, best_score, best_name, best_edges = None, -1.0, "none", None
    all_quads: list[np.ndarray] = []
    for name, edges in edge_maps(small, cfg).items():
        for q in find_quads(edges, cfg):
            all_quads.append(q)
            sc = score_quad(small, lab, q, edges)
            if sc > best_score:
                best, best_score, best_name, best_edges = q, sc, name, edges
    if best is None:
        return None, "none", all_quads, None
    return order_points(best), f"multi:{best_name}", all_quads, best_edges


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


def detect_corners(img: np.ndarray, cfg: ScanConfig | None = None) -> tuple[np.ndarray | None, str]:
    """Лише пошук кутів (етапи 1–3), без корекції перспективи. Використовується для оцінки."""
    cfg = cfg or ScanConfig()
    small, blurred, ratio = preprocess(img, cfg)
    if cfg.strategy == "multi":
        small_corners, method, _, _ = find_document_multi(small, cfg)
    else:
        edges, _ = detect_edges(blurred, cfg)
        small_corners, method = find_document_corners(edges, cfg)
    return (None if small_corners is None else small_corners * ratio), method


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
        if cfg.strategy == "multi":
            small_corners, method, cands, win_edges = find_document_multi(small, cfg)
            if win_edges is not None:
                stages["3_edges"] = win_edges  # карта країв, що дала переможця
            vis = small.copy()
            for q in cands:  # усі кандидати — блакитним
                cv2.polylines(vis, [q.astype(np.int32)], True, (255, 200, 0), 2)
            stages["3b_candidates"] = vis
        else:
            small_corners, method = find_document_corners(edges, cfg)
        if small_corners is None:
            return ScanResult(False, "none", None, None, None, stages,
                              f"Документ не знайдено (Canny {lo}/{hi}). "
                              "Спробуйте ручний режим --manual або інші пороги.")
        corners = small_corners * ratio

    stages["4_contour"] = draw_corners(img, corners,
                                       (0, 255, 0) if method in ("contour", "manual") or method.startswith("multi")
                                       else (0, 165, 255))
    warped = four_point_transform(img, corners)
    stages["5_warped"] = warped
    output = enhance(warped, cfg)
    stages["6_output"] = output
    info = f"Canny {lo}/{hi}" if cfg.strategy == "simple" else "multi"
    return ScanResult(True, method, corners, warped, output, stages, f"OK ({method}, {info})")


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
