"""Алгоритм сканування документа.

Конвеєр:
    1. preprocess            — зменшення, grayscale, розмиття Гауса
    2. пошук країв           — Canny (+ морфологічне закриття)
    3. пошук 4 кутів         — контури -> approxPolyDP -> чотирикутник
    4. four_point_transform  — гомографія (корекція перспективи)
    5. enhance               — адаптивна бінаризація / CLAHE

Дві стратегії пошуку кутів (ScanConfig.strategy):
    simple — одна карта країв (Canny на grayscale з автопорогом), найбільший 4-кутник;
    multi  — (за замовчуванням)
             1) кандидати-контури з кількох карт країв (grayscale, кожен канал BGR,
                різні пороги, Otsu);
             2) кандидати з прямих ліній (LSD): 4-кутник з перетинів двох пар прямих —
                стійкий до перекриття частини сторони рукою чи предметом;
             3) фільтр правдоподібності (кути, площа, положення кутів);
             4) вибір найкращого за оцінкою «контраст × опора на краї × площа^p»
                (кандидати з прямих — з вагою line_weight; якщо переможець з прямих лежить
                усередині найкращого контура — обирається контур);
             5) (опційно, refine) уточнення сторін: пряма через точки країв уздовж кожної
                сторони (fitLine), кути — перетини сусідніх прямих.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    # --- стратегія multi (значення підібрані src/tune.py на частині tune, див. docs/tuning_results.json) ---
    use_lines: bool = True          # додаткові кандидати з прямих ліній (LSD)
    filter_quads: bool = True       # відкидати неправдоподібні 4-кутники
    min_angle: float = 50.0         # мін. внутрішній кут 4-кутника, градуси (макс = 180 - min)
    refine: bool = False            # уточнення сторін за точками країв (fitLine); на tune не дало приросту
    score_area_pow: float = 0.5     # вага площі в оцінці кандидата
    score_support_pow: float = 1.0  # вага опори на краї
    side_support_min: float = 0.0   # кожна сторона кандидата-контуру має лежати на краях хоча б на цю частку
    line_side_min: float = 0.7      # те саме для кандидатів з прямих (суворіше: прямі легко «склеїти» з чужих)
    line_weight: float = 0.5        # множник оцінки кандидатів з прямих
    line_per_group: int = 6         # скільки найдовших прямих кожного напрямку перебирати
    line_inside_thr: float = 0.9    # якщо 4-кутник з прямих на цю частку лежить усередині кращого
                                    # контура — обирається контур (прямі всередині = колонки, рамки)
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


def quad_angles(q: np.ndarray) -> np.ndarray:
    """Внутрішні кути 4-кутника в градусах."""
    out = []
    for i in range(4):
        v1, v2 = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        c = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        out.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.array(out)


def is_plausible(q: np.ndarray, shape, cfg: ScanConfig) -> bool:
    """Чи схожий 4-кутник на сфотографований аркуш.

    - кути в межах [min_angle, 180 - min_angle] (сильна перспектива дає ~50–130°);
    - площа від min_area до 95 % кадру (весь кадр — зазвичай рамка, а не документ);
    - кути не далі ніж 10 % розміру кадру за його межами.
    """
    h, w = shape[:2]
    q = order_points(q)
    a = quad_angles(q)
    if a.min() < cfg.min_angle or a.max() > 180 - cfg.min_angle:
        return False
    area = cv2.contourArea(q) / (h * w)
    if not (cfg.min_area_ratio <= area <= 0.95):
        return False
    m = 0.1 * max(h, w)
    return bool((q[:, 0] > -m).all() and (q[:, 0] < w + m).all()
                and (q[:, 1] > -m).all() and (q[:, 1] < h + m).all())


def side_supports(q: np.ndarray, dil_edges: np.ndarray, n: int = 40) -> np.ndarray:
    """Частка точок кожної сторони, що лежать на (розширених) краях. Точки поза кадром — не враховуються."""
    h, w = dil_edges.shape[:2]
    t = np.linspace(0.05, 0.95, n)[:, None]
    out = []
    for k in range(4):
        pp = q[k] + (q[(k + 1) % 4] - q[k]) * t
        inside = (pp[:, 0] >= 0) & (pp[:, 0] < w) & (pp[:, 1] >= 0) & (pp[:, 1] < h)
        if inside.sum() < n * 0.2:   # сторона майже вся поза кадром — не можемо перевірити
            out.append(1.0)
            continue
        xi, yi = pp[inside, 0].astype(int), pp[inside, 1].astype(int)
        out.append(float(dil_edges[yi, xi].mean()))
    return np.array(out)


def quad_features(small: np.ndarray, lab: np.ndarray, quad: np.ndarray,
                  edges: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Ознаки кандидата: (контраст, опора кожної сторони [4], частка площі кадру).

    контраст — відстань між середнім кольором (Lab) усередині 4-кутника та у смузі 15 px навколо;
    опора    — частка точок кожної сторони, що лежать на знайдених краях.
    """
    h, w = small.shape[:2]
    inside = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
    ring = cv2.dilate(inside, _kernel(15)) - inside
    if not inside.any() or not ring.any():
        return 0.0, np.zeros(4), 0.0
    contrast = float(np.linalg.norm(lab[inside > 0].mean(0) - lab[ring > 0].mean(0)))
    sides = side_supports(quad, cv2.dilate(edges, _kernel(3)) > 0)
    return contrast, sides, cv2.contourArea(quad) / (h * w)


def score_from_features(contrast: float, sides: np.ndarray, area: float, from_lines: bool,
                        cfg: ScanConfig) -> float:
    """Оцінка = контраст × (середня опора)^b × площа^p  (× line_weight для кандидатів з прямих).

    Кандидат відкидається (оцінка 0), якщо хоча б одна сторона має опору менше порогу:
    така сторона не лежить на реальній межі.
    """
    side_min = cfg.line_side_min if from_lines else cfg.side_support_min
    if sides.min() < side_min:
        return 0.0
    sc = contrast * float(sides.mean()) ** cfg.score_support_pow * area ** cfg.score_area_pow
    return sc * cfg.line_weight if from_lines else sc


def score_quad(small: np.ndarray, lab: np.ndarray, quad: np.ndarray, edges: np.ndarray,
               cfg: ScanConfig | None = None, from_lines: bool = False) -> float:
    cfg = cfg or ScanConfig()
    return score_from_features(*quad_features(small, lab, quad, edges), from_lines, cfg)


# --------------------------------------------------------------------------- #
# Кандидати з прямих ліній (LSD)
# --------------------------------------------------------------------------- #
def _line_through(seg: np.ndarray) -> np.ndarray:
    """Пряма через відрізок у однорідних координатах (a, b, c): ax + by + c = 0."""
    return np.cross([seg[0], seg[1], 1.0], [seg[2], seg[3], 1.0])


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None  # паралельні
    return p[:2] / p[2]


def detect_segments(small: np.ndarray, cfg: ScanConfig, max_lines: int = 40) -> np.ndarray:
    """Відрізки LSD на яскравості та кожному каналі BGR, без майже-дублікатів."""
    h, w = small.shape[:2]
    lsd = cv2.createLineSegmentDetector()
    k = cfg.blur_ksize | 1
    segs = []
    for ch in [cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), *cv2.split(small)]:
        found = lsd.detect(cv2.GaussianBlur(ch, (k, k), 0))[0]
        if found is not None:
            segs.append(found.reshape(-1, 4))
    if not segs:
        return np.empty((0, 4), np.float32)
    segs = np.vstack(segs)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = length >= 0.08 * min(h, w)
    segs, length = segs[keep], length[keep]
    out: list[np.ndarray] = []
    for i in np.argsort(-length):  # від найдовших; дублікати (той самий край на різних каналах) — геть
        s = segs[i]
        ang = np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180
        mid = (s[:2] + s[2:]) / 2
        dup = False
        for t in out:
            ta = np.degrees(np.arctan2(t[3] - t[1], t[2] - t[0])) % 180
            d_ang = min(abs(ang - ta), 180 - abs(ang - ta))
            line = _line_through(t)
            dist = abs(line[0] * mid[0] + line[1] * mid[1] + line[2]) / np.hypot(line[0], line[1])
            if d_ang < 4 and dist < 6:
                dup = True
                break
        if not dup:
            out.append(s)
        if len(out) >= max_lines:
            break
    return np.array(out, np.float32).reshape(-1, 4)


def line_quads(small: np.ndarray, cfg: ScanConfig, support_edges: np.ndarray,
               keep: int = 10) -> list[np.ndarray]:
    """4-кутники з перетинів двох пар прямих.

    Відрізки ділимо на дві групи за напрямком (k-means на подвоєному куті — так 0° і 180°
    вважаються одним напрямком), з кожної групи беремо пару прямих, кути — їхні перетини.
    Перший відбір — швидка оцінка опори на краї (вибірка точок уздовж периметра).
    """
    segs = detect_segments(small, cfg)
    if len(segs) < 4:
        return []
    h, w = small.shape[:2]
    ang = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])
    feats = np.stack([np.cos(2 * ang), np.sin(2 * ang)], 1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    cv2.setRNGSeed(0)  # k-means++ випадково обирає початкові центри — фіксуємо, щоб результат повторювався
    _, labels, _ = cv2.kmeans(feats, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    groups = []
    for g in (0, 1):
        idx = np.where(labels == g)[0]
        groups.append(idx[np.argsort(-length[idx])][:cfg.line_per_group])
    if min(len(g) for g in groups) < 2:
        return []

    lines = [_line_through(sg) for sg in segs]
    dil = cv2.dilate(support_edges, _kernel(3)) > 0
    min_sep = 0.1 * min(h, w)
    scored = []
    for i1 in range(len(groups[0])):
        for i2 in range(i1 + 1, len(groups[0])):
            a1, a2 = groups[0][i1], groups[0][i2]
            for j1 in range(len(groups[1])):
                for j2 in range(j1 + 1, len(groups[1])):
                    b1, b2 = groups[1][j1], groups[1][j2]
                    pts = [_intersect(lines[a1], lines[b1]), _intersect(lines[a1], lines[b2]),
                           _intersect(lines[a2], lines[b2]), _intersect(lines[a2], lines[b1])]
                    if any(p is None for p in pts):
                        continue
                    q = order_points(np.array(pts, np.float32))
                    if np.linalg.norm(q[0] - q[1]) < min_sep or np.linalg.norm(q[1] - q[2]) < min_sep:
                        continue
                    if not cv2.isContourConvex(q.reshape(-1, 1, 2)) or not is_plausible(q, small.shape, cfg):
                        continue
                    sides = side_supports(q, dil, 30)
                    if sides.min() < cfg.line_side_min:
                        continue
                    area = cv2.contourArea(q) / (h * w)
                    scored.append((sides.mean() * area ** cfg.score_area_pow, q))
    scored.sort(key=lambda x: -x[0])
    return [q for _, q in scored[:keep]]


# --------------------------------------------------------------------------- #
# Уточнення сторін
# --------------------------------------------------------------------------- #
def refine_quad(q: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Уточнює 4-кутник: для кожної сторони бере точки країв у вузькій смузі вздовж неї,
    проводить пряму методом найменших квадратів з робастною функцією Хубера (cv2.fitLine),
    нові кути — перетини сусідніх прямих. Якщо точок замало або результат дивний — лишає як є.
    """
    q = order_points(q)
    h, w = edges.shape[:2]
    ys, xs = np.nonzero(edges)
    if len(xs) < 20:
        return q
    pts = np.stack([xs, ys], 1).astype(np.float32)
    band = max(3.0, 0.012 * np.hypot(h, w))
    new_lines = []
    for k in range(4):
        p0, p1 = q[k], q[(k + 1) % 4]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        d = d / (L + 1e-9)
        n = np.array([-d[1], d[0]])
        rel = pts - p0
        t = rel @ d / (L + 1e-9)
        dist = np.abs(rel @ n)
        sel = pts[(dist <= band) & (t > 0.1) & (t < 0.9)]
        if len(sel) >= max(15, 0.3 * L):
            vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            new_lines.append(np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0]))
        else:
            new_lines.append(_line_through(np.array([*p0, *p1])))
    corners = []
    for k in range(4):
        p = _intersect(new_lines[k - 1], new_lines[k])
        if p is None:
            return q
        corners.append(p)
    r = order_points(np.array(corners, np.float32))
    if not cv2.isContourConvex(r.reshape(-1, 1, 2)) or np.abs(r - q).max() > 3 * band:
        return q
    return r


def collect_candidates(small: np.ndarray, cfg: ScanConfig,
                       eps_list: tuple[float, ...] | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Усі кандидати: (джерело, 4-кутник, карта країв для оцінки опори)."""
    maps = edge_maps(small, cfg)
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    for eps in eps_list or (cfg.approx_eps,):
        c = cfg if eps == cfg.approx_eps else replace(cfg, approx_eps=eps)
        for name, edges in maps.items():
            for q in find_quads(edges, c):
                candidates.append((name, q, edges))
    if cfg.use_lines:
        union = maps["bgr_low"] | maps["gray_low"]
        for q in line_quads(small, cfg, union):
            candidates.append(("lines", q, union))
    return candidates


def _inside_fraction(inner: np.ndarray, outer: np.ndarray) -> float:
    """Яка частка площі inner лежить усередині outer."""
    a = inner.reshape(-1, 1, 2).astype(np.float32)
    b = outer.reshape(-1, 1, 2).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter / max(cv2.contourArea(a), 1e-6))


def find_document_multi(small: np.ndarray, cfg: ScanConfig
                        ) -> tuple[np.ndarray | None, str, list[np.ndarray], np.ndarray | None]:
    """Повертає (кути, метод, усі кандидати, карта країв переможця)."""
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidates = collect_candidates(small, cfg)
    if cfg.filter_quads:
        candidates = [c for c in candidates if is_plausible(c[1], small.shape, cfg)]
    all_quads = [c[1] for c in candidates]
    scores = [score_quad(small, lab, q, e, cfg, src == "lines") for src, q, e in candidates]
    if not candidates or max(scores) <= 0:
        return None, "none", all_quads, None
    best_i = int(np.argmax(scores))
    if candidates[best_i][0] == "lines":
        contour_i = [i for i, c in enumerate(candidates) if c[0] != "lines" and scores[i] > 0]
        if contour_i:
            ci = max(contour_i, key=lambda i: scores[i])
            if _inside_fraction(candidates[best_i][1], candidates[ci][1]) >= cfg.line_inside_thr:
                best_i = ci
    name, best, edges = candidates[best_i]
    best = order_points(best)
    if cfg.refine:
        best = refine_quad(best, edges)
    return best, f"multi:{name}", all_quads, edges


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
