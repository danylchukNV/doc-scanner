"""Веб-демонстрація сканера документів (Streamlit).

Запуск:
    pip install streamlit
    streamlit run app.py

Вкладки:
    Сканування  — фото (завантаження / камера / приклад), поворот / дзеркало ->
                  знайдений контур, результат, усі етапи;
                  параметри алгоритму змінюються в бічній панелі; ручне коригування кутів.
    Порівняння  — стратегії simple і multi на одному зображенні.
    Датасет     — випадкові приклади з розміченого датасету: еталон (зелений) vs знайдене (червоний), IoU.
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from datasets import filter_split, load_dataset  # noqa: E402
from evaluate import polygon_iou  # noqa: E402
from scanner import ScanConfig, detect_corners, draw_corners, scan  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGE_TITLES = {
    "1_original": "1. Оригінал",
    "2_gray_blur": "2. Сірий + розмиття Гауса",
    "3_edges": "3. Карта країв (переможця)",
    "3b_candidates": "3б. Усі кандидати",
    "4_contour": "4. Знайдений документ",
    "5_warped": "5. Корекція перспективи",
    "6_output": "6. Підсилення",
}

st.set_page_config(page_title="Сканер документів", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Допоміжне
# --------------------------------------------------------------------------- #
def to_rgb(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def limit_size(img: np.ndarray, max_side: int = 2000) -> np.ndarray:
    """Дуже великі фото зменшуємо — на якість пошуку не впливає (він працює на 500 px)."""
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


ROTATIONS = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


def orientation_ui(img: np.ndarray, key: str) -> tuple[np.ndarray, str]:
    """Кнопки повороту на 90° і дзеркального відображення. Стан — окремо для кожного фото."""
    rk, fk = f"rot_{key}", f"flip_{key}"
    st.session_state.setdefault(rk, 0)
    st.session_state.setdefault(fk, False)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 3])
    if c1.button("↺ 90°", key=f"l_{key}", help="Повернути проти годинникової стрілки"):
        st.session_state[rk] = (st.session_state[rk] - 1) % 4
    if c2.button("↻ 90°", key=f"r_{key}", help="Повернути за годинниковою стрілкою"):
        st.session_state[rk] = (st.session_state[rk] + 1) % 4
    if c3.button("⇋ Дзеркало", key=f"f_{key}", help="Відобразити по горизонталі"):
        st.session_state[fk] = not st.session_state[fk]
    if c4.button("Скинути", key=f"z_{key}"):
        st.session_state[rk], st.session_state[fk] = 0, False
    k, flip = st.session_state[rk], st.session_state[fk]
    if flip:
        img = cv2.flip(img, 1)
    if k:
        img = cv2.rotate(img, ROTATIONS[k])
    c5.caption(f"Поворот: {k * 90}°{' · дзеркально' if flip else ''}")
    return img, f"{k}{int(flip)}"


def example_files() -> list[Path]:
    out = []
    for folder in (ROOT / "samples", ROOT / "samples" / "synthetic"):
        if folder.exists():
            out += sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    return out


def encode_png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


@st.cache_resource(show_spinner="Читаю розмітку датасету...")
def cached_dataset(folder: str, split: str, frame_step: int):
    fmt, samples = load_dataset(Path(folder), "auto", frame_step)
    return fmt, [s for s in filter_split(samples, split) if s.corners is not None]


# --------------------------------------------------------------------------- #
# Бічна панель: параметри
# --------------------------------------------------------------------------- #
def sidebar_config() -> ScanConfig:
    st.sidebar.header("⚙️ Параметри")
    d = ScanConfig()
    strategy = st.sidebar.radio("Стратегія пошуку", ["multi", "simple"], horizontal=True,
                                help="multi — кілька карт країв + прямі + вибір кращого кандидата; "
                                     "simple — одна карта Canny на grayscale")
    mode = st.sidebar.selectbox("Підсилення результату", ["bw", "color", "none"],
                                format_func={"bw": "Ч/б (adaptive threshold)", "color": "Колір (CLAHE)",
                                             "none": "Без підсилення"}.get)
    cfg = replace(d, strategy=strategy, mode=mode)

    with st.sidebar.expander("Попередня обробка і краї", expanded=False):
        cfg.blur_ksize = st.slider("Ядро розмиття Гауса", 1, 15, d.blur_ksize, step=2)
        cfg.morph_ksize = st.slider("Ядро морфологічного закриття (0 — вимк.)", 0, 15, d.morph_ksize)
        cfg.approx_eps = st.slider("ε approxPolyDP (частка периметра)", 0.005, 0.08, d.approx_eps, 0.005)
        cfg.min_area_ratio = st.slider("Мін. площа документа (частка кадру)", 0.01, 0.5, d.min_area_ratio, 0.01)
        if strategy == "simple":
            auto = st.checkbox("Автопоріг Canny (за медіаною)", True)
            if auto:
                cfg.canny_sigma = st.slider("σ автопорогу", 0.05, 1.0, d.canny_sigma, 0.01)
            else:
                lo, hi = st.slider("Пороги Canny (низький, високий)", 0, 255, (50, 150))
                cfg.canny_low, cfg.canny_high = lo, hi
        else:
            cfg.canny_sigma = st.slider("σ автопорогу (карта gray_auto)", 0.05, 1.0, d.canny_sigma, 0.01)

    if strategy == "multi":
        with st.sidebar.expander("Стратегія multi", expanded=False):
            cfg.use_lines = st.checkbox("Кандидати з прямих (LSD)", d.use_lines)
            cfg.line_weight = st.slider("Вага кандидатів з прямих", 0.0, 1.5, d.line_weight, 0.05)
            cfg.line_inside_thr = st.slider("Прямі всередині контура → обрати контур (поріг)",
                                            0.5, 1.01, d.line_inside_thr, 0.01)
            cfg.filter_quads = st.checkbox("Фільтр правдоподібності", d.filter_quads)
            cfg.min_angle = st.slider("Мін. внутрішній кут, °", 10.0, 80.0, d.min_angle, 1.0)
            cfg.score_area_pow = st.slider("Вага площі в оцінці (степінь)", 0.0, 1.0, d.score_area_pow, 0.05)
            cfg.refine = st.checkbox("Уточнення сторін (fitLine)", d.refine)

    if mode == "bw":
        with st.sidebar.expander("Адаптивна бінаризація", expanded=False):
            cfg.block_size = st.slider("Розмір вікна (blockSize)", 3, 101, d.block_size, step=2)
            cfg.thresh_c = st.slider("Константа C", -10, 40, d.thresh_c)
    elif mode == "color":
        with st.sidebar.expander("CLAHE", expanded=False):
            cfg.clahe_clip = st.slider("clipLimit", 0.5, 8.0, d.clahe_clip, 0.5)
    return cfg


# --------------------------------------------------------------------------- #
# Вибір зображення
# --------------------------------------------------------------------------- #
def pick_image() -> tuple[np.ndarray | None, str]:
    src = st.radio("Джерело зображення", ["Приклад", "Завантажити файл", "Камера"], horizontal=True)
    if src == "Завантажити файл":
        f = st.file_uploader("Фото документа", type=["jpg", "jpeg", "png", "bmp", "webp"])
        return (decode(f.getvalue()), f.name) if f else (None, "")
    if src == "Камера":
        f = st.camera_input("Зробіть фото документа")
        return (decode(f.getvalue()), "camera.png") if f else (None, "")
    files = example_files()
    if not files:
        st.info("Немає прикладів у samples/. Згенеруйте: `python src/make_synthetic.py`")
        return None, ""
    p = st.selectbox("Приклад", files, format_func=lambda p: str(p.relative_to(ROOT / "samples")))
    return decode(p.read_bytes()), p.name


# --------------------------------------------------------------------------- #
# Вкладка «Сканування»
# --------------------------------------------------------------------------- #
def manual_corners_ui(img: np.ndarray, auto: np.ndarray | None, key: str) -> np.ndarray | None:
    with st.expander("✋ Ручне коригування кутів", expanded=False):
        use = st.checkbox("Використати ручні кути", key=f"use_{key}")
        h, w = img.shape[:2]
        init = auto if auto is not None else np.array([[0.1 * w, 0.1 * h], [0.9 * w, 0.1 * h],
                                                       [0.9 * w, 0.9 * h], [0.1 * w, 0.9 * h]])
        pts = []
        cols = st.columns(4)
        for i, (name, col) in enumerate(zip(("TL", "TR", "BR", "BL"), cols)):
            with col:
                st.markdown(f"**{name}**")
                x = st.slider("x, %", 0.0, 100.0, float(np.clip(init[i][0] / w * 100, 0, 100)), 0.5,
                              key=f"x{i}_{key}")
                y = st.slider("y, %", 0.0, 100.0, float(np.clip(init[i][1] / h * 100, 0, 100)), 0.5,
                              key=f"y{i}_{key}")
                pts.append([x / 100 * w, y / 100 * h])
        st.caption("Початкові значення — кути, знайдені автоматично. Зміна фото скидає повзунки.")
        return np.array(pts, np.float32) if use else None


def tab_scan(cfg: ScanConfig) -> None:
    img, name = pick_image()
    if img is None:
        return
    img = limit_size(img)
    img, orient = orientation_ui(img, f"scan_{name}_{img.shape}")
    key = f"{name}_{img.shape}_{orient}"

    t0 = time.perf_counter()
    res = scan(img, cfg)
    ms = (time.perf_counter() - t0) * 1000

    manual = manual_corners_ui(img, res.corners, key)
    if manual is not None:
        res = scan(img, cfg, manual_corners=manual)

    c1, c2, c3 = st.columns(3)
    c1.metric("Результат", "знайдено" if res.ok else "не знайдено")
    c2.metric("Метод (джерело переможця)", res.method)
    c3.metric("Час", f"{ms:.0f} мс")

    left, right = st.columns(2)
    with left:
        st.subheader("Знайдений документ")
        st.image(to_rgb(res.stages.get("4_contour", img)), width="stretch")
    with right:
        st.subheader("Скан")
        if res.ok:
            st.image(to_rgb(res.output), width="stretch")
            st.download_button("⬇️ Завантажити скан (PNG)", encode_png(res.output),
                               file_name=f"{Path(name).stem}_scan.png", mime="image/png")
        else:
            st.warning(res.message + "  \nСпробуйте змінити параметри або ручне коригування кутів.")

    st.subheader("Етапи обробки")
    stages = [(k, v) for k, v in res.stages.items()]
    for row in range(0, len(stages), 3):
        cols = st.columns(3)
        for col, (k, v) in zip(cols, stages[row:row + 3]):
            col.image(to_rgb(v), caption=STAGE_TITLES.get(k, k), width="stretch")


# --------------------------------------------------------------------------- #
# Вкладка «Порівняння»
# --------------------------------------------------------------------------- #
def tab_compare(cfg: ScanConfig) -> None:
    st.caption("Та сама картинка, дві стратегії пошуку. Параметри з бічної панелі застосовуються до обох.")
    img, name = pick_image_compare()
    if img is None:
        return
    img = limit_size(img)
    img, _ = orientation_ui(img, f"cmp_{name}_{img.shape}")
    cols = st.columns(2)
    for col, strategy in zip(cols, ("simple", "multi")):
        c = replace(cfg, strategy=strategy)
        t0 = time.perf_counter()
        res = scan(img, c)
        ms = (time.perf_counter() - t0) * 1000
        with col:
            st.subheader(strategy)
            st.caption(f"{res.method} · {ms:.0f} мс")
            st.image(to_rgb(res.stages.get("4_contour", img)), width="stretch")
            if res.ok:
                st.image(to_rgb(res.output), width="stretch")
            else:
                st.warning("Документ не знайдено")


def pick_image_compare() -> tuple[np.ndarray | None, str]:
    files = example_files()
    up = st.file_uploader("Своє фото (необов'язково)", type=["jpg", "jpeg", "png", "bmp", "webp"],
                          key="cmp_up")
    if up:
        return decode(up.getvalue()), up.name
    if not files:
        return None, ""
    p = st.selectbox("Або приклад", files, key="cmp_sel",
                     format_func=lambda p: str(p.relative_to(ROOT / "samples")))
    return decode(p.read_bytes()), p.name


# --------------------------------------------------------------------------- #
# Вкладка «Датасет»
# --------------------------------------------------------------------------- #
def tab_dataset(cfg: ScanConfig) -> None:
    st.caption("Приклади з розміченого датасету. **Зелений** — еталонні кути, **червоний** — знайдені "
               "сканером. Успіх — IoU ≥ 0.9.")
    defaults = {
        "Mendeley (All My Dataset)": ROOT / "docs" / "datasets" / "All My Dataset",
        "SmartDoc 2015": ROOT / "docs" / "datasets" / "SmartDoc" / "testDataset",
        "DocCorner (витягнутий)": ROOT / "datasets" / "doccorner_sample",
        "Синтетика": ROOT / "samples" / "synthetic",
    }
    available = {k: v for k, v in defaults.items() if v.exists()}
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        choice = st.selectbox("Датасет", list(available) + ["Інша папка..."])
        folder = available.get(choice)
        if folder is None:
            folder = Path(st.text_input("Шлях до папки з розміткою", str(ROOT / "docs" / "datasets")))
    with c2:
        split = st.selectbox("Частина", ["test", "tune", "all"])
    with c3:
        n = st.slider("Скільки прикладів", 1, 12, 6)
    if not folder.exists():
        st.info("Папку не знайдено. Датасети лежать у docs/datasets/ (див. README).")
        return
    try:
        fmt, samples = cached_dataset(str(folder), split, 50)
    except Exception as e:  # noqa: BLE001
        st.error(f"Не вдалося прочитати датасет: {e}")
        return
    if not samples:
        st.info("У цій частині немає зображень з розміткою.")
        return

    if st.button("🎲 Інші випадкові приклади") or "ds_seed" not in st.session_state:
        st.session_state.ds_seed = random.randint(0, 10 ** 6)
    picked = random.Random(st.session_state.ds_seed).sample(samples, min(n, len(samples)))
    st.caption(f"Формат: {fmt}, зображень у частині «{split}»: {len(samples)}")

    ious = []
    cols = st.columns(3)
    for i, s in enumerate(picked):
        img = s.load()
        pred, method = detect_corners(img, cfg)
        iou = polygon_iou(pred, s.corners) if pred is not None else 0.0
        ious.append(iou)
        vis = draw_corners(img, s.corners, (0, 255, 0))
        if pred is not None:
            vis = draw_corners(vis, pred, (0, 0, 255))
        mark = "✅" if iou >= 0.9 else ("⚠️" if iou >= 0.5 else "❌")
        with cols[i % 3]:
            st.image(to_rgb(limit_size(vis, 900)), width="stretch",
                     caption=f"{mark} IoU {iou:.2f} · {method} · {s.name}")
    st.metric("Успіх на цих прикладах", f"{sum(i >= 0.9 for i in ious)}/{len(ious)}",
              help="Повна оцінка на всьому датасеті: python src/evaluate.py ...")


# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📄 Інтелектуальний сканер документів")
    st.caption("Canny / контури / LSD-прямі → вибір кандидата → гомографія → адаптивна бінаризація")
    cfg = sidebar_config()
    t1, t2, t3 = st.tabs(["Сканування", "Порівняння simple / multi", "Датасет"])
    with t1:
        tab_scan(cfg)
    with t2:
        tab_compare(cfg)
    with t3:
        tab_dataset(cfg)


main()
