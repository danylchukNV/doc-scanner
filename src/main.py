"""CLI сканера документів.

Приклади:
    python src/main.py samples/01_a4_dark_top.jpg
    python src/main.py samples/02_a4_dark_angle.jpg --mode color --show
    python src/main.py samples/09_white_on_white.jpg --manual
    python src/main.py samples/ --debug            # пакетна обробка всієї папки
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import (ScanConfig, make_debug_montage, read_image, scan,  # noqa: E402
                     write_image)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сканер документів (OpenCV)")
    p.add_argument("input", help="зображення або папка із зображеннями")
    p.add_argument("-o", "--out", default=str(ROOT / "output"), help="папка для результатів")
    p.add_argument("--mode", choices=["bw", "color", "none"], default="bw",
                   help="підсилення: bw — адаптивна бінаризація, color — CLAHE, none — без")
    p.add_argument("--debug", action="store_true", help="зберегти зображення з усіма етапами")
    p.add_argument("--show", action="store_true", help="показати етапи у вікні")
    p.add_argument("--manual", action="store_true", help="вибрати 4 кути мишкою")
    g = p.add_argument_group("параметри алгоритму")
    g.add_argument("--strategy", choices=["multi", "simple"], default="multi",
                   help="multi — кілька карт країв і вибір кращого (за замовч.), simple — одна карта")
    g.add_argument("--canny", nargs=2, type=int, metavar=("LOW", "HIGH"),
                   help="ручні пороги Canny (за замовчуванням — автопоріг)")
    g.add_argument("--sigma", type=float, default=0.33, help="σ автопорогу Canny")
    g.add_argument("--blur", type=int, default=5, help="ядро розмиття Гауса")
    g.add_argument("--morph", type=int, default=5, help="ядро морфологічного закриття, 0 — вимк.")
    g.add_argument("--eps", type=float, default=0.02, help="epsilon approxPolyDP (частка периметра)")
    g.add_argument("--min-area", type=float, default=0.05, help="мін. частка площі кадру")
    g.add_argument("--block", type=int, default=21, help="blockSize adaptiveThreshold")
    g.add_argument("--C", type=int, default=10, help="константа C adaptiveThreshold")
    g.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="будь-який параметр ScanConfig, напр. --set use_lines=false min_angle=40")
    return p.parse_args()


def build_config(a: argparse.Namespace) -> ScanConfig:
    from evaluate import apply_overrides
    return apply_overrides(ScanConfig(
        blur_ksize=a.blur,
        canny_low=a.canny[0] if a.canny else None,
        canny_high=a.canny[1] if a.canny else None,
        canny_sigma=a.sigma,
        morph_ksize=a.morph,
        approx_eps=a.eps,
        min_area_ratio=a.min_area,
        mode=a.mode,
        strategy=a.strategy,
        block_size=a.block,
        thresh_c=a.C,
    ), a.set)


def select_corners_manually(img: np.ndarray) -> np.ndarray | None:
    """Вікно, де користувач клікає 4 кути. Enter — підтвердити, R — скинути, Esc — скасувати."""
    max_h = 800
    scale = min(1.0, max_h / img.shape[0])
    view = cv2.resize(img, None, fx=scale, fy=scale)
    pts: list[tuple[int, int]] = []
    win = "Click 4 corners | Enter=OK  R=reset  Esc=cancel"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = view.copy()
        for i, pt in enumerate(pts):
            cv2.circle(vis, pt, 6, (0, 0, 255), -1)
            if i:
                cv2.line(vis, pts[i - 1], pt, (0, 255, 0), 2)
        if len(pts) == 4:
            cv2.line(vis, pts[3], pts[0], (0, 255, 0), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(pts) == 4:
            break
        if key in (ord("r"), ord("R")):
            pts.clear()
        if key == 27:
            pts.clear()
            break
    cv2.destroyWindow(win)
    return np.array(pts, dtype=np.float32) / scale if len(pts) == 4 else None


def show_stages(stages: dict[str, np.ndarray]) -> None:
    montage = make_debug_montage(stages)
    scale = min(1.0, 1600 / montage.shape[1])
    cv2.imshow("Stages (any key to close)", cv2.resize(montage, None, fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_one(path: Path, cfg: ScanConfig, a: argparse.Namespace, out_dir: Path) -> dict:
    img = read_image(path)
    manual = select_corners_manually(img) if a.manual else None
    res = scan(img, cfg, manual_corners=manual)

    if res.ok:
        ext = ".png" if cfg.mode == "bw" else ".jpg"
        write_image(out_dir / f"{path.stem}_scan{ext}", res.output)
    if a.debug:
        write_image(out_dir / "debug" / f"{path.stem}_stages.jpg", make_debug_montage(res.stages))
    if a.show:
        show_stages(res.stages)

    status = "OK " if res.ok else "ERR"
    print(f"[{status}] {path.name}: {res.message}")
    size = f"{res.output.shape[1]}x{res.output.shape[0]}" if res.ok else ""
    return {"file": path.name, "ok": res.ok, "method": res.method, "output_size": size,
            "message": res.message}


def main() -> int:
    a = parse_args()
    cfg = build_config(a)
    src = Path(a.input)
    out_dir = Path(a.out)

    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    elif src.is_file():
        files = [src]
    else:
        print(f"Не знайдено: {src}")
        return 2
    if not files:
        print("Немає зображень для обробки")
        return 2

    rows = []
    for f in files:
        try:
            rows.append(process_one(f, cfg, a, out_dir))
        except Exception as e:  # одне погане фото не зупиняє пакетну обробку
            print(f"[ERR] {f.name}: {e}")
            rows.append({"file": f.name, "ok": False, "method": "error",
                         "output_size": "", "message": str(e)})

    if len(rows) > 1:
        report = out_dir / "report.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(report, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        ok = sum(r["ok"] for r in rows)
        print(f"\nУспішно: {ok}/{len(rows)}. Звіт: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
