"""Оцінка точності на синтетичних даних з відомими кутами.

Метрика: середня відстань (px) між знайденими та справжніми кутами.
Кути поза кадром (випадок cut_off) не можуть бути знайдені — це очікувана невдача.

Запуск:
    python src/evaluate.py                                # синтетичні тести
    python src/evaluate.py datasets/doccorner_sample      # будь-яка папка з ground_truth.json

Записи з null у ground_truth.json — зображення без документа: успіх, якщо сканер
нічого не знайшов (перевірка хибних спрацьовувань).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import ScanConfig, order_points, read_image, scan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SYN = ROOT / "samples" / "synthetic"
TOL_PX = 15  # допуск, щоб вважати результат успішним


def main() -> int:
    ap = argparse.ArgumentParser(description="Оцінка точності пошуку кутів")
    ap.add_argument("folder", nargs="?", default=str(SYN), help="папка із зображеннями та ground_truth.json")
    folder = Path(ap.parse_args().folder)
    gt_path = folder / "ground_truth.json"
    if not gt_path.exists():
        print(f"Немає {gt_path}. Для синтетики: python src/make_synthetic.py")
        return 2
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    cfg = ScanConfig()

    print(f"{'файл':<28}{'метод':<15}{'похибка, px':>12}  результат")
    print("-" * 68)
    passed = 0
    for name, corners in gt.items():
        res = scan(read_image(folder / name), cfg)
        if corners is None:  # негативний приклад
            good = not res.ok
            passed += good
            print(f"{name:<28}{res.method:<15}{'—':>12}  {'вірно: документа немає' if good else 'ХИБНЕ СПРАЦЮВАННЯ'}")
            continue
        if not res.ok:
            print(f"{name:<28}{res.method:<15}{'—':>12}  НЕ ЗНАЙДЕНО")
            continue
        err = float(np.linalg.norm(res.corners - order_points(np.array(corners)), axis=1).mean())
        good = err <= TOL_PX
        passed += good
        print(f"{name:<28}{res.method:<15}{err:>12.1f}  {'успіх' if good else 'неточно'}")
    print(f"\nУспішно (похибка ≤ {TOL_PX}px): {passed}/{len(gt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
