"""Оцінка точності на синтетичних даних з відомими кутами.

Метрика: середня відстань (px) між знайденими та справжніми кутами.
Кути поза кадром (випадок cut_off) не можуть бути знайдені — це очікувана невдача.

Запуск:  python src/evaluate.py
"""
from __future__ import annotations

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
    gt_path = SYN / "ground_truth.json"
    if not gt_path.exists():
        print("Спершу запустіть: python src/make_synthetic.py")
        return 2
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    cfg = ScanConfig()

    print(f"{'файл':<28}{'метод':<15}{'похибка, px':>12}  результат")
    print("-" * 68)
    passed = 0
    for name, corners in gt.items():
        res = scan(read_image(SYN / name), cfg)
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
