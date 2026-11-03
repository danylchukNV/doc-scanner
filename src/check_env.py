"""Перевірка середовища та тестових даних (етап 1).

Запуск:  python src/check_env.py
"""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    print(f"Python : {sys.version.split()[0]}")
    print(f"OpenCV : {cv2.__version__}")
    print(f"NumPy  : {np.__version__}")

    files = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in EXTS)
    print(f"\nТестових зображень у samples/: {len(files)}")
    bad = 0
    for p in files:
        # imdecode замість imread — коректно читає шляхи з кирилицею у Windows
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [!] {p.name}: не вдалося прочитати")
            bad += 1
        else:
            h, w = img.shape[:2]
            print(f"  [ok] {p.name}: {w}x{h}")

    if len(files) < 5:
        print("\nПотрібно щонайменше 5 зображень (див. docs/test_photos_checklist.md)")
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
