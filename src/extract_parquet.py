"""Витягує зображення та розмітку кутів із parquet-файлів DocCornerDataset (Hugging Face).

Пише .jpg/.png у папку і ground_truth.json у тому ж форматі, що й make_synthetic.py
({"файл": [[x,y] TL, TR, BR, BL]} у пікселях), тож результат одразу читає evaluate.py.

Приклади:
    python src/extract_parquet.py docs/datasets/DocCornerDataset/test/data-00006.parquet
    python src/extract_parquet.py docs/datasets/DocCornerDataset/test --limit 300
    python src/extract_parquet.py docs/datasets/DocCornerDataset/test --limit 50 --negatives

Файли читаються пакетами (pyarrow.iter_batches), тому вся таблиця з картинками
не завантажується в пам'ять одночасно.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    raise SystemExit("Потрібен pyarrow:  pip install pyarrow")

ROOT = Path(__file__).resolve().parent.parent
CORNER_COLS = ["corner_tl_x", "corner_tl_y", "corner_tr_x", "corner_tr_y",
               "corner_br_x", "corner_br_y", "corner_bl_x", "corner_bl_y"]


def image_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def image_bytes(cell) -> bytes | None:
    """Колонка image у HF-датасетах — словник {"bytes": ..., "path": ...} або сирі байти."""
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    return cell


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Витягнути зображення з parquet DocCornerDataset")
    p.add_argument("input", help=".parquet файл або папка з ними")
    p.add_argument("-o", "--out", default=str(ROOT / "datasets" / "doccorner_sample"),
                   help="куди зберегти (за замовчуванням datasets/doccorner_sample)")
    p.add_argument("--limit", type=int, default=0, help="випадкова вибірка N зображень (0 = всі)")
    p.add_argument("--negatives", action="store_true",
                   help="додати також зображення БЕЗ документа (is_negative)")
    p.add_argument("--seed", type=int, default=42, help="seed випадкової вибірки")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    src = Path(a.input)
    files = sorted(src.glob("*.parquet")) if src.is_dir() else [src]
    if not files or not files[0].exists():
        print(f"Не знайдено parquet: {src}")
        return 2

    # 1. Дешевий прохід: читаємо лише колонку is_negative, щоб вибрати рядки
    candidates: list[tuple[int, int]] = []  # (індекс файлу, індекс рядка)
    for fi, f in enumerate(files):
        neg = pq.read_table(f, columns=["is_negative"]).column("is_negative").to_pylist()
        candidates += [(fi, ri) for ri, n in enumerate(neg) if a.negatives or not n]
    print(f"Файлів: {len(files)}, придатних рядків: {len(candidates)}")

    if a.limit and a.limit < len(candidates):
        random.Random(a.seed).shuffle(candidates)
        candidates = candidates[:a.limit]
    wanted: dict[int, set[int]] = {}
    for fi, ri in candidates:
        wanted.setdefault(fi, set()).add(ri)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt: dict[str, list[list[float]] | None] = {}
    saved = skipped = 0

    # 2. Читаємо пакетами лише потрібні рядки
    for fi, f in enumerate(files):
        rows = wanted.get(fi)
        if not rows:
            continue
        pf = pq.ParquetFile(f)
        cols = ["image", "filename", "is_negative"] + CORNER_COLS
        offset = 0
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                if offset + i not in rows:
                    continue
                data = image_bytes(d["image"][i])
                if not data:
                    skipped += 1
                    continue
                name = f"{f.stem}_{offset + i:05d}{image_ext(data)}"
                (out / name).write_bytes(data)

                if d["is_negative"][i]:
                    gt[name] = None  # документа немає
                else:
                    pts = np.array([d[c][i] for c in CORNER_COLS], dtype=np.float64).reshape(4, 2)
                    if pts.max() <= 1.0:  # нормалізовані координати -> пікселі
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        h, w = img.shape[:2]
                        pts *= [w, h]
                    gt[name] = pts.round(1).tolist()
                saved += 1
            offset += batch.num_rows
        print(f"  {f.name}: готово")

    (out / "ground_truth.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    print(f"\nЗбережено {saved} зображень у {out}" + (f", пропущено {skipped}" if skipped else ""))
    print(f"Розмітка: {out / 'ground_truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
