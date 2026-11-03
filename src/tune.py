"""Підбір параметрів стратегії multi на частині датасетів «tune».

Ідея: найдовша операція — генерація кандидатів (карти країв, контури, прямі).
Тому для кожного зображення кандидати й їхні ознаки (контраст, опора сторін, площа,
кути, уточнений варіант) рахуються ОДИН раз з найм'якшими налаштуваннями, а потім
сотні комбінацій параметрів перевіряються майже миттєво — лише перерахунком оцінки.

Підсумкову якість обраних параметрів потрібно міряти на частині «test» (evaluate.py --split test),
яку цей скрипт не бачить.

Приклад (усе за один запуск):
    python src/tune.py "docs/datasets/All My Dataset" docs/datasets/SmartDoc/testDataset --frame-step 50

Частинами (якщо запуск не може тривати довго): спершу підготувати кандидатів шматками,
потім перебір по збереженому кешу:
    python src/tune.py <папки...> --parts 6 --part 0     # ... --part 5
    python src/tune.py <папки...> --parts 6 --search
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import filter_split, load_dataset  # noqa: E402
from evaluate import polygon_iou  # noqa: E402
from scanner import (ScanConfig, _inside_fraction, collect_candidates, is_plausible,  # noqa: E402
                     order_points, preprocess, quad_features, refine_quad, score_from_features)

ROOT = Path(__file__).resolve().parent.parent

# Сітка параметрів, що перебираються
GRID = {
    "approx_eps": [0.02, 0.03],
    "min_angle": [30.0, 40.0, 50.0],
    "side_support_min": [0.0, 0.3, 0.5],
    "line_side_min": [0.5, 0.7, 0.85],
    "line_weight": [0.5, 0.7, 1.0],
    "line_inside_thr": [0.9, 1.01],   # 1.01 = правило «прямі всередині контура» вимкнене
    "score_area_pow": [0.25, 0.5],
    "refine": [False, True],
}
EPS_LIST = tuple(GRID["approx_eps"])


def collect_with_eps(small, cfg):
    """Варіант collect_candidates, що позначає, з яким epsilon знайдено контур."""
    out = []
    for eps in EPS_LIST:
        c = replace(cfg, approx_eps=eps, use_lines=False)
        for src, q, e in collect_candidates(small, c):
            out.append((src, q, e, eps))
    if cfg.use_lines:
        for src, q, e in collect_candidates(small, replace(cfg, approx_eps=EPS_LIST[0])):
            if src == "lines":
                out.append((src, q, e, None))
    return out


def prepare_eps(samples, base: ScanConfig):
    prepared = []
    loose = replace(base, line_side_min=min(GRID["line_side_min"]))
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        try:
            img = s.load()
        except Exception as e:
            print(f"  [!] {s.name}: {e}")
            continue
        small, _, ratio = preprocess(img, base)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        gt = s.corners / ratio
        cands = []
        for src, q, edges, eps in collect_with_eps(small, loose):
            q = order_points(q)
            contrast, sides, area = quad_features(small, lab, q, edges)
            cands.append(dict(src=src, eps=eps, contrast=contrast, sides=sides, area=area,
                              iou=polygon_iou(q, gt), iou_ref=polygon_iou(refine_quad(q, edges), gt),
                              quad=q, shape=small.shape))
        prepared.append((s.name, cands))
        if i % 25 == 0:
            print(f"  ...{i}/{len(samples)}  ({(time.time() - t0) / i:.2f} с/зобр.)")
    return prepared


def evaluate_config(prepared, cfg: ScanConfig) -> np.ndarray:
    """IoU обраного кандидата для кожного зображення при даних параметрах
    (та сама логіка вибору, що й у scanner.find_document_multi)."""
    out = []
    for _, cands in prepared:
        best_c, sc_c, best_l, sc_l = None, 0.0, None, 0.0
        for c in cands:
            if c["eps"] is not None and c["eps"] != cfg.approx_eps:
                continue
            if cfg.filter_quads and not c.setdefault(("pl", cfg.min_angle),
                                                     is_plausible(c["quad"], c["shape"], cfg)):
                continue
            sc = score_from_features(c["contrast"], c["sides"], c["area"], c["src"] == "lines", cfg)
            if c["src"] == "lines":
                if sc > sc_l:
                    best_l, sc_l = c, sc
            elif sc > sc_c:
                best_c, sc_c = c, sc
        best = best_c if sc_c >= sc_l else best_l
        if best is best_l and best_c is not None \
                and _inside_fraction(best_l["quad"], best_c["quad"]) >= cfg.line_inside_thr:
            best = best_c
        out.append(0.0 if best is None else (best["iou_ref"] if cfg.refine else best["iou"]))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Підбір параметрів на частині tune")
    ap.add_argument("folders", nargs="+", help="папки датасетів (Mendeley, SmartDoc, gt)")
    ap.add_argument("--frame-step", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="макс. зображень з кожного датасету")
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", default=str(ROOT / "docs" / "tuning_results.json"))
    ap.add_argument("--parts", type=int, default=0, help="поділити підготовку на N частин")
    ap.add_argument("--part", type=int, default=None, help="підготувати лише частину з цим номером")
    ap.add_argument("--search", action="store_true", help="лише перебір по кешу частин")
    ap.add_argument("--cache-dir", default=str(ROOT / "output" / "tune_cache"))
    a = ap.parse_args()

    base = ScanConfig()
    cache = Path(a.cache_dir)
    datasets = {}
    for folder in a.folders:
        name = Path(folder).name
        if a.search:
            datasets[name] = []
            for f in sorted(cache.glob(f"{name}_part*.pkl")):
                datasets[name] += pickle.loads(f.read_bytes())
            print(f"{name}: {len(datasets[name])} зображень з кешу")
            continue
        fmt, samples = load_dataset(Path(folder), "auto", a.frame_step)
        samples = [s for s in filter_split(samples, "tune") if s.corners is not None]
        if a.limit and len(samples) > a.limit:
            samples = random.Random(a.seed).sample(samples, a.limit)
        if a.part is not None:
            samples = samples[a.part::a.parts]
        print(f"{name}: {len(samples)} зображень tune — рахую кандидатів...")
        datasets[name] = prepare_eps(samples, base)
        if a.part is not None:
            cache.mkdir(parents=True, exist_ok=True)
            (cache / f"{name}_part{a.part:02d}.pkl").write_bytes(pickle.dumps(datasets[name]))
    if a.part is not None:
        print("Частину збережено в кеш.")
        return 0

    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))
    print(f"\nПеребір {len(combos)} комбінацій...")
    results = []
    for vals in combos:
        cfg = replace(base, **dict(zip(keys, vals)))
        per = {name: float((evaluate_config(prep, cfg) >= a.iou).mean()) for name, prep in datasets.items()}
        results.append((float(np.mean(list(per.values()))), per, dict(zip(keys, vals))))
    results.sort(key=lambda r: -r[0])

    cur = {k: getattr(base, k) for k in keys}
    cur_per = {n: float((evaluate_config(p, base) >= a.iou).mean()) for n, p in datasets.items()}
    print(f"\nПоточні параметри: {cur}\n  успіх: {cur_per}  середнє {np.mean(list(cur_per.values())):.1%}")
    print(f"\nТоп-10 (успіх IoU ≥ {a.iou}, середнє по датасетах):")
    for score, per, params in results[:10]:
        per_s = ", ".join(f"{n} {v:.1%}" for n, v in per.items())
        print(f"  {score:.1%}  [{per_s}]  {params}")

    # вплив кожного параметра: найкращий результат при кожному його значенні
    print("\nВплив параметрів (найкращий результат при фіксованому значенні):")
    for k in keys:
        row = []
        for v in GRID[k]:
            row.append(f"{v}: {max(r[0] for r in results if r[2][k] == v):.1%}")
        print(f"  {k:<18} " + "  ".join(row))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "iou_threshold": a.iou,
        "tune_sizes": {n: len(p) for n, p in datasets.items()},
        "current": {"params": cur, "success": cur_per},
        "best": {"params": results[0][2], "success": results[0][1]},
        "top10": [{"mean": r[0], "success": r[1], "params": r[2]} for r in results[:10]],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nЗбережено: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
