"""Завантажувачі датасетів з розміткою кутів документа.

Кожен завантажувач повертає список Sample: назва, функція завантаження зображення
та еталонні кути (4x2, TL TR BR BL у пікселях) або None, якщо документа немає.

Підтримувані формати:
    gt        — папка з ground_truth.json (синтетика, extract_parquet.py)
    mendeley  — "A New Image Dataset for Document Corner Localization":
                поруч з кожним X.jpg лежить X.csv: x1,y1,...,x4,y4,модель_телефона
                (порядок кутів TL, BL, BR, TR)
    smartdoc  — SmartDoc 2015 Challenge 1: X.avi + X.gt.xml з кутами для кожного кадру
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from scanner import order_points, read_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    name: str
    load: Callable[[], np.ndarray]
    corners: np.ndarray | None


# --------------------------------------------------------------------------- #
def load_gt_folder(folder: Path) -> list[Sample]:
    gt = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
    return [Sample(name, (lambda p=folder / name: read_image(p)),
                   None if c is None else order_points(np.array(c)))
            for name, c in gt.items()]


# --------------------------------------------------------------------------- #
def load_mendeley(folder: Path) -> list[Sample]:
    samples = []
    # suffix у дублікатів виду "(631).jpg 2" — ".jpg 2", тож вони відсіюються автоматично
    for img_path in sorted(folder.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        csv_path = img_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as fh:
            row = next(csv.reader(fh), None)
        if not row or len(row) < 8:
            continue
        pts = np.array([float(v) for v in row[:8]]).reshape(4, 2)
        samples.append(Sample(img_path.name, (lambda p=img_path: read_image(p)), order_points(pts)))
    return samples


# --------------------------------------------------------------------------- #
def _read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)  # у розмітці кадри нумеруються з 1
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Не вдалося прочитати кадр {index} з {video}")
    return frame


def load_smartdoc(folder: Path, frame_step: int = 30) -> list[Sample]:
    """Бере кожен frame_step-й кадр кожного відео (сусідні кадри майже однакові)."""
    samples = []
    for xml_path in sorted(folder.rglob("*.gt.xml")):
        video = xml_path.with_name(xml_path.name.replace(".gt.xml", ".avi"))
        if not video.exists():
            continue
        root = ET.parse(xml_path).getroot()
        for fr in root.iter("frame"):
            idx = int(fr.get("index"))
            if (idx - 1) % frame_step or fr.get("rejected") == "true":
                continue
            p = {pt.get("name"): (float(pt.get("x")), float(pt.get("y"))) for pt in fr.iter("point")}
            pts = np.array([p["tl"], p["tr"], p["br"], p["bl"]], dtype=np.float32)
            name = f"{xml_path.parent.name}/{video.stem}#{idx:03d}"
            samples.append(Sample(name, (lambda v=video, i=idx: _read_frame(v, i)), pts))
    return samples


# --------------------------------------------------------------------------- #
def detect_format(folder: Path) -> str:
    if (folder / "ground_truth.json").exists():
        return "gt"
    if next(folder.rglob("*.gt.xml"), None):
        return "smartdoc"
    if next(folder.rglob("*.csv"), None):
        return "mendeley"
    raise ValueError(f"Не вдалося визначити формат датасету в {folder}")


def load_dataset(folder: Path, fmt: str = "auto", frame_step: int = 30) -> tuple[str, list[Sample]]:
    fmt = detect_format(folder) if fmt == "auto" else fmt
    if fmt == "gt":
        return fmt, load_gt_folder(folder)
    if fmt == "mendeley":
        return fmt, load_mendeley(folder)
    if fmt == "smartdoc":
        return fmt, load_smartdoc(folder, frame_step)
    raise ValueError(f"Невідомий формат: {fmt}")
