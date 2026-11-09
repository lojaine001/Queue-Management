"""Load the overhead fisheye datasets (WEPDTOF, CEPDOF) with their labels.

Expected layout (as extracted from the BU VIP lab archives):

    <root>/WEPDTOF/WEPDTOF/frames/<scene>/<scene>_000001.jpg
    <root>/WEPDTOF/WEPDTOF/annotations/<scene>.json
    <root>/WEPDTOF/WEPDTOF/ROI_maps/<scene>.png        white = area that is scored
    <root>/CEPDOF/CEPDOF/<Video>/<Video>_000001.jpg
    <root>/CEPDOF/CEPDOF/annotations/<Video>.json

Labels follow the CEPDOF format: COCO-like JSON, one annotation per person per frame,
bbox = [cx, cy, w, h, angle] (angle in degrees, clockwise) and a person_id that stays
the same for one person across frames. Only frames listed in the JSON are used, since
some scenes have unlabelled frames on disk.

Mac leftovers in the archives (__MACOSX, ._*, .DS_Store) are ignored.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = os.environ.get('FISHEYE_DATA', r'D:\data_new_system')
DATASETS = ('WEPDTOF', 'CEPDOF')

# WEPDTOF scenes closest to a shop checkout / sales floor.
SHOP_SCENES = ('convenience_store', 'jewelry_store', 'jewelry_store_2', 'printing_store',
               'repair_store', 'street_grocery', 'tech_store', 'empty_store')


@dataclass
class Scene:
    dataset: str
    name: str
    frames_dir: Path
    ann_path: Path
    roi_path: Path = None


@dataclass
class Frame:
    file_name: str
    path: Path
    width: int
    height: int
    # one row per labelled person: cx, cy, w, h, angle_deg
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), np.float32))
    person_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))


def _is_junk(name):
    return name.startswith('._') or name == '.DS_Store' or name == '__MACOSX'


def _find_dir(parent, name):
    """Case-insensitive lookup (CEPDOF has Edge_cases/ next to Edge_Cases.json)."""
    for p in parent.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def list_scenes(dataset, root=DEFAULT_ROOT):
    base = Path(root) / dataset / dataset
    ann_dir = base / 'annotations'
    if not ann_dir.is_dir():
        raise FileNotFoundError(f'{ann_dir} not found (set --root or FISHEYE_DATA)')
    scenes = []
    for ann in sorted(ann_dir.glob('*.json')):
        if _is_junk(ann.name):
            continue
        name = ann.stem
        if dataset == 'WEPDTOF':
            frames_dir = base / 'frames' / name
            roi = base / 'ROI_maps' / f'{name}.png'
            scenes.append(Scene(dataset, name, frames_dir, ann, roi if roi.exists() else None))
        else:
            frames_dir = _find_dir(base, name)
            if frames_dir is None:
                continue
            scenes.append(Scene(dataset, name, frames_dir, ann, None))
    return scenes


def get_scene(dataset, name, root=DEFAULT_ROOT):
    for s in list_scenes(dataset, root):
        if s.name.lower() == name.lower():
            return s
    raise KeyError(f'{dataset} has no scene {name!r}')


def load_frames(scene):
    """Labelled frames of a scene in time order (sorted by file name)."""
    with open(scene.ann_path, encoding='utf-8') as f:
        d = json.load(f)
    frames = {}
    for im in d['images']:
        frames[im['id']] = Frame(im['file_name'], scene.frames_dir / im['file_name'],
                                 int(im['width']), int(im['height']))
    per_image = {}
    for a in d['annotations']:
        per_image.setdefault(a['image_id'], []).append(a)
    for image_id, anns in per_image.items():
        fr = frames.get(image_id)
        if fr is None:
            continue
        fr.boxes = np.array([a['bbox'][:5] for a in anns], np.float32).reshape(-1, 5)
        fr.person_ids = np.array([a['person_id'] for a in anns], np.int64)
    return sorted(frames.values(), key=lambda fr: fr.file_name)


def load_roi(scene, width, height):
    """Boolean mask of the scored area, or None when the scene has no ROI map."""
    if scene.roi_path is None:
        return None
    m = cv2.imread(str(scene.roi_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m > 127


def box_corners(box):
    """4x2 corners of a rotated label box [cx, cy, w, h, angle_deg]."""
    cx, cy, w, h, a = (float(v) for v in box)
    return cv2.boxPoints(((cx, cy), (w, h), a))


def inside_box(box, x, y):
    return cv2.pointPolygonTest(box_corners(box).astype(np.float32), (float(x), float(y)), False) >= 0


def inside_roi(roi, x, y):
    if roi is None:
        return True
    h, w = roi.shape
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < w and 0 <= yi < h and bool(roi[yi, xi])
