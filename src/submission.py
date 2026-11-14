"""Turn predicted masks into the competition's per-filament CSV submission.

The model predicts a single binary mask per image, but the competition scores
individual filaments. Connected components bridge the two: each component
becomes one submitted filament.
"""
import csv
from pathlib import Path

import cv2 as cv
import numpy as np
from pycocotools import mask as mask_util

SUBMISSION_COLUMNS = ["filament_id", "segmentation_rle"]


def mask_to_instances(mask: np.ndarray, min_area: int = 50) -> list[np.ndarray]:
    """Split a binary mask into per-filament masks, dropping specks.

    Uses 8-connectivity because barbs meet the filament spine diagonally and
    4-connectivity would shear them off into separate components.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels = cv.connectedComponents(mask, connectivity=8)
    instances = []
    for label in range(1, count):
        instance = (labels == label).astype(np.uint8)
        if instance.sum() >= min_area:
            instances.append(instance)
    return instances


def unpack_point(point) -> tuple[float, float, int]:
    """(high, low, min_area) from an operating point.

    A point is (threshold, min_area), a plain cut, or (high, min_area, low) for
    hysteresis. The two-value form is the three-value form with low == high.
    """
    if len(point) == 2:
        threshold, min_area = point
        return threshold, threshold, min_area
    high, min_area, low = point
    if low > high:
        raise ValueError(f"hysteresis needs low <= high, got low={low} high={high}")
    return high, low, min_area


def point_name(point) -> str:
    """Short label for an operating point, as used in log columns."""
    high, low, min_area = unpack_point(point)
    if low == high:
        return f"t{high}_a{min_area}"
    return f"t{high}_l{low}_a{min_area}"


def probability_to_instances(probability: np.ndarray, point, merge_gap: int = 0) -> list[np.ndarray]:
    """Per-filament masks from a probability map at one operating point.

    With `merge_gap`, kept pieces within that many px of one another are then
    joined into one filament (see merge_fragments). The area and confidence
    filters apply to the pieces first, so a speck that was dropped cannot
    bridge two real fragments.

    Hysteresis thresholding: a filament is a connected region above `low` that
    reaches `high` somewhere. The best plain threshold sits at the top of the
    range tried (0.95), which keeps only the confident cores of filaments and
    so cuts them short; growing those cores down to a lower threshold recovers
    their extent without admitting the faint blobs that never reach `high`.
    Requiring a confident pixel is also a per-filament confidence filter, which
    is what min_area otherwise has to stand in for - and min_area alone costs
    0.115 PQ even on perfect masks at 600 px, because a fifth of the annotated
    filaments are smaller than that.
    """
    high, low, min_area = unpack_point(point)
    mask = (np.asarray(probability) >= low).astype(np.uint8)
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return []
    if low == high:
        peaks = np.full(count, np.inf)
    else:
        # only foreground pixels can raise a component's peak; there are few of them
        inside = mask.astype(bool)
        peaks = np.zeros(count)
        np.maximum.at(peaks, labels[inside], np.asarray(probability, dtype=np.float64)[inside])
    instances = [
        (labels == label).astype(np.uint8)
        for label in range(1, count)
        if stats[label, cv.CC_STAT_AREA] >= min_area and peaks[label] >= high
    ]
    return merge_fragments(instances, merge_gap)


def merge_fragments(instances: list[np.ndarray], gap: int) -> list[np.ndarray]:
    """Join instances that lie within `gap` px of one another into one filament.

    The network often predicts one filament as a chain of pieces, and each piece
    alone covers too little of the annotated outline to reach IoU 0.5: a third
    of the near misses on validation were overlapped by more than one
    prediction. Only the original pixels are kept - growing the masks instead
    cost 0.03 PQ per pixel, because the predicted widths are already right.
    """
    if gap <= 0 or len(instances) < 2:
        return instances
    # dilating every piece by half the gap makes pieces at most `gap` apart touch
    radius = (gap + 1) // 2
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    union = np.maximum.reduce(instances)
    _, groups = cv.connectedComponents(cv.dilate(union, kernel), connectivity=8)
    merged: dict[int, np.ndarray] = {}
    for instance in instances:
        group = int(groups[instance.astype(bool)][0])
        merged[group] = instance if group not in merged else np.maximum(merged[group], instance)
    return list(merged.values())


def encode_instances(instances: list[np.ndarray]) -> list[str]:
    """COCO RLE counts for each instance, as text ready for the CSV."""
    return [
        mask_util.encode(np.asfortranarray(instance))["counts"].decode("ascii")
        for instance in instances
    ]


def rows_from_instances(image_id: str, instances: list[np.ndarray]) -> list[tuple[str, str]]:
    """Submission rows for already-extracted instances.

    Split out so validation can count instances and emit rows without
    re-deriving them, while still going through this exact code path.
    """
    return [
        (f"{image_id}_{n}", rle)
        for n, rle in enumerate(encode_instances(instances), start=1)
    ]


def rows_for_image(image_id: str, mask: np.ndarray, min_area: int = 50) -> list[tuple[str, str]]:
    """Submission rows for one image: (filament_id, segmentation_rle) per filament."""
    return rows_from_instances(image_id, mask_to_instances(mask, min_area))


def write_submission(rows, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUBMISSION_COLUMNS)
        writer.writerows(rows)
