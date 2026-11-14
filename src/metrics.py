"""Local Panoptic Quality scoring, mirroring the organisers' evaluation notebook.

The scoring logic here is deliberately a faithful port of the notebook shipped
with the competition rather than an independent implementation: a local score
is only useful if it agrees with the one on the leaderboard.

Ground-truth filament ids are ``<annotator>-<image>_<n>`` and prediction ids are
``<image>_<n>``, so an image annotated by three people is scored three times
against the same prediction.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

IOU_THRESHOLD = 0.5


def get_overlap_matrices(
    gt_layers: torch.Tensor,
    pred_layers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU and Dice between every GT and predicted layer of one image."""
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]

    # an explicit size rather than -1: reshaping an empty stack with -1 is ambiguous
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)

    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection

    zero = torch.tensor(0.0, device=gt_layers.device)
    iou = torch.where(union == 0, zero, intersection / union)
    dice = torch.where(union == 0, zero, 2 * intersection / (gt_areas + pred_areas))
    return iou, dice


def rles_to_layers(rles: list[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    return mask_util.decode(dicts).transpose(2, 0, 1).astype(np.float32)


def _label_map(
    rles: list[str],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Instances as a single integer image, where pixel value n marks instance n.

    Decoding one instance at a time keeps peak memory at one uint8 frame plus
    one int32 frame, rather than the n x height x width float32 stack the dense
    route built. Areas are read back out of the label map so that they always
    agree with the intersections computed from it.

    A pixel can only have one owner here, so the map loses information wherever
    two instances overlap. That is rare but real - one validation annotator-image
    has two filaments sharing 31% of its foreground - so the total area claimed
    by the instances is compared against the area the map ends up holding, and
    the third return value reports whether anything was lost. Callers fall back
    to the dense route when it was.
    """
    labels = np.zeros((height, width), dtype=np.int32)
    claimed = 0
    for n, rle in enumerate(rles, start=1):
        keep = mask_util.decode({"size": [height, width], "counts": rle}).astype(bool)
        claimed += int(keep.sum())
        labels[keep] = n
    areas = np.bincount(labels.ravel(), minlength=len(rles) + 1)[1:].astype(np.float64)
    return labels, areas, claimed == int(areas.sum())


def _overlap_from_labels(
    gt_labels: np.ndarray,
    gt_areas: np.ndarray,
    pred_labels: np.ndarray,
    pred_areas: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every pairwise IoU and Dice of one image, from a single pass over its pixels.

    Pairing each pixel's two labels into one code and counting the codes yields
    the whole intersection matrix at once. The dense route instead multiplied a
    n_gt x (height*width) matrix by a (height*width) x n_pred one - and since
    filaments cover about 0.4% of a frame, 99.6% of that arithmetic multiplied
    zeros. This is exact only where instances do not overlap, which `_label_map`
    reports on and `overlap_table` checks before choosing this route.
    """
    n_gt, n_pred = len(gt_areas), len(pred_areas)
    if n_gt == 0 or n_pred == 0:
        empty = torch.zeros((n_gt, n_pred), dtype=torch.float32)
        return empty, empty.clone()

    codes = gt_labels.ravel().astype(np.int64) * (n_pred + 1) + pred_labels.ravel()
    counts = np.bincount(codes, minlength=(n_gt + 1) * (n_pred + 1))
    intersection = counts.reshape(n_gt + 1, n_pred + 1)[1:, 1:].astype(np.float64)

    total = gt_areas[:, None] + pred_areas[None, :]
    union = total - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    dice = np.divide(2.0 * intersection, total, out=np.zeros_like(intersection), where=total > 0)
    return (torch.from_numpy(iou).float(), torch.from_numpy(dice).float())


def gt_label_cache(
    gt_df: pd.DataFrame,
    height: int = 2048,
    width: int = 2048,
) -> dict[str, tuple[np.ndarray, np.ndarray, bool]]:
    """Label maps for every annotator-image of the ground truth, built once.

    The validation ground truth is fixed for the whole run, so decoding it afresh
    on every epoch and every operating point is pure repetition. One cache costs
    about 16 MB per annotator-image held as int32.
    """
    keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    return {
        annotator_image: _label_map(
            gt_df.loc[keys == annotator_image, "segmentation_rle"].tolist(), height, width)
        for annotator_image in keys.unique()
    }


def overlap_table(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = 2048,
    width: int = 2048,
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray, bool]] | None = None,
) -> pd.DataFrame:
    """One row per annotator-image, holding its GT-vs-prediction overlap matrices.

    Both metrics are read off this table, so building it once per operating point
    and deriving PQ and Dice from it halves the work - see the *_from_table
    functions. Pass `gt_cache` from `gt_label_cache` to reuse the ground-truth
    label maps across epochs.

    Images whose instances overlap go the dense route, which is slower but places
    no single-owner constraint on a pixel. On this dataset that is one validation
    image in 164, so the table stays exact at the fast route's speed.
    """
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_keys = (
        pred_df["filament_id"].str.split("_", n=1).str[0]
        if len(pred_df)
        else pd.Series([], dtype=object)
    )

    # predictions are shared by every annotator of a photograph, so a label map
    # built for one annotator serves the rest
    pred_maps: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
    pred_rle_cache: dict[str, list[str]] = {}

    rows = []
    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rles = gt_df.loc[gt_keys == annotator_image, "segmentation_rle"].tolist()

        if gt_cache is not None and annotator_image in gt_cache:
            gt_labels, gt_areas, gt_exact = gt_cache[annotator_image]
        else:
            gt_labels, gt_areas, gt_exact = _label_map(gt_rles, height, width)

        if image_id not in pred_maps:
            pred_rle_cache[image_id] = (
                pred_df.loc[pred_keys == image_id, "segmentation_rle"].tolist()
                if len(pred_df)
                else []
            )
            pred_maps[image_id] = _label_map(pred_rle_cache[image_id], height, width)
        pred_labels, pred_areas, pred_exact = pred_maps[image_id]

        if gt_exact and pred_exact:
            iou, dice = _overlap_from_labels(gt_labels, gt_areas, pred_labels, pred_areas)
        else:
            iou, dice = get_overlap_matrices(
                torch.from_numpy(rles_to_layers(gt_rles, height, width)),
                torch.from_numpy(rles_to_layers(pred_rle_cache[image_id], height, width)),
            )
        rows.append(
            {
                "annotator_image": annotator_image,
                "iou_matrix": iou,
                "dice_matrix": dice,
                "n_gt": len(gt_rles),
                "n_pred": len(pred_rle_cache[image_id]),
            }
        )
    return pd.DataFrame(rows)


def panoptic_quality_from_table(table: pd.DataFrame) -> float:
    """PQ = sum(IoU over true positives) / (TP + 0.5*FP + 0.5*FN).

    Takes a prebuilt overlap table so that a caller scoring several operating
    points, or both metrics, pays for the overlaps only once.
    """
    tp_scores: list[float] = []
    fp_count = 0
    fn_count = 0

    for row in table.itertuples(index=False):
        if row.n_gt == 0:
            fp_count += row.n_pred
            continue
        if row.n_pred == 0:
            fn_count += row.n_gt
            continue

        hits = row.iou_matrix > IOU_THRESHOLD
        tp_scores.extend(row.iou_matrix[hits].tolist())
        fp_count += int((hits.sum(dim=0) == 0).sum().item())
        fn_count += int((hits.sum(dim=1) == 0).sum().item())

    denominator = len(tp_scores) + 0.5 * fp_count + 0.5 * fn_count
    return sum(tp_scores) / denominator if denominator > 0 else 0.0


def panoptic_quality(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = 2048,
    width: int = 2048,
) -> float:
    """PQ of a prediction set, building its overlap table on the way."""
    return panoptic_quality_from_table(overlap_table(gt_df, pred_df, height, width))


def gt_df_from_coco(
    json_path: str | Path,
    image_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build a ground-truth dataframe in submission format from COCO annotations."""
    coco = json.loads(Path(json_path).read_text())
    sizes = {img["id"]: (img["height"], img["width"]) for img in coco["images"]}

    per_image: dict[str, list[str]] = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        if image_ids is not None and image_id not in image_ids:
            continue
        height, width = sizes[image_id]
        rles = mask_util.frPyObjects(ann["segmentation"], height, width)
        merged = mask_util.merge(rles)
        per_image.setdefault(image_id, []).append(merged["counts"].decode("ascii"))

    rows = [
        (f"{image_id}_{n}", rle)
        for image_id, rles in per_image.items()
        for n, rle in enumerate(rles, start=1)
    ]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def mean_best_dice_from_table(table: pd.DataFrame) -> float:
    """Mean over ground-truth filaments of the best Dice any prediction achieves.

    A diagnostic, not the score: the leaderboard uses Panoptic Quality. It is
    kept because the two disagree usefully. PQ charges half a point for every
    unmatched prediction, so it rewards a conservative model; this asks only how
    well each true filament was covered and never counts the extras. Reading
    them side by side says whether a change found more filaments or merely
    stopped guessing.

    A filament nothing overlaps contributes 0, so misses are what this punishes.
    """
    scores: list[float] = []
    for row in table.itertuples(index=False):
        if row.n_gt == 0:
            continue
        if row.n_pred == 0:
            scores.extend([0.0] * row.n_gt)
            continue
        scores.extend(row.dice_matrix.max(dim=1).values.tolist())

    return float(np.mean(scores)) if scores else 0.0


def mean_best_dice(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = 2048,
    width: int = 2048,
) -> float:
    """Mean best Dice of a prediction set, building its overlap table on the way."""
    return mean_best_dice_from_table(overlap_table(gt_df, pred_df, height, width))
