"""Choose the operating point for a submission from trained checkpoints.

Predicts the validation photographs once - optionally under eight-way TTA and
averaged over several checkpoints - caches the probability maps, and scores a
wide grid of (high, min_area, low) points with the organisers' Panoptic Quality.

Training validates a handful of points because each one costs a full overlap
table. Here that cost is paid once per *low* threshold: the components of
`probability >= low` are fixed, and `high` and `min_area` only decide which of
them are kept. Since predicted components never overlap one another, dropping
some leaves the IoU of the rest unchanged, so every (high, min_area) is read off
the same table by selecting columns.

    python sweep.py --checkpoint runs/filanet3/best.pt --tta \
        --images processed/train --annotations <train json> --cache runs/filanet3/sweep_tta
"""
import argparse
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import torch

from dataset import FilamentDataset, split_by_image
from inference import load_model, predict_probability
from metrics import (IOU_THRESHOLD, _overlap_from_labels, get_overlap_matrices,
                     gt_df_from_coco, gt_label_cache, overlap_table,
                     panoptic_quality_from_table, rles_to_layers)
from submission import point_name, probability_to_instances, rows_from_instances

LOWS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)
HIGHS = (0.8, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)
MIN_AREAS = (0, 50, 100, 200, 300, 400, 600, 800, 1000)
# merging pieces up to 20 px apart took filanet3 from 0.364 to 0.375
MERGE_GAPS = (0, 10, 20, 30, 40)


def components(probability: np.ndarray, low: float):
    """Label map of `probability >= low`, with each component's area and peak."""
    mask = (probability >= low).astype(np.uint8)
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    inside = mask.astype(bool)
    peaks = np.zeros(count)
    np.maximum.at(peaks, labels[inside], probability[inside].astype(np.float64))
    return labels, stats[1:, cv.CC_STAT_AREA].astype(np.float64), peaks[1:]


def iou_against(gt_entry, gt_rles, labels, areas, height, width) -> np.ndarray:
    """IoU of every GT filament of one annotator-image against every component."""
    gt_labels, gt_areas, gt_exact = gt_entry
    if gt_exact:
        return _overlap_from_labels(gt_labels, gt_areas, labels, areas)[0].numpy()
    # overlapping GT instances: the dense route, as overlap_table does
    layers = np.stack([(labels == n) for n in range(1, len(areas) + 1)]).astype(np.float32) \
        if len(areas) else np.zeros((0, height, width), np.float32)
    return get_overlap_matrices(torch.from_numpy(rles_to_layers(gt_rles, height, width)),
                                torch.from_numpy(layers))[0].numpy()


def selection_counts(tables, high: float, min_area: int) -> np.ndarray:
    """(sum of matched IoU, TP, FP, FN) per table, keeping only components that
    reach `high` with at least `min_area` px.

    Mirrors panoptic_quality_from_table on the column subset. Kept per table
    rather than summed so that bootstrap_interval can resample photographs.
    """
    counts = np.zeros((len(tables), 4))
    for n, (iou, areas, peaks) in enumerate(tables):
        keep = (areas >= min_area) & (peaks >= high)
        if not keep.any():
            counts[n, 3] = iou.shape[0]
            continue
        hits = iou[:, keep] > IOU_THRESHOLD
        counts[n] = (float(iou[:, keep][hits].sum()), hits.sum(),
                     (hits.sum(axis=0) == 0).sum(), (hits.sum(axis=1) == 0).sum())
    return counts


def pq_from_counts(counts: np.ndarray) -> float:
    tp_sum, tp, fp, fn = counts.sum(axis=0)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return float(tp_sum / denominator) if denominator > 0 else 0.0


def pq_of_selection(tables, high: float, min_area: int) -> float:
    """PQ when only components reaching `high` with at least `min_area` px are kept."""
    return pq_from_counts(selection_counts(tables, high, min_area))


def bootstrap_interval(counts: np.ndarray, photo_of_row: np.ndarray, draws: int = 1000,
                       seed: int = 0) -> tuple[float, float]:
    """95% interval of PQ under resampling the validation photographs.

    106 photographs is a small sample: on filanet3 and filanet6 this interval
    was about +/-0.025 wide, wider than the gap between the two models'
    validation scores and wider than the gap between their leaderboard scores.
    Photographs, not annotator-images, are what is resampled, since the
    annotators of one photograph are scored against the same prediction.
    """
    photos = np.unique(photo_of_row)
    per_photo = np.stack([counts[photo_of_row == p].sum(axis=0) for p in photos])
    rng = np.random.default_rng(seed)
    scores = [pq_from_counts(per_photo[rng.integers(0, len(photos), len(photos))])
              for _ in range(draws)]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def robust_scores(scores: dict) -> dict:
    """Each plain or hysteresis point's worst PQ over itself and its neighbours.

    `scores` maps (high, low, min_area) to PQ. A neighbour is one grid step
    away in `low` or in `min_area`. The best single point is what the sweep
    used to submit, and on filanet6 it sat on a cliff: 0.375 at low 0.98, 0.327
    at low 0.5, and the leaderboard gave 0.28 where filanet3, whose best point
    sat on a plateau, gave 0.32 from the same validation score. A point whose
    neighbours are also good survives the small shift between validation and
    test; this measures that.
    """
    robust = {}
    for (high, low, min_area), pq in scores.items():
        li, ai = LOWS.index(low), MIN_AREAS.index(min_area)
        worst = pq
        for dl, da in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if not (0 <= li + dl < len(LOWS) and 0 <= ai + da < len(MIN_AREAS)):
                continue
            neighbour = (high, LOWS[li + dl], MIN_AREAS[ai + da])
            if neighbour in scores:
                worst = min(worst, scores[neighbour])
        robust[(high, low, min_area)] = worst
    return robust


def merge_counts(probabilities, by_photo, gt_df, gt_keys, gt_cache, point, gap):
    """Per-annotator-image counts at one point with fragment merging, and the
    photograph each row belongs to."""
    tables, photo_of_row = [], []
    for image_id, annotator_images in by_photo.items():
        probability = probabilities[image_id].astype(np.float32)
        height, width = probability.shape
        labels = np.zeros((height, width), np.int32)
        instances = probability_to_instances(probability, point, gap)
        for n, instance in enumerate(instances, start=1):
            labels[instance.astype(bool)] = n
        areas = np.array([float(i.sum()) for i in instances])
        for annotator_image in annotator_images:
            rles = gt_df.loc[gt_keys == annotator_image, "segmentation_rle"].tolist()
            iou = iou_against(gt_cache[annotator_image], rles, labels, areas, height, width)
            tables.append((iou, areas, np.ones(len(areas))))
            photo_of_row.append(image_id)
    return selection_counts(tables, high=0.0, min_area=0), np.array(photo_of_row)


def pq_with_merge(probabilities, by_photo, gt_df, gt_keys, gt_cache, point, gap) -> float:
    """PQ at one operating point with fragment merging, scored from label maps.

    The same instances the submission path produces, but their overlaps come
    straight from a label map rather than through RLE strings and back.
    """
    return pq_from_counts(merge_counts(probabilities, by_photo, gt_df, gt_keys,
                                       gt_cache, point, gap)[0])


def cached_probabilities(args, val_data, device):
    """{image_id: probability} for each validation photograph, from disk where possible."""
    args.cache.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoints": [str(p) for p in args.checkpoint], "tta": args.tta,
            "val_fraction": args.val_fraction, "seed": args.seed}
    meta_path = args.cache / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"{args.cache} holds predictions from a different setup "
                         f"({meta_path.read_text()}); choose another --cache")
    meta_path.write_text(json.dumps(meta))

    models = None
    probabilities = {}
    seen = set()
    for index in range(len(val_data)):
        image_id = val_data.ids[index].split("-", 1)[1]
        if image_id in seen:
            continue
        seen.add(image_id)
        path = args.cache / f"{image_id}.npy"
        if not path.exists():
            if models is None:
                models = [load_model(p, device) for p in args.checkpoint]
            image, _ = val_data[index]
            with torch.no_grad():
                probability = predict_probability(models, image.unsqueeze(0).to(device), args.tta)
            np.save(path, probability.cpu().numpy().astype(np.float16))
            print(f"  predicted {len(seen)} photographs", end="\r", flush=True)
        probabilities[image_id] = path
    print()
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                        help="one or more checkpoints; several are averaged")
    parser.add_argument("--images", type=Path, required=True, help="preprocessed image directory")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO annotation json")
    parser.add_argument("--cache", type=Path, required=True,
                        help="directory for the cached probability maps and results")
    parser.add_argument("--tta", action="store_true", help="average all eight orientations")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="must match the training run, or validation leaks into train")
    parser.add_argument("--seed", type=int, default=0, help="must match the training run")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--merge-candidates", type=int, default=10,
                        help="how many of the leading points to retry with each merge gap")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_ids = sorted(img["id"] for img in json.loads(args.annotations.read_text())["images"])
    _, val_ids = split_by_image(all_ids, args.val_fraction, args.seed)
    val_data = FilamentDataset(args.images, args.annotations, ids=val_ids)
    gt_df = gt_df_from_coco(args.annotations, image_ids=set(val_ids))
    gt_cache = gt_label_cache(gt_df)
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    by_photo: dict[str, list[str]] = {}
    for annotator_image in gt_keys.unique():
        by_photo.setdefault(annotator_image.split("-", 1)[1], []).append(annotator_image)

    probabilities = cached_probabilities(args, val_data, device)

    # one table per low threshold: (iou, component areas, component peaks) per annotator-image
    tables: dict[float, list] = {low: [] for low in LOWS}
    for n, (image_id, annotator_images) in enumerate(by_photo.items(), start=1):
        probability = np.load(probabilities[image_id]).astype(np.float32)
        height, width = probability.shape
        for low in LOWS:
            labels, areas, peaks = components(probability, low)
            for annotator_image in annotator_images:
                rles = gt_df.loc[gt_keys == annotator_image, "segmentation_rle"].tolist()
                iou = iou_against(gt_cache[annotator_image], rles, labels, areas, height, width)
                tables[low].append((iou, areas, peaks))
        print(f"  scored {n}/{len(by_photo)} photographs", end="\r", flush=True)
    print()

    scores = {}   # (high, low, min_area) -> PQ
    for low in LOWS:
        for high in sorted(set(HIGHS) | {low}):
            if high < low:
                continue
            for min_area in MIN_AREAS:
                scores[(high, low, min_area)] = pq_of_selection(tables[low], high, min_area)
    robust = robust_scores(scores)

    def as_point(key):
        high, low, min_area = key
        return (high, min_area) if high == low else (high, min_area, low)

    # ranked by the worst PQ among each point's neighbours, not by its own PQ
    ranked = sorted(scores, key=lambda k: (robust[k], scores[k]), reverse=True)
    with open(args.cache / "sweep.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pq", "robust_pq", "point", "high", "low", "min_area"])
        for key in ranked:
            writer.writerow([f"{scores[key]:.6f}", f"{robust[key]:.6f}",
                             point_name(as_point(key)), *key])

    best_single = max(scores, key=scores.get)
    print(f"best single point: PQ {scores[best_single]:.4f} at {point_name(as_point(best_single))}"
          f" (worst neighbour {robust[best_single]:.4f})")
    print(f"top {args.top} of {len(scores)} points by worst-neighbour PQ:")
    for key in ranked[:args.top]:
        print(f"  robust {robust[key]:.4f}  PQ {scores[key]:.4f}  {point_name(as_point(key))}")

    # Stage two: fragment merging on the leading points. Merging decides which
    # pieces form one filament, so it cannot be read off the tables above.
    loaded = {image_id: np.load(path) for image_id, path in probabilities.items()}
    merged: dict[tuple, float] = {}
    for key in ranked[:args.merge_candidates]:
        for gap in MERGE_GAPS:
            pq = pq_with_merge(loaded, by_photo, gt_df, gt_keys, gt_cache, as_point(key), gap)
            merged[(key, gap)] = pq
            print(f"  merge gap {gap:>2} px  PQ {pq:.4f}  {point_name(as_point(key))}", flush=True)

    # a gap is judged with its neighbouring gaps too, for the same reason
    def gap_robust(entry):
        key, gap = entry
        i = MERGE_GAPS.index(gap)
        near = [MERGE_GAPS[j] for j in (i - 1, i + 1) if 0 <= j < len(MERGE_GAPS)]
        return min([merged[entry]] + [merged[(key, g)] for g in near])

    final = sorted(merged, key=lambda e: (min(gap_robust(e), robust[e[0]]), merged[e]), reverse=True)
    with open(args.cache / "sweep_merge.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pq", "robust_pq", "point", "merge_gap"])
        writer.writerows([f"{merged[e]:.6f}", f"{min(gap_robust(e), robust[e[0]]):.6f}",
                          point_name(as_point(e[0])), e[1]] for e in final)

    # the fast paths above must agree with the scoring the leaderboard uses
    (best_key, best_gap) = final[0]
    best_point, best_pq = as_point(best_key), merged[final[0]]
    counts, photo_of_row = merge_counts(loaded, by_photo, gt_df, gt_keys, gt_cache,
                                        best_point, best_gap)
    low_ci, high_ci = bootstrap_interval(counts, photo_of_row)
    rows = []
    for image_id in by_photo:
        probability = loaded[image_id].astype(np.float32)
        rows.extend(rows_from_instances(
            image_id, probability_to_instances(probability, best_point, best_gap)))
    pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    check = panoptic_quality_from_table(overlap_table(gt_df, pred_df, gt_cache=gt_cache))
    print(f"chosen: PQ {best_pq:.4f} (95% CI {low_ci:.3f}-{high_ci:.3f} over photographs, "
          f"worst neighbour {min(gap_robust(final[0]), robust[best_key]):.4f}) "
          f"at {point_name(best_point)}, merge gap {best_gap} px")
    print(f"check through the submission path: PQ {check:.4f}"
          + ("" if abs(check - best_pq) < 1e-4 else "  <-- DISAGREES with the sweep"))
    high, min_area, *rest = best_point
    print(f"for predict.py: --point {','.join(str(v) for v in (high, min_area, *rest))} "
          f"--merge-gap {best_gap}")
    print(f"all points written to {args.cache / 'sweep.csv'} and {args.cache / 'sweep_merge.csv'}")


if __name__ == "__main__":
    main()
