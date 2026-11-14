"""Train FilaNet on the MAGFiLO filament dataset.

Validation deliberately builds its rows through the same submission code path
the real submission uses, so validation and submission cannot drift apart. It
then scores them with the organisers' Panoptic Quality, which is what the
leaderboard reports. Mean best-Dice per filament is logged alongside because
the two disagree sharply about the operating point - PQ charges half a point
for every unmatched prediction and mean Dice ignores them entirely - and
--select-by chooses which one picks best.pt.

Run the preprocessing once first:
    python img_prepro_v2.py <train_images> <processed>
"""
import argparse
import copy
import csv
import faulthandler
import json
import math
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TARGETS, FilamentDataset, split_by_image
from inference import predict_probability
from losses import ClDiceLoss, FilamentLoss
from metrics import (gt_df_from_coco, gt_label_cache, mean_best_dice_from_table,
                     overlap_table, panoptic_quality_from_table)
from model import FilaNet
from submission import point_name, probability_to_instances, rows_from_instances


class EvalResult(NamedTuple):
    scores: dict          # Panoptic Quality, what the organisers' notebook reports
    dice_scores: dict     # mean best-Dice per GT filament, a diagnostic beside PQ
    val_loss: float
    diagnostics: dict

# The leaderboard scores PQ, so the grid brackets PQ's optimum, measured by
# sweeping a trained model's cached predictions: min_area peaks near 600 and
# falls away by 2000, and threshold is still climbing at 0.95, which says the
# model is over-confident - so 0.97 is tried too. Mean Dice is reported
# alongside but wants the opposite corner - it never counts a false positive,
# so it prefers every prediction kept - and (0.7, 600) keeps an eye on that.
# Three-value points are hysteresis (high, min_area, low): see
# probability_to_instances. This is a small grid to track training by; the
# operating point for a submission comes from sweep.py, which tries hundreds.
DEFAULT_OPERATING_POINTS = [
    (0.7, 600), (0.9, 600), (0.95, 300), (0.95, 600), (0.97, 300),
    (0.95, 300, 0.5), (0.95, 100, 0.5), (0.97, 100, 0.7),
]


def gpu_telemetry() -> tuple[int | None, int | None]:
    """Current GPU temperature and SM clock, or (None, None) if unavailable.

    Worth logging on this machine: the card throttles from 2100 MHz to under
    750 MHz once it reaches its 98C limit, which roughly doubles epoch time.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(", ")
        return int(out[0]), int(out[1])
    except Exception:
        return None, None


def format_progress(index: int, n_batches: int, running_loss: float, elapsed: float) -> str:
    """One in-epoch progress line, including the rate so throttling is visible."""
    done = index + 1
    rate = elapsed / done
    eta = rate * (n_batches - done) / 60
    return (f"    batch {done}/{n_batches}  loss {running_loss:.4f}  "
            f"{rate:.2f}s/it  ETA {eta:.1f} min")


DIAGNOSTIC_KEYS = ["prob_p50", "prob_p999", "components"]


def should_validate(epoch: int, start_epoch: int, total_epochs: int, val_every: int) -> bool:
    """Whether epoch `epoch` should be scored, given --val-every.

    Validation is the expensive half of an epoch here - 10.4 minutes against 6.5
    of training, and 42 against 6.5 before the overlap table was rewritten - so
    skipping it buys epochs. Two epochs are always scored regardless: the first
    of a run, because a collapse should be visible immediately rather than
    `val_every` epochs later, and the last, because otherwise the final model is
    never scored at all.
    """
    if val_every <= 1:
        return True
    return (epoch - start_epoch) % val_every == 0 or epoch == total_epochs - 1


def log_columns(points) -> list[str]:
    thresholds = sorted({p[0] for p in points})
    return (
        ["epoch", "train_loss", "val_loss"]
        + [f"dice_{point_name(p)}" for p in points]
        + [f"pq_{point_name(p)}" for p in points]
        + ["best_dice_point", "best_dice", "best_pq_point", "best_pq"]
        + DIAGNOSTIC_KEYS
        + [f"fg_pct_t{t}" for t in thresholds]
        + ["train_minutes", "val_minutes", "gpu_temp", "gpu_clock"]
    )


def log_row(epoch, train_loss, val_loss, scores, points,
            train_minutes, val_minutes, gpu_temp, gpu_clock, diagnostics=None,
            dice_scores=None) -> list:
    dice_scores = dice_scores or {}
    best_pq_point = max(scores, key=scores.get) if scores else None
    best_dice_point = max(dice_scores, key=dice_scores.get) if dice_scores else None
    diagnostics = diagnostics or {}
    thresholds = sorted({p[0] for p in points})
    return (
        [epoch, train_loss, val_loss]
        + [dice_scores.get(p, float("nan")) for p in points]
        + [scores.get(p, float("nan")) for p in points]
        + [
            str(best_dice_point),
            dice_scores[best_dice_point] if best_dice_point is not None else float("nan"),
            str(best_pq_point),
            scores[best_pq_point] if best_pq_point is not None else float("nan"),
        ]
        + [diagnostics.get(k, float("nan")) for k in DIAGNOSTIC_KEYS]
        + [diagnostics.get(f"fg_pct_t{t}", float("nan")) for t in thresholds]
        + [
            train_minutes,
            val_minutes,
            "" if gpu_temp is None else gpu_temp,
            "" if gpu_clock is None else gpu_clock,
        ]
    )


def ensure_log_header(path: Path, columns: list[str]) -> None:
    """Make sure the log starts with `columns`, preserving any earlier run.

    A run whose column set has changed must not append rows the header cannot
    describe, so a stale header is rotated aside rather than overwritten.
    """
    path = Path(path)
    if path.exists():
        with open(path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing == columns:
            return
        backup = path.with_suffix(path.suffix + f".{int(time.time())}.bak")
        path.rename(backup)
        print(f"log columns changed; previous log moved to {backup}", flush=True)

    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(columns)


def should_step(batch_index: int, accumulation: int, n_batches: int) -> bool:
    """Whether to apply the optimizer after this batch.

    Steps on every accumulation boundary, and always on the final batch so the
    tail of an epoch that does not divide evenly is not silently discarded.
    """
    return (batch_index + 1) % accumulation == 0 or batch_index == n_batches - 1


def learning_rate_factor(step: int, warmup_steps: int, total_steps: int,
                         schedule: str = "cosine", floor: float = 0.01) -> float:
    """Multiplier on the base learning rate after `step` optimizer steps.

    A linear warmup, then either constant or a cosine decay to `floor`. The runs
    so far held Adam at 1e-4 throughout, and their PQ moved by several points
    from one epoch to the next (filanet3 fell from 0.33 to 0.30 and back
    between epochs 8 and 10) - noise a decaying rate settles.
    """
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model, name: str, lr: float, weight_decay: float):
    """Adam as before, or AdamW with decay kept off norms and biases."""
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


class ModelEMA:
    """An exponential moving average of a model's weights, used for validation.

    The effective decay ramps up as (1 + n) / (10 + n) towards `decay`, so the
    average is not dominated by the random initialisation for the first few
    hundred steps - a run here is only a few thousand steps long.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for averaged, current in zip(self.module.state_dict().values(),
                                     model.state_dict().values()):
            if averaged.dtype.is_floating_point:
                averaged.mul_(decay).add_(current.detach(), alpha=1.0 - decay)
            else:
                averaged.copy_(current)


def save_checkpoint(path, model, optimizer, epoch: int, best_pq: float,
                    scheduler=None, ema: "ModelEMA | None" = None) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_pq": best_pq,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            # what validation scored, and so what inference.load_model prefers
            "ema": ema.module.state_dict() if ema is not None else None,
            "ema_updates": ema.updates if ema is not None else 0,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    ema: "ModelEMA | None" = None) -> tuple[int, float]:
    """Restore a checkpoint, including whatever optimizer, schedule and average it holds.

    Checkpoints written before the schedule and EMA existed carry neither; the
    schedule is then left where the caller put it, and the average starts from
    the restored weights.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None:
        ema.module.load_state_dict(state["ema"] if state.get("ema") is not None else state["model"])
        ema.updates = state.get("ema_updates", 0)
    return state["epoch"], state["best_pq"]


def initialise_from(model: nn.Module, state_dict: dict) -> None:
    """Start `model` from another run's weights, for fine-tuning.

    Unlike --resume this restores no optimizer, schedule or epoch: the run
    starts over from these weights with a fresh warmup. The one shape allowed
    to differ is the output head, so that a mask-only model can seed one that
    also predicts the centreline; each new output starts as a copy of the old
    mask output, which keeps the mask prediction intact on the first step.
    """
    own = model.state_dict()
    if set(own) != set(state_dict):
        raise ValueError("checkpoint is from a different architecture")
    for key, value in state_dict.items():
        if value.shape == own[key].shape:
            own[key] = value
        elif key.startswith("head.") and value.shape[1:] == own[key].shape[1:]:
            own[key] = value[:1].repeat(own[key].shape[0], *([1] * (value.dim() - 1)))
        else:
            raise ValueError(f"{key}: checkpoint {tuple(value.shape)}, "
                             f"model {tuple(own[key].shape)}")
    model.load_state_dict(own)


def evaluate(
    model,
    dataset,
    gt_df: pd.DataFrame,
    operating_points=DEFAULT_OPERATING_POINTS,
    device: str = "cuda",
    height: int = 2048,
    width: int = 2048,
    criterion=None,
    gt_cache=None,
    tta: bool = False,
) -> "EvalResult":
    """Both metrics at each (threshold, min_area), from a single forward pass.

    Threshold and minimum area are free to choose after training, and the two
    metrics want opposite settings - mean Dice ignores false positives and so
    wants everything kept, PQ charges for them and so wants them filtered. They
    are swept together here rather than fixed in advance.
    """
    model.eval()
    rows: dict[tuple, list[tuple[str, str]]] = {p: [] for p in operating_points}
    losses: list[float] = []
    # a collapsed model is only visible in these: probability with no spread,
    # and either the whole frame or nothing predicted as foreground
    p50s: list[float] = []
    p999s: list[float] = []
    fg_pct: dict[float, list[float]] = {p[0]: [] for p in operating_points}
    components: dict[tuple, list[int]] = {p: [] for p in operating_points}

    # The same photograph appears once per annotator. It must be predicted once:
    # emitting a set of filaments per annotator would make the duplicates
    # compete as false positives against each other.
    seen: set[str] = set()

    with torch.no_grad():
        for index in range(len(dataset)):
            image_id = dataset.ids[index].split("-", 1)[1]
            if image_id in seen:
                continue
            seen.add(image_id)

            image, target = dataset[index]
            batch = image.unsqueeze(0).to(device)
            logits = model(batch)
            if criterion is not None:
                # loss on the plain view only, so it stays comparable across runs
                losses.append(criterion(logits, target.unsqueeze(0).to(device)).item())
            probability = predict_probability(model, batch, tta=tta,
                                              identity_logits=logits).cpu().numpy()

            p50s.append(float(np.percentile(probability, 50)))
            # foreground is ~0.3% of a frame, so p99 still sits in background;
            # p99.9 is the percentile that actually lands inside a filament
            p999s.append(float(np.percentile(probability, 99.9)))

            for threshold in fg_pct:
                fg_pct[threshold].append(100.0 * float((probability >= threshold).mean()))
            for point in operating_points:
                instances = probability_to_instances(probability, point)
                components[point].append(len(instances))
                rows[point].extend(rows_from_instances(image_id, instances))

    frames = {
        point: pd.DataFrame(r, columns=["filament_id", "segmentation_rle"])
        for point, r in rows.items()
    }
    # One overlap table per operating point, serving both metrics. Computing them
    # separately meant twelve tables an epoch instead of six, and validation was
    # 87% of this run's wall clock while training was 13% - the model reached
    # epoch 6 of 200 in eight hours and its PQ was still climbing.
    tables = {
        point: overlap_table(gt_df, frame, height=height, width=width, gt_cache=gt_cache)
        for point, frame in frames.items()
    }
    scores = {point: panoptic_quality_from_table(t) for point, t in tables.items()}
    # not the score - the leaderboard uses PQ - but it disagrees with PQ about the
    # operating point badly enough to be worth reading: see DEFAULT_OPERATING_POINTS
    dice_scores = {point: mean_best_dice_from_table(t) for point, t in tables.items()}
    best_point = max(scores, key=scores.get) if scores else None
    diagnostics = {
        "prob_p50": float(np.mean(p50s)) if p50s else float("nan"),
        "prob_p999": float(np.mean(p999s)) if p999s else float("nan"),
        "components": float(np.mean(components[best_point])) if best_point else float("nan"),
    }
    for threshold, values in fg_pct.items():
        diagnostics[f"fg_pct_t{threshold}"] = float(np.mean(values)) if values else float("nan")

    return EvalResult(scores, dice_scores,
                      float(np.mean(losses)) if losses else float("nan"), diagnostics)


def train_one_epoch(model, loader, criterion, optimizer, device, accumulation: int,
                    report_every: int = 25, scheduler=None, ema: "ModelEMA | None" = None) -> float:
    model.train()
    n_batches = len(loader)
    total = 0.0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for index, (image, target) in enumerate(loader):
        image, target = image.to(device), target.to(device)
        loss = criterion(model(image), target) / accumulation
        loss.backward()
        total += loss.item() * accumulation

        if should_step(index, accumulation, n_batches):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        if report_every and (index + 1) % report_every == 0:
            # flushed, so a run watched over ssh or in a log file stays live
            print(format_progress(index, n_batches, total / (index + 1), time.time() - start),
                  flush=True)

    return total / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="preprocessed image directory")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO annotation json")
    parser.add_argument("--out", type=Path, default=Path("runs/filanet"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.3, help="clDice weight, paper's [0, 0.5]")
    parser.add_argument("--tversky-beta", type=float, default=0.7,
                        help="weight on missed filaments; 0.5 is plain Dice")
    parser.add_argument("--bce-weight", type=float, default=1.0,
                        help="pixelwise term holding the background down; 0 gives the paper's Eq. 3")
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW only; not applied to norms and biases")
    parser.add_argument("--schedule", choices=("cosine", "constant"), default="cosine",
                        help="learning rate after warmup; constant is what earlier runs used")
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema", type=float, default=0.999,
                        help="decay of the weight average that is validated and saved; 0 disables")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop", type=int, default=None,
                        help="train on random windows of this size instead of whole "
                             "photographs; buys optimizer steps, must divide by 64")
    parser.add_argument("--crop-foreground-prob", type=float, default=0.7,
                        help="fraction of windows centred on a filament")
    parser.add_argument("--target", choices=TARGETS, default="union",
                        help="how a photograph's annotators become one training target; "
                             "'annotator' trains on each of them separately")
    parser.add_argument("--deformable-from", type=int, default=0,
                        help="stages finer than this use plain convs; 2 makes the "
                             "1/1 and 1/2 stages plain, which is where the cost is")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-every", type=int, default=1,
                        help="score every Nth epoch; the first and last are always scored")
    parser.add_argument("--val-tta", action="store_true",
                        help="validate on the mean of all eight orientations; eight "
                             "forward passes per photograph instead of one")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-every", type=int, default=25,
                        help="batches between in-epoch progress lines; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select-by", choices=("pq", "dice"), default="pq",
                        help="metric that picks best.pt; the leaderboard scores PQ")
    parser.add_argument("--no-augment", action="store_true",
                        help="train on the photographs as they are, for comparison")
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoints from before AdamW resume only with --optimizer adam")
    parser.add_argument("--init", type=Path, default=None,
                        help="fine-tune: start from this checkpoint's weights (its EMA if "
                             "it has one) with a fresh optimizer and schedule")
    parser.add_argument("--spine-weight", type=float, default=0.0,
                        help="weight of the auxiliary centreline loss; 0 trains the mask alone")
    parser.add_argument("--size-balance", action="store_true",
                        help="weight the BCE on small filaments' pixels up (see dataset.size_weight)")
    args = parser.parse_args()
    if args.init and args.resume:
        raise SystemExit("--init and --resume are exclusive")

    if args.crop and args.crop % 64:
        raise SystemExit(f"--crop must be divisible by 64, got {args.crop}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(
        img["id"] for img in json.loads(args.annotations.read_text())["images"]
    )
    train_ids, val_ids = split_by_image(all_ids, args.val_fraction, args.seed)
    spine = args.spine_weight > 0
    train_data = FilamentDataset(args.images, args.annotations, ids=train_ids,
                                 augment=not args.no_augment, crop=args.crop,
                                 crop_foreground_prob=args.crop_foreground_prob,
                                 target=args.target, spine=spine,
                                 size_balance=args.size_balance)
    # always per annotator: evaluate() predicts each photograph once and scores
    # it against every annotator, exactly as the leaderboard does. The extra
    # channels are kept so that validation loss is the training loss.
    val_data = FilamentDataset(args.images, args.annotations, ids=val_ids, spine=spine,
                               size_balance=args.size_balance)
    # A previous run died at 17:23 on 2026-09-28 with exception 0xc0000005 - an
    # access violation inside torch_cpu.dll, which no try/except can catch and
    # which left nothing on disk to read. faulthandler installs a native signal
    # handler, so the next one writes a Python traceback here instead.
    crash_log = open(args.out / "crash.txt", "w")
    faulthandler.enable(file=crash_log)

    val_gt = gt_df_from_coco(args.annotations, image_ids=set(val_ids))
    # the validation ground truth is fixed for the whole run, so its label maps
    # are decoded once here rather than on every epoch and operating point
    gt_cache = gt_label_cache(val_gt)

    photographs = len({i.split("-", 1)[1] for i in val_ids})
    augmentation = "as-is" if args.no_augment else "8 orientations"
    window = f"{args.crop}px windows" if args.crop else "whole 2048px frames"
    steps = -(-len(train_data) // args.batch_size) // args.accumulation
    print(f"{window}, batch {args.batch_size} x {args.accumulation} "
          f"-> {steps} optimizer steps per epoch")
    unit = "annotator-images" if args.target == "annotator" else f"photographs ({args.target} target)"
    print(f"train {len(train_data)} {unit} ({augmentation}) | "
          f"val {len(val_ids)} over {photographs} photographs, {len(val_gt)} filaments")

    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        drop_last=args.batch_size > 1)
    model = FilaNet(deformable_from=args.deformable_from,
                    out_channels=2 if spine else 1).to(device)
    if args.init:
        source = torch.load(args.init, map_location="cpu", weights_only=False)
        initialise_from(model, source["ema"] if source.get("ema") is not None else source["model"])
        print(f"initialised from {args.init} (epoch {source['epoch']})")
    criterion = FilamentLoss(ClDiceLoss(alpha=args.alpha, bce_weight=args.bce_weight,
                                        tversky_beta=args.tversky_beta),
                             train_data.channels, spine_weight=args.spine_weight)
    print(f"target channels {train_data.channels}"
          + (f", centreline loss weight {args.spine_weight}" if spine else ""))
    optimizer = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    # should_step also steps on an epoch's ragged tail, hence the ceiling
    steps_per_epoch = math.ceil(len(loader) / args.accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(args.warmup_epochs * steps_per_epoch)
    # a resumed checkpoint from before the schedule existed carries no position
    # in it, so the schedule is offset to the epoch it resumes at
    step_offset = 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step + step_offset, warmup_steps,
                                                     total_steps, args.schedule))
    ema = ModelEMA(model, args.ema) if args.ema > 0 else None
    print(f"{args.optimizer} lr {args.lr} {args.schedule} over {total_steps} steps "
          f"({warmup_steps} warmup)" + (f", EMA {args.ema}" if ema else ""))

    start_epoch, best_score = 0, 0.0
    if args.resume and args.resume.exists():
        start_epoch, best_score = load_checkpoint(args.resume, model, optimizer, scheduler, ema)
        start_epoch += 1
        if scheduler.last_epoch == 0:
            # nothing restored the schedule's position: jump it to this epoch
            step_offset = start_epoch * steps_per_epoch
            for group, base in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base * learning_rate_factor(step_offset, warmup_steps,
                                                          total_steps, args.schedule)
        print(f"resumed from {args.resume} at epoch {start_epoch}, "
              f"best {args.select_by} {best_score:.4f}")

    points = DEFAULT_OPERATING_POINTS
    log_path = args.out / "log.csv"
    ensure_log_header(log_path, log_columns(points))

    for epoch in range(start_epoch, args.epochs):
        print(f"epoch {epoch}/{args.epochs - 1}", flush=True)
        t0 = time.time()
        loss = train_one_epoch(model, loader, criterion, optimizer, device,
                               args.accumulation, args.report_every, scheduler, ema)
        train_minutes = (time.time() - t0) / 60
        temp, clock = gpu_telemetry()

        scoring = should_validate(epoch, start_epoch, args.epochs, args.val_every)
        thermal = f"  gpu {temp}C {clock}MHz" if temp is not None else ""

        if scoring:
            t1 = time.time()
            # the averaged weights are what gets scored, saved as best and submitted
            scored = ema.module if ema is not None else model
            scores, dice_scores, val_loss, diag = evaluate(scored, val_data, val_gt, points,
                                                           device, criterion=criterion,
                                                           gt_cache=gt_cache, tta=args.val_tta)
            val_minutes = (time.time() - t1) / 60

            best_pq_point = max(scores, key=scores.get)
            best_dice_point = max(dice_scores, key=dice_scores.get)
            pq = scores[best_pq_point]
            dice = dice_scores[best_dice_point]
            # the leaderboard scores PQ; mean Dice is logged alongside because it
            # prefers a different operating point, which is worth knowing
            selected = dice if args.select_by == "dice" else pq
            selected_point = best_dice_point if args.select_by == "dice" else best_pq_point

            dice_detail = "  ".join(
                f"{point_name(p)}: {v:.4f}" for p, v in sorted(dice_scores.items()))
            pq_detail = "  ".join(
                f"{point_name(p)}: {v:.4f}" for p, v in sorted(scores.items()))
            print(f"  train loss {loss:.4f} ({train_minutes:.1f} min)  "
                  f"val loss {val_loss:.4f}  ({val_minutes:.1f} min){thermal}\n"
                  f"  Dice {dice:.4f} at {best_dice_point}    PQ {pq:.4f} at {best_pq_point}\n"
                  f"  Dice  {dice_detail}\n"
                  f"  PQ    {pq_detail}", flush=True)
        else:
            # the row is still written, with the validation columns left empty, so
            # that one line per epoch holds and train_loss stays plottable
            scores, dice_scores, diag = {}, {}, {}
            val_loss, val_minutes, selected = float("nan"), 0.0, None
            print(f"  train loss {loss:.4f} ({train_minutes:.1f} min)  "
                  f"not scored this epoch{thermal}", flush=True)

        with open(log_path, "a", newline="") as f:
            val_loss_cell = "" if selected is None else f"{val_loss:.6f}"
            csv.writer(f).writerow(log_row(epoch, f"{loss:.6f}", val_loss_cell, scores,
                                           points, f"{train_minutes:.2f}", f"{val_minutes:.2f}",
                                           temp, clock, diag, dice_scores=dice_scores))

        save_checkpoint(args.out / "last.pt", model, optimizer, epoch, best_score)
        if selected is not None and selected > best_score:
            best_score = selected
            save_checkpoint(args.out / "best.pt", model, optimizer, epoch, best_score)
            print(f"           new best {args.select_by} {best_score:.4f} at {selected_point}"
                  f" -> {args.out / 'best.pt'}")

    print(f"done. best {args.select_by} {best_score:.4f}")


if __name__ == "__main__":
    main()
