"""clDice: a topology-preserving loss for tubular structures.

Implements Shit et al., "clDice - a Novel Topology-Preserving Loss Function for
Tubular Structure Segmentation" (arXiv:2003.07311). Ordinary Dice weights every
pixel equally, so a filament broken in two loses only the few pixels at the
break. clDice compares morphological *skeletons*, where that same break severs
the centreline and is punished accordingly.

Computed on the foreground channel only, which the paper notes is sufficient for
highly imbalanced data (section 4.3) - filaments occupy ~0.3% of an image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

EPS = 1e-6

# Measured on the MAGFiLO annotations, the thickest point of each filament has
# a median radius of 8 px, p90 15, p99 26, max 47. Iterated erosion never
# reaches the centre of anything fatter than k, leaving an empty skeleton and a
# vacuous clDice of 1 that contributes no loss - so k covers the fattest
# filament observed. Iterations beyond a structure's radius are no-ops, and
# with checkpointing the extra steps cost about 0.02 GB each at 2048x2048.
DEFAULT_K = 47


def _min_pool(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _max_pool(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _peel(x: torch.Tensor, skeleton: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One erosion step, accumulating whatever the opening leaves behind."""
    x = _min_pool(x)
    delta = F.relu(x - _max_pool(_min_pool(x)))
    return x, skeleton + (1.0 - skeleton) * delta


def soft_skeleton(x: torch.Tensor, k: int = DEFAULT_K, checkpoint: bool = True) -> torch.Tensor:
    """Differentiable morphological skeleton (paper, Algorithm 1).

    Iterated min/max pooling stands in for erosion and dilation, so the whole
    thing stays differentiable. `k` must exceed the radius of the thickest
    structure, or skeletonization stops before reaching its centre and returns
    an empty skeleton.

    Each iteration retains several full-resolution tensors for the backward
    pass, which at 2048x2048 costs about 0.17 GB per step. `checkpoint` trades
    that for recomputation, which is cheap here because the operations are
    pooling rather than convolution.
    """
    skeleton = F.relu(x - _max_pool(_min_pool(x)))

    for _ in range(k):
        if checkpoint and x.requires_grad:
            x, skeleton = _checkpoint(_peel, x, skeleton, use_reentrant=False)
        else:
            x, skeleton = _peel(x, skeleton)

    return skeleton


def soft_tversky(pred: torch.Tensor, target: torch.Tensor,
                 beta: float = 0.5) -> torch.Tensor:
    """Tversky index, averaged over the batch. beta=0.5 is exactly soft Dice.

    Dice charges the same for a missed filament as for a spurious one. This
    model's errors are not symmetric: at PQ's operating point it produced 715
    false negatives against 505 false positives, so the misses are what cost it
    score. `beta` is the weight on false negatives and `1 - beta` the weight on
    false positives, so beta above 0.5 makes missing a filament the more
    expensive mistake and pushes the network to predict more freely.
    """
    dims = tuple(range(1, pred.ndim))
    alpha = 1.0 - beta
    true_positive = (pred * target).sum(dims)
    false_positive = (pred * (1.0 - target)).sum(dims)
    false_negative = ((1.0 - pred) * target).sum(dims)
    return ((true_positive + EPS)
            / (true_positive + alpha * false_positive + beta * false_negative + EPS)).mean()


def soft_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice overlap, averaged over the batch."""
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dims)
    total = pred.sum(dims) + target.sum(dims)
    return ((2.0 * intersection + EPS) / (total + EPS)).mean()


def soft_cldice(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = DEFAULT_K,
    checkpoint: bool = True,
) -> torch.Tensor:
    """Centreline Dice (paper, Algorithm 2), averaged over the batch.

    Topology precision asks how much of the prediction's skeleton falls inside
    the true mask; topology sensitivity asks the reverse. clDice is their
    harmonic mean.
    """
    dims = tuple(range(1, pred.ndim))
    pred_skeleton = soft_skeleton(pred, k, checkpoint)
    target_skeleton = soft_skeleton(target, k, checkpoint)

    # Epsilon belongs in the denominator only. Putting it in the numerator too
    # makes an empty predicted skeleton score a perfect precision of 1, and a
    # constant prediction has no skeleton - so the network can score ~0.85 by
    # painting the whole frame, which is a local minimum it will find. The paper
    # says as much: clDice is undefined when both terms vanish and should be
    # "extended continuously with the value 0".
    precision = (pred_skeleton * target).sum(dims) / (pred_skeleton.sum(dims) + EPS)
    sensitivity = (target_skeleton * pred).sum(dims) / (target_skeleton.sum(dims) + EPS)

    cldice = 2.0 * precision * sensitivity / (precision + sensitivity + EPS)

    # With no foreground to compare there is no topology either, so leave the
    # dice term to handle such a sample rather than inventing a penalty.
    has_foreground = target.sum(dims) > 0
    return torch.where(has_foreground, cldice, torch.ones_like(cldice)).mean()


class ClDiceLoss(nn.Module):
    """Paper Eq. 3, L = (1 - alpha)(1 - softDice) + alpha(1 - soft_clDice), plus BCE.

    Takes raw logits so it pairs directly with the model's output head.

    The BCE term is a deviation from the paper, and it is the difference between
    this loss training and not training on 2048x2048 frames whose foreground is
    0.36%. Both of the paper's terms are region-normalized, so neither supplies a
    per-pixel gradient:

      - clDice's gradient on background pixels is exactly zero. Its skeleton is a
        ridge detector, relu(x - open(x)), and a flat background has no ridges,
        so a prediction of 0.9 everywhere with faint ridges along the filaments
        satisfies both its precision and sensitivity terms at any level.
      - soft-Dice's background gradient is -dice/(sum_p + sum_t), which at this
        size is about 1e-10 per pixel.

    A convnet moves shared weights, so what decides the outcome is the sum of
    those gradients over every pixel. Measured on this data, clDice pulls the
    level up with 3.1e-2 while soft-Dice pushes down with 4.7e-4 - so adding
    clDice at alpha=0.3 *reverses* the sign, and the network raises every pixel
    together. That is not hypothetical: it is what the first eleven-epoch run
    did, reaching 99.99% predicted foreground and PQ 0.0 while its loss fell.

    BCE restores the missing per-pixel term. Section 4.3 of the paper allows
    foreground-only clDice because of "sparsity of foreground", an assumption
    that a saturated prediction violates - so the guard belongs here.

    The paper's soft-Dice is generalised to Tversky, which is the same thing at
    tversky_beta=0.5. The default of 0.7 weights a missed filament more heavily
    than a spurious one, because that is the shape of the measured error: 715
    false negatives against 505 false positives at PQ's operating point, with
    the misses concentrated in small filaments.
    """

    def __init__(self, alpha: float = 0.3, k: int = DEFAULT_K, checkpoint: bool = True,
                 bce_weight: float = 1.0, tversky_beta: float = 0.7):
        super().__init__()
        if not 0.0 <= alpha <= 0.5:
            raise ValueError(f"alpha must lie in [0, 0.5] as the paper specifies, got {alpha}")
        if not 0.0 < tversky_beta < 1.0:
            raise ValueError(f"tversky_beta must lie in (0, 1), got {tversky_beta}")
        self.alpha = alpha
        self.k = k
        self.checkpoint = checkpoint
        self.bce_weight = bce_weight
        self.tversky_beta = tversky_beta

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                pixel_weight: torch.Tensor | None = None) -> torch.Tensor:
        """`pixel_weight`, if given, weights the BCE term pixel by pixel and is
        normalised to keep its mean at 1, so it moves emphasis between pixels
        without changing the term's overall scale."""
        pred = torch.sigmoid(logits)
        loss = 1.0 - soft_tversky(pred, target, self.tversky_beta)
        if self.alpha > 0.0:
            cldice_term = 1.0 - soft_cldice(pred, target, self.k, self.checkpoint)
            loss = (1.0 - self.alpha) * loss + self.alpha * cldice_term
        if self.bce_weight:
            if pixel_weight is not None:
                pixel_weight = pixel_weight / pixel_weight.mean().clamp_min(EPS)
            loss = loss + self.bce_weight * F.binary_cross_entropy_with_logits(
                logits, target, weight=pixel_weight)
        return loss


class FilamentLoss(nn.Module):
    """The training loss over a multi-channel target, as FilamentDataset lays it out.

    Channel 0 of the target is the mask, scored by ClDiceLoss against channel 0
    of the logits. A "weight" channel, when present, weights that loss's BCE
    term pixel by pixel (see dataset.size_weight). A "spine" channel trains the
    model's second output against the annotated centrelines with BCE and soft
    Dice, scaled by `spine_weight`.

    The centreline is an auxiliary target: it is never submitted, but it is the
    one part of a filament annotators largely agree on - outlines vary between
    annotators far more than the line along the middle does - and predicting
    it asks the network to trace each filament end to end, the continuity that
    connected components need and that fragmented predictions lack.
    """

    def __init__(self, mask_loss: ClDiceLoss, channels: list[str], spine_weight: float = 0.0):
        super().__init__()
        if spine_weight and "spine" not in channels:
            raise ValueError("spine_weight needs a 'spine' target channel")
        self.mask_loss = mask_loss
        self.channels = list(channels)
        self.spine_weight = spine_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = (target[:, self.channels.index("weight")].unsqueeze(1)
                  if "weight" in self.channels else None)
        loss = self.mask_loss(logits[:, :1], target[:, :1], pixel_weight=weight)
        if self.spine_weight:
            spine_logits = logits[:, 1:2]
            spine = target[:, self.channels.index("spine")].unsqueeze(1)
            spine_term = (F.binary_cross_entropy_with_logits(spine_logits, spine)
                          + 1.0 - soft_dice(torch.sigmoid(spine_logits), spine))
            loss = loss + self.spine_weight * spine_term
        return loss
