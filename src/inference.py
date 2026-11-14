"""Probability maps from one or several trained models, with test-time augmentation.

Training already shows the network all eight orientations of the square (see
dataset.dihedral), so at inference each photograph can be predicted in all
eight and the answers averaged once each is turned back. Averaging several
checkpoints is the same operation one level up, so both live here, and every
caller - validation, sweep.py and predict.py - goes through it.
"""
from pathlib import Path

import torch

from model import FilaNet

# (quarter turns, mirrored) for every symmetry of the square; the first is identity
ORIENTATIONS = [(k, mirror) for mirror in (False, True) for k in range(4)]


def orient(x: torch.Tensor, k: int, mirror: bool) -> torch.Tensor:
    """The same transform dataset.dihedral applies: mirror first, then turn."""
    if mirror:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


def unorient(x: torch.Tensor, k: int, mirror: bool) -> torch.Tensor:
    """Undo `orient`: turn back first, then mirror back."""
    if k:
        x = torch.rot90(x, -k, dims=(-2, -1))
    return torch.flip(x, dims=[-1]) if mirror else x


def predict_probability(models, image: torch.Tensor, tta: bool = False,
                        identity_logits: torch.Tensor | None = None) -> torch.Tensor:
    """Mean foreground probability over `models` and, with `tta`, all eight orientations.

    `image` is a (1, C, H, W) batch already on the models' device; the result is
    (H, W) on that device. Probabilities are averaged rather than logits, so one
    over-confident view cannot outvote the rest.

    `identity_logits`, when the caller has already run the first model on the
    unturned image (validation does, for its loss), is reused instead of
    computing that view again.
    """
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    orientations = ORIENTATIONS if tta else ORIENTATIONS[:1]
    total = None
    for index, model in enumerate(models):
        for k, mirror in orientations:
            if index == 0 and (k, mirror) == (0, False) and identity_logits is not None:
                logits = identity_logits
            else:
                logits = model(orient(image, k, mirror))
            # channel 0 is the mask; a second channel, if any, is the centreline
            probability = unorient(torch.sigmoid(logits[:, :1]), k, mirror)
            total = probability if total is None else total + probability
    return total[0, 0] / (len(models) * len(orientations))


def load_model(path: str | Path, device: str = "cuda", use_ema: bool = True) -> FilaNet:
    """A trained FilaNet from a checkpoint, in eval mode.

    Prefers the averaged (EMA) weights when the checkpoint has them, since those
    are what validation scored.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state.get("ema") if use_ema and state.get("ema") is not None else state["model"]
    return FilaNet.from_state_dict(weights).to(device).eval()
