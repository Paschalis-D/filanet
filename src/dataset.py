from pathlib import Path
from typing import Optional
import json
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, functional
from pycocotools import mask as mask_utils


def split_by_image(
    image_ids: list[str],
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split annotator-image ids into train and validation, dividing on the photograph.

    Ids look like ``<annotator>-<image>`` and the same photograph is annotated by
    up to three people. Splitting on the raw id would put one annotator's copy in
    train and another's copy of the same photograph in validation, so every copy
    of a photograph is kept on the same side.
    """
    stems = sorted({image_id.split("-", 1)[1] for image_id in image_ids})
    random.Random(seed).shuffle(stems)
    val_stems = set(stems[: round(len(stems) * val_fraction)])

    train = [i for i in image_ids if i.split("-", 1)[1] not in val_stems]
    val = [i for i in image_ids if i.split("-", 1)[1] in val_stems]
    return train, val


def dihedral(image: torch.Tensor, mask: torch.Tensor, k: int, mirror: bool):
    """Apply one of the eight square symmetries to an image and its mask together.

    A quarter turn plus an optional mirror gives the dihedral group of the
    square: eight orientations, all exactly label-preserving, so one annotated
    photograph becomes eight training samples without inventing any pixels.

    The solar disk is circular and centred by preprocessing, so a quarter turn
    leaves it where it was - unlike an arbitrary angle, which would need
    interpolation and would blur filaments only a few pixels across. Mirroring
    reverses filament chirality, which is fine for predicting masks but would be
    wrong if this model were ever asked to classify handedness.
    """
    if mirror:
        image, mask = torch.flip(image, dims=[-1]), torch.flip(mask, dims=[-1])
    if k:
        image = torch.rot90(image, k, dims=(-2, -1))
        mask = torch.rot90(mask, k, dims=(-2, -1))
    # rot90 and flip return views, and some convolution paths need contiguity
    return image.contiguous(), mask.contiguous()


def random_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    size: int,
    foreground_prob: float = 0.7,
):
    """Take a random `size`x`size` window from an image and its mask together.

    Training on windows rather than whole photographs buys optimizer steps. A
    2048x2048 frame at batch 1 gives 245 steps an epoch, which is nowhere near
    enough to converge a network this size; a 512 window is a sixteenth of the
    pixels, so the same hardware fits a batch of eight and takes roughly eight
    times as many steps in the same hour. It also gives BatchNorm more than one
    image to estimate its statistics from.

    Filaments cover 0.36% of a frame, so a uniform window would miss every
    filament about two thirds of the time and most steps would carry no
    detection signal at all. `foreground_prob` of the windows are therefore
    centred on a filament pixel, and the rest are uniform - those still matter,
    because they are what teaches the network what is *not* a filament.
    """
    height, width = mask.shape[-2:]
    if size >= height or size >= width:
        return image, mask

    limit_y, limit_x = height - size, width - size
    foreground = torch.nonzero(mask[0] > 0) if foreground_prob > 0 else None
    if foreground is not None and len(foreground) and torch.rand(()) < foreground_prob:
        centre = foreground[int(torch.randint(len(foreground), ()))]
        top = int(centre[0]) - size // 2
        left = int(centre[1]) - size // 2
        top = max(0, min(top, limit_y))
        left = max(0, min(left, limit_x))
    else:
        top = int(torch.randint(limit_y + 1, ()))
        left = int(torch.randint(limit_x + 1, ()))

    window = (..., slice(top, top + size), slice(left, left + size))
    return image[window].contiguous(), mask[window].contiguous()


# How the annotators of one photograph are combined into a training target.
# "annotator" keeps one sample per annotator, as the data ships. The others give
# one sample per photograph. Measured on the training annotations, converting
# each consensus to instances and scoring it against every annotator - which is
# what the leaderboard does - gives union 0.62, majority 0.58 and unanimity 0.47
# PQ, while one annotator scored against the others gets 0.34. Training on each
# annotator separately shows the network up to three contradictory masks of the
# same photograph and so pulls it towards their average.
TARGETS = ("annotator", "union", "majority", "soft")


# Centrelines are drawn this many px wide. A one-pixel line is too thin a target
# to learn from at 2048px, and filaments are 13px across at their median, so
# 3px stays well inside every filament but its narrowest tips.
SPINE_WIDTH = 3

# Small filaments are what the model misses: on validation 55% of those under
# 600px were never touched by any prediction, against 4% of those over 4000px.
# A pixel-wise loss counts a filament by its pixels, so a 300px filament weighs
# a twentieth of a 6000px one, while PQ counts each filament once. Pixels of a
# filament smaller than the reference area are weighted up by the square root
# of how much smaller - a 300px filament by 2.2, a 100px one by 3.9 - capped,
# so a stray tiny annotation cannot dominate a batch. The reference is roughly
# the median annotated filament (1228px).
SIZE_REFERENCE_AREA = 1500
SIZE_MAX_WEIGHT = 4.0


def size_weight(area: int) -> float:
    """Loss weight on the pixels of a filament of `area` px; 1 at and above the reference."""
    return float(np.clip(np.sqrt(SIZE_REFERENCE_AREA / max(area, 1)), 1.0, SIZE_MAX_WEIGHT))


def combine_annotators(masks: list[torch.Tensor], target: str) -> torch.Tensor:
    """One target from the masks several annotators drew on the same photograph.

    "majority" needs strictly more than half, so with two annotators it is their
    intersection. "soft" is the fraction of annotators who marked each pixel.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    stack = torch.stack(masks)
    if target == "annotator" or len(masks) == 1:
        return stack[0]
    if target == "union":
        return stack.amax(dim=0)
    if target == "majority":
        return (stack.sum(dim=0) * 2 > len(masks)).float()
    return stack.mean(dim=0)


def group_by_photograph(ids: list[str]) -> list[list[str]]:
    """Annotator-image ids grouped by photograph, in order of first appearance."""
    groups: dict[str, list[str]] = {}
    for image_id in ids:
        groups.setdefault(image_id.split("-", 1)[1], []).append(image_id)
    return list(groups.values())


class FilamentDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        json_path: Optional[Path] = None,
        is_train: bool = True,
        ids: Optional[list[str]] = None,
        augment: bool = False,
        crop: Optional[int] = None,
        crop_foreground_prob: float = 0.7,
        target: str = "annotator",
        spine: bool = False,
        size_balance: bool = False,
    ):
        """`spine` and `size_balance` add channels to the target, in the order
        given by `self.channels`: the mask is always channel 0, then the
        filament centrelines, then a per-pixel loss weight."""
        self.images_dir = Path(images_dir)
        self.is_train = is_train
        self.channels = ["mask"] + (["spine"] if spine else []) + (["weight"] if size_balance else [])
        # off by default: validation has to measure the task, not a variant of it
        self.augment = augment
        self.crop = crop
        self.crop_foreground_prob = crop_foreground_prob
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        self.target = target
        self.to_tensor = ToTensor()

        if is_train and json_path is None:
            raise ValueError("json_path is required when is_train=True")

        if is_train:
            with open(json_path, "r") as f:
                coco = json.load(f)

            # image_id -> (file_name, height, width)
            self.image_meta: dict[str, tuple[str, int, int]] = {
                img["id"]: (img["file_name"], img["height"], img["width"])
                for img in coco["images"]
            }

            # image_id -> list of polygon segmentations (one per filament), and
            # the centreline polyline drawn along each, in the same order
            self.id_to_segmentations: dict[str, list] = {}
            self.id_to_spines: dict[str, list] = {}
            for ann in coco["annotations"]:
                self.id_to_segmentations.setdefault(ann["image_id"], []).append(
                    ann["segmentation"]
                )
                self.id_to_spines.setdefault(ann["image_id"], []).append(ann.get("spine") or [])

            # Stable, precomputed index -> image_id mapping
            self.ids: list[str] = sorted(self.image_meta.keys()) if ids is None else list(ids)
        else:
            self.image_meta = {}
            self.id_to_segmentations = {}
            self.id_to_spines = {}
            self.ids = sorted(p.name for p in self.images_dir.iterdir()
                              if p.suffix.lower() in {".jpg", ".jpeg"})

        # the annotator-image ids behind each sample: one each, or a photograph's worth
        self.samples: list[list[str]] = (
            [[i] for i in self.ids] if target == "annotator" or not is_train
            else group_by_photograph(self.ids)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_mask(self, image_id: str, height: int, width: int) -> torch.Tensor:
        canvas = np.zeros((height, width), dtype=np.uint8)
        for polygon in self.id_to_segmentations.get(image_id, []):
            rles = mask_utils.frPyObjects(polygon, height, width)
            rle = mask_utils.merge(rles)
            canvas = np.maximum(canvas, mask_utils.decode(rle))
        return torch.from_numpy(canvas).unsqueeze(0).float()

    def _build_spine(self, image_id: str, height: int, width: int) -> torch.Tensor:
        canvas = np.zeros((height, width), dtype=np.uint8)
        for spine in self.id_to_spines.get(image_id, []):
            points = np.round(np.asarray(spine, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
            if len(points) >= 2:
                # OpenCV draws thickness t as t + 1 px wide for t > 1 - thickness 3
                # is a 5px line - so one less gives SPINE_WIDTH
                cv.polylines(canvas, [points], isClosed=False, color=1,
                             thickness=max(SPINE_WIDTH - 1, 1))
        return torch.from_numpy(canvas).unsqueeze(0).float()

    def _build_weight(self, image_id: str, height: int, width: int) -> torch.Tensor:
        weight = np.ones((height, width), dtype=np.float32)
        for polygon in self.id_to_segmentations.get(image_id, []):
            instance = mask_utils.decode(mask_utils.merge(
                mask_utils.frPyObjects(polygon, height, width))).astype(bool)
            area = int(instance.sum())
            if area:
                weight[instance] = np.maximum(weight[instance], size_weight(area))
        return torch.from_numpy(weight).unsqueeze(0)

    def _build_target(self, keys: list[str], height: int, width: int) -> torch.Tensor:
        parts = [combine_annotators([self._build_mask(k, height, width) for k in keys], self.target)]
        if "spine" in self.channels:
            # a 3px line's caps reach past the end of its filament, and a
            # centreline outside the mask would contradict channel 0
            spine = combine_annotators([self._build_spine(k, height, width) for k in keys],
                                       self.target)
            parts.append(torch.minimum(spine, parts[0]))
        if "weight" in self.channels:
            # a pixel any annotator put in a small filament keeps that filament's weight
            parts.append(torch.stack([self._build_weight(k, height, width) for k in keys]).amax(0))
        return torch.cat(parts)

    def __getitem__(self, index: int):
        keys = self.samples[index]
        key = keys[0]

        if self.is_train:
            # every annotator-image of a photograph points at the same file
            file_name, height, width = self.image_meta[key]
        else:
            file_name = key

        img_path = self.images_dir / file_name
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image = self.to_tensor(img)  # (1, H, W), float32 in [0, 1]

        if not self.is_train:
            return image, key

        mask = self._build_target(keys, height, width)
        if self.augment:
            image, mask = dihedral(
                image, mask,
                k=int(torch.randint(4, ())),
                mirror=bool(torch.randint(2, ())),
            )
        if self.crop:
            image, mask = random_crop(image, mask, self.crop, self.crop_foreground_prob)
        return image, mask
    
if __name__ == "__main__":
    data = FilamentDataset(images_dir=Path("filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train\\train_images"), json_path=Path("filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train\\MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    data_to_plot = 0
    print(len(data))
    print(data[data_to_plot])
    
    img, mask = data[data_to_plot]
    img = img.numpy().squeeze()
    mask = mask.numpy().squeeze()
    cv.imshow("Image", img)
    cv.imshow("Mask", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    overlay = cv.addWeighted(img, 0.5, mask, 0.5, 0)
    cv.imshow("Overlay", overlay)
    cv.waitKey(0)
    cv.destroyAllWindows()
    