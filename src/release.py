"""Load the released FilaNet weights, which ship as several files.

GitHub refuses files over 100 MB and the network's weights are 125 MB, so the
state dict is split across files listed in weights/manifest.json, each with its
SHA-256. They are checked and joined here into exactly the tensors the
submission was made with.
"""
import hashlib
import json
from pathlib import Path

import torch

from model import FilaNet


def load_state_dict(weights_dir: str | Path) -> dict:
    """The full state dict from the shards in `weights_dir`, after checking their hashes."""
    weights_dir = Path(weights_dir)
    manifest = json.loads((weights_dir / "manifest.json").read_text())
    state_dict = {}
    for entry in manifest["files"]:
        path = weights_dir / entry["name"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"{path} is corrupt or incomplete (sha256 {digest})")
        state_dict.update(torch.load(path, map_location="cpu", weights_only=True))
    return state_dict


def load_released_model(weights_dir: str | Path, device: str = "cuda") -> FilaNet:
    """The trained FilaNet, in eval mode on `device`."""
    return FilaNet.from_state_dict(load_state_dict(weights_dir)).to(device).eval()
