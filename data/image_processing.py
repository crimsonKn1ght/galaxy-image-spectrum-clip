"""Image preprocessing. Two transforms share this module (the domain seam for non-RGB inputs).

- ``clip_image_transform``: convert a multi-band (g, r, z) survey cutout to a CLIP-ready
  3-channel 224x224 tensor via a per-band arcsinh stretch (the standard Legacy Survey display
  transform) plus CLIP normalization. Used by the real image tower.
- ``raw_image_transform``: per-channel standardize the array with no resize. Used by the flatten
  image encoder on the synthetic smoke path (keeps CLIP weights out of the CPU smoke).
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F

# OpenAI CLIP normalization constants.
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def _to_three_bands(array: np.ndarray) -> np.ndarray:
    """Return a 3-channel array: keep the first three bands, or repeat the last if there are fewer."""
    channels = array.shape[0]
    if channels == 3:
        return array
    if channels > 3:
        return array[:3]
    pad = [array[-1:]] * (3 - channels)
    return np.concatenate([array, *pad], axis=0)


def clip_image_transform(image_cfg: Dict) -> Callable[[np.ndarray], torch.Tensor]:
    """Build a ``(C, H, W) array -> (3, size, size) tensor`` transform for the CLIP tower."""
    size = int(image_cfg.get("image_size", 224))
    scale = float(image_cfg.get("arcsinh_scale", 0.1))

    def transform(array: np.ndarray) -> torch.Tensor:
        array = _to_three_bands(np.asarray(array, dtype=np.float32))
        stretched = torch.from_numpy(np.arcsinh(array / scale).astype(np.float32))
        flat = stretched.view(3, -1)
        low = flat.min(dim=1).values.view(3, 1, 1)
        high = flat.max(dim=1).values.view(3, 1, 1)
        normalized = (stretched - low) / (high - low + 1e-6)
        resized = F.interpolate(
            normalized.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
        return (resized - _CLIP_MEAN) / _CLIP_STD

    return transform


def raw_image_transform(image_cfg: Dict) -> Callable[[np.ndarray], torch.Tensor]:
    """Build a ``(C, H, W) array -> (C, H, W) tensor`` transform (per-channel standardize, no resize)."""

    def transform(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        flat = tensor.view(tensor.shape[0], -1)
        mean = flat.mean(dim=1).view(-1, 1, 1)
        std = flat.std(dim=1).view(-1, 1, 1)
        return (tensor - mean) / (std + 1e-6)

    return transform


def _to_three_bands_batch(images: torch.Tensor) -> torch.Tensor:
    """Return a (B, 3, H, W) batch: keep the first three bands, or repeat the last if there are fewer."""
    channels = images.shape[1]
    if channels == 3:
        return images
    if channels > 3:
        return images[:, :3]
    pad = images[:, -1:].repeat(1, 3 - channels, 1, 1)
    return torch.cat([images, pad], dim=1)


def clip_batch_transform(images: np.ndarray, image_cfg: Dict, device: "torch.device") -> torch.Tensor:
    """Batched, on-device equivalent of ``clip_image_transform`` for feature precomputation.

    Takes a (B, C, H, W) array and returns a (B, 3, size, size) tensor on ``device``. Mirrors the
    per-image transform exactly (per-image, per-channel arcsinh stretch -> min-max -> resize -> CLIP
    normalize) but as one batched GPU op, which is far faster than transforming images one at a time.
    """
    size = int(image_cfg.get("image_size", 224))
    scale = float(image_cfg.get("arcsinh_scale", 0.1))
    x = torch.as_tensor(np.asarray(images, dtype=np.float32), device=device)
    x = _to_three_bands_batch(x)
    x = torch.arcsinh(x / scale)
    flat = x.flatten(2)
    low = flat.min(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    high = flat.max(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    x = (x - low) / (high - low + 1e-6)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - _CLIP_MEAN.to(device)) / _CLIP_STD.to(device)


def raw_batch_transform(images: np.ndarray, device: "torch.device") -> torch.Tensor:
    """Batched, on-device equivalent of ``raw_image_transform`` (per-channel standardize, no resize)."""
    x = torch.as_tensor(np.asarray(images, dtype=np.float32), device=device)
    flat = x.flatten(2)
    mean = flat.mean(dim=2).unsqueeze(-1).unsqueeze(-1)
    std = flat.std(dim=2).unsqueeze(-1).unsqueeze(-1)
    return (x - mean) / (std + 1e-6)
