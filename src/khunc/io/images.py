from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path) -> torch.Tensor:
    with Image.open(Path(path).expanduser()) as image:
        pixels = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div(255)


def save_image(image: torch.Tensor, path: str | Path) -> Path:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("save_image expects one image")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected an RGB CHW tensor")
    pixels = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.numpy()).save(path)
    return path
