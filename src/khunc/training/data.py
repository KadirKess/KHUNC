import random
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from ..config import TrainConfig
from ..io.images import load_image

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int, augmentation: TrainConfig | None = None):
        self.paths = paths
        if augmentation is not None and augmentation.augment:
            self.transform = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        image_size, scale=augmentation.crop_scale, ratio=(1.0, 1.0), antialias=True
                    ),
                    v2.RandomHorizontalFlip(p=augmentation.horizontal_flip),
                    v2.RandomVerticalFlip(p=augmentation.vertical_flip),
                    v2.ColorJitter(
                        brightness=augmentation.color_jitter,
                        contrast=augmentation.color_jitter,
                        saturation=augmentation.color_jitter,
                    ),
                ]
            )
        else:
            self.transform = v2.Resize((image_size, image_size), antialias=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_image(self.paths[index])).clamp(0, 1)


@dataclass
class DataSplits:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    manifest: dict


def _image_paths(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _EXTENSIONS)


def create_data(config: TrainConfig) -> DataSplits:
    root = Path(config.data_dir).expanduser().resolve()
    if not (root / "train").is_dir() or not (root / "val").is_dir():
        raise FileNotFoundError(f"Expected Imagenette train/ and val/ directories under {root}")
    train, validation = [], []
    rng = random.Random(config.seed)
    classes = sorted(p for p in (root / "train").iterdir() if p.is_dir())
    if not classes:
        raise ValueError("Imagenette train/ has no class directories")
    for directory in classes:
        paths = _image_paths(directory)
        if len(paths) < 2:
            raise ValueError(f"Need at least two training images in {directory.name}")
        rng.shuffle(paths)
        count = max(1, min(len(paths) - 1, round(len(paths) * config.validation_fraction)))
        validation.extend(paths[:count])
        train.extend(paths[count:])
    test = _image_paths(root / "val")
    if not test:
        raise ValueError("Imagenette val/ has no test images")
    groups = {"train": train, "validation": validation, "test": test}
    manifest = {
        "root": str(root),
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "splits": {
            name: [str(p.relative_to(root)) for p in paths] for name, paths in groups.items()
        },
    }
    loaders = {}
    for name, paths in groups.items():
        loaders[name] = DataLoader(
            ImageDataset(paths, config.image_size, config if name == "train" else None),
            batch_size=config.batch_size,
            shuffle=name == "train",
            num_workers=config.num_workers,
            pin_memory=config.accelerator != "cpu" and torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
            generator=torch.Generator().manual_seed(config.seed),
        )
    return DataSplits(loaders["train"], loaders["validation"], loaders["test"], manifest)


def download_imagenette(root="data", size="full") -> Path:
    names = {"full": "imagenette2", "320": "imagenette2-320", "160": "imagenette2-160"}
    if str(size) not in names:
        raise ValueError("size must be full, 320 or 160")
    name = names[str(size)]
    root = Path(root).expanduser()
    target = root / name
    if (target / "train").is_dir() and (target / "val").is_dir():
        return target
    if target.exists():
        raise FileExistsError(f"Incomplete dataset at {target}; choose a different root")
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://s3.amazonaws.com/fast-ai-imageclas/{name}.tgz"
    with tempfile.TemporaryDirectory(prefix="imagenette-", dir=root) as directory:
        staging = Path(directory)
        archive_path = staging / "dataset.tgz"
        with urlopen(url, timeout=60) as source, archive_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        with tarfile.open(archive_path) as archive:
            for member in archive:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != name:
                    raise ValueError("Unexpected path in Imagenette archive")
                destination = staging / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                else:
                    raise ValueError("Unexpected non-file member in Imagenette archive")
        extracted = staging / name
        if not (extracted / "train").is_dir() or not (extracted / "val").is_dir():
            raise ValueError("Downloaded archive is not an Imagenette dataset")
        extracted.rename(target)
    return target
