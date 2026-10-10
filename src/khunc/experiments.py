import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from .config import TrainConfig


def write_json(path: Path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def environment() -> dict:
    packages = {}
    for name in ["khunc", "torch", "torchvision", "pytorch-lightning", "pytorch-msssim", "numpy"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


@dataclass(frozen=True)
class Experiment:
    path: Path

    @classmethod
    def create(cls, name: str, root: str | Path = "experiments", config=None) -> "Experiment":
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-_")
        if not slug:
            raise ValueError("Experiment name must contain letters or numbers")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = Path(root).expanduser() / f"{timestamp}-{slug}-{uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        for directory in ["checkpoints", "logs", "evaluations", "plots", "images"]:
            (path / directory).mkdir()
        if config is not None:
            values = config.to_dict() if isinstance(config, TrainConfig) else config
            write_json(path / "config.json", values)
        write_json(path / "environment.json", environment())
        write_json(path / "status.json", {"state": "created"})
        return cls(path)

    def load_config(self) -> TrainConfig:
        return TrainConfig.from_dict(json.loads((self.path / "config.json").read_text()))

    def set_status(self, state: str, **details):
        write_json(self.path / "status.json", {"state": state, **details})
