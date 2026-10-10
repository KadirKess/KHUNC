import hashlib
from dataclasses import asdict
from pathlib import Path

import torch

from ..config import ModelConfig
from ..models.network import KHUNC


def choose_device(device: str | torch.device = "auto") -> torch.device:
    if str(device) != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_fingerprint(model: KHUNC) -> bytes:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.digest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: str | Path, device: str | torch.device = "auto") -> KHUNC:
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {name.replace("_orig_mod.", ""): value for name, value in state.items()}
    if all(name.startswith("model.") for name in state):
        state = {name[6:]: value for name, value in state.items()}
    config = ModelConfig(
        base_channels=state["encoder.inc.double_conv.0.weight"].shape[0],
        latent_channels=state["encoder.outc.conv.weight"].shape[0],
        encoder_up_layers=sum(f"encoder.up{index}.up.weight" in state for index in range(1, 5)),
    )
    model = KHUNC(config)
    model.load_state_dict(state, strict=True)
    return model.to(choose_device(device)).eval()


def save_model(model: KHUNC, path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"state_dict": state, "model_config": asdict(model.config)}, path)
    return path
