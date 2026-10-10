from .benchmark.evaluation import (
    EvaluationResult,
    compare_quantization,
    download_kodak,
    evaluate,
    evaluate_kodak,
    ssimulacra2,
)
from .codec.compression import Codec, compress, decompress, inspect_compressed
from .config import ModelConfig, QuantizationConfig, TrainConfig
from .experiments import Experiment
from .io.images import load_image, save_image
from .models.checkpoints import load_model, save_model
from .models.network import KHUNC

__all__ = [
    "KHUNC",
    "Codec",
    "ModelConfig",
    "QuantizationConfig",
    "TrainConfig",
    "Experiment",
    "EvaluationResult",
    "compress",
    "decompress",
    "inspect_compressed",
    "load_model",
    "save_model",
    "load_image",
    "save_image",
    "download_kodak",
    "evaluate",
    "evaluate_kodak",
    "compare_quantization",
    "ssimulacra2",
    "train",
    "test_model",
    "download_imagenette",
]


def train(config: TrainConfig, *, checkpoint=None, resume_from=None):
    from .training.trainer import train as run

    return run(config, checkpoint=checkpoint, resume_from=resume_from)


def test_model(model: KHUNC, config: TrainConfig) -> dict:
    from .training.trainer import test_model as run

    return run(model, config)


def download_imagenette(root="data", size="full"):
    from .training.data import download_imagenette as download

    return download(root, size)
