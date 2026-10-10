import json
from dataclasses import replace

import pytest
import torch
from PIL import Image

from khunc import ModelConfig, QuantizationConfig, TrainConfig, train
from khunc import test_model as run_test
from khunc.codec.quantization import quantize_dequantize
from khunc.models.network import KHUNC
from khunc.training.data import create_data
from khunc.training.trainer import TrainingModule


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "imagenette"
    for split, count in [("train", 4), ("val", 2)]:
        for category in ["class1", "class2"]:
            folder = root / split / category
            folder.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (32, 32), (index * 40, 80, 120)).save(folder / f"{index}.png")
    return TrainConfig(
        data_dir=root,
        output_dir=tmp_path / "runs",
        model=ModelConfig(base_channels=2),
        image_size=32,
        loss="mse",
        epochs=1,
        batch_size=2,
        num_workers=0,
        accelerator="cpu",
    )


def test_split_is_reproducible_disjoint_and_excludes_official_validation(config):
    first = create_data(config)
    second = create_data(config)
    assert first.manifest == second.manifest
    groups = {name: set(paths) for name, paths in first.manifest["splits"].items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert len(groups["train"] | groups["validation"]) == 8
    assert all(path.startswith("val/") for path in groups["test"])


@pytest.mark.parametrize("method", ["none", "ste", "noise"])
def test_encoder_and_decoder_learn_with_each_surrogate(config, method):
    config = replace(
        config, quantization=QuantizationConfig(storage="uint8", bits=6, training=method)
    )
    module = TrainingModule(KHUNC(config.model), config)
    loss = module.training_step(torch.rand(2, 3, 32, 32), 0)
    loss.backward()
    for network in [module.model.encoder, module.model.decoder]:
        gradients = [value.grad for value in network.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
        )
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_noise_training_uses_real_rounding_during_validation(config):
    config = replace(
        config, quantization=QuantizationConfig(storage="uint8", bits=2, training="noise")
    )
    module = TrainingModule(KHUNC(config.model), config).eval()
    image = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        z = module.model.encoder(image)
        reconstruction = module.model.decoder(quantize_dequantize(z, config.quantization))
        expected = module.reconstruction_loss(image, reconstruction)
        actual = module.validation_step(image, 0)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("augment", [False, True])
def test_training_saves_and_resumes_a_complete_experiment(config, augment):
    config = replace(config, augment=augment)
    first = train(config)
    assert first.best_checkpoint.is_file()
    assert (first.experiment.path / "checkpoints" / "last.ckpt").is_file()
    assert (first.experiment.path / "logs" / "training" / "metrics.csv").is_file()
    assert first.experiment.load_config() == config
    state = json.loads((first.experiment.path / "status.json").read_text())
    assert state["state"] == "trained"
    metrics = run_test(first.model, config)
    assert metrics["test_loss"] >= 0
    resume_path = first.experiment.path / "checkpoints/last.ckpt"
    if not augment:
        legacy = torch.load(resume_path, weights_only=True)
        for key in ["augment", "crop_scale", "horizontal_flip", "vertical_flip", "color_jitter"]:
            legacy["hyper_parameters"]["config"].pop(key)
        resume_path = first.experiment.path / "checkpoints/legacy.ckpt"
        torch.save(legacy, resume_path)
    resumed = train(replace(config, epochs=2), resume_from=resume_path)
    checkpoint = torch.load(resumed.best_checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert (
        checkpoint["global_step"]
        > torch.load(first.best_checkpoint, weights_only=True)["global_step"]
    )


def test_plateau_reduces_learning_rate_and_stops_training(config, monkeypatch):
    config = replace(config, epochs=10, lr_scheduler_patience=0, early_stopping_patience=2)
    monkeypatch.setattr(
        TrainingModule,
        "reconstruction_loss",
        lambda self, image, reconstructed: reconstructed.mean() * 0 + 1,
    )
    result = train(config)
    last = torch.load(result.experiment.path / "checkpoints/last.ckpt", weights_only=True)
    assert last["epoch"] == 2
    assert last["optimizer_states"][0]["param_groups"][0]["lr"] < config.learning_rate


def test_augmentation_varies_training_images_but_not_validation_or_test(config):
    for split in ["train", "val"]:
        for path in config.data_dir.joinpath(split).rglob("*.png"):
            pixels = torch.randint(0, 256, (72, 96, 3), dtype=torch.uint8).numpy()
            Image.fromarray(pixels).save(path)
    augmented = create_data(replace(config, augment=True))
    baseline = create_data(config)
    assert augmented.manifest == baseline.manifest
    samples = [augmented.train.dataset[0] for _ in range(3)]
    assert not torch.equal(samples[0], samples[1])
    for sample in samples:
        assert sample.shape == (3, config.image_size, config.image_size)
        assert sample.dtype == torch.float32
        assert torch.isfinite(sample).all() and 0 <= sample.min() <= sample.max() <= 1
    for name in ["validation", "test"]:
        dataset = getattr(augmented, name).dataset
        assert torch.equal(dataset[0], dataset[0])
        assert torch.equal(dataset[0], getattr(baseline, name).dataset[0])


@pytest.mark.parametrize("workers", [0, 2])
def test_augmented_batches_are_reproducible_with_the_same_seed(config, workers):
    config = replace(config, augment=True, num_workers=workers)
    torch.manual_seed(config.seed)
    first = next(iter(create_data(config).train))
    torch.manual_seed(config.seed)
    second = next(iter(create_data(config).train))
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "settings",
    [
        {"crop_scale": (0.0, 1.0)},
        {"crop_scale": (0.8, 0.4)},
        {"horizontal_flip": 1.1},
        {"vertical_flip": -0.1},
        {"color_jitter": -0.1},
    ],
)
def test_invalid_augmentation_settings_are_rejected(config, settings):
    with pytest.raises(ValueError):
        replace(config, **settings)
