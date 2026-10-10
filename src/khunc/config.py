from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    latent_channels: int = 3
    base_channels: int = 32
    encoder_up_layers: int = 2

    def __post_init__(self):
        if self.latent_channels < 1 or self.base_channels < 1:
            raise ValueError("Channel counts must be positive")
        if self.encoder_up_layers not in range(5):
            raise ValueError("encoder_up_layers must be between 0 and 4")

    @property
    def decoder_up_layers(self) -> int:
        return 4 - self.encoder_up_layers

    @property
    def downsampling_factor(self) -> int:
        return 2**self.decoder_up_layers


@dataclass(frozen=True)
class QuantizationConfig:
    storage: Literal["float32", "float16", "bfloat16", "uint8"] = "float32"
    bits: int = 8
    training: Literal["none", "ste", "noise"] = "none"

    def __post_init__(self):
        if self.storage not in {"float32", "float16", "bfloat16", "uint8"}:
            raise ValueError("storage must be float32, float16, bfloat16 or uint8")
        if not 1 <= self.bits <= 8:
            raise ValueError("bits must be between 1 and 8")
        if self.training not in {"none", "ste", "noise"}:
            raise ValueError("training must be none, ste or noise")
        if self.training == "noise" and self.storage != "uint8":
            raise ValueError("Additive noise requires uint8 storage")
        if self.training != "none" and self.storage == "float32":
            raise ValueError("Float32 storage does not need a quantization surrogate")


@dataclass(frozen=True)
class TrainConfig:
    name: str = "float32-baseline"
    data_dir: str | Path = "data/imagenette2"
    output_dir: str | Path = "experiments"
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    image_size: int = 384
    augment: bool = False
    crop_scale: tuple[float, float] = (0.5, 1.0)
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    color_jitter: float = 0.1
    validation_fraction: float = 0.1
    seed: int = 42
    num_workers: int = 2
    loss: Literal["ms_ssim", "mse", "ms_ssim_mse"] = "ms_ssim"
    mse_weight: float = 0.1
    accelerator: str = "auto"
    precision: str = "32-true"
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.1
    early_stopping_patience: int | None = 20
    compile_model: bool = False

    def __post_init__(self):
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        object.__setattr__(self, "crop_scale", tuple(self.crop_scale))
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if self.image_size < 16 or self.image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16")
        if len(self.crop_scale) != 2 or not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1:
            raise ValueError("crop_scale must contain two ordered fractions in (0, 1]")
        if not 0 <= self.horizontal_flip <= 1 or not 0 <= self.vertical_flip <= 1:
            raise ValueError("Flip probabilities must be between 0 and 1")
        if not 0 <= self.color_jitter <= 1:
            raise ValueError("color_jitter must be between 0 and 1")
        if self.loss not in {"ms_ssim", "mse", "ms_ssim_mse"}:
            raise ValueError("Unknown reconstruction loss")
        if self.loss != "mse" and self.image_size <= 160:
            raise ValueError("MS-SSIM needs image_size greater than 160")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.num_workers < 0 or self.mse_weight < 0:
            raise ValueError("num_workers and mse_weight cannot be negative")
        if self.precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise ValueError("Unsupported training precision")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.lr_scheduler_patience < 0 or not 0 < self.lr_scheduler_factor < 1:
            raise ValueError("Invalid learning rate scheduler settings")

    def to_dict(self) -> dict:
        values = asdict(self)
        values["data_dir"] = str(Path(self.data_dir).expanduser())
        values["output_dir"] = str(Path(self.output_dir).expanduser())
        return values

    @classmethod
    def from_dict(cls, values: dict) -> "TrainConfig":
        values = dict(values)
        values["model"] = ModelConfig(**values.get("model", {}))
        values["quantization"] = QuantizationConfig(**values.get("quantization", {}))
        return cls(**values)
