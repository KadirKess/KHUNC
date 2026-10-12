import copy
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_msssim import ms_ssim

from ..codec.quantization import quantize_dequantize, training_latent
from ..config import TrainConfig
from ..experiments import Experiment, write_json
from ..models.checkpoints import file_digest, load_model
from ..models.network import KHUNC, pad_image
from ..training.data import create_data


class TrainingModule(pl.LightningModule):
    def __init__(self, model: KHUNC, config: TrainConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters({"config": config.to_dict()})

    def reconstruction_loss(self, image, reconstructed):
        # MS-SSIM's variance estimates need float32 even under mixed precision.
        with torch.autocast(device_type=image.device.type, enabled=False):
            image, reconstructed = image.float(), reconstructed.float()
            mse = torch.nn.functional.mse_loss(reconstructed, image)
            if self.config.loss == "mse":
                return mse
            perceptual = 1 - ms_ssim(image, reconstructed, data_range=1.0)
            if self.config.loss == "ms_ssim_mse":
                return perceptual + self.config.mse_weight * mse
            return perceptual

    def _step(self, image, stage):
        height, width = image.shape[-2:]
        z = self.model.encoder(pad_image(image))
        if stage == "train":
            z = training_latent(z, self.config.quantization)
        else:
            z = quantize_dequantize(z, self.config.quantization)
        reconstruction = self.model.decoder(z)[..., :height, :width]
        loss = self.reconstruction_loss(image, reconstruction)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=image.shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


@dataclass
class TrainingResult:
    experiment: Experiment
    best_checkpoint: Path
    model: KHUNC


def train(
    config: TrainConfig, *, checkpoint=None, resume_from=None, progress_refresh_rate=20
) -> TrainingResult:
    """checkpoint starts fine-tuning; resume_from restores a KHUNC training run."""
    if checkpoint is not None and resume_from is not None:
        raise ValueError("Choose checkpoint or resume_from, not both")
    pl.seed_everything(config.seed, workers=True)
    model = load_model(checkpoint, device="cpu") if checkpoint else KHUNC(config.model)
    if model.config != config.model:
        raise ValueError("Checkpoint architecture differs from config.model")
    if resume_from:
        saved = torch.load(Path(resume_from).expanduser(), map_location="cpu", weights_only=True)
        previous = saved.get("hyper_parameters", {}).get("config")
        if previous is None:
            raise ValueError("Use checkpoint= to fine-tune an original notebook checkpoint")
        previous = TrainConfig.from_dict(previous).to_dict()
        current = config.to_dict()
        ignored = {"name", "output_dir", "epochs", "accelerator", "num_workers"}
        if any(previous[key] != current[key] for key in current if key not in ignored):
            raise ValueError("Resuming requires the same data, model and training configuration")
    splits = create_data(config)
    experiment = Experiment.create(config.name, config.output_dir, config)
    write_json(experiment.path / "splits.json", splits.manifest)
    source = checkpoint or resume_from
    if source:
        write_json(
            experiment.path / "source_checkpoint.json",
            {
                "path": str(Path(source).expanduser().resolve()),
                "sha256": file_digest(source),
                "mode": "resume" if resume_from else "fine_tune",
            },
        )
    module = TrainingModule(model, config)
    callback = ModelCheckpoint(
        dirpath=experiment.path / "checkpoints",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    latest = ModelCheckpoint(
        dirpath=experiment.path / "checkpoints",
        filename="last",
        save_top_k=1,
        enable_version_counter=False,
        every_n_epochs=1,
    )
    callbacks = [callback, latest, TQDMProgressBar(refresh_rate=progress_refresh_rate, leave=False)]
    if config.early_stopping_patience is not None:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience))
    if config.compile_model:
        module.model = torch.compile(module.model)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        devices=1,
        precision=config.precision,
        callbacks=callbacks,
        logger=CSVLogger(experiment.path / "logs", name="training", version=""),
        default_root_dir=experiment.path,
        deterministic="warn",
    )
    experiment.set_status("training")
    try:
        trainer.fit(
            module,
            train_dataloaders=splits.train,
            val_dataloaders=splits.validation,
            ckpt_path=str(Path(resume_from).expanduser()) if resume_from else None,
        )
        if not callback.best_model_path:
            raise RuntimeError("Training produced no validated checkpoint")
        best = Path(callback.best_model_path)
        experiment.set_status(
            "trained",
            best_checkpoint=str(best),
            best_validation_loss=float(callback.best_model_score),
        )
    except BaseException as error:
        experiment.set_status(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error)
        )
        raise
    return TrainingResult(experiment, best, load_model(best, device="cpu"))


def test_model(model: KHUNC, config: TrainConfig, *, progress_refresh_rate=20) -> dict:
    splits = create_data(config)
    module = TrainingModule(copy.deepcopy(model), config)
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=1,
        precision=config.precision,
        callbacks=[TQDMProgressBar(refresh_rate=progress_refresh_rate, leave=False)],
        logger=False,
        enable_checkpointing=False,
    )
    return trainer.test(module, dataloaders=splits.test, verbose=False)[0]
