from time import monotonic

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping


class StatusLines(Callback):
    def __init__(self, best_loss=float("inf")):
        self.pending = None
        self.best_loss = best_loss

    def state_dict(self):
        return {"best_loss": self.best_loss}

    def load_state_dict(self, state_dict):
        self.best_loss = state_dict["best_loss"]

    def _print(self, trainer, message):
        if trainer.is_global_zero:
            print(message, flush=True)

    def _start(self, trainer, stage, total):
        self.stage = stage
        self.total = total
        self.started = self.updated = monotonic()
        self._print(trainer, f"{stage}: starting, 0/{total} batches")

    def _batch(self, trainer, batch_idx):
        now = monotonic()
        if now - self.updated >= 15 or batch_idx + 1 == self.total:
            self._print(
                trainer,
                f"{self.stage}: {batch_idx + 1}/{self.total} batches, "
                f"{now - self.started:.0f}s elapsed",
            )
            self.updated = now

    def _summary(self, trainer):
        if self.pending is None:
            return
        epoch, train_loss, val_loss, elapsed = self.pending
        self.pending = None
        improvement = "improved" if val_loss < self.previous_best else "did not improve"
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        lr_change = f"reduced from {self.previous_lr:.6g}" if lr < self.previous_lr else "unchanged"
        # Plateau reduces after patience + 1 bad epochs; EarlyStopping stops at patience.
        remaining = max(0, scheduler.patience + 1 - scheduler.num_bad_epochs)
        early = next((c for c in trainer.callbacks if isinstance(c, EarlyStopping)), None)
        stopping = "EarlyStopping disabled"
        if early is not None:
            stopping = (
                f"EarlyStopping: {early.wait_count}/{early.patience} checks without improvement"
            )
            if trainer.should_stop:
                stopping += ", stop requested"
        self._print(
            trainer,
            f"Epoch {epoch}/{trainer.max_epochs} complete in {elapsed:.0f}s: "
            f"train_loss={train_loss:.6g}, val_loss={val_loss:.6g} ({improvement}), "
            f"best={min(self.previous_best, val_loss):.6g}\n"
            f"LR={lr:.6g} ({lr_change}); ReduceLROnPlateau: "
            f"{scheduler.num_bad_epochs}/{scheduler.patience + 1} bad epochs, "
            f"{remaining} more before reduction; {stopping}",
        )

    def on_train_epoch_start(self, trainer, pl_module):
        # Lightning steps the plateau scheduler after on_train_epoch_end.
        self._summary(trainer)
        self.previous_best = self.best_loss
        self.previous_lr = trainer.optimizers[0].param_groups[0]["lr"]
        self.epoch_started = monotonic()
        self._start(
            trainer,
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} training",
            trainer.num_training_batches,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch(trainer, batch_idx)

    def on_validation_epoch_start(self, trainer, pl_module):
        stage = "Sanity validation" if trainer.sanity_checking else "Validation"
        counts = (
            trainer.num_sanity_val_batches if trainer.sanity_checking else trainer.num_val_batches
        )
        self._start(trainer, stage, sum(counts))

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._batch(trainer, batch_idx)

    def on_validation_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self._print(trainer, f"{self.stage} complete: val_loss={float(loss):.6g}")

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.best_loss = min(self.best_loss, float(metrics["val_loss"]))
        self.pending = (
            trainer.current_epoch + 1,
            float(metrics["train_loss_epoch"]),
            float(metrics["val_loss"]),
            monotonic() - self.epoch_started,
        )

    def on_train_end(self, trainer, pl_module):
        self._summary(trainer)

    def on_test_epoch_start(self, trainer, pl_module):
        self._start(trainer, "Test", sum(trainer.num_test_batches))

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._batch(trainer, batch_idx)

    def on_test_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("test_loss")
        if loss is not None:
            self._print(trainer, f"Test complete: test_loss={float(loss):.6g}")

    def on_exception(self, trainer, pl_module, exception):
        self._print(trainer, f"Run interrupted: {type(exception).__name__}: {exception}")
