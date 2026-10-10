import csv
from pathlib import Path

from .benchmark.evaluation import EvaluationResult
from .io.images import load_image


def _save(figure, path):
    if path is not None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
    return figure


def plot_reconstruction(original, reconstructed, *, path=None):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, image, title in zip(axes, [original, reconstructed], ["Original", "Reconstructed"]):
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if image.ndim == 4:
            image = image[0]
        axis.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    return _save(figure, path)


def plot_training(metrics_path: str | Path, *, path=None):
    import matplotlib.pyplot as plt

    with Path(metrics_path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    figure, axis = plt.subplots(figsize=(8, 4))
    for key, label in [("train_loss_epoch", "Training"), ("val_loss", "Validation")]:
        samples = [
            (float(row["epoch"]), float(row[key]))
            for row in rows
            if row.get("epoch") and row.get(key)
        ]
        if samples:
            axis.plot(*zip(*samples), label=label)
    axis.set(xlabel="Epoch", ylabel="Reconstruction loss")
    axis.legend()
    figure.tight_layout()
    return _save(figure, path)


def plot_quantization(comparison: EvaluationResult, *, path=None):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    for mode, values in comparison.summary.items():
        x, y = values["mean_compressed_bytes"] / 1024, values["mean_ssimulacra2"]
        axis.scatter(x, y)
        axis.annotate(mode, (x, y), xytext=(5, 5), textcoords="offset points")
    axis.set(xlabel="Mean stored size, KiB", ylabel="Mean SSIMULACRA2")
    figure.tight_layout()
    return _save(figure, path)
