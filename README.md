# KHUNC: Kinda Half U-Net Compression

Kinda Half U-Net Compression is an image autoencoder with a three-channel latent at one quarter of the input width and height by default. Use Python functions to train, compress, reconstruct, and compare experiments in a notebook.

## Setup

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KadirKess/KHUNC/blob/main/notebooks/experiments.ipynb)

Use Python 3.11 through 3.13 and uv.

```sh
git clone https://github.com/KadirKess/KHUNC.git
cd KHUNC
uv sync --extra notebook
uv run --extra notebook jupyter lab notebooks/experiments.ipynb
```

Open the GitHub notebook using the Colab button above each time you want the latest version. A saved Drive copy is independent and does not update when the repository changes. Experiment outputs still save to your own Drive. The setup cell updates the cloned Python code; use a fresh runtime after code changes so already-imported modules are reloaded.

The notebook includes a Colab setup cell. It installs KHUNC into the running Python environment so training can use Colab's GPU. Set `output_dir` to a folder in mounted Google Drive to keep experiment files after a session ends.

SSIMULACRA2 is an external executable used by the Python evaluator. On macOS, install it with `brew install jpeg-xl`. The notebook includes a Colab installer for the upstream implementation. Pass `binary="/path/to/ssimulacra2"` if it is not on your PATH. Evaluation raises an error if the metric is unavailable.

## Train and fine-tune

Edit a Python configuration directly in the notebook. Each run saves a snapshot; you do not need to maintain configuration files by hand.

```python
from khunc import TrainConfig, QuantizationConfig, download_imagenette, train

dataset = download_imagenette(size="full")
config = TrainConfig(
    name="float16",
    data_dir=dataset,
    output_dir="experiments",
    quantization=QuantizationConfig(storage="float16"),
    learning_rate=1e-4,
    epochs=10,
)
result = train(config, checkpoint="best_model.ckpt")
model = result.model
```

`train(config)` starts from new weights. `checkpoint=` fine-tunes saved weights with a new optimizer. `resume_from=` restores a KHUNC training checkpoint, including optimizer and scheduler state. When resuming, `epochs` is the total epoch target, not an additional count. The original notebook checkpoint supports fine-tuning and inference, but not optimizer resumption through the new trainer.

Defaults reproduce the original architecture, 384 x 384 tensor resizing, MS-SSIM loss, AdamW learning rate of 0.001, and 50 epochs. The default experiment uses float32 latents. ReduceLROnPlateau monitors `val_loss`, with `lr_scheduler_patience=10` and `lr_scheduler_factor=0.1`. EarlyStopping uses `early_stopping_patience=20`; set it to `None` to disable it. Both use validation with actual stored quantization. Compilation is opt-in through `compile_model=True`.

Imagenette's training directory is split into training and validation sets within each class, using `validation_fraction` and `seed`. Its existing `val` directory is the test set. The exact file lists are saved with each run. The old notebook used that directory for validation, so it is not an untouched test set for a checkpoint selected by the old notebook. Kodak is a separate final benchmark.

```python
from khunc import test_model

test_metrics = test_model(model, config)
```

Training always produces fixed-size batches. The notebook enables augmentation by default; `TrainConfig` defaults to resize-only training for compatibility with existing experiments. Set `augment=True` to use random square crops at varied scales, horizontal and vertical flips, and mild brightness, contrast, and saturation changes. Both the input and reconstruction target are the same augmented image. Validation and Imagenette testing use deterministic resizing, and Kodak uses native dimensions. These augmentations aim to improve generalization; measure their effect against `augment=False` using the same split and seed.

Augmentation settings are plain notebook variables: `crop_scale=(0.5, 1.0)` specifies the fraction of source area before resizing, `horizontal_flip=0.5` and `vertical_flip=0.5` specify probabilities, and `color_jitter=0.1` sets the strength of color changes. Set a flip or jitter value to zero to disable it. Square crops preserve aspect ratio before resizing. Parameters are saved with each experiment. See torchvision's [RandomResizedCrop](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomResizedCrop.html) and [ColorJitter](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ColorJitter.html).

## Architecture experiments

The encoder always has four downsampling layers. Set `ModelConfig(encoder_up_layers=1)` for one encoder upsampling layer and three decoder upsampling layers. The default is two in each. Values from 0 to 4 are supported; the decoder gets the remaining layers. With three latent channels, 1/3 stores one quarter as many values as 2/2. Changing the split requires training a new model. The original checkpoint uses 2/2.

```python
from dataclasses import replace
from khunc import ModelConfig

smaller_latent = replace(config, name="split-1-3", model=ModelConfig(encoder_up_layers=1))
result = train(smaller_latent)
```

## Latent storage and quantization

The notebook uses plain settings such as `latent_dtype = "float16"` and `training_precision = "32-true"`. The former changes the saved compressed latent; the latter changes arithmetic during training. Input images still use ordinary RGB pixels. The Python API stores these choices in `QuantizationConfig` and `TrainConfig` so experiments can be saved and resumed.

| Setting | Meaning |
| --- | --- |
| `storage="float32"` | Original float32 latent baseline |
| `storage="float16"` | Store half as many bytes as float32, decode with float32 calculations |
| `storage="bfloat16"` | Store the latent as bfloat16, decode with float32 calculations |
| `storage="uint8", bits=8` | Store 256 possible values per latent channel |
| `storage="uint8", bits=6` | Store 64 possible values; pack codes into six bits each |
| `training="none"` | Train on the float latent, quantize only for validation and inference |
| `training="ste"` | Round in the forward pass and approximate the gradient as identity |
| `training="noise"` | Add uniform noise scaled to the quantization step during training |

Bit depths from 1 to 8 are supported. For integer storage, each image and channel has its own minimum and scale. Both are stored as float32 metadata. Ranges are computed from detached latents in the training surrogates. STE and noise are approximations; compare them experimentally. Validation and testing always use actual rounding at the configured bit depth, even when training uses noise.

`precision="bf16-mixed"` controls training arithmetic on compatible hardware. It is independent of latent storage and does not simulate uint8 quantization. The default is `precision="32-true"`. The optional reconstruction losses are `"mse"` and `"ms_ssim_mse"`; the latter adds `mse_weight * MSE` to the MS-SSIM loss.

STE means straight-through estimator: round or cast the latent in the forward pass, then approximate its gradient as identity. This follows the idea described by [Bengio et al., 2013](https://arxiv.org/abs/1308.3432). Noise replaces integer rounding during training with uniform noise within half a quantization step, following [Ballé et al., 2017](https://arxiv.org/abs/1611.01704). Here the step is computed per image and channel; it is an adaptation, not a reproduction of their full compression model. `training="none"` leaves the training latent unchanged. Noise is available for integer storage; STE also supports float16 and bfloat16 casts.

## Compress and decompress

```python
from pathlib import Path
from khunc import Codec, QuantizationConfig, load_model, inspect_compressed, save_image

model = load_model("best_model.ckpt")
codec = Codec(model, QuantizationConfig(storage="float16"))
compressed = codec.encode("input.png")
Path("image.khunc").write_bytes(compressed)
reconstructed = codec.decode("image.khunc")
save_image(reconstructed, "reconstructed.png")
print(inspect_compressed(compressed))
```

The default `Codec(model)` uses float32 storage. Create a new `Codec` after modifying model weights. Reuse it for many images. `encode` accepts an image path, a uint8 RGB tensor, or a float RGB tensor in `[0, 1]`. `decode` returns a float32 CHW tensor on the CPU. Inputs are padded to a multiple of 16 and reconstructions are cropped to their original dimensions.

The `.khunc` format stores the original dimensions, latent dimensions, precision, model fingerprint, quantization parameters, and a checksum. Model weights are shared and are not included in each image. Decoding rejects a different model fingerprint or a damaged file. Latent codes below eight bits are bit-packed; there is currently no entropy coding. This is lossy compression of the source image, even though the stored latent codes themselves are preserved exactly.

## Evaluate and plot

```python
from khunc import evaluate_kodak, compare_quantization, download_kodak
from khunc.plotting import plot_quantization

kodak = download_kodak()
comparison = compare_quantization(
    model,
    kodak,
    configurations=[
        QuantizationConfig(storage="float32"),
        QuantizationConfig(storage="float16"),
        QuantizationConfig(storage="bfloat16"),
        QuantizationConfig(storage="uint8", bits=8),
        QuantizationConfig(storage="uint8", bits=6),
    ],
    experiment=result.experiment,
    repeats=3,
)
figure = plot_quantization(
    comparison, path=result.experiment.path / "plots" / "quantization.png"
)
```

`evaluate_kodak(model, quantization=...)` downloads the 24 Kodak images and runs a single configuration. `evaluate(model, image_paths, ...)` evaluates any image set. The evaluator decodes the saved compressed files and scores RGB8 PNG reconstructions with SSIMULACRA2. It saves per-image scores, their mean and range, stored sizes, bits per pixel, and mean compression and decompression times. Stored sizes include format overhead and quantization metadata.

Timings use batch size one, warmup for each image shape, and CUDA or MPS synchronization. They include quantization, in-memory serialization, and memory transfers. Input loading and input transfer to the device, disk I/O, PNG saving, metric calculation, and codec initialization are outside the measurements. Results record the device, package versions, executable hash, model fingerprint, source image hashes, and timing settings. Compare performance on the same device.

## Experiment files

Each run creates a unique folder under `output_dir`:

```text
experiments/<timestamp>-<name>-<id>/
    config.json
    environment.json
    splits.json
    status.json
    checkpoints/
        best.ckpt
        last.ckpt
    logs/training/metrics.csv
    evaluations/
        <mode>-<id>/
            evaluation.json
            results.csv
            summary.json
            reference/
            compressed/
            reconstructed/
    plots/
    images/
```

Reload a configuration with `Experiment(path).load_config()`. Training and evaluation report failed or interrupted runs in their saved status files. Datasets, checkpoints, and experiment outputs are excluded from version control.

## Package layout

`models/` holds the networks and checkpoint loading, `codec/` handles quantization and compressed files, `training/` holds data splits and training, `benchmark/` holds evaluation, and `io/` handles image loading and saving. Configuration, experiment folders, and plotting stay at the package root. The notebook uses the public functions exported by `khunc`.

## Development

```sh
uv run pytest
uv run ruff check .
```

## Reference

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), 2015.
