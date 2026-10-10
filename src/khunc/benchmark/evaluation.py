import csv
import math
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

from PIL import Image

from ..codec.compression import Codec, inspect_compressed, synchronize
from ..config import QuantizationConfig
from ..experiments import Experiment, environment, write_json
from ..io.images import load_image, save_image
from ..models.checkpoints import file_digest
from ..models.network import KHUNC


def download_kodak(root: str | Path = "data/kodak") -> Path:
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 25):
        path = root / f"kodim{index:02d}.png"
        if path.exists():
            with Image.open(path) as image:
                image.verify()
            continue
        url = f"https://r0k.us/graphics/kodak/kodak/{path.name}"
        with tempfile.NamedTemporaryFile(dir=root, suffix=".png") as temporary:
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, temporary)
            temporary.flush()
            with Image.open(temporary.name) as image:
                if image.size not in {(768, 512), (512, 768)}:
                    raise ValueError(f"Unexpected Kodak dimensions in {url}")
                image.verify()
            shutil.copyfile(temporary.name, path)
    return root


def ssimulacra2(reference: str | Path, reconstruction: str | Path, binary="ssimulacra2") -> float:
    executable = shutil.which(str(binary))
    if executable is None:
        raise FileNotFoundError("SSIMULACRA2 is not installed; pass its executable path as binary")
    result = subprocess.run(
        [executable, str(reference), str(reconstruction)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    value = float(result.stdout.strip())
    if not math.isfinite(value):
        raise ValueError("SSIMULACRA2 returned a non-finite score")
    return value


def quantization_name(config: QuantizationConfig) -> str:
    return f"uint8-{config.bits}bit" if config.storage == "uint8" else config.storage


@dataclass
class EvaluationResult:
    path: Path
    rows: list[dict]
    summary: dict


def _paths(images) -> list[Path]:
    if isinstance(images, (str, Path)):
        source = Path(images).expanduser()
        if source.is_dir():
            extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            paths = sorted(
                p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in extensions
            )
        else:
            paths = [source]
    else:
        paths = [Path(p).expanduser() for p in images]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError("Provide at least one existing image file")
    return paths


def _time_call(call, device):
    synchronize(device)
    start = time.perf_counter()
    result = call()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000


def _summary(rows):
    keys = [
        "ssimulacra2",
        "compressed_bytes",
        "payload_bytes",
        "bits_per_pixel",
        "encode_ms",
        "decode_ms",
        "source_file_bytes",
    ]
    summary = {f"mean_{key}": statistics.mean(row[key] for row in rows) for key in keys}
    scores = [row["ssimulacra2"] for row in rows]
    summary.update(
        images=len(rows),
        min_ssimulacra2=min(scores),
        max_ssimulacra2=max(scores),
        std_ssimulacra2=statistics.pstdev(scores),
    )
    return summary


def _write_rows(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    model: KHUNC,
    images,
    *,
    quantization: QuantizationConfig | None = None,
    experiment: Experiment | None = None,
    output_dir="experiments",
    device="auto",
    repeats=3,
    warmup=2,
    binary="ssimulacra2",
    verbose=True,
) -> EvaluationResult:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    if shutil.which(str(binary)) is None:
        raise FileNotFoundError("Install SSIMULACRA2 or pass binary= with its executable path")
    paths = _paths(images)
    codec = Codec(model, quantization, device)
    mode = quantization_name(codec.quantization)
    experiment = experiment or Experiment.create(f"evaluate-{mode}", output_dir)
    folder = experiment.path / "evaluations" / f"{mode}-{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    for name in ["reference", "compressed", "reconstructed"]:
        (folder / name).mkdir()
    metadata = {
        "state": "running",
        "environment": environment(),
        "device": str(codec.device),
        "model_sha256": codec.fingerprint.hex(),
        "quantization": asdict(codec.quantization),
        "binary": str(Path(shutil.which(str(binary))).resolve()),
        "binary_sha256": file_digest(shutil.which(str(binary))),
        "repeats": repeats,
        "warmup_roundtrips_per_shape": warmup,
        "timing": "Synchronized batch-size-one encode/decode, including quantization and memory "
        "transfers. Input loading and input transfer to device, disk I/O, PNG writing "
        "and metric computation are excluded. Model weights are excluded from file sizes.",
        "sources": [{"path": str(p.resolve()), "sha256": file_digest(p)} for p in paths],
    }
    write_json(folder / "evaluation.json", metadata)
    rows, warmed = [], set()
    try:
        for index, path in enumerate(paths, 1):
            image = load_image(path).to(codec.device)
            shape = tuple(image.shape[-2:])
            if shape not in warmed:
                for _ in range(warmup):
                    codec.decode(codec.encode(image))
                synchronize(codec.device)
                warmed.add(shape)
            encodings = [
                _time_call(lambda: codec.encode(image), codec.device) for _ in range(repeats)
            ]
            encoded = encodings[-1][0]
            stem = f"{index:04d}-{path.stem}"
            compressed_path = folder / "compressed" / f"{stem}.khunc"
            compressed_path.write_bytes(encoded)
            stored = compressed_path.read_bytes()
            decodings = [
                _time_call(lambda: codec.decode(stored), codec.device) for _ in range(repeats)
            ]
            reference = save_image(image, folder / "reference" / f"{stem}.png")
            reconstruction = save_image(decodings[-1][0], folder / "reconstructed" / f"{stem}.png")
            info = inspect_compressed(stored)
            row = {
                "image": str(path.resolve()),
                "mode": mode,
                "height": shape[0],
                "width": shape[1],
                "source_file_bytes": path.stat().st_size,
                "compressed_bytes": compressed_path.stat().st_size,
                "payload_bytes": info["payload_bytes"],
                "bits_per_pixel": info["bits_per_pixel"],
                "encode_ms": statistics.mean(sample[1] for sample in encodings),
                "decode_ms": statistics.mean(sample[1] for sample in decodings),
                "ssimulacra2": ssimulacra2(reference, reconstruction, binary),
            }
            rows.append(row)
            _write_rows(folder / "results.csv", rows)
            if verbose:
                print(f"{index}/{len(paths)} {path.name}: {row['ssimulacra2']:.2f}")
        summary = _summary(rows)
        write_json(folder / "summary.json", summary)
        metadata["state"] = "complete"
        write_json(folder / "evaluation.json", metadata)
        return EvaluationResult(folder, rows, summary)
    except BaseException as error:
        metadata.update(
            state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        write_json(folder / "evaluation.json", metadata)
        raise


def evaluate_kodak(model: KHUNC, *, kodak_dir="data/kodak", **kwargs) -> EvaluationResult:
    folder = download_kodak(kodak_dir)
    paths = [folder / f"kodim{index:02d}.png" for index in range(1, 25)]
    return evaluate(model, paths, **kwargs)


def compare_quantization(
    model: KHUNC,
    images,
    configurations: list[QuantizationConfig],
    *,
    experiment: Experiment | None = None,
    output_dir="experiments",
    **kwargs,
) -> EvaluationResult:
    if not configurations:
        raise ValueError("Provide at least one quantization configuration")
    names = [quantization_name(config) for config in configurations]
    if len(set(names)) != len(names):
        raise ValueError("Each storage mode and bit depth must be unique")
    paths = _paths(images)
    experiment = experiment or Experiment.create("quantization-comparison", output_dir)
    folder = experiment.path / "evaluations" / f"comparison-{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    rows, summary = [], {}
    for config in configurations:
        result = evaluate(model, paths, quantization=config, experiment=experiment, **kwargs)
        rows.extend(result.rows)
        summary[quantization_name(config)] = {**result.summary, "path": str(result.path)}
        _write_rows(folder / "results.csv", rows)
        write_json(folder / "summary.json", summary)
    return EvaluationResult(folder, rows, summary)
