import json
import shutil

import pytest
from PIL import Image

from khunc import QuantizationConfig, compare_quantization, evaluate, ssimulacra2


@pytest.mark.skipif(
    shutil.which("ssimulacra2") is None, reason="SSIMULACRA2 executable unavailable"
)
def test_evaluation_uses_saved_files_and_records_sizes(small_model, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 48), (120, 80, 40)).save(source)
    assert ssimulacra2(source, source) == pytest.approx(100)
    result = compare_quantization(
        small_model,
        [source],
        [
            QuantizationConfig(storage="float32"),
            QuantizationConfig(storage="float16"),
            QuantizationConfig(storage="uint8", bits=6),
        ],
        output_dir=tmp_path / "runs",
        device="cpu",
        repeats=1,
        warmup=1,
        verbose=False,
    )
    assert len(result.rows) == 3
    assert result.summary["float16"]["mean_payload_bytes"] == (
        result.summary["float32"]["mean_payload_bytes"] / 2
    )
    assert (
        result.summary["float32"]["mean_compressed_bytes"]
        > result.summary["uint8-6bit"]["mean_compressed_bytes"]
    )
    for mode in result.summary:
        folder = result.summary[mode]["path"]
        metadata = json.loads((__import__("pathlib").Path(folder) / "evaluation.json").read_text())
        assert metadata["state"] == "complete"
        assert metadata["binary_sha256"]
    assert (result.path / "results.csv").is_file()


def test_missing_metric_fails_without_silent_fallback(small_model, tmp_path):
    with pytest.raises(FileNotFoundError, match="SSIMULACRA2"):
        evaluate(small_model, [], output_dir=tmp_path, binary="missing-ssimulacra2")
