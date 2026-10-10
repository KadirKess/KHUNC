import pytest
import torch

from khunc import (
    KHUNC,
    Codec,
    ModelConfig,
    QuantizationConfig,
    inspect_compressed,
    load_model,
    save_model,
)
from khunc.models.network import pad_image


@pytest.mark.parametrize("encoder_up_layers", range(5))
def test_architecture_splits_round_trip_and_reload(tmp_path, encoder_up_layers):
    config = ModelConfig(base_channels=2, encoder_up_layers=encoder_up_layers)
    model = KHUNC(config).eval()
    image = torch.rand(3, 31, 47)
    with torch.no_grad():
        latent = model.encoder(pad_image(image.unsqueeze(0)))
    factor = config.downsampling_factor
    assert latent.shape == (1, 3, 32 // factor, 48 // factor)
    path = save_model(model, tmp_path / "model.pt")
    restored = load_model(path, device="cpu")
    assert restored.config == config
    codec = Codec(restored, QuantizationConfig(storage="float32"), device="cpu")
    content = codec.encode(image)
    assert inspect_compressed(content)["downsampling_factor"] == factor
    with torch.no_grad():
        assert torch.equal(codec.decode(content), model(image.unsqueeze(0))[0])
    integer_codec = Codec(restored, QuantizationConfig(storage="uint8", bits=6), device="cpu")
    assert integer_codec.decode(integer_codec.encode(image)).shape == image.shape
