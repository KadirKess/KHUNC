import pytest
import torch

from khunc import Codec, QuantizationConfig, inspect_compressed
from khunc.codec.compression import _pack_codes, _unpack_codes


@pytest.mark.parametrize(
    "storage,bits",
    [("float32", 8), ("float16", 8), ("bfloat16", 8)] + [("uint8", bits) for bits in range(1, 9)],
)
def test_files_decode_without_original_latents(small_model, tmp_path, storage, bits):
    image = torch.rand(3, 31, 47)
    codec = Codec(small_model, QuantizationConfig(storage=storage, bits=bits), "cpu")
    path = tmp_path / "image.khunc"
    path.write_bytes(codec.encode(image))
    reloaded_codec = Codec(small_model, device="cpu")
    reconstructed = reloaded_codec.decode(path)
    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == torch.float32
    assert torch.isfinite(reconstructed).all()
    info = inspect_compressed(path)
    assert info["storage"] == storage
    assert info["compressed_bytes"] == path.stat().st_size
    assert info["compressed_bytes"] == info["payload_bytes"] + info["metadata_bytes"]
    if storage == "float32":
        with torch.inference_mode():
            assert torch.equal(reconstructed, small_model(image.unsqueeze(0))[0])
    elif storage == "float16":
        with torch.inference_mode():
            from khunc.models.network import pad_image

            latent = small_model.encoder(pad_image(image.unsqueeze(0)))
            expected = small_model.decoder(latent.half().float())[0, :, :31, :47]
        assert torch.equal(reconstructed, expected)
        assert info["payload_bytes"] == latent.numel() * 2


def test_float16_storage_rejects_overflow(small_model):
    with torch.no_grad():
        small_model.encoder.outc.conv.bias.fill_(70000)
    codec = Codec(small_model, QuantizationConfig(storage="float16"), "cpu")
    with pytest.raises(ValueError, match="float16 range"):
        codec.encode(torch.rand(3, 32, 48))


@pytest.mark.parametrize("bits", range(1, 9))
def test_bit_packing_handles_partial_bytes(bits):
    codes = torch.randint(0, 2**bits, (7,), dtype=torch.uint8).numpy()
    payload = _pack_codes(codes, bits)
    restored = _unpack_codes(payload, len(codes), bits)
    assert len(payload) == (len(codes) * bits + 7) // 8
    assert (restored == codes).all()


def test_corrupted_file_is_rejected(small_model):
    codec = Codec(small_model, device="cpu")
    content = bytearray(codec.encode(torch.rand(3, 32, 48)))
    content[-5] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        codec.decode(bytes(content))


def test_wrong_decoder_weights_are_rejected(small_model):
    codec = Codec(small_model, device="cpu")
    content = codec.encode(torch.rand(3, 32, 48))
    with torch.no_grad():
        small_model.decoder.outc.conv.bias.add_(0.1)
    different_codec = Codec(small_model, device="cpu")
    with pytest.raises(ValueError, match="different model weights"):
        different_codec.decode(content)


def test_rgb8_input_matches_normalized_float_input(small_model):
    codec = Codec(small_model, device="cpu")
    image = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
    assert codec.encode(image) == codec.encode(image.float() / 255)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        codec.encode(image.float())
