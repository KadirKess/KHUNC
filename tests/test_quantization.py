import pytest
import torch

from khunc import QuantizationConfig
from khunc.codec.quantization import (
    dequantize,
    quantization_parameters,
    quantize,
    quantize_dequantize,
    training_latent,
)


@pytest.mark.parametrize("bits", range(1, 9))
def test_quantization_error_is_bounded_per_image_and_channel(bits):
    z = torch.randn(2, 3, 17, 19)
    z[1] *= 100
    z[0, 1] = 3
    codes, offset, scale = quantize(z, bits)
    restored = dequantize(codes, offset, scale)
    assert codes.dtype == torch.uint8
    assert codes.max() <= 2**bits - 1
    assert torch.all((restored - z).abs() <= scale / 2 + 1e-5)
    assert torch.equal(restored[0, 1], z[0, 1])


@pytest.mark.parametrize("bits", [2, 6, 8])
def test_ste_uses_exact_codes_forward_and_nonzero_gradients_backward(bits):
    z = torch.randn(2, 3, 17, 19, requires_grad=True)
    config = QuantizationConfig(storage="uint8", bits=bits, training="ste")
    result = training_latent(z, config)
    assert torch.equal(result, quantize_dequantize(z, config))
    result.sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))


def test_noise_tracks_the_quantization_step_and_retains_gradients():
    z = torch.randn(2, 3, 17, 19, requires_grad=True)
    config = QuantizationConfig(storage="uint8", bits=6, training="noise")
    _, scale = quantization_parameters(z, config.bits)
    result = training_latent(z, config)
    assert not torch.equal(result, z)
    assert torch.all((result - z).abs() <= scale / 2 + 1e-6)
    result.sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
def test_float_storage_ste_matches_cast_and_preserves_gradients(storage):
    z = torch.randn(2, 3, 17, 19, requires_grad=True)
    config = QuantizationConfig(storage=storage, training="ste")
    result = training_latent(z, config)
    assert torch.equal(result, z.to(getattr(torch, storage)).float())
    result.sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))
