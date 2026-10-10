import torch

from ..config import QuantizationConfig


def quantization_parameters(z: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")
    if z.ndim != 4 or not torch.isfinite(z).all():
        raise ValueError("Expected a finite NCHW latent")
    values = z.detach().float()
    offset = values.amin(dim=(-2, -1), keepdim=True)
    scale = (values.amax(dim=(-2, -1), keepdim=True) - offset) / (2**bits - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return offset, scale


def quantize(z: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offset, scale = quantization_parameters(z, bits)
    codes = ((z.float() - offset) / scale).round().clamp(0, 2**bits - 1).to(torch.uint8)
    return codes, offset, scale


def dequantize(codes: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.float() * scale + offset


def quantize_dequantize(z: torch.Tensor, config: QuantizationConfig) -> torch.Tensor:
    if config.storage == "float32":
        return z.float()
    if config.storage == "float16":
        return z.to(torch.float16).float()
    if config.storage == "bfloat16":
        return z.to(torch.bfloat16).float()
    return dequantize(*quantize(z, config.bits))


def training_latent(z: torch.Tensor, config: QuantizationConfig) -> torch.Tensor:
    if config.training == "none":
        return z
    if config.training == "noise":
        _, scale = quantization_parameters(z, config.bits)
        return z.float() + torch.empty_like(z, dtype=torch.float32).uniform_(-0.5, 0.5) * scale
    rounded = quantize_dequantize(z, config)
    # Exact rounded values forward; identity approximation backward.
    return rounded.detach() + (z.float() - z.float().detach())
