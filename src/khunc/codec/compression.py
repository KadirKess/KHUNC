import struct
import zlib
from pathlib import Path

import numpy as np
import torch

from ..codec.quantization import dequantize, quantize
from ..config import QuantizationConfig
from ..io.images import load_image
from ..models.checkpoints import choose_device, model_fingerprint
from ..models.network import KHUNC, pad_image

_HEADER = struct.Struct("<8sBBIIIII32s")
_CHECKSUM = struct.Struct("<I")
_MAGIC = b"KHUNC001"
_STORAGE = {"float32": 0, "bfloat16": 1, "uint8": 2, "float16": 3}


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _pack_codes(values: np.ndarray, bits: int) -> bytes:
    codes = values.reshape(-1)
    if bits == 8:
        return codes.tobytes()
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint8)
    bit_values = ((codes[:, None] >> shifts) & 1).reshape(-1)
    return np.packbits(bit_values, bitorder="big").tobytes()


def _unpack_codes(payload: bytes, count: int, bits: int) -> np.ndarray:
    values = np.frombuffer(payload, dtype=np.uint8)
    if bits == 8:
        return values.copy()
    bit_values = np.unpackbits(values, bitorder="big", count=count * bits).reshape(count, bits)
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint8)
    return (bit_values * weights).sum(axis=1).astype(np.uint8)


def _serialize(z, shape, config, fingerprint):
    z = z.detach().float().cpu().contiguous()
    if not torch.isfinite(z).all():
        raise ValueError("The encoder produced non-finite values")
    _, channels, height, width = z.shape
    bits = {"float32": 32, "float16": 16, "bfloat16": 16}.get(config.storage, config.bits)
    header = _HEADER.pack(
        _MAGIC, _STORAGE[config.storage], bits, channels, height, width, *shape, fingerprint
    )
    if config.storage == "float32":
        payload = z.numpy().astype("<f4", copy=False).tobytes()
    elif config.storage == "float16":
        values = z.to(torch.float16)
        if not torch.isfinite(values).all():
            raise ValueError("The latent exceeds the float16 range; use float32 or bfloat16")
        payload = values.numpy().astype("<f2", copy=False).tobytes()
    elif config.storage == "bfloat16":
        values = z.to(torch.bfloat16).view(torch.uint16).numpy()
        payload = values.astype("<u2", copy=False).tobytes()
    else:
        codes, offset, scale = quantize(z, config.bits)
        metadata = torch.cat((offset.flatten(), scale.flatten())).numpy().astype("<f4").tobytes()
        payload = metadata + _pack_codes(codes.numpy(), config.bits)
    content = header + payload
    return content + _CHECKSUM.pack(zlib.crc32(content))


def _read_content(source: bytes | str | Path) -> bytes:
    if isinstance(source, bytes):
        return source
    return Path(source).expanduser().read_bytes()


def _parse(content):
    if len(content) < _HEADER.size + _CHECKSUM.size:
        raise ValueError("Truncated KHUNC file")
    data, checksum = content[: -_CHECKSUM.size], content[-_CHECKSUM.size :]
    if zlib.crc32(data) != _CHECKSUM.unpack(checksum)[0]:
        raise ValueError("KHUNC file checksum mismatch")
    magic, storage, bits, channels, height, width, oh, ow, fingerprint = _HEADER.unpack_from(data)
    if magic != _MAGIC or storage not in _STORAGE.values():
        raise ValueError("Unsupported KHUNC file format")
    if min(channels, height, width, oh, ow) < 1:
        raise ValueError("Invalid image dimensions")
    padded = (((oh + 15) // 16) * 16, ((ow + 15) // 16) * 16)
    if not any(
        (height, width) == (padded[0] // factor, padded[1] // factor) for factor in (1, 2, 4, 8, 16)
    ):
        raise ValueError("Latent dimensions do not match the original image")
    if (storage == 0 and bits != 32) or (storage in {1, 3} and bits != 16):
        raise ValueError("Invalid floating-point storage precision")
    if storage == 2 and not 1 <= bits <= 8:
        raise ValueError("Invalid quantization bit depth")
    count = channels * height * width
    metadata_bytes = channels * 8 if storage == 2 else 0
    expected = _HEADER.size + metadata_bytes + (count * bits + 7) // 8
    if len(data) != expected:
        raise ValueError("KHUNC payload length does not match its header")
    return storage, bits, channels, height, width, oh, ow, fingerprint, data[_HEADER.size :]


def _deserialize(content):
    storage, bits, channels, height, width, oh, ow, fingerprint, payload = _parse(content)
    shape = (1, channels, height, width)
    if storage == 0:
        values = np.frombuffer(payload, dtype="<f4").copy().reshape(shape)
        z = torch.from_numpy(values)
    elif storage == 3:
        values = np.frombuffer(payload, dtype="<f2").copy().reshape(shape)
        z = torch.from_numpy(values).float()
    elif storage == 1:
        values = np.frombuffer(payload, dtype="<u2").copy().reshape(shape)
        z = torch.from_numpy(values).view(torch.bfloat16).float()
    else:
        metadata = np.frombuffer(payload[: channels * 8], dtype="<f4").copy()
        offset = torch.from_numpy(metadata[:channels]).reshape(1, channels, 1, 1)
        scale = torch.from_numpy(metadata[channels:]).reshape(1, channels, 1, 1)
        if (
            not torch.isfinite(offset).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Invalid quantization metadata")
        codes = _unpack_codes(payload[channels * 8 :], channels * height * width, bits)
        z = dequantize(torch.from_numpy(codes.reshape(shape)), offset, scale)
    if not torch.isfinite(z).all():
        raise ValueError("Non-finite latent in KHUNC file")
    return z, (oh, ow), fingerprint


def inspect_compressed(source: bytes | str | Path) -> dict:
    content = _read_content(source)
    storage, bits, channels, height, width, oh, ow, fingerprint, _ = _parse(content)
    return {
        "storage": next(name for name, code in _STORAGE.items() if code == storage),
        "bits": bits,
        "original_shape": (oh, ow),
        "latent_shape": (channels, height, width),
        "downsampling_factor": ((oh + 15) // 16) * 16 // height,
        "model_sha256": fingerprint.hex(),
        "compressed_bytes": len(content),
        "payload_bytes": (channels * height * width * bits + 7) // 8,
        "metadata_bytes": _HEADER.size + _CHECKSUM.size + (channels * 8 if storage == 2 else 0),
        "bits_per_pixel": len(content) * 8 / (oh * ow),
    }


class Codec:
    """Create after training; reuse while the model weights stay fixed."""

    def __init__(self, model: KHUNC, quantization: QuantizationConfig | None = None, device="auto"):
        self.device = choose_device(device)
        self.model = model.float().to(self.device).eval()
        self.quantization = quantization or QuantizationConfig()
        self.fingerprint = model_fingerprint(self.model)

    @torch.inference_mode()
    def encode(self, image: torch.Tensor | str | Path) -> bytes:
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[:2] != (1, 3) or min(image.shape[-2:]) < 1:
            raise ValueError("Expected a single RGB image in CHW or NCHW format")
        if image.dtype == torch.uint8:
            image = image.float().div(255)
        if not image.is_floating_point():
            raise ValueError("Images must be uint8 or floating point")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating-point image values must be finite and in [0, 1]")
        image = image.to(device=self.device, dtype=torch.float32)
        z = self.model.encoder(pad_image(image))
        return _serialize(z, tuple(image.shape[-2:]), self.quantization, self.fingerprint)

    @torch.inference_mode()
    def decode(self, source: bytes | str | Path) -> torch.Tensor:
        z, (height, width), fingerprint = _deserialize(_read_content(source))
        if fingerprint != self.fingerprint:
            raise ValueError("This image was encoded with different model weights")
        factor = self.model.config.downsampling_factor
        expected = (
            self.model.config.latent_channels,
            ((height + 15) // 16) * 16 // factor,
            ((width + 15) // 16) * 16 // factor,
        )
        if tuple(z.shape[1:]) != expected:
            raise ValueError("Latent dimensions do not match the decoder architecture")
        image = self.model.decoder(z.to(self.device))
        return image[0, :, :height, :width].cpu()


def compress(model: KHUNC, image, *, quantization=None, device="auto", path=None) -> bytes:
    content = Codec(model, quantization, device).encode(image)
    if path is not None:
        path = Path(path).expanduser()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return content


def decompress(model: KHUNC, source, *, device="auto") -> torch.Tensor:
    return Codec(model, device=device).decode(source)
