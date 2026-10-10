import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        x = F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return self.conv(torch.cat((skip, x), dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class EncoderUNet(nn.Module):
    def __init__(self, latent_channels=3, base_channels=32, up_layers=2):
        super().__init__()
        width = base_channels
        self.inc = DoubleConv(3, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        self.down4 = Down(width * 8, width * 16)
        self.up_layers = up_layers
        for index in range(1, up_layers + 1):
            channels = width * 2 ** (5 - index)
            setattr(self, f"up{index}", Up(channels, channels // 2))
        self.outc = OutConv(width * 2 ** (4 - up_layers), latent_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = x5
        skips = [x4, x3, x2, x1]
        for index in range(1, self.up_layers + 1):
            x = getattr(self, f"up{index}")(x, skips[index - 1])
        return self.outc(x)


class DecoderUNet(nn.Module):
    def __init__(self, latent_channels=3, base_channels=32, up_layers=2):
        super().__init__()
        width = base_channels
        self.up_layers = up_layers
        channels = width * 2 ** (up_layers + 1)
        self.inc = DoubleConv(latent_channels, channels)
        for index in range(1, up_layers + 1):
            setattr(self, f"up{index}", nn.ConvTranspose2d(channels, channels, 2, stride=2))
            setattr(self, f"conv{index}", DoubleConv(channels, channels // 2))
            channels //= 2
        self.outc = OutConv(width * 2, 3)

    def forward(self, z):
        x = self.inc(z)
        for index in range(1, self.up_layers + 1):
            x = getattr(self, f"conv{index}")(getattr(self, f"up{index}")(x))
        return torch.sigmoid(self.outc(x))


def pad_image(x: torch.Tensor) -> torch.Tensor:
    height, width = x.shape[-2:]
    return F.pad(x, (0, (-width) % 16, 0, (-height) % 16), mode="replicate")


class KHUNC(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = EncoderUNet(
            self.config.latent_channels, self.config.base_channels, self.config.encoder_up_layers
        )
        self.decoder = DecoderUNet(
            self.config.latent_channels, self.config.base_channels, self.config.decoder_up_layers
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        z = self.encoder(pad_image(x))
        return self.decoder(z)[..., :height, :width]
