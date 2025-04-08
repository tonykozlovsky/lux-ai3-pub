from typing import Callable, Tuple

import torch
from torch import nn


class SELayerNoMask(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 16):
        super(SELayerNoMask, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlockNoMask(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            normalize: bool = False,
            activation: Callable = nn.ReLU,
            squeeze_excitation: bool = True,
            **conv2d_kwargs
    ):
        super(ResidualBlockNoMask, self).__init__()

        # Calculate "same" padding
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://www.wolframalpha.com/input/?i=i%3D%28i%2B2x-k-%28k-1%29%28d-1%29%2Fs%29+%2B+1&assumption=%22i%22+-%3E+%22Variable%22
        assert "padding" not in conv2d_kwargs.keys()
        k = kernel_size
        d = conv2d_kwargs.get("dilation", 1)
        s = conv2d_kwargs.get("stride", 1)
        padding = (k - 1) * (d + s - 1) / (2 * s)
        assert padding == int(padding), f"padding should be an integer, was {padding:.2f}"
        padding = int(padding)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs
        )
        # We use LayerNorm here since the size of the input "images" may vary based on the board size
        #self.norm1 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.norm1 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.act1 = activation()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs
        )
        #self.norm2 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.norm2 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.final_act = activation()

        if in_channels != out_channels:
            self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.change_n_channels = nn.Identity()

        if squeeze_excitation:
            self.squeeze_excitation = SELayerNoMask(out_channels)
        else:
            self.squeeze_excitation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.act1(self.norm1(x))
        x = self.conv2(x)
        x = self.squeeze_excitation(self.norm2(x))
        x = x + self.change_n_channels(identity)
        return self.final_act(x)


class ParallelDilationResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            normalize: bool = False,
            activation: Callable = nn.ReLU,
            squeeze_excitation: bool = True,
            **conv2d_kwargs
    ):
        super(ParallelDilationResidualBlock, self).__init__()

        # Calculate "same" padding
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://www.wolframalpha.com/input/?i=i%3D%28i%2B2x-k-%28k-1%29%28d-1%29%2Fs%29+%2B+1&assumption=%22i%22+-%3E+%22Variable%22
        assert "padding" not in conv2d_kwargs.keys()
        assert "dilation" not in conv2d_kwargs.keys()
        k = kernel_size
        d_main = 1
        s = conv2d_kwargs.get("stride", 1)
        padding_main = (k - 1) * (d_main + s - 1) / (2 * s)
        assert padding_main == int(padding_main), f"padding should be an integer, was {padding_main:.2f}"
        padding_main = int(padding_main)

        d_dilation = 2
        padding_dilation = (k - 1) * (d_dilation + s - 1) / (2 * s)
        assert padding_dilation == int(padding_dilation), f"padding should be an integer, was {padding_dilation:.2f}"
        padding_dilation = int(padding_dilation)

        # Main branch
        self.conv1_main = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding_main, padding_main),
            dilation=(d_main, d_main),
            **conv2d_kwargs
        )
        # We use LayerNorm here since the size of the input "images" may vary based on the board size
        self.norm1_main = nn.Identity()
        self.act1_main = activation()

        self.conv2_main = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding_main, padding_main),
            dilation=(d_main, d_main),
            **conv2d_kwargs
        )
        self.norm2_main = nn.Identity()
        if squeeze_excitation:
            self.squeeze_excitation_main = SELayerNoMask(out_channels)
        else:
            self.squeeze_excitation_main = nn.Identity()

        # Dilated branch
        self.conv1_dilation = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding_dilation, padding_dilation),
            dilation=(d_dilation, d_dilation),
            **conv2d_kwargs
        )
        # We use LayerNorm here since the size of the input "images" may vary based on the board size
        self.norm1_dilation = nn.Identity()
        self.act1_dilation = activation()

        self.conv2_dilation = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding_dilation, padding_dilation),
            dilation=(d_dilation, d_dilation),
            **conv2d_kwargs
        )
        self.norm2_dilation = nn.Identity()
        if squeeze_excitation:
            self.squeeze_excitation_dilation = SELayerNoMask(out_channels)
        else:
            self.squeeze_excitation_dilation = nn.Identity()

        self.final_act = activation()
        if in_channels != out_channels:
            self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.change_n_channels = nn.Identity()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_orig = x

        # Main branch
        x_main = self.conv1_main(x_orig)
        x_main = self.act1_main(self.norm1_main(x_main))
        x_main = self.conv2_main(x_main)
        x_main = self.squeeze_excitation_main(self.norm2_main(x_main))

        # Dilated branch
        x_dilation = self.conv1_dilation(x_orig)
        x_dilation = self.act1_dilation(self.norm1_dilation(x_dilation))
        x_dilation = self.conv2_dilation(x_dilation)
        x_dilation = self.squeeze_excitation_dilation(self.norm2_dilation(x_dilation))

        x = x_main + x_dilation + self.change_n_channels(x_orig)
        return self.final_act(x)
