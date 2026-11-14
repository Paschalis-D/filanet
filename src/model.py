import math

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

# TODO: Test later if aditionally to the offsets the network could benefit from learninf a mask too.

# Channel widths per encoder stage, finest first. The earlier (16, 32, 64, 128,
# 256, 512) put 0.1% of the encoder's parameters at full resolution and 73.8% at
# 1/32, while a filament measures 13px across at full resolution and 1.8px at
# 1/32 - capacity allocated almost inversely to where the structure is legible.
# Those narrow early widths existed only to fit whole 2048px frames into 16GB;
# training on 512px windows costs half the activations of a full frame, so the
# budget can move to where the filaments are. This costs about the same memory
# as the full-frame configuration that already fitted.
DEFAULT_WIDTHS = (32, 64, 128, 256, 384, 512)

# Filaments cover 0.36% of a MAGFiLO frame, measured over the 1154 training
# annotations. The output head starts at this prior's log-odds so an untrained
# model predicts a sparse foreground instead of calling the whole frame
# positive and having to walk six logits down to find the data.
FOREGROUND_PRIOR = 0.0036

# GroupNorm rather than BatchNorm, because training happens on 512px windows and
# inference on whole 2048px frames and those are not the same distribution: a
# window averages 0.47 intensity with 9% black sky, a frame 0.30 with 41%.
# BatchNorm carries running statistics from one into the other, and doing so
# drove predicted foreground from 0.6% to 52% while the training loss fell.
# GroupNorm normalises each sample on its own, so there is nothing to carry and
# no gap between training and evaluation - and it does not degrade at the small
# batches this model is limited to.
NORM_GROUPS = 8

# "batch" exists only so that checkpoints trained before the switch to GroupNorm
# (runs/filanet3 and earlier) can still be loaded and scored; never train with it.
NORMS = ("group", "batch")


def _norm(channels: int, kind: str = "group") -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    groups = NORM_GROUPS
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DeformedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm="group"):
        """Deformed Convolution layers recieve the output of a regular convolution
        and use it as a learned offset map that is then aplied in the deform_conv2d
        """
        super(DeformedConvolution, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Keeps spatial dimensions the same and ensure out_channels have the necessary size (ex. for a kernel of 3 the out channels would be 18)
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=kernel_size//2)
        # Stride 1: keeps spatial dimensions the same, changes channel count. Downsampling is a separate pooling step.
        self.deform_conv = DeformConv2d(in_channels, self.out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.norm = _norm(self.out_channels, norm)
        self.relu = nn.ReLU()

    def forward(self, x):
        offsets = self.offset_conv(x)
        x = self.deform_conv(x, offsets)
        x = self.norm(x)
        return self.relu(x)


class PlainConvolution(nn.Module):
    """An ordinary stride-1 conv, norm and ReLU, interchangeable with DeformedConvolution."""

    def __init__(self, in_channels, out_channels, kernel_size, norm="group"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = _norm(out_channels, norm)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """Two stride-1 convs: in_channels -> out_channels -> out_channels.

    Deformable unless `deformable` is False.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, deformable=True, norm="group"):
        super().__init__()
        block = DeformedConvolution if deformable else PlainConvolution
        self.conv1 = block(in_channels, out_channels, kernel_size, norm)
        self.conv2 = block(out_channels, out_channels, kernel_size, norm)

    def forward(self, x):
        return self.conv2(self.conv1(x))


# Stages, finest first, below this index use plain convolutions. Deformable
# convolutions are the expensive part of this network and most of their cost
# sits at full resolution, where there are the most pixels; a filament is 13px
# across there and a 3x3 grid already lies inside it. 0 keeps the network
# deformable everywhere, as the design specified; 2 makes the 1/1 and 1/2 stages
# plain, on both the encoder and decoder side.
DEFAULT_DEFORMABLE_FROM = 0


class Encoder(nn.Module):
    def __init__(self, in_channels=1, widths=DEFAULT_WIDTHS, kernel_size=3,
                 deformable_from=DEFAULT_DEFORMABLE_FROM, norm="group"):
        super(Encoder, self).__init__()
        channels = [in_channels, *widths]
        self.stages = nn.ModuleList([
            DoubleConv(channels[i], channels[i + 1], kernel_size,
                       deformable=i >= deformable_from, norm=norm)
            for i in range(len(widths))
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        for stage in self.stages:
            x = stage(x)
            skip_connections.append(x)
            x = self.pool(x)
        return skip_connections, x


class EdgePrior(nn.Module):
    """Learned edge map from the raw input, projected onto the bottleneck token grid."""

    def __init__(self, in_channels=1, embed_dim=512, kernel_size=3):
        super().__init__()
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size // 2)
        self.project = nn.Conv2d(1, embed_dim, kernel_size=1)

    def forward(self, x, out_hw):
        edge = torch.sigmoid(self.edge_conv(x))
        edge = nn.functional.interpolate(edge, size=out_hw, mode="bilinear", align_corners=False)
        edge = self.project(edge)
        return edge.flatten(2).transpose(1, 2)


class EGMHSA(nn.Module):
    """Multi-head self-attention with the edge prior added to Q and K in place of
    positional encodings."""

    def __init__(self, dim=512, heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, edge):
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens + edge, tokens + edge, tokens)
        tokens = self.norm(tokens + self.dropout(attended))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class Bottleneck(nn.Module):
    def __init__(self, channels=512, heads=4, dropout=0.1, norm="group"):
        super().__init__()
        self.conv = DoubleConv(channels, channels, norm=norm)
        self.attention1 = EGMHSA(channels, heads, dropout)
        self.attention2 = EGMHSA(channels, heads, dropout)

    def forward(self, x, edge):
        z = self.conv(x)
        z = self.attention1(z, edge)
        return self.attention2(z, edge)


class Decoder(nn.Module):
    def __init__(self, widths=DEFAULT_WIDTHS, bottleneck_channels=512, kernel_size=3,
                 deformable_from=DEFAULT_DEFORMABLE_FROM, norm="group"):
        super().__init__()
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        in_channels = bottleneck_channels
        # coarsest level first; level i runs at the resolution of encoder stage i
        for i in reversed(range(len(widths))):
            width = widths[i]
            self.ups.append(nn.ConvTranspose2d(in_channels, width, kernel_size=2, stride=2))
            self.convs.append(DoubleConv(2 * width, width, kernel_size,
                                         deformable=i >= deformable_from, norm=norm))
            in_channels = width

    def forward(self, z, skips):
        for up, conv, skip in zip(self.ups, self.convs, reversed(skips)):
            z = up(z)
            z = conv(torch.cat([skip, z], dim=1))
        return z


class FilaNet(nn.Module):
    def __init__(self, in_channels=1, widths=DEFAULT_WIDTHS, heads=4, dropout=0.1,
                 deformable_from=DEFAULT_DEFORMABLE_FROM, norm="group", out_channels=1):
        """`out_channels` is 1 for the mask alone, or 2 to also predict the
        filament centreline (see losses.FilamentLoss); channel 0 is always the mask."""
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        self.encoder = Encoder(in_channels, widths, deformable_from=deformable_from, norm=norm)
        self.edge_prior = EdgePrior(in_channels, embed_dim=widths[-1])
        self.bottleneck = Bottleneck(widths[-1], heads, dropout, norm=norm)
        self.decoder = Decoder(widths, bottleneck_channels=widths[-1],
                               deformable_from=deformable_from, norm=norm)
        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)
        nn.init.constant_(
            self.head.bias, math.log(FOREGROUND_PRIOR / (1.0 - FOREGROUND_PRIOR))
        )

    @classmethod
    def from_state_dict(cls, state_dict: dict, **kwargs) -> "FilaNet":
        """A FilaNet shaped to fit `state_dict`, with those weights loaded.

        Checkpoints only hold weights, and the architecture has changed under
        them - widths, norm and which stages are deformable - so the shape is
        read back off the weights rather than assumed. `kwargs` go to the
        constructor for what weights cannot reveal, such as `heads`.
        """
        # before GroupNorm the norm layer was called `batch_norm`
        state_dict = {k.replace(".batch_norm.", ".norm."): v for k, v in state_dict.items()}
        n_stages = len({k.split(".")[2] for k in state_dict if k.startswith("encoder.stages.")})
        widths = []
        deformable = []
        for i in range(n_stages):
            prefix = f"encoder.stages.{i}.conv1."
            is_deformable = prefix + "deform_conv.weight" in state_dict
            weight = state_dict[prefix + ("deform_conv.weight" if is_deformable else "conv.weight")]
            widths.append(weight.shape[0])
            deformable.append(is_deformable)
        deformable_from = deformable.index(True) if True in deformable else n_stages
        norm = "batch" if any(k.endswith("running_mean") for k in state_dict) else "group"
        in_channels = state_dict["edge_prior.edge_conv.weight"].shape[1]
        out_channels = state_dict["head.weight"].shape[0]

        model = cls(in_channels, tuple(widths), deformable_from=deformable_from, norm=norm,
                    out_channels=out_channels, **kwargs)
        model.load_state_dict(state_dict)
        return model

    def forward(self, x):
        skips, encoded = self.encoder(x)
        edge = self.edge_prior(x, out_hw=encoded.shape[-2:])
        z = self.bottleneck(encoded, edge)
        return self.head(self.decoder(z, skips))


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FilaNet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {params / 1e6:.1f}M")

    x = torch.randn(1, 1, 2048, 2048, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    logits = model(x)
    logits.sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"output: {tuple(logits.shape)}")
    print(f"fwd+bwd: {time.time() - start:.2f}s")
