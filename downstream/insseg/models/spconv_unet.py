"""
Author: Xiaoyang Wu
Email: xiaoyang.wu.cs@gmail.com

Sparse UNet driven by spconv2
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `README.md` to install spconv2.`')


def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else
                      torch.tensor([i] * o) for i, o in enumerate(offset)],
                     dim=0).long().to(offset.device)


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self,
                 in_channels,
                 embed_channels,
                 stride=1,
                 norm_fn=None,
                 indice_key=None,
                 bias=False,
                 ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels)
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels, embed_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels, embed_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SpUNetBase(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=32,
                 channels=(32, 64, 128, 256, 256, 128, 96, 96),
                 layers=(2, 3, 4, 6, 2, 2, 2, 2)):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=5, padding=1, bias=False, indice_key='stem'),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(enc_channels, channels[s], kernel_size=2, stride=2, bias=False,
                                    indice_key=f"spconv{s + 1}"),
                norm_fn(channels[s]),
                nn.ReLU()
            ))
            self.enc.append(spconv.SparseSequential(OrderedDict([
                # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                # if i == 0 else
                (f"block{i}", block(channels[s], channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                for i in range(layers[s])
            ])))

            # decode num_stages
            self.up.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(channels[len(channels) - s - 2], dec_channels,
                                           kernel_size=2, bias=False, indice_key=f"spconv{s + 1}"),
                norm_fn(dec_channels),
                nn.ReLU()
            ))
            self.dec.append(spconv.SparseSequential(OrderedDict([
                (f"block{i}", block(dec_channels + enc_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"))
                if i == 0 else
                (f"block{i}", block(dec_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"))
                for i in range(layers[len(channels) - s - 1])
            ])))
            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.final = spconv.SubMConv3d(channels[-1], out_channels, kernel_size=1, padding=1, bias=True) \
            if out_channels > 0 else spconv.Identity()

        self.offset_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            norm_fn(channels[-1]),
            nn.ReLU(),
            nn.Linear(channels[-1], 3)
        )

    def forward(self, discrete_coord, feat):
        sparse_shape = torch.add(torch.max(discrete_coord[:, 1:], dim=0).values, 1).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=discrete_coord.contiguous(),
            spatial_shape=sparse_shape,
            batch_size=discrete_coord[:, 0].max().tolist() + 1
        )
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        # dec forward
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)

        offset_pred = self.offset_head(x.features)
        x = self.final(x)
        return offset_pred, x.features
