# -*- coding: utf-8 -*-
"""
MCSF-ResUNet
-------------
A lightweight 2D medical image segmentation network built on a ResNet-34
encoder and three task-oriented modules:

1) MRRA: Multi-Receptive Residual Adapter
2) DCCF: Decoder-guided Contrastive Cross-scale Fusion
3) SASC: Semantic-conditioned Axial Structure Calibration

Compatible with Python 3.7, PyTorch 1.13.1 and torchvision 0.14.1.
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, activation=True):
        super(ConvBNAct, self).__init__()
        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
        layers = [
            nn.Conv2d(
                in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_ch),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch, kernel_size=kernel_size, padding=padding,
                dilation=dilation, groups=in_ch, bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MRRA(nn.Module):
    """Multi-Receptive Residual Adapter.

    It is only placed after ResNet-34 layer3 and layer4. Three inexpensive
    depthwise branches provide different receptive fields. A pixel-wise gate
    selects a scale at each spatial position. A learnable residual scale is
    initialized conservatively so the pretrained representation is not
    destroyed at the beginning of fine-tuning.
    """

    def __init__(self, channels, reduction=4):
        super(MRRA, self).__init__()
        inner = max(channels // reduction, 32)
        self.reduce = ConvBNAct(channels, inner, kernel_size=1, padding=0)
        self.branch3 = DepthwiseSeparableConv(inner, inner, kernel_size=3, dilation=1)
        self.branch5 = DepthwiseSeparableConv(inner, inner, kernel_size=5, dilation=1)
        self.branch_d2 = DepthwiseSeparableConv(inner, inner, kernel_size=3, dilation=2)
        gate_hidden = max(inner // 4, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(inner * 3, gate_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 3, kernel_size=1, bias=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(inner, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        z = self.reduce(x)
        b1 = self.branch3(z)
        b2 = self.branch5(z)
        b3 = self.branch_d2(z)
        cat = torch.cat([b1, b2, b3], dim=1)
        weights = torch.softmax(self.gate(cat), dim=1)
        fused = (
            b1 * weights[:, 0:1] +
            b2 * weights[:, 1:2] +
            b3 * weights[:, 2:3]
        )
        return F.relu(x + self.gamma * self.project(fused), inplace=True)


class DCCF(nn.Module):
    """Decoder-guided Contrastive Cross-scale Fusion.

    Inputs:
      current: same-level encoder feature
      deep:    adjacent deeper encoder feature
      query:   upsampled decoder feature at the current resolution

    The module combines same-level detail, adjacent semantic context and a
    local contrast residual. The decoder query only guides the mixture weights,
    keeping the skip path residual and stable.
    """

    def __init__(self, current_ch, deep_ch, query_ch, out_ch):
        super(DCCF, self).__init__()
        self.current_proj = ConvBNAct(current_ch, out_ch, kernel_size=1, padding=0)
        self.deep_proj = ConvBNAct(deep_ch, out_ch, kernel_size=1, padding=0)
        self.query_proj = ConvBNAct(query_ch, out_ch, kernel_size=1, padding=0)

        gate_ch = max(out_ch // 8, 8)
        self.g_cur = nn.Conv2d(out_ch, gate_ch, kernel_size=1, bias=False)
        self.g_deep = nn.Conv2d(out_ch, gate_ch, kernel_size=1, bias=False)
        self.g_query = nn.Conv2d(out_ch, gate_ch, kernel_size=1, bias=False)
        self.g_contrast = nn.Conv2d(out_ch, gate_ch, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(gate_ch * 4, gate_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                gate_ch, gate_ch, kernel_size=3, padding=1,
                groups=gate_ch, bias=False
            ),
            nn.BatchNorm2d(gate_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_ch, 3, kernel_size=1, bias=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                out_ch, out_ch, kernel_size=3, padding=1,
                groups=out_ch, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, current, deep, query):
        target_size = current.shape[-2:]
        cur = self.current_proj(current)
        dep = self.deep_proj(deep)
        dep = F.interpolate(dep, size=target_size, mode="bilinear", align_corners=False)
        qry = self.query_proj(query)
        if qry.shape[-2:] != target_size:
            qry = F.interpolate(qry, size=target_size, mode="bilinear", align_corners=False)

        contrast = cur - F.avg_pool2d(cur, kernel_size=3, stride=1, padding=1)
        gate_input = torch.cat([
            self.g_cur(cur),
            self.g_deep(dep),
            self.g_query(qry),
            self.g_contrast(contrast),
        ], dim=1)
        weights = torch.softmax(self.gate(gate_input), dim=1)
        mixed = (
            cur * weights[:, 0:1] +
            dep * weights[:, 1:2] +
            contrast * weights[:, 2:3]
        )
        return F.relu(cur + self.gamma * self.refine(mixed), inplace=True)


class SASC(nn.Module):
    """Semantic-conditioned Axial Structure Calibration.

    This lightweight high-resolution module is inspired by directional
    encoding, but adds two task-specific cues: a local contrast residual and a
    deeper decoder semantic condition. It is applied only once near the output
    to keep computation low.
    """

    def __init__(self, channels, semantic_ch, reduction=8):
        super(SASC, self).__init__()
        hidden = max(channels // reduction, 8)
        self.semantic_proj = ConvBNAct(semantic_ch, channels, kernel_size=1, padding=0)
        self.reduce_h = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
        )
        self.reduce_w = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
        )
        self.expand_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.expand_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x, semantic):
        sem = self.semantic_proj(semantic)
        sem = F.interpolate(sem, size=x.shape[-2:], mode="bilinear", align_corners=False)
        contrast = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        conditioned = x + sem + contrast

        desc_h = conditioned.mean(dim=3, keepdim=True)
        desc_w = conditioned.mean(dim=2, keepdim=True)
        gate_h = torch.sigmoid(self.expand_h(self.reduce_h(desc_h)))
        gate_w = torch.sigmoid(self.expand_w(self.reduce_w(desc_w)))

        spatial_input = torch.cat([
            contrast.mean(dim=1, keepdim=True),
            sem.mean(dim=1, keepdim=True),
        ], dim=1)
        gate_s = self.spatial_gate(spatial_input)
        calibrated = x * gate_h * gate_w + contrast * gate_s
        return F.relu(x + self.gamma * self.out_proj(calibrated), inplace=True)


class DecoderRefine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderRefine, self).__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=3),
            DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3),
        )
        self.short = ConvBNAct(in_ch, out_ch, kernel_size=1, padding=0, activation=False)

    def forward(self, x):
        return F.relu(self.block(x) + self.short(x), inplace=True)


class UpProject(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpProject, self).__init__()
        self.project = ConvBNAct(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x, size=None):
        x = self.project(x)
        if size is None:
            return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ["state_dict", "model", "model_state_dict", "net"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def load_resnet34_pretrained(backbone, weight_path):
    if not weight_path:
        return {"loaded": 0, "missing": [], "unexpected": []}
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            "ResNet-34 pretrained weight was not found: {}".format(weight_path)
        )
    checkpoint = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)
    cleaned = OrderedDict()
    for key, value in state.items():
        new_key = key
        for prefix in ["module.", "backbone.", "encoder."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        if new_key.startswith("fc."):
            continue
        cleaned[new_key] = value
    incompatible = backbone.load_state_dict(cleaned, strict=False)
    loaded = len(cleaned) - len(incompatible.unexpected_keys)
    return {
        "loaded": loaded,
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }


class MCSFResUNet(nn.Module):
    def __init__(self, num_classes=9, pretrained_path=None):
        super(MCSFResUNet, self).__init__()
        try:
            backbone = models.resnet34(weights=None)
        except TypeError:
            backbone = models.resnet34(pretrained=False)

        self.pretrained_info = load_resnet34_pretrained(backbone, pretrained_path)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.mrra3 = MRRA(256)
        self.mrra4 = MRRA(512)

        self.up4 = UpProject(512, 256)
        self.fuse3 = DCCF(256, 512, 256, 256)
        self.dec3 = DecoderRefine(512, 256)

        self.up3 = UpProject(256, 128)
        self.fuse2 = DCCF(128, 256, 128, 128)
        self.dec2 = DecoderRefine(256, 128)

        self.up2 = UpProject(128, 64)
        self.fuse1 = DCCF(64, 128, 64, 64)
        self.dec1 = DecoderRefine(128, 64)

        self.up1 = UpProject(64, 32)
        self.fuse0 = DCCF(64, 64, 32, 32)
        self.dec0 = DecoderRefine(64, 32)
        self.sasc = SASC(32, 64)

        self.final_refine = nn.Sequential(
            ConvBNAct(32, 32, kernel_size=3),
            DepthwiseSeparableConv(32, 32, kernel_size=3),
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        # The auxiliary heads are always instantiated so checkpoint loading is
        # identical during training and testing.
        self.aux_d2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux_d1 = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_new_modules()

    def _init_new_modules(self):
        pretrained_names = {
            "conv1", "bn1", "layer1", "layer2", "layer3", "layer4"
        }
        for name, module in self.named_modules():
            root = name.split(".")[0] if name else ""
            if root in pretrained_names:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encoder_parameters(self):
        modules = [
            self.conv1, self.bn1, self.layer1, self.layer2,
            self.layer3, self.layer4
        ]
        for module in modules:
            for parameter in module.parameters():
                yield parameter

    def new_parameters(self):
        encoder_param_ids = set(id(p) for p in self.encoder_parameters())
        for parameter in self.parameters():
            if id(parameter) not in encoder_param_ids:
                yield parameter

    def set_shallow_encoder_trainable(self, trainable):
        modules = [self.conv1, self.bn1, self.layer1]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    def keep_frozen_shallow_eval(self):
        self.conv1.eval()
        self.bn1.eval()
        self.layer1.eval()

    def forward(self, x):
        input_size = x.shape[-2:]

        e0 = self.relu(self.bn1(self.conv1(x)))       # 1/2, 64
        e1 = self.layer1(self.maxpool(e0))            # 1/4, 64
        e2 = self.layer2(e1)                          # 1/8, 128
        e3_raw = self.layer3(e2)                      # 1/16, 256
        e3 = self.mrra3(e3_raw)
        e4_raw = self.layer4(e3_raw)                  # 1/32, 512
        e4 = self.mrra4(e4_raw)

        q3 = self.up4(e4, size=e3.shape[-2:])
        s3 = self.fuse3(e3, e4, q3)
        d3 = self.dec3(torch.cat([q3, s3], dim=1))

        q2 = self.up3(d3, size=e2.shape[-2:])
        s2 = self.fuse2(e2, e3, q2)
        d2 = self.dec2(torch.cat([q2, s2], dim=1))

        q1 = self.up2(d2, size=e1.shape[-2:])
        s1 = self.fuse1(e1, e2, q1)
        d1 = self.dec1(torch.cat([q1, s1], dim=1))

        q0 = self.up1(d1, size=e0.shape[-2:])
        s0 = self.fuse0(e0, e1, q0)
        d0 = self.dec0(torch.cat([q0, s0], dim=1))
        d0 = self.sasc(d0, d1)

        out_feature = F.interpolate(
            d0, size=input_size, mode="bilinear", align_corners=False
        )
        out_feature = self.final_refine(out_feature)
        out = self.head(out_feature)

        aux1 = self.aux_d1(d1)
        aux2 = self.aux_d2(d2)
        aux1 = F.interpolate(aux1, size=input_size, mode="bilinear", align_corners=False)
        aux2 = F.interpolate(aux2, size=input_size, mode="bilinear", align_corners=False)

        return {"out": out, "aux1": aux1, "aux2": aux2}


if __name__ == "__main__":
    model = MCSFResUNet(num_classes=9, pretrained_path=None)
    model.eval()
    sample = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(sample)
    for key, value in outputs.items():
        print(key, tuple(value.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
