from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ECAAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.act(y)


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        gate = self.act(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * gate


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class ResidualScale(nn.Module):
    def __init__(self, init_value: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * residual


class ResidualECAAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, init_gamma: float = 0.1):
        super().__init__()
        self.eca = ECAAttention(channels, kernel_size=kernel_size)
        self.scale = ResidualScale(init_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale(x, self.eca(x) - x)


class StackedResidualECAAttention(nn.Module):
    def __init__(self, channels: int, depth: int = 2, init_gamma: float = 0.08):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualECAAttention(channels, init_gamma=init_gamma) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ResidualCoordinateAttention(nn.Module):
    def __init__(self, channels: int, init_gamma: float = 0.1):
        super().__init__()
        self.coord = CoordinateAttention(channels)
        self.scale = ResidualScale(init_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale(x, self.coord(x) - x)


class ResidualCBAMLite(nn.Module):
    def __init__(self, channels: int, init_gamma: float = 0.1):
        super().__init__()
        self.channel = ECAAttention(channels)
        self.spatial = SpatialGate()
        self.scale = ResidualScale(init_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.spatial(self.channel(x))
        return self.scale(x, refined - x)


class ResidualDepthwiseRefine(nn.Module):
    def __init__(self, channels: int, init_gamma: float = 0.1):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.scale = ResidualScale(init_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale(x, self.refine(x))


class ConvNeXtV2AttentionVariant(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        attention: str,
        head: str = "linear",
        input_size: int = 224,
    ):
        super().__init__()
        self.input_size = input_size
        self.encoder = timm.create_model("convnextv2_tiny", pretrained=pretrained, num_classes=0)

        self.encoder.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)
            dummy_feat = self.encoder.forward_features(dummy_input)
            self.feature_dim = self._infer_feature_dim(dummy_feat)

        if attention == "eca":
            self.attention = ECAAttention(self.feature_dim)
        elif attention == "se":
            self.attention = SEAttention(self.feature_dim)
        elif attention == "spatial":
            self.attention = SpatialGate()
        elif attention == "coordinate":
            self.attention = CoordinateAttention(self.feature_dim)
        elif attention == "res_eca":
            self.attention = ResidualECAAttention(self.feature_dim)
        elif attention == "res_eca2":
            self.attention = StackedResidualECAAttention(self.feature_dim, depth=2)
        elif attention == "res_coordinate":
            self.attention = ResidualCoordinateAttention(self.feature_dim)
        elif attention == "res_cbam":
            self.attention = ResidualCBAMLite(self.feature_dim)
        elif attention == "res_dwconv":
            self.attention = ResidualDepthwiseRefine(self.feature_dim)
        elif attention == "none":
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unsupported attention: {attention}")

        if head == "mlp":
            hidden = max(self.feature_dim // 2, 128)
            self.head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, num_classes),
            )
        elif head == "linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        else:
            raise ValueError(f"Unsupported head: {head}")

    @staticmethod
    def _infer_feature_dim(feat: torch.Tensor) -> int:
        if feat.dim() == 4:
            if feat.shape[1] > feat.shape[2] and feat.shape[1] > feat.shape[3]:
                return feat.shape[1]
            return feat.shape[3]
        if feat.dim() == 3:
            return feat.shape[2]
        return feat.shape[-1]

    def _to_nchw(self, raw_feat: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if raw_feat.dim() != 4:
            raise ValueError(f"Expected a 4D feature map, got shape {tuple(raw_feat.shape)}")
        if raw_feat.shape[1] == self.feature_dim:
            return raw_feat, False
        return raw_feat.permute(0, 3, 1, 2).contiguous(), True

    @staticmethod
    def _restore_layout(feat: torch.Tensor, was_nhwc: bool) -> torch.Tensor:
        if was_nhwc:
            return feat.permute(0, 2, 3, 1).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        raw_feat = self.encoder.forward_features(x)
        if raw_feat.dim() == 4:
            feat_nchw, was_nhwc = self._to_nchw(raw_feat)
            attn_feat = self.attention(feat_nchw)
            pooled = attn_feat.mean(dim=[-2, -1])
            returned_feat = self._restore_layout(attn_feat, was_nhwc)
        elif raw_feat.dim() == 3:
            pooled = raw_feat.mean(dim=1)
            returned_feat = raw_feat
        else:
            pooled = raw_feat
            returned_feat = raw_feat

        logits = self.head(pooled)
        return logits, returned_feat
