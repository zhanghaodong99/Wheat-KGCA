from __future__ import annotations

import torch
import torch.nn as nn


class SemanticResidualGate(nn.Module):
    """Fuse VLM/expert semantic evidence into visual logits with a gated residual.

    This module keeps the visual classifier as the main decision path. The
    semantic branch only contributes a bounded residual, so ablation can test
    whether structured agronomic evidence improves the visual baseline.
    """

    def __init__(
        self,
        visual_dim: int,
        semantic_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.semantic_encoder = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(visual_dim + hidden_dim),
            nn.Linear(visual_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid(),
        )
        self.semantic_residual = nn.Linear(hidden_dim, num_classes)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(
        self,
        visual_feature: torch.Tensor,
        visual_logits: torch.Tensor,
        semantic_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        semantic_feature = self.semantic_encoder(semantic_vector)
        gate = self.gate(torch.cat([visual_feature, semantic_feature], dim=-1))
        residual = self.semantic_residual(semantic_feature)
        fused_logits = visual_logits + self.residual_scale * gate * residual
        aux = {
            "semantic_feature": semantic_feature,
            "semantic_gate": gate,
            "semantic_residual": residual,
        }
        return fused_logits, aux


class FeatureWiseSemanticGate(nn.Module):
    """Modulate visual features before the classifier using semantic evidence."""

    def __init__(
        self,
        visual_dim: int,
        semantic_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, visual_feature: torch.Tensor, semantic_vector: torch.Tensor) -> torch.Tensor:
        gate = self.gate(semantic_vector)
        return visual_feature * (1.0 + self.residual_scale * gate)
