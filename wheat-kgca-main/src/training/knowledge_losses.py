from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_STAGE_ORDER = (
    "seedling",
    "tillering",
    "stem_elongation",
    "flag_leaf_booting",
    "heading",
    "flowering",
    "grain_filling",
    "maturity",
)


def adjacent_mask(num_classes: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(num_classes, device=device)
    return (idx[:, None] - idx[None, :]).abs() == 1


class NonAdjacentSuppressionLoss(nn.Module):
    """Penalize high probability assigned to non-adjacent growth stages."""

    def __init__(self, num_classes: int = 8, margin: float = 0.05) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        class_idx = torch.arange(self.num_classes, device=logits.device)
        distance = (class_idx[None, :] - target[:, None]).abs()
        non_adjacent = distance > 1
        penalty = torch.relu(probs - self.margin) * non_adjacent.float()
        denom = non_adjacent.float().sum().clamp_min(1.0)
        return penalty.sum() / denom


class AdjacentPairConsistencyLoss(nn.Module):
    """Encourage adjacent-pair auxiliary logits to match boundary-pair targets."""

    def __init__(self, adjacent_pairs: tuple[tuple[int, int], ...] | None = None) -> None:
        super().__init__()
        self.adjacent_pairs = adjacent_pairs or ((0, 1), (2, 3), (4, 5), (5, 6))

    def forward(self, pair_logits: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for a, b in self.adjacent_pairs:
            key = f"{a}_{b}"
            if key not in pair_logits:
                continue
            mask = (target == a) | (target == b)
            if not mask.any():
                continue
            binary_target = (target[mask] == b).long()
            losses.append(F.cross_entropy(pair_logits[key][mask], binary_target))
        if not losses:
            return torch.zeros((), device=target.device, dtype=torch.float32)
        return torch.stack(losses).mean()


class SemanticAnchorLoss(nn.Module):
    """Align predicted distribution with VLM/expert semantic stage evidence."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, semantic_probs: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        targets = semantic_probs / semantic_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.kl_div(log_probs, targets, reduction="batchmean") * (self.temperature**2)


class WheatKnowledgeLoss(nn.Module):
    """Combined training objective used by Wheat-KGCA ablation experiments."""

    def __init__(
        self,
        lambda_non_adjacent: float = 0.05,
        lambda_pair: float = 0.1,
        lambda_semantic: float = 0.0,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.lambda_non_adjacent = lambda_non_adjacent
        self.lambda_pair = lambda_pair
        self.lambda_semantic = lambda_semantic
        self.non_adjacent = NonAdjacentSuppressionLoss(num_classes=num_classes)
        self.pair = AdjacentPairConsistencyLoss()
        self.semantic = SemanticAnchorLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pair_logits: dict[str, torch.Tensor] | None = None,
        semantic_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ce = F.cross_entropy(logits, target)
        non_adj = self.non_adjacent(logits, target)
        pair = self.pair(pair_logits or {}, target)
        if semantic_probs is None or self.lambda_semantic == 0:
            semantic = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            semantic = self.semantic(logits, semantic_probs)
        total = (
            ce
            + self.lambda_non_adjacent * non_adj
            + self.lambda_pair * pair
            + self.lambda_semantic * semantic
        )
        parts = {
            "loss_ce": ce.detach(),
            "loss_non_adjacent": non_adj.detach(),
            "loss_pair": pair.detach(),
            "loss_semantic": semantic.detach(),
        }
        return total, parts
