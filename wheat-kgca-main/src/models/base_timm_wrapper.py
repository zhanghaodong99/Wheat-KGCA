import torch
import torch.nn as nn
import timm


class TimmBackboneWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        timm_model_name: str,
        num_classes: int = 6,
        pretrained: bool = True,
        input_size: int = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.timm_model_name = timm_model_name
        self.input_size = input_size
        self.encoder = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        self.encoder.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)
            dummy_feat = self.encoder.forward_features(dummy_input)
            self.feature_dim = self._infer_feature_dim(dummy_feat)

        self.head = nn.Linear(self.feature_dim, num_classes)

    @staticmethod
    def _infer_feature_dim(feat: torch.Tensor) -> int:
        if feat.dim() == 4:
            if feat.shape[1] > feat.shape[2] and feat.shape[1] > feat.shape[3]:
                return feat.shape[1]
            return feat.shape[3]
        if feat.dim() == 3:
            return feat.shape[2]
        return feat.shape[-1]

    def _pool_features(self, raw_feat: torch.Tensor) -> torch.Tensor:
        if raw_feat.dim() == 4:
            if raw_feat.shape[1] != self.feature_dim:
                raw_feat = raw_feat.permute(0, 3, 1, 2).contiguous()
            return raw_feat.mean(dim=[-2, -1])
        if raw_feat.dim() == 3:
            return raw_feat.mean(dim=1)
        return raw_feat

    def forward(self, x):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        raw_feat = self.encoder.forward_features(x)
        feat_glob = self._pool_features(raw_feat)
        out = self.head(feat_glob)
        return out, raw_feat
