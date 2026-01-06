from __future__ import annotations

import torch
import torch.nn as nn


class ConvNeXtComplexityRegressor(nn.Module):
    """
    Pure PyTorch regression model: ConvNeXt-V2 backbone + MLP regression head + sigmoid.
    Output is a scalar score in [0, 1].
    """

    def __init__(
        self,
        *,
        model_name: str = "convnextv2_nano.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency `timm`.") from exc

        self.model_name = model_name

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feature_dim = int(self.backbone.num_features)
        self.feature_dim = feature_dim

        self.regressor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        score = self.regressor(features)
        return score.squeeze(-1)
