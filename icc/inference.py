from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ICCHParams:
    model_name: str
    num_classes: int
    feature_dim: int


class ConvNeXtICC(nn.Module):
    """
    Pure-PyTorch inference module expected by ICAR.

    State dict is expected to have (at minimum) keys:
    - backbone.*
    - classifier.*
    """

    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        try:
            import timm  # local import: keep inference import surface minimal
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency `timm`. Install it to construct ConvNeXtICC."
            ) from exc

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove timm head
        )

        feature_dim = int(getattr(self.backbone, "num_features"))
        self.feature_dim = feature_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    @staticmethod
    def infer_hparams_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> ICCHParams:
        classifier_weight = state_dict.get("classifier.1.weight")
        if classifier_weight is None:
            raise ValueError(
                "Could not infer hparams: missing `classifier.1.weight` in state_dict."
            )
        num_classes, feature_dim = int(classifier_weight.shape[0]), int(
            classifier_weight.shape[1]
        )
        return ICCHParams(
            model_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
            num_classes=num_classes,
            feature_dim=feature_dim,
        )


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Make exported state dict directly loadable by `ConvNeXtICC`.

    Handles common training-time prefixes (Lightning/DataParallel).
    """
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        if new_key.startswith("net."):
            new_key = new_key[len("net.") :]
        stripped[new_key] = value
    return stripped


def load_icc_checkpoint(
    checkpoint_path: str,
    *,
    device: Optional[Union[str, torch.device]] = None,
    model_name: Optional[str] = None,
    num_classes: int = 2,
    strict: bool = True,
) -> Tuple[ConvNeXtICC, Dict[str, Any]]:
    """
    Load an ICAR-compatible `ICC.pt` (or a raw state dict).

    Accepted checkpoint formats:
    1) `{"state_dict": ..., "hparams": {...}, ...}`
    2) raw state dict mapping param_name -> tensor
    """
    map_location = device if device is not None else "cpu"
    obj = torch.load(checkpoint_path, map_location=map_location)

    metadata: Dict[str, Any] = {}
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
        metadata = {k: v for k, v in obj.items() if k != "state_dict"}
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(obj).__name__}")

    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint: `state_dict` is not a dict.")

    state_dict = _strip_known_prefixes(state_dict)

    hparams = metadata.get("hparams") if isinstance(metadata, dict) else None
    if isinstance(hparams, dict):
        model_name = model_name or hparams.get("model_name")
        num_classes = int(hparams.get("num_classes", num_classes))

    if model_name is None:
        model_name = "convnextv2_tiny.fcmae_ft_in22k_in1k"

    model = ConvNeXtICC(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model = model.to(device)
    model.eval()
    return model, metadata

