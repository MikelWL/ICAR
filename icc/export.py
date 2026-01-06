from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from icc.inference import _strip_known_prefixes


def _extract_state_dict_and_hparams(obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Handle:
    - Lightning checkpoint dicts (with `state_dict` and `hyper_parameters`)
    - already-exported ICC dicts (with `state_dict` and `hparams`)
    - raw state dict
    """
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
        meta = {k: v for k, v in obj.items() if k != "state_dict"}
        hparams = {}
        if isinstance(meta.get("hparams"), dict):
            hparams = dict(meta["hparams"])
        elif isinstance(meta.get("hyper_parameters"), dict):
            hparams = dict(meta["hyper_parameters"])
        meta["hparams"] = hparams
        return state_dict, meta

    if isinstance(obj, dict):
        return obj, {"hparams": {}}

    raise ValueError(f"Unsupported checkpoint type: {type(obj).__name__}")


def export_ckpt_to_icc_pt(
    ckpt_path: str,
    out_path: str,
    *,
    model_architecture: Optional[str] = None,
) -> str:
    """
    Convert a Lightning `.ckpt` into ICAR-compatible `ICC.pt`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict, meta = _extract_state_dict_and_hparams(ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint: `state_dict` is not a dict.")

    state_dict = _strip_known_prefixes(state_dict)
    state_dict = {
        k: v for k, v in state_dict.items() if k.startswith(("backbone.", "classifier."))
    }
    if not state_dict:
        raise ValueError(
            "Export produced an empty state_dict; expected keys under `backbone.*` and `classifier.*`."
        )

    if model_architecture is None:
        model_architecture = (
            meta.get("hparams", {}).get("model_name")
            or meta.get("hyper_parameters", {}).get("model_name")
            or "convnextv2_tiny.fcmae_ft_in22k_in1k"
        )

    # Populate hparams for provenance (and ICAR convenience)
    hparams = dict(meta.get("hparams") or {})
    if "model_name" not in hparams:
        hparams["model_name"] = model_architecture
    if "num_classes" not in hparams:
        linear_w = state_dict.get("classifier.1.weight")
        if linear_w is not None:
            hparams["num_classes"] = int(linear_w.shape[0])
    if "feature_dim" not in hparams:
        linear_w = state_dict.get("classifier.1.weight")
        if linear_w is not None:
            hparams["feature_dim"] = int(linear_w.shape[1])

    out_obj = {
        "state_dict": state_dict,
        "hparams": hparams,
        "model_architecture": model_architecture,
    }

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export Lightning ICC checkpoint to ICAR ICC.pt")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to Lightning .ckpt")
    parser.add_argument("--out", required=True, type=str, help="Output ICC.pt path")
    parser.add_argument(
        "--model_architecture",
        default="",
        type=str,
        help="Override model_architecture string (optional)",
    )
    args = parser.parse_args()

    model_arch = args.model_architecture.strip() or None
    export_ckpt_to_icc_pt(args.ckpt, args.out, model_architecture=model_arch)


if __name__ == "__main__":
    main()
