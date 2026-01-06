from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from .data import IC9600Dataset
from .metrics import compute_ic9600_metrics
from .preprocess import get_eval_transforms
from .lit import IC9600RegressionLightning


@torch.no_grad()
def _predict(
    model: torch.nn.Module, loader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    model.to(device)
    preds: List[float] = []
    targets: List[float] = []
    filenames: List[str] = []
    for batch in loader:
        images, y, names = batch
        images = images.to(device)
        p = model(images).detach().cpu().numpy().tolist()
        preds.extend(p)
        targets.extend(y.numpy().tolist())
        filenames.extend(list(names))
    return np.array(preds), np.array(targets), filenames


def main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    ckpt_path = args.checkpoint
    model = IC9600RegressionLightning.load_from_checkpoint(ckpt_path)

    dataset = IC9600Dataset(
        txt_path=args.ic9600_test_txt,
        img_dir=args.ic9600_img_dir,
        transform=get_eval_transforms(img_size=args.img_size),
        return_filenames=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    preds, targets, filenames = _predict(model, loader, device=device)
    metrics = compute_ic9600_metrics(preds, targets)

    df = pd.DataFrame(
        {
            "filename": filenames,
            "ground_truth": targets,
            "prediction": preds,
        }
    )
    df["error"] = df["prediction"] - df["ground_truth"]
    df["abs_error"] = np.abs(df["error"])
    df.to_csv(output_dir / "predictions.csv", index=False)

    (output_dir / "metrics.json").write_text(json.dumps(metrics.as_dict(), indent=2))

    print("IC9600 evaluation (ICNet-style):")
    print(f"  RMSE:     {metrics.rmse:.4f}")
    print(f"  RMAE:     {metrics.rmae:.4f}")
    print(f"  Pearson:  {metrics.pearson:.4f}")
    print(f"  Spearman: {metrics.spearman:.4f}")
    print(f"Wrote: {(output_dir / 'metrics.json')}")
    print(f"Wrote: {(output_dir / 'predictions.csv')}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate ConvNeXt regression model on IC9600")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--ic9600_test_txt", required=True, type=str)
    p.add_argument("--ic9600_img_dir", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    raise SystemExit(main(p.parse_args()))
