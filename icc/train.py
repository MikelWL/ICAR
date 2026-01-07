from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from icc.data.datamodule import ICCDataModule
from icc.export import export_ckpt_to_icc_pt
from icc.lit import ICCLightningModule
from icc.thresholds import tune_binary_threshold


def set_seed(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))

    datamodule = ICCDataModule(
        data_dir=args.data_dir,
        annotations_file=args.annotations_file,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
    )
    datamodule.setup()

    model = ICCLightningModule(
        model_name=args.model_name,
        pretrained=True,
        num_classes=2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
    )

    tb_logger = TensorBoardLogger(save_dir=str(logs_dir), name="", default_hp_metric=False)

    callbacks = [
        ModelCheckpoint(dirpath=str(checkpoints_dir), filename="last", save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=int(args.early_stopping_patience),
                mode="min",
                verbose=True,
            )
        )

    trainer = pl.Trainer(
        max_epochs=int(args.num_epochs),
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=20,
        precision=16 if args.half_precision else 32,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)

    tuned_threshold = None
    if args.tune_threshold:
        device = next(model.parameters()).device
        probs = []
        targets = []
        model.eval()
        for batch in datamodule.val_dataloader():
            images, batch_targets = batch[:2]
            images = images.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(images)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.append(batch_probs.detach().cpu())
            targets.append(batch_targets.detach().cpu())

        if probs:
            probs_pos = torch.cat(probs, dim=0)
            targets_pos = torch.cat(targets, dim=0)
            result = tune_binary_threshold(
                probs_pos,
                targets_pos,
                metric=args.threshold_metric,
                thresholds=torch.linspace(0.0, 1.0, int(args.threshold_grid_steps)),
            )
            tuned_threshold = float(result.best_threshold)
            (output_dir / "threshold_tuning.json").write_text(
                json.dumps(
                    {
                        "metric": result.metric,
                        "best_threshold": result.best_threshold,
                        "best_metric_value": result.best_metric_value,
                    },
                    indent=2,
                )
            )

    trainer.test(model=model, datamodule=datamodule)

    final_ckpt = checkpoints_dir / "final_tested.ckpt"
    trainer.save_checkpoint(str(final_ckpt))

    # Save split indices for reproducibility
    split_info = {
        "seed": int(args.seed),
        "train_indices": list(map(int, datamodule.train_indices)),
        "val_indices": list(map(int, datamodule.val_indices)),
        "test_indices": list(map(int, datamodule.test_indices)),
    }
    (output_dir / "split_indices.json").write_text(json.dumps(split_info, indent=2))

    # Export ICAR-compatible ICC.pt
    export_path = output_dir / "ICC.pt"
    export_ckpt_to_icc_pt(
        str(final_ckpt),
        str(export_path),
        model_architecture=args.model_name,
        threshold=tuned_threshold,
    )

    print(f"Wrote Lightning checkpoint: {final_ckpt}")
    print(f"Wrote ICAR ICC checkpoint:  {export_path}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train ICC (ConvNeXt-V2 binary complexity classifier)")
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--annotations_file", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)

    p.add_argument("--model_name", type=str, default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--scheduler", type=str, default="cosine")
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--half_precision",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use mixed precision (fp16) when supported.",
        )
    else:  # pragma: no cover
        group = p.add_mutually_exclusive_group()
        group.add_argument(
            "--half_precision",
            action="store_true",
            default=True,
            help="Use mixed precision (fp16) when supported.",
        )
        group.add_argument(
            "--no_half_precision",
            dest="half_precision",
            action="store_false",
            help="Disable mixed precision.",
        )

    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--tune_threshold",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Tune and embed a default probability threshold using the val split.",
        )
    else:  # pragma: no cover
        group = p.add_mutually_exclusive_group()
        group.add_argument(
            "--tune_threshold",
            action="store_true",
            default=True,
            help="Tune and embed a default probability threshold using the val split.",
        )
        group.add_argument(
            "--no_tune_threshold",
            dest="tune_threshold",
            action="store_false",
            help="Disable threshold tuning.",
        )
    p.add_argument(
        "--threshold_metric",
        type=str,
        default="f1",
        choices=["f1", "balanced_accuracy", "accuracy", "youden_j"],
        help="Metric to maximize when tuning the threshold on the val split.",
    )
    p.add_argument(
        "--threshold_grid_steps",
        type=int,
        default=1001,
        help="Number of threshold values in [0,1] to scan (default: 1001).",
    )

    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--early_stopping",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable early stopping on `val_loss`.",
        )
    else:  # pragma: no cover
        group = p.add_mutually_exclusive_group()
        group.add_argument(
            "--early_stopping",
            action="store_true",
            default=True,
            help="Enable early stopping on `val_loss`.",
        )
        group.add_argument(
            "--no_early_stopping",
            dest="early_stopping",
            action="store_false",
            help="Disable early stopping.",
        )
    p.add_argument("--early_stopping_patience", type=int, default=5)

    raise SystemExit(main(p.parse_args()))
