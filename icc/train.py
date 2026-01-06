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
