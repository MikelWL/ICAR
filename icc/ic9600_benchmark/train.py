from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .data import IC9600RegressionDataModule
from .lit import IC9600RegressionLightning


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

    datamodule = IC9600RegressionDataModule(
        ic9600_train_txt=args.ic9600_train_txt,
        ic9600_test_txt=args.ic9600_test_txt,
        ic9600_img_dir=args.ic9600_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        val_split=args.val_split,
        seed=args.seed,
        augment=args.augment,
    )
    datamodule.setup()

    model = IC9600RegressionLightning(
        model_name=args.model_name,
        pretrained=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        dropout=args.dropout,
    )

    tb_logger = TensorBoardLogger(save_dir=str(logs_dir), name="", default_hp_metric=False)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoints_dir),
            filename="best",
            monitor="val_pearson",
            mode="max",
            save_top_k=1,
            save_last=False,
        ),
        ModelCheckpoint(dirpath=str(checkpoints_dir), filename="last", save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

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

    final_ckpt = checkpoints_dir / "final_model.ckpt"
    trainer.save_checkpoint(str(final_ckpt))

    # Save a small "best val" snapshot for convenience/provenance
    best = {
        "seed": int(args.seed),
        "model_name": args.model_name,
        "img_size": int(args.img_size),
        "batch_size": int(args.batch_size),
        "val_split": float(args.val_split),
        "best_model_path": str(getattr(trainer.checkpoint_callback, "best_model_path", "")),
    }
    (output_dir / "run_config.json").write_text(json.dumps(best, indent=2))

    print(f"Wrote checkpoints under: {checkpoints_dir}")
    print(f"Final model: {final_ckpt}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train ConvNeXt regression model on IC9600 (ICNet-style benchmark)"
    )
    p.add_argument("--ic9600_train_txt", required=True, type=str)
    p.add_argument("--ic9600_test_txt", required=True, type=str)
    p.add_argument("--ic9600_img_dir", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)

    p.add_argument("--model_name", type=str, default="convnextv2_nano.fcmae_ft_in22k_in1k")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--augment", action="store_true", default=True)

    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--dropout", type=float, default=0.2)

    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument("--half_precision", action=argparse.BooleanOptionalAction, default=True)
    else:  # pragma: no cover
        p.add_argument("--half_precision", action="store_true", default=True)
        p.add_argument("--no-half-precision", dest="half_precision", action="store_false")

    p.add_argument("--seed", type=int, default=42)

    raise SystemExit(main(p.parse_args()))
