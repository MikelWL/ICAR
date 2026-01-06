from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, ROC


class ICCLightningModule(pl.LightningModule):
    """
    Lightning training wrapper that keeps the same parameter naming as ICAR:
    - `backbone.*`
    - `classifier.*`
    """

    def __init__(
        self,
        *,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        num_classes: int = 2,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-5,
        scheduler: str = "cosine",
    ):
        super().__init__()
        self.save_hyperparameters()

        import timm

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feature_dim = int(self.backbone.num_features)
        self.feature_dim = feature_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

        metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="binary", num_classes=2),
                "precision": Precision(task="binary", num_classes=2),
                "recall": Recall(task="binary", num_classes=2),
                "f1": F1Score(task="binary", num_classes=2),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.train_roc = ROC(task="binary")
        self.val_roc = ROC(task="binary")
        self.test_roc = ROC(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def _step(self, batch, stage: str):
        images, targets = batch[:2]
        logits = self(images)
        loss = F.cross_entropy(logits, targets)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            self.log_dict(self.train_metrics(preds, targets), prog_bar=True, on_epoch=True)
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.train_roc.update(probs[:, 1], targets)
        elif stage == "val":
            self.log_dict(self.val_metrics(preds, targets), prog_bar=True, on_epoch=True)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            self.val_roc.update(probs[:, 1], targets)
        else:
            self.log_dict(self.test_metrics(preds, targets), on_epoch=True)
            self.log("test_loss", loss, on_epoch=True)
            self.test_roc.update(probs[:, 1], targets)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _log_auc(self, roc: ROC, key: str, prog_bar: bool):
        fpr, tpr, _ = roc.compute()
        auc_value = torch.trapz(tpr, fpr)
        self.log(key, auc_value, prog_bar=prog_bar, on_epoch=True)
        roc.reset()

    def on_train_epoch_end(self):
        self._log_auc(self.train_roc, "train_roc_auc", prog_bar=True)

    def on_validation_epoch_end(self):
        self._log_auc(self.val_roc, "val_roc_auc", prog_bar=True)

    def on_test_epoch_end(self):
        self._log_auc(self.test_roc, "test_roc_auc", prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        return optimizer

