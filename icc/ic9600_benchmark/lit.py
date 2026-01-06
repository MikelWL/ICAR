from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .metrics import compute_ic9600_metrics
from .model import ConvNeXtComplexityRegressor


class IC9600RegressionLightning(pl.LightningModule):
    def __init__(
        self,
        *,
        model_name: str = "convnextv2_nano.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler: str = "cosine",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConvNeXtComplexityRegressor(
            model_name=model_name, pretrained=pretrained, dropout=dropout
        )

        self._val_preds = []
        self._val_targets = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch[:2]
        preds = self(images)
        loss = F.mse_loss(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch[:2]
        preds = self(images)
        loss = F.mse_loss(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(targets.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).numpy()
        targets = torch.cat(self._val_targets).numpy()
        m = compute_ic9600_metrics(preds, targets)
        self.log("val_rmse", m.rmse, prog_bar=True)
        self.log("val_rmae", m.rmae, prog_bar=True)
        self.log("val_pearson", m.pearson, prog_bar=True)
        self.log("val_spearman", m.spearman, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
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
