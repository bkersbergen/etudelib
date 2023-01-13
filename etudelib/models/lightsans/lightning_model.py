import torch
import torch.nn.functional as F

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import optim
import pytorch_lightning as pl
import logging
from abc import ABC
from .torch_model import LightSANs

logger = logging.getLogger(__name__)
__all__ = ["LightSANs", "LightSANsLightning"]


@MODEL_REGISTRY
class LightSANs(pl.LightningModule, ABC):
    def __init__(self, backbone):
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class LightSANsLightning(LightSANs):
    pass
