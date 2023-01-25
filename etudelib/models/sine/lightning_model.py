import logging
from abc import ABC
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from .torch_model import SINEModel

logger = logging.getLogger(__name__)
__all__ = ["SINE", "SINELightning"]


@MODEL_REGISTRY
class SINE(pl.LightningModule, ABC):
    def __init__(self,
                 embedding_size: int,
                 layer_norm_eps: float,
                 prototype_size: int,
                 interest_size: int,
                 tau_ratio: float,
                 reg_loss_ratio: float,
                 max_seq_length: int,
                 n_items: int,
                 ):
        super().__init__()
        self.model = SINEModel(embedding_size,
                               layer_norm_eps,
                               prototype_size,
                               interest_size,
                               tau_ratio,
                               reg_loss_ratio,
                               max_seq_length,
                               n_items,
                               )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.optimizer.lr)

    def get_backbone(self):
        return self.model


class SINELightning(SINE):
    """Torch Lightning Module for the SINE model.
        Args:
            hparams (Union[DictConfig, ListConfig]): Model params
        """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            embedding_size=hparams.model.embedding_size,
            layer_norm_eps=hparams.model.layer_norm_eps,
            prototype_size=hparams.model.prototype_size,
            interest_size=hparams.model.interest_size,
            tau_ratio=hparams.model.tau_ratio,
            reg_loss_ratio=hparams.model.reg_loss_ratio,
            max_seq_length=hparams.dataset.max_seq_length,
            n_items=hparams.dataset.n_items,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
