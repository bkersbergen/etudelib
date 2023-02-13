import logging
from abc import ABC
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from .torch_model import SASRecModel

logger = logging.getLogger(__name__)
__all__ = ["SASRec", "SASRecLightning"]


class SASRec(pl.LightningModule, ABC):
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 hidden_size: int,
                 inner_size: int,
                 hidden_dropout_prob: float,
                 attn_dropout_prob: float,
                 hidden_act: str,
                 layer_norm_eps: float,
                 initializer_range: float,
                 max_seq_length: int,
                 n_items: int,
                 ):
        super().__init__()
        self.model = SASRecModel(
            n_layers,
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
            initializer_range,
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


class SASRecLightning(SASRec):
    """Torch Lightning Module for the STAMP model.
        Args:
            hparams (Union[DictConfig, ListConfig]): Model params
        """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            n_layers=hparams.model.n_layers,
            n_heads=hparams.model.n_heads,
            hidden_size=hparams.model.hidden_size,
            inner_size=hparams.model.inner_size,
            hidden_dropout_prob=hparams.model.hidden_dropout_prob,
            attn_dropout_prob=hparams.model.attn_dropout_prob,
            hidden_act=hparams.model.hidden_act,
            layer_norm_eps=hparams.model.layer_norm_eps,
            initializer_range=hparams.model.initializer_range,
            max_seq_length=hparams.dataset.max_seq_length,
            n_items=hparams.dataset.n_items,
            )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
