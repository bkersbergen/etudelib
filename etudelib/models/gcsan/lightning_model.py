import logging
from abc import ABC
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from .torch_model import GCSANModel
from ..loss import EmbLoss, embLoss_fn

logger = logging.getLogger(__name__)
__all__ = ["GCSAN", "GCSANLightning"]


class GCSAN(pl.LightningModule, ABC):
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 hidden_size: int,
                 inner_size: int,
                 hidden_dropout_prob: float,
                 attn_dropout_prob: float,
                 hidden_act: str,
                 layer_norm_eps: float,
                 step: int,
                 weight: float,
                 reg_weight: float,
                 initializer_range: float,
                 n_items: int,
                    ):
        super().__init__()

        self.model = GCSANModel(n_layers,
                 n_heads,
                 hidden_size,
                 inner_size,
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 hidden_act,
                 layer_norm_eps,
                 step,
                 weight,
                 reg_weight,
                 initializer_range,
                 n_items,
                )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.cross_entropy(y_hat, y)
        reg_loss = embLoss_fn(self.model.item_embedding.weight)
        total_loss = loss + self.model.reg_weight * reg_loss
        self.log('train_loss', total_loss, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.cross_entropy(y_hat, y)
        reg_loss = embLoss_fn(self.model.item_embedding.weight)
        total_loss = loss + self.model.reg_weight * reg_loss
        self.log('valid_loss', total_loss, on_step=True)


    def test_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.cross_entropy(y_hat, y)
        reg_loss = embLoss_fn(self.model.item_embedding.weight)
        total_loss = loss + self.model.reg_weight * reg_loss
        self.log('test_loss', total_loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.optimizer.lr)

    def get_backbone(self):
        return self.model


class GCSANLightning(GCSAN):
    """Torch Lightning Module for the GCSAN model.
        Args:
            hparams (Union[DictConfig, ListConfig]): Model params
        """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(n_layers=hparams.model.n_layers,
                 n_heads=hparams.model.n_heads,
                 hidden_size=hparams.model.hidden_size,
                 inner_size=hparams.model.inner_size,
                 hidden_dropout_prob=hparams.model.hidden_dropout_prob,
                 attn_dropout_prob=hparams.model.attn_dropout_prob,
                 hidden_act=hparams.model.hidden_act,
                 layer_norm_eps=hparams.model.layer_norm_eps,
                 step=hparams.model.step,
                 weight=hparams.model.weight,
                 reg_weight=hparams.model.reg_weight,
                 initializer_range=hparams.model.initializer_range,
                 n_items=hparams.dataset.n_items,
                         )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
