from abc import ABC
import logging
from typing import List, Optional, Tuple, Union
from omegaconf import DictConfig, ListConfig

import torch
import torch.nn.functional as F

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import optim
import pytorch_lightning as pl
from .torch_model import LightSANsModel

logger = logging.getLogger(__name__)
__all__ = ["LightSANs", "LightSANsLightning"]


@MODEL_REGISTRY
class LightSANs(pl.LightningModule, ABC):
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 k_interests: int,
                 hidden_size: int,
                 inner_size: int,
                 hidden_dropout_prob: float,
                 attn_dropout_prob: float,
                 hidden_act: str,
                 layer_norm_eps: float,
                 initializer_range: float,
                 max_seq_length: int,
                 n_items: int,
                 topk: int,
                 ):
        super().__init__()

        self.model = LightSANsModel(n_layers,
                 n_heads,
                 k_interests,
                 hidden_size,
                 inner_size,
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 hidden_act,
                 layer_norm_eps,
                 initializer_range,
                 max_seq_length,
                 n_items,
                 topk)

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


class LightSANsLightning(LightSANs):
    """Torch Lightning Module for the LightSANs model.
        Args:
            hparams (Union[DictConfig, ListConfig]): Model params
        """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(n_layers=hparams.model.n_layers,
                         n_heads=hparams.model.n_heads,
                         k_interests=hparams.model.k_interests,
                         hidden_size=hparams.model.hidden_size,
                         inner_size=hparams.model.inner_size,
                         hidden_dropout_prob=hparams.model.hidden_dropout_prob,
                         attn_dropout_prob=hparams.model.attn_dropout_prob,
                         hidden_act=hparams.model.hidden_act,
                         layer_norm_eps=hparams.model.layer_norm_eps,
                         initializer_range=hparams.model.initializer_range,
                         max_seq_length=hparams.data.max_seq_length,
                         n_items=hparams.data.n_items,
                         topk=hparams.model.topk,
                         )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
