import logging
from abc import ABC
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from .torch_model import CORE_KNNModel

logger = logging.getLogger(__name__)
__all__ = ["CORE_KNN", "CORE_KNNLightning"]


class CORE_KNN(pl.LightningModule, ABC):
    def __init__(self,
                 embedding_size: int,
                 dnn_type: str,
                 sess_dropout: float,
                 item_dropout: float,
                 temperature: float,
                 n_layers: int,
                 n_heads: int,
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

        self.model = CORE_KNNModel(embedding_size,
                 dnn_type,
                 sess_dropout,
                 item_dropout,
                 temperature,
                 n_layers,
                 n_heads,
                 inner_size,
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 hidden_act,
                 layer_norm_eps,
                 initializer_range,
                 max_seq_length,
                 n_items)

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


class CORE_KNNLightning(CORE_KNN):
    """Torch Lightning Module for the CORE_KNN model.
        Args:
            hparams (Union[DictConfig, ListConfig]): Model params
        """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(embedding_size=hparams.model.embedding_size,
                         dnn_type=hparams.model.dnn_type,
                         sess_dropout=hparams.model.sess_dropout,
                         item_dropout=hparams.model.item_dropout,
                         temperature=hparams.model.temperature,
                         n_layers=hparams.model.n_layers,
                         n_heads=hparams.model.n_heads,
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
