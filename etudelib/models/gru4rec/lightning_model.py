import logging
from abc import ABC
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from .torch_model import GRU4RecModel

logger = logging.getLogger(__name__)
__all__ = ["GRU4Rec", "GRU4RecLightning"]


@MODEL_REGISTRY
class GRU4Rec(pl.LightningModule, ABC):
    def __init__(self,
                 embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout_prob: float,
                 n_items: int,
                 ):
        super().__init__()

        self.model = GRU4RecModel(embedding_size,
                                  hidden_size,
                                  num_layers,
                                  dropout_prob,
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


class GRU4RecLightning(GRU4Rec):
    """Torch Lightning Module for the GRU4Rec model.
        Args:
            hparams (Union[DictConfig, ListConfig]): Model params
        """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            embedding_size=hparams.model.embedding_size,
            hidden_size=hparams.model.hidden_size,
            num_layers=hparams.model.num_layers,
            dropout_prob=hparams.model.dropout_prob,
            n_items=hparams.dataset.n_items,
                         )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
