import logging
from abc import ABC
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from .torch_model import RANDOMModel

logger = logging.getLogger(__name__)
__all__ = ["RANDOM", "RANDOMLightning"]


class RANDOM(pl.LightningModule, ABC):
    def __init__(self, 
                 n_items: int,
                 embedding_size: int,
                ):
        super().__init__()
        self.model = RANDOMModel(n_items, 
                                 embedding_size, 
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


class RANDOMLightning(RANDOM):
    """Torch Lightning Module for the RANDOM model.
        Args:
            hparams (Union[DictConfig, ListConfig]): Model params
        """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            n_items=hparams.dataset.n_items,
            embedding_size=hparams.model.embedding_size,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
