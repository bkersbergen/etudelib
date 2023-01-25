import logging
from abc import ABC
from typing import Union

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from .torch_model import RepeatNetModel

logger = logging.getLogger(__name__)
__all__ = ["RepeatNet", "RepeatNetLightning"]


@MODEL_REGISTRY
class RepeatNet(pl.LightningModule, ABC):
    def __init__(self, embedding_size: int,
                 hidden_size: int,
                 joint_train: bool,
                 dropout_prob: float,
                 n_items: int,
                 max_seq_length: int,
                 ):
        super().__init__()
        self.joint_train = joint_train
        self.model = RepeatNetModel(
            embedding_size,
            hidden_size,
            dropout_prob,
            n_items,
            max_seq_length,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.nll_loss((y_hat + 1e-8).log(), y, ignore_index=0)
        if self.joint_train is True:
            loss += self.repeat_explore_loss(item_seq, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.nll_loss((y_hat + 1e-8).log(), y, ignore_index=0)
        if self.joint_train is True:
            loss += self.repeat_explore_loss(item_seq, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        item_seq, item_seq_len, y = batch
        y_hat = self.model(item_seq, item_seq_len)
        loss = F.nll_loss((y_hat + 1e-8).log(), y, ignore_index=0)
        if self.joint_train is True:
            loss += self.repeat_explore_loss(item_seq, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.optimizer.lr)

    def get_backbone(self):
        return self.model

    def repeat_explore_loss(self, item_seq, pos_item):

        batch_size = item_seq.size(0)
        repeat, explore = torch.zeros(batch_size), torch.ones(
            batch_size
        )
        index = 0
        for seq_item_ex, pos_item_ex in zip(item_seq, pos_item):
            if pos_item_ex in seq_item_ex:
                repeat[index] = 1
                explore[index] = 0
            index += 1
        repeat_loss = torch.mul(
            repeat.unsqueeze(1), torch.log(self.repeat_explore[:, 0] + 1e-8)
        ).mean()
        explore_loss = torch.mul(
            explore.unsqueeze(1), torch.log(self.repeat_explore[:, 1] + 1e-8)
        ).mean()

        return (-repeat_loss - explore_loss) / 2


class RepeatNetLightning(RepeatNet):
    """Torch Lightning Module for the RepeatNet model.
        Args:
            hparams (Union[DictConfig, ListConfig]): Model params
        """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            embedding_size=hparams.model.embedding_size,
            hidden_size=hparams.model.hidden_size,
            joint_train=hparams.model.joint_train,
            dropout_prob=hparams.model.dropout_prob,
            n_items=hparams.dataset.n_items,
            max_seq_length=hparams.dataset.max_seq_length,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
