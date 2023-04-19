# -*- coding: utf-8 -*-

r"""
NOOP
################################################
A No-Operations model. Used as a zero measurement


"""

import torch
from torch import nn


class NOOPModel(nn.Module):
    r"""Allways return a fixed list of numbers.
    """

    def __init__(self,):
        super(NOOPModel, self).__init__()

        self.result = torch.arange(0, 21)

    def forward(self, item_seq, _item_seq_len):
        batch_size = item_seq.shape[0]
        return self.result.repeat(batch_size, 1)
