# -*- coding: utf-8 -*-

r"""
NOOP
################################################
A RANDOM model. Used as a zero measurement


"""

import torch
from torch import nn


class RANDOMModel(nn.Module):
    r"""Allways return a list of numbers of size n_items.
    """

    def __init__(self,
                 n_items: int,
                 ):
        super(RANDOMModel, self).__init__()

        self.result = torch.arange(0, n_items)

    def forward(self, item_seq, _item_seq_len):
        batch_size = item_seq.shape[0]
        return self.result.repeat(batch_size, 1)
