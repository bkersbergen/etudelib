# -*- coding: utf-8 -*-

r"""
NOOP
################################################
A No-Operations model. Used as a zero measurement


"""

import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_


class NOOPModel(nn.Module):
    r"""Allways return a list of random numbers of size n_items.
    """

    def __init__(self,
                 n_items: int,
                 ):
        super(NOOPModel, self).__init__()

        generator = torch.Generator()
        generator.manual_seed(n_items)
        # load parameters info
        self.result = torch.rand(n_items, generator=generator).unsqueeze(0)

    def forward(self, item_seq, item_seq_len):
        return self.result
