# -*- coding: utf-8 -*-

r"""
NOOP
################################################
A TOPKONLY model. Used as a zero measurement


"""
import torch
from torch import nn


class TOPKONLYModel(nn.Module):
    r"""this is our baseline that is dependent on the value of C for measuring the latency..
    TODO:
        1) This is our baseline that is dependent on the value of C for measuring the latency.
        2) Randomly arrange the number of items in the initialize step to prevent inference-consistency-warnings during conversion
    """

    def __init__(self,
                 n_items: int,
                 ):
        super(TOPKONLYModel, self).__init__()

        # register buffer automatically moved to the same device as the module when it is loaded
        self.register_buffer('data', torch.randint(n_items, size=(n_items,), dtype=torch.float32))

    def forward(self, item_seq, _item_seq_len):
        shape = (item_seq.size(0),) + self.data.shape
        result = self.data.unsqueeze(0).expand(shape)
        return result

