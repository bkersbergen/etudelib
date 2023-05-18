# -*- coding: utf-8 -*-

r"""
NOOP
################################################
A RANDOM model. Used as a zero measurement


"""

import torch
from torch import nn


class RANDOMModel(nn.Module):
    r"""Return random query and random embedding matrix
    """

    def __init__(self,
                 n_items: int,
                 embedding_size: int,
                 ):
        super(RANDOMModel, self).__init__()

        self.n_items = n_items
        self.embedding_size = embedding_size
        torch.manual_seed(self.n_items)
        self.item_embedding = nn.Embedding(
            n_items, self.embedding_size, padding_idx=0
        )

    def forward(self, item_seq, _item_seq_len):
        result = torch.rand((1, self.embedding_size), device=item_seq.device)
        
        return result
