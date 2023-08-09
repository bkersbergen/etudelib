# -*- coding: utf-8 -*-
# @Time   : 2020/8/25 19:56
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/9/15, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

r"""
NARM
################################################

Reference:
    Jing Li et al. "Neural Attentive Session-based Recommendation." in CIKM 2017.

Reference code:
    https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch

"""

import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_


class NARMModel(nn.Module):
    r"""NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
    and capture the user’s main purpose in the current session.

    """

    def __init__(self,
                 embedding_size: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout_probs: float,
                 n_items: int,
                 ):
        super(NARMModel, self).__init__()

        # load parameters info
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_probs = dropout_probs

        # define layers and loss
        self.item_embedding = nn.Embedding(
                n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_probs[0])
        self.gru = nn.GRU(
            self.embedding_size,
            self.hidden_size,
            self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs[1])
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, item_seq, item_seq_len):

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        seq_output = self.b(c_t)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]).to(torch.int64)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

