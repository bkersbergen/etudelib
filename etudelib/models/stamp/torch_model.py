# -*- coding: utf-8 -*-
# @Time   : 2020/9/8 19:24
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/10/2
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

r"""
STAMP
################################################

Reference:
    Qiao Liu et al. "STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation." in KDD 2018.

"""

import torch
from torch import nn
from torch.nn.init import normal_


class STAMPModel(nn.Module):
    r"""STAMP is capable of capturing users’ general interests from the long-term memory of a session context,
    whilst taking into account users’ current interests from the short-term memory of the last-clicks.


    Note:

        According to the test results, we made a little modification to the score function mentioned in the paper,
        and did not use the final sigmoid activation function.

    """

    def __init__(self, embedding_size: int, n_items: int):
        super(STAMPModel, self).__init__()

        # load parameters info
        self.embedding_size = embedding_size

        # define layers and loss
        self.item_embedding = nn.Embedding(
            n_items, self.embedding_size, padding_idx=0
        )
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w0 = nn.Linear(self.embedding_size, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=True)
        self.mlp_a = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.mlp_b = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.loss_fct = nn.CrossEntropyLoss()

        # # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        last_inputs = self.gather_indexes(item_seq_emb, item_seq_len - 1)
        org_memory = item_seq_emb
        ms = torch.div(torch.sum(org_memory, dim=1), item_seq_len.unsqueeze(1).float())
        alpha = self.count_alpha(org_memory, last_inputs, ms)
        vec = torch.matmul(alpha.unsqueeze(1), org_memory)
        ma = vec.squeeze(1) + ms
        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last_inputs))
        seq_output = hs * ht
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

    def count_alpha(self, context, aspect, output):
        r"""This is a function that count the attention weights

        Args:
            context(torch.FloatTensor): Item list embedding matrix, shape of [batch_size, time_steps, emb]
            aspect(torch.FloatTensor): The embedding matrix of the last click item, shape of [batch_size, emb]
            output(torch.FloatTensor): The average of the context, shape of [batch_size, emb]

        Returns:
            torch.Tensor:attention weights, shape of [batch_size, time_steps]
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(
            -1, timesteps, self.embedding_size
        )
        output_3dim = output.repeat(1, timesteps).view(
            -1, timesteps, self.embedding_size
        )
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]).to(torch.int64)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

