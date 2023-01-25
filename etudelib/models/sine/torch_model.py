# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 11:10
# @Author  : Jingqi Gao
# @Email   : jgaoaz@connect.ust.hk

r"""
SINE
################################################

Reference:
    Qiaoyu Tan et al. "Sparse-Interest Network for Sequential Recommendation." in WSDM 2021.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

torch.autograd.set_detect_anomaly(True)


class SINEModel(nn.Module):

    def __init__(self,
                 embedding_size: int,
                 layer_norm_eps: float,
                 prototype_size: int,
                 interest_size: int,
                 tau_ratio: float,
                 reg_loss_ratio: float,
                 max_seq_length: int,
                 n_items: int,
                 ):
        super(SINEModel, self).__init__()

        # load parameters info
        self.embedding_size = embedding_size
        self.layer_norm_eps = layer_norm_eps

        self.D = self.embedding_size
        self.L = prototype_size  # 500 for movie-len dataset
        self.k = interest_size  # 4 for movie-len dataset
        self.tau = tau_ratio  # 0.1 in paper
        self.reg_loss_ratio = reg_loss_ratio  # 0.1 in paper

        self.initializer_range = 0.01

        self.max_seq_length = max_seq_length

        self.w1 = self._init_weight((self.D, self.D))
        self.w2 = self._init_weight(self.D)
        self.w3 = self._init_weight((self.D, self.D))
        self.w4 = self._init_weight(self.D)

        self.C = nn.Embedding(self.L, self.D)

        self.w_k_1 = self._init_weight((self.k, self.D, self.D))
        self.w_k_2 = self._init_weight((self.k, self.D))
        self.item_embedding = nn.Embedding(n_items, self.D, padding_idx=0)
        self.ln2 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.ln4 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weight(self, shape):
        mat = np.random.normal(0, self.initializer_range, shape)
        return torch.tensor(mat, dtype=torch.float32)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def calculate_reg_loss(self):
        C_mean = torch.mean(self.C.weight, dim=1, keepdim=True)
        C_reg = self.C.weight - C_mean
        C_reg = C_reg.matmul(C_reg.T) / self.D
        return (torch.norm(C_reg) ** 2 - torch.norm(torch.diag(C_reg)) ** 2) / 2

    def forward(self, item_seq, _item_seq_len):
        x_u = self.item_embedding(item_seq)  # [B, N, D]
        # concept activation
        # sort by inner product
        x = torch.matmul(x_u, self.w1)
        x = torch.tanh(x)
        x = torch.matmul(x, self.w2)
        a = F.softmax(x, dim=1)
        z_u = torch.matmul(a.unsqueeze(2).transpose(1, 2), x_u).transpose(1, 2)
        s_u = torch.matmul(self.C.weight, z_u)
        s_u = s_u.squeeze(2)
        idx = s_u.argsort(1)[:, -self.k:]
        s_u_idx = s_u.sort(1)[0][:, -self.k:]
        c_u = self.C(idx)
        sigs = torch.sigmoid(s_u_idx.unsqueeze(2).repeat(1, 1, self.embedding_size))
        C_u = c_u.mul(sigs)

        # intention assignment
        # use matrix multiplication instead of cos()
        w3_x_u_norm = F.normalize(x_u.matmul(self.w3), p=2, dim=2)
        C_u_norm = self.ln2(C_u)
        P_k_t = torch.bmm(w3_x_u_norm, C_u_norm.transpose(1, 2))
        P_k_t_b = F.softmax(P_k_t, dim=2)
        P_k_t_b_t = P_k_t_b.transpose(1, 2)

        # attention weighting
        a_k = x_u.unsqueeze(1).repeat(1, self.k, 1, 1).matmul(self.w_k_1)
        P_t_k = F.softmax(
            torch.tanh(a_k)
            .matmul(self.w_k_2.reshape(self.k, self.embedding_size, 1))
            .squeeze(3),
            dim=2,
        )

        # interest embedding generation
        mul_p = P_k_t_b_t.mul(P_t_k)
        x_u_re = x_u.unsqueeze(1).repeat(1, self.k, 1, 1)
        mul_p_re = mul_p.unsqueeze(3)
        delta_k = x_u_re.mul(mul_p_re).sum(2)
        delta_k = F.normalize(delta_k, p=2, dim=2)

        # prototype sequence
        x_u_bar = P_k_t_b.matmul(C_u)
        C_apt = F.softmax(torch.tanh(x_u_bar.matmul(self.w3)).matmul(self.w4), dim=1)
        C_apt = C_apt.reshape(-1, 1, self.max_seq_length).matmul(x_u_bar)
        C_apt = self.ln4(C_apt)

        # aggregation weight
        e_k = delta_k.bmm(C_apt.reshape(-1, self.embedding_size, 1)) / self.tau
        e_k_u = F.softmax(e_k.squeeze(2), dim=1)
        v_u = e_k_u.unsqueeze(2).mul(delta_k).sum(dim=1)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            v_u, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
