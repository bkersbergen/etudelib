# -*- coding: utf-8 -*-
# @Time   : 2020/10/4 16:55
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

r"""
GCSAN
################################################

Reference:
    Chengfeng Xu et al. "Graph Contextualized Self-Attention Network for Session-based Recommendation." in IJCAI 2019.

"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from etudelib.models.layers import TransformerEncoder
from etudelib.models.loss import EmbLoss, BPRLoss


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size: int, step: int = 1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )
        self.linear_edge_out = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via gated graph neural network.

        Args:
            A (torch.FloatTensor): The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden (torch.FloatTensor): The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = torch.matmul(A[:, :, : A.size(1)], self.linear_edge_in(hidden))
        input_out = torch.matmul(
            A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden)
        )
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class GCSANModel(nn.Module):
    r"""GCSAN captures rich local dependencies via graph neural network,
     and learns long-range dependencies by applying the self-attention mechanism.

    Note:

        In the original paper, the attention mechanism in the self-attention layer is a single head,
        for the reusability of the project code, we use a unified transformer component.
        According to the experimental results, we only applied regularization to embedding.
    """

    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 hidden_size: int,
                 inner_size: int,
                 hidden_dropout_prob: float,
                 attn_dropout_prob: float,
                 hidden_act: str,
                 layer_norm_eps: float,
                 step: int,
                 weight: float,
                 reg_weight: float,
                 initializer_range: float,
                 n_items: int,
                    ):
        super(GCSANModel, self).__init__()

        # load parameters info
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size  # same as embedding_size
        self.inner_size = inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps

        self.step = step
        self.weight = weight
        assert 0 <= self.weight <= 1
        self.reg_weight = reg_weight
        self.initializer_range = initializer_range

        # define layers and loss
        self.item_embedding = nn.Embedding(
            n_items, self.hidden_size, padding_idx=0
        )
        self.gnn = GNN(self.hidden_size, self.step)
        self.self_attention = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_slice(self, item_seq):
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq_np = item_seq.cpu().numpy()

        for u_input in item_seq_np:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.tensor(alias_inputs, dtype=torch.int64, device=item_seq.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.tensor(np.array(A), dtype=torch.float32, device=item_seq.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.tensor(items, dtype=torch.int64, device=item_seq.device)

        return alias_inputs, A, items

    def forward(self, item_seq, item_seq_len):
        alias_inputs, A, items = self._get_slice(item_seq)
        hidden = self.item_embedding(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.hidden_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        a = seq_hidden
        attention_mask = self.get_attention_mask(item_seq)

        outputs = self.self_attention(a, attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        at = self.gather_indexes(output, item_seq_len - 1)
        seq_output = self.weight * at + (1 - self.weight) * ht
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores


    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0).double()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask == 0.0, -10000.0, extended_attention_mask)
        return extended_attention_mask


    #
    # def calculate_loss(self, interaction):
    #     item_seq = interaction[self.ITEM_SEQ]
    #     item_seq_len = interaction[self.ITEM_SEQ_LEN]
    #     seq_output = self.forward(item_seq, item_seq_len)
    #     pos_items = interaction[self.POS_ITEM_ID]
    #     test_item_emb = self.item_embedding.weight
    #     logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
    #     loss = self.loss_fct(logits, pos_items)
    #     reg_loss = self.reg_loss(self.item_embedding.weight)
    #     total_loss = loss + self.reg_weight * reg_loss
    #     return total_loss
    #
    # def predict(self, interaction):
    #     item_seq = interaction[self.ITEM_SEQ]
    #     item_seq_len = interaction[self.ITEM_SEQ_LEN]
    #     test_item = interaction[self.ITEM_ID]
    #     seq_output = self.forward(item_seq, item_seq_len)
    #     test_item_emb = self.item_embedding(test_item)
    #     scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
    #     return scores
    #
    # def full_sort_predict(self, interaction):
    #     item_seq = interaction[self.ITEM_SEQ]
    #     item_seq_len = interaction[self.ITEM_SEQ_LEN]
    #     seq_output = self.forward(item_seq, item_seq_len)
    #     test_items_emb = self.item_embedding.weight
    #     scores = torch.matmul(
    #         seq_output, test_items_emb.transpose(0, 1)
    #     )  # [B, n_items]
    #     return scores
