"""
LightSANs
################################################
Reference:
    Xin-Yan Fan et al. "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." in SIGIR 2021.
Reference:
    https://github.com/BELIEVEfxy/LightSANs
"""

import torch.nn as nn
import torch.nn.functional as F

from etudelib.models.layers import LightTransformerEncoder


class LightSANs(nn.Module):
    def __init__(self, config):
        super(LightSANs, self).__init__()
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.k_interests = config["k_interests"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.seq_len = self.max_seq_length
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = LightTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            k_interests=self.k_interests,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)