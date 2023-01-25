# -*- coding: utf-8 -*-
# @Time     : 2020/11/22 8:30
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun, Fan xinyan
# @Email    : shaoweiqi@ruc.edu.cn, xinyan.fan@ruc.edu.cn

r"""
RepeatNet
################################################

Reference:
    Pengjie Ren et al. "RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation."
    in AAAI 2019

Reference code:
    https://github.com/PengjieRen/RepeatNet.

"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_


class RepeatNetModel(nn.Module):
    r"""
    RepeatNet explores a hybrid encoder with an repeat module and explore module
    repeat module is used for finding out the repeat consume in sequential recommendation
    explore module is used for exploring new items for recommendation

    """

    def __init__(self,
                 embedding_size: int,
                 hidden_size:int,
                 dropout_prob: float,
                 n_items: int,
                 max_seq_length: int,
                 ):

        super(RepeatNetModel, self).__init__()

        # load parameters
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # define the layers and loss function
        self.item_matrix = nn.Embedding(
                n_items, self.embedding_size, padding_idx=0
        )
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.repeat_explore_mechanism = Repeat_Explore_Mechanism(            hidden_size=self.hidden_size,
            seq_len=max_seq_length,
            dropout_prob=self.dropout_prob,
        )
        self.repeat_recommendation_decoder = Repeat_Recommendation_Decoder(
            hidden_size=self.hidden_size,
            seq_len=max_seq_length,
            num_item=n_items,
            dropout_prob=self.dropout_prob,
        )
        self.explore_recommendation_decoder = Explore_Recommendation_Decoder(
            hidden_size=self.hidden_size,
            seq_len=max_seq_length,
            num_item=n_items,
            dropout_prob=self.dropout_prob,
        )

        # init the weight of the module
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, item_seq, item_seq_len):

        batch_seq_item_embedding = self.item_matrix(item_seq)
        # batch_size * seq_len == embedding ==>> batch_size * seq_len * embedding_size

        all_memory, _ = self.gru(batch_seq_item_embedding)
        last_memory = self.gather_indexes(all_memory, item_seq_len - 1)
        # all_memory: batch_size * item_seq * hidden_size
        # last_memory: batch_size * hidden_size
        timeline_mask = item_seq == 0

        self.repeat_explore = self.repeat_explore_mechanism.forward(
            all_memory=all_memory, last_memory=last_memory
        )
        # batch_size * 2
        repeat_recommendation_decoder = self.repeat_recommendation_decoder.forward(
            all_memory=all_memory,
            last_memory=last_memory,
            item_seq=item_seq,
            mask=timeline_mask,
        )
        # batch_size * num_item
        explore_recommendation_decoder = self.explore_recommendation_decoder.forward(
            all_memory=all_memory,
            last_memory=last_memory,
            item_seq=item_seq,
            mask=timeline_mask,
        )
        # batch_size * num_item
        prediction = repeat_recommendation_decoder * self.repeat_explore[
            :, 0
        ].unsqueeze(1) + explore_recommendation_decoder * self.repeat_explore[
            :, 1
        ].unsqueeze(
            1
        )
        # batch_size * num_item

        return prediction


    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)



class Repeat_Explore_Mechanism(nn.Module):
    def __init__(self, hidden_size, seq_len, dropout_prob):
        super(Repeat_Explore_Mechanism, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.Wre = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ure = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vre = nn.Linear(hidden_size, 1, bias=False)
        self.Wcre = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, all_memory, last_memory):
        """
        calculate the probability of Repeat and explore
        """
        all_memory_values = all_memory

        all_memory = self.dropout(self.Ure(all_memory))

        last_memory = self.dropout(self.Wre(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_ere = self.tanh(all_memory + last_memory)

        output_ere = self.Vre(output_ere)
        alpha_are = nn.Softmax(dim=1)(output_ere)
        alpha_are = alpha_are.repeat(1, 1, self.hidden_size)
        output_cre = alpha_are * all_memory_values
        output_cre = output_cre.sum(dim=1)

        output_cre = self.Wcre(output_cre)

        repeat_explore_mechanism = nn.Softmax(dim=-1)(output_cre)

        return repeat_explore_mechanism


class Repeat_Recommendation_Decoder(nn.Module):
    def __init__(self, hidden_size, seq_len, num_item, dropout_prob):
        super(Repeat_Recommendation_Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_item = num_item
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vr = nn.Linear(hidden_size, 1)

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        """
        calculate the force of repeat
        """
        all_memory = self.dropout(self.Ur(all_memory))

        last_memory = self.dropout(self.Wr(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_er = self.tanh(last_memory + all_memory)

        output_er = self.Vr(output_er).squeeze(2)

        if mask is not None:
            output_er.masked_fill_(mask, -1e9)

        output_er = nn.Softmax(dim=-1)(output_er)
        output_er = output_er.unsqueeze(1)

        map_matrix = Explore_Recommendation_Decoder.build_map(item_seq, max_index=self.num_item)
        output_er = torch.matmul(output_er, map_matrix).squeeze(1)
        repeat_recommendation_decoder = output_er.squeeze(1)

        return repeat_recommendation_decoder


class Explore_Recommendation_Decoder(nn.Module):
    def __init__(self, hidden_size, seq_len, num_item, dropout_prob):
        super(Explore_Recommendation_Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_item = num_item
        self.We = nn.Linear(hidden_size, hidden_size)
        self.Ue = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.Ve = nn.Linear(hidden_size, 1)
        self.matrix_for_explore = nn.Linear(
            2 * self.hidden_size, self.num_item, bias=False
        )

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        """
        calculate the force of explore
        """
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.Ue(all_memory))

        last_memory = self.dropout(self.We(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_ee = self.tanh(all_memory + last_memory)
        output_ee = self.Ve(output_ee).squeeze(-1)

        if mask is not None:
            output_ee.masked_fill_(mask, -1e9)

        output_ee = output_ee.unsqueeze(-1)

        alpha_e = nn.Softmax(dim=1)(output_ee)
        alpha_e = alpha_e.repeat(1, 1, self.hidden_size)
        output_e = (alpha_e * all_memory_values).sum(dim=1)
        output_e = torch.cat([output_e, last_memory_values], dim=1)
        output_e = self.dropout(self.matrix_for_explore(output_e))

        map_matrix = Explore_Recommendation_Decoder.build_map(item_seq, max_index=self.num_item)
        explore_mask = torch.bmm(
            (item_seq > 0).float().unsqueeze(1), map_matrix
        ).squeeze(1)
        output_e = output_e.masked_fill(explore_mask.bool(), float("-inf"))
        explore_recommendation_decoder = nn.Softmax(1)(output_e)

        return explore_recommendation_decoder

    @staticmethod
    def build_map(b_map, max_index=None):
        """
        project the b_map to the place where it in should be like this:
            item_seq A: [3,4,5]   n_items: 6

            after map: A

            [0,0,1,0,0,0]

            [0,0,0,1,0,0]

            [0,0,0,0,1,0]

            batch_size * seq_len ==>> batch_size * seq_len * n_item

        use in RepeatNet:

        [3,4,5] matmul [0,0,1,0,0,0]

                       [0,0,0,1,0,0]

                       [0,0,0,0,1,0]

        ==>>> [0,0,3,4,5,0] it works in the RepeatNet when project the seq item into all items

        batch_size * 1 * seq_len matmul batch_size * seq_len * n_item ==>> batch_size * 1 * n_item
        """
        batch_size, b_len = b_map.size()
        if max_index is None:
            max_index = b_map.max() + 1
        if torch.cuda.is_available():
            b_map_ = torch.FloatTensor(batch_size, b_len, max_index).fill_(0)
        else:
            b_map_ = torch.zeros(batch_size, b_len, max_index)
        b_map_.scatter_(2, b_map.unsqueeze(2), 1.0)
        b_map_.requires_grad = False
        return b_map_
