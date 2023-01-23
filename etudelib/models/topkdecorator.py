import torch
from torch import nn


class TopKDecorator(nn.Module):
    """
    This PyTorch model decorates an existing Model enabling the model to be converted to JIT or ONNX with torch.topk support
    """

    def __init__(self, recommender_model, topk: int):
        super(TopKDecorator, self).__init__()
        self.recommender_model = recommender_model
        self.topk: int = topk

    def forward(self, item_seq, item_seq_len):
        scores = self.recommender_model.forward(item_seq, item_seq_len)
        return torch.topk(scores, self.topk)
