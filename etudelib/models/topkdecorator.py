import torch
from torch import nn
from etudelib.models.faiss_index import faiss_index
import torch.nn.functional as F

class TopKDecorator(nn.Module):
    """
    This PyTorch model decorates an existing Model enabling the model to be converted to JIT or ONNX with torch.topk support
    """

    def __init__(self, recommender_model, topk: int = 21, topk_method: str = "torch_exact", faiss_index_dir: str = ""):
        super(TopKDecorator, self).__init__()
        self.recommender_model = recommender_model
        self.topk: int = topk
        self.topk_method = topk_method
        if not hasattr(recommender_model, 'item_embedding'):
            print("No item embedding found. Switched to 'topk' mode for topk")
            self.topk_method = "topk"
        if self.topk_method == "faiss":
            embedding_matrix = recommender_model.item_embedding.weight.detach()
            self.index = faiss_index(embedding_matrix, faiss_index_dir)

    def forward(self, item_seq, item_seq_len):
        query = self.recommender_model.forward(item_seq, item_seq_len)
        if self.topk_method == "mm+topk":
            if self.recommender_model.__class__.__name__ == 'COREModel':
                index = F.normalize(self.recommender_model.item_embedding.weight, dim=-1).transpose(0, 1)
            else:
                index = self.recommender_model.item_embedding.weight.transpose(0, 1)
            dot_product = torch.matmul(query, index)
            result = torch.topk(dot_product, self.topk)
        elif self.topk_method == "mm":
            index = self.recommender_model.item_embedding.weight.transpose(0, 1)
            result = torch.matmul(query, index)
        elif self.topk_method == "topk":
            result = torch.topk(query, self.topk)
        elif self.topk_method == "faiss":
            result = self.index.search(query, self.topk)
        
        return result
