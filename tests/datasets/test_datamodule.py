import torch

from etudelib.data.synthetic.synthetic import SyntheticDataset


class TestDataModule:
    def test_synthetic(self):
        max_seq_length = 3
        dataset_ut = SyntheticDataset(qty_interactions=10, qty_sessions=5, n_items=8, max_seq_length=max_seq_length)
        random_idx = 6
        item_seq, session_length, next_item = dataset_ut.__getitem__(random_idx)
        assert type(item_seq) == torch.Tensor
        assert type(session_length) == torch.Tensor
        assert type(next_item) == torch.Tensor
        assert item_seq.shape == torch.Size([max_seq_length])


