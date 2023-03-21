import torch

from etudelib.data.synthetic.synthetic import SyntheticDataset


class TestDataModule:
    def test_happyflow_bolcom(self):
        max_seq_length = 20
        qty_interactions = 100
        n_items=10_000_000
        dataset_ut = SyntheticDataset(qty_interactions=qty_interactions, qty_sessions=qty_interactions, n_items=n_items, max_seq_length=max_seq_length, param_source='bolcom')
        for idx in range(qty_interactions):
            item_seq, session_length, next_item = dataset_ut.__getitem__(idx)
            assert type(item_seq) == torch.Tensor
            assert type(session_length) == torch.Tensor
            assert type(next_item) == torch.Tensor
            assert item_seq.shape == torch.Size([max_seq_length])




    def test_happyflow_rsc15(self):
        max_seq_length = 20
        qty_interactions = 100
        n_items = 10_000_000
        dataset_ut = SyntheticDataset(qty_interactions=qty_interactions, qty_sessions=qty_interactions, n_items=n_items, max_seq_length=max_seq_length, param_source='rsc15')
        for idx in range(qty_interactions):
            item_seq, session_length, next_item = dataset_ut.__getitem__(idx)
            assert type(item_seq) == torch.Tensor
            assert type(session_length) == torch.Tensor
            assert type(next_item) == torch.Tensor
            assert item_seq.shape == torch.Size([max_seq_length])

