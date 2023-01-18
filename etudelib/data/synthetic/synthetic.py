from torch.utils.data import Dataset
import numpy as np
import torch


class SyntheticDataset(Dataset):
    def __init__(self, qty_interactions, qty_sessions, n_items, max_seq_length):
        self.qty_interactions = qty_interactions
        self.qty_sessions = qty_sessions
        self.n_items = n_items
        self.max_seq_length = max_seq_length

    def __len__(self):
        return self.qty_interactions

    # This returns given an index the i-th sample and label
    def __getitem__(self, _idx):
        session_length = np.random.randint(1, self.max_seq_length + 1)
        item_seq = np.random.randint(1, self.n_items, session_length)
        item_seq.resize(self.max_seq_length, refcheck=False)
        # return (torch.as_tensor(item_seq), torch.as_tensor(session_length)), torch.as_tensor(43)
        next_item = self.n_items - 1  # the id of the last item in the catalog
        return torch.tensor(item_seq), torch.tensor(session_length), torch.tensor(next_item)
