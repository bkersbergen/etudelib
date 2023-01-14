from torch.utils.data import Dataset
import numpy as np
import torch


class SyntheticDataset(Dataset):
    def __init__(self, qty_interactions, qty_sessions, qty_catalog_items, max_seq_length):
        self.qty_interactions = qty_interactions
        self.qty_sessions = qty_sessions
        self.qty_catalog_items = qty_catalog_items
        self.max_seq_length = max_seq_length

    def __len__(self):
        return self.qty_interactions

    # This returns given an index the i-th sample and label
    def __getitem__(self, _idx):
        session_length = np.random.randint(1, self.max_seq_length + 1)
        item_seq = np.random.randint(0, self.qty_catalog_items + 1, session_length)
        item_seq.resize(self.max_seq_length, refcheck=False)
        # return (torch.as_tensor(item_seq), torch.as_tensor(session_length)), torch.as_tensor(43)
        next_item = self.qty_catalog_items
        return torch.tensor(item_seq), torch.tensor(session_length), torch.tensor(next_item)
