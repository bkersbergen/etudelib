from torch.utils.data import Dataset
import numpy as np
import scipy.stats as st
import torch


class SyntheticDataset(Dataset):
    def __init__(self, qty_interactions, qty_sessions, n_items, max_seq_length):
        self.qty_interactions = qty_interactions
        self.qty_sessions = qty_sessions
        self.n_items = n_items
        self.max_seq_length = max_seq_length

        self.rng = np.random.default_rng()

        distribution_name = 'powerlaw'
        distribution = getattr(st, distribution_name)
        session_fit_params = (0.07004263982467046, 1.9999999999999998, 277.00000000000006)
        session_f_new = distribution.rvs(*session_fit_params, size=qty_sessions)
        session_f_new = np.ceil(session_f_new).astype(int)
        self.session_p_new = session_f_new / np.sum(session_f_new)

        item_fit_params = (0.040690006010909816, 0.9999999999999999, 2335.0000000000005)
        item_f_new = distribution.rvs(*item_fit_params, size=n_items)
        item_f_new = np.ceil(item_f_new).astype(int)
        self.item_p_new = item_f_new / np.sum(item_f_new)

    def __len__(self):
        return self.qty_interactions

    # This returns given an index the i-th sample and label
    def __getitem__(self, _idx):
        session_length = self.rng.choice(len(self.session_p_new),
                                         size=1,
                                         replace=True,
                                         p=self.session_p_new)

        item_seq = self.rng.choice(len(self.item_p_new),
                                   size=session_length + 1,
                                   replace=True,
                                   p=self.item_p_new)

        next_item = item_seq[-1]  # next item is the last item generated list
        evolving_items = np.array(item_seq[:-1])
        evolving_items.resize(self.max_seq_length, refcheck=False)
        return torch.tensor(evolving_items), torch.tensor(session_length), torch.tensor(next_item)
