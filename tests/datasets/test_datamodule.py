from etudelib.data.synthetic.synthetic import SyntheticDataset


class TestDataModule:
    def test_synthetic(self):
        dataset_ut = SyntheticDataset(qty_interactions=10, qty_sessions=5, qty_catalog_items=8, max_seq_length=3)
        result = dataset_ut.__getitem__(6)
        print(result)
