from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.models.lightsans.lightning_model import LightSANs
from etudelib.models.lightsans.torch_model import LightSANsModel


def main(hparams):
    print(hparams)
    qty_interactions = 10000
    n_items = 5000
    max_seq_length = 43
    qty_sessions = qty_interactions
    batch_size = 32

    train_ds = SyntheticDataset(qty_interactions=qty_interactions,
                                qty_sessions=qty_sessions,
                                n_items=n_items,
                                max_seq_length=max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    backbone = LightSANsModel(n_layers=2,
                              n_heads=2,
                              k_interests=15,
                              hidden_size=64,
                              inner_size=256,
                              hidden_dropout_prob=0.5,
                              attn_dropout_prob=0.5,
                              hidden_act="gelu",
                              layer_norm_eps=1e-12,
                              initializer_range=0.02,
                              max_seq_length=max_seq_length,
                              n_items=n_items,
                              topk=21,
                              )

    model = LightSANs(backbone)

    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=3,
        callbacks=[TQDMProgressBar()],
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="mps")
    parser.add_argument("--devices", default=1)
    args = parser.parse_args()

    main(args)
