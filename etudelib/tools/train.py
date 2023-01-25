import logging
import warnings
import os
from argparse import ArgumentParser, Namespace

from omegaconf import OmegaConf, DictConfig, ListConfig
from importlib import import_module

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.utils.loggers import configure_logger

logger = logging.getLogger("etudelib")


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="sine", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    args = parser.parse_args()
    return args


def train():
    """Train an session based recommendation based on a provided configuration file."""
    args = get_args()
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config_path = os.path.join("../..", f"etudelib/models/{args.model}/config.yaml".lower())
    config = OmegaConf.load(config_path)

    if config.get('project', {}).get("seed") is not None:
        seed_everything(config.project.seed)

    qty_interactions = 10000
    n_items = 5000
    max_seq_length = 43
    qty_sessions = qty_interactions
    batch_size = 32

    config['data'] = {}
    config['data']['n_items'] = n_items
    config['data']['max_seq_length'] = max_seq_length

    logger.info(config)

    train_ds = SyntheticDataset(qty_interactions=qty_interactions,
                                qty_sessions=qty_sessions,
                                n_items=n_items,
                                max_seq_length=max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_ds = SyntheticDataset(qty_interactions=qty_interactions,
                              qty_sessions=qty_sessions,
                              n_items=n_items,
                              max_seq_length=max_seq_length)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    test_ds = SyntheticDataset(qty_interactions=qty_interactions,
                               qty_sessions=qty_sessions,
                               n_items=n_items,
                               max_seq_length=max_seq_length)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    module = import_module(f"etudelib.models.{config.model.name}.lightning_model".lower())
    model = getattr(module, f"{config.model.name}Lightning")(config)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=5)],
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)
    trainer.save_checkpoint("../../results/lightsans.chkpoint")


if __name__ == "__main__":
    train()
