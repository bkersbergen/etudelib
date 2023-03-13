from datetime import datetime
import logging
import warnings
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
from importlib import import_module

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.deploy.export import save_eager_model, load_eager_model, export, ExportMode
from etudelib.models.topkdecorator import TopKDecorator
from etudelib.tools.benchmarker.microbenchmarker import MicroBenchmark
from etudelib.utils.loggers import configure_logger

import onnxruntime as ort

logger = logging.getLogger(__name__)


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


def microbenchmark():
    """Microbenchmarks a session based recommendation based on a provided configuration file."""
    args = get_args()
    basedir = "../.."
    projectdir = Path(basedir, 'project/benchmark')
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config_path = os.path.join(basedir, f"etudelib/models/{args.model}/config.yaml".lower())
    config = OmegaConf.load(config_path)

    if config.get('project', {}).get("seed") is not None:
        seed_everything(config.project.seed)

    qty_interactions = 1000
    n_items = 50000
    max_seq_length = 43
    qty_sessions = qty_interactions
    batch_size = 32
    device_type = 'cuda'

    config['dataset'] = {}
    config['dataset']['n_items'] = n_items
    config['dataset']['max_seq_length'] = max_seq_length

    logger.info(config)

    train_ds = SyntheticDataset(qty_interactions=qty_interactions,
                                qty_sessions=qty_sessions,
                                n_items=n_items,
                                max_seq_length=max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    module = import_module(f"etudelib.models.{config.model.name}.lightning_model".lower())
    model = getattr(module, f"{config.model.name}Lightning")(config)

    trainer = Trainer(
        accelerator='gpu' if device_type == 'cuda' else 'cpu',
        devices=1,
        max_epochs=1,
        callbacks=[TQDMProgressBar(refresh_rate=5)],
    )

    trainer.fit(model, train_loader)

    eager_model = model.get_backbone()

    eager_model = TopKDecorator(eager_model, topk=21)
    eager_model.eval()
    params = []
    for name, parameter in eager_model.named_parameters():
        if parameter.requires_grad:
            # param = parameter.numel()
            params.append([name, parameter.shape])
    for idx, layer in enumerate(params):
        print(idx, layer)
    # eager_path = save_eager_model(eager_model.to('cpu'), Path(projectdir))

    # eager_model = load_eager_model(eager_path, device='cpu')
    # print(eager_model)

    benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    item_seq, session_length, next_item = next(iter(benchmark_loader))

    import torch
    logger.info('eager mode {}'.format(config.model.name))

    bench = MicroBenchmark()
    eager_cpu_results = bench.benchmark_pytorch_predictions(eager_model, benchmark_loader, 'cpu')
    results = {'modelname': config.model.name,
               'runtime': 'eager_model_cpu',
               'latency_df': eager_cpu_results,
               }
    bench.write_results(results, projectdir / 'results')

    logger.info('JIT freeze mode {}'.format(config.model.name))
    model_input = (item_seq, session_length)
    jit_cpu_model = torch.jit.freeze(torch.jit.trace(eager_model, model_input))
    jit_cpu_results = bench.benchmark_pytorch_predictions(jit_cpu_model, benchmark_loader, 'cpu')
    results = {'modelname': config.model.name,
               'runtime': 'jit_model_cpu',
               'latency_df': jit_cpu_results,
               }
    bench.write_results(results, projectdir / 'results')

    logger.info('JIT optimize mode {}'.format(config.model.name))
    jitopt_model = torch.jit.optimize_for_inference(torch.jit.trace(eager_model, model_input))
    jitopt_cpu_results = bench.benchmark_pytorch_predictions(jitopt_model, benchmark_loader, 'cpu')
    results = {'modelname': config.model.name,
               'runtime': 'jitopt_model_cpu',
               'latency_df': jitopt_cpu_results,
               }
    bench.write_results(results, projectdir / 'results')

    logger.info('ONNX mode {}'.format(config.model.name))
    onnx_path = export(eager_model, model_input, ExportMode.ONNX, projectdir)
    providers = ['CPUExecutionProvider']
    ort_sess = ort.InferenceSession(onnx_path, providers=providers)
    onnx_cpu_results = bench.benchmark_onnxed_predictions(ort_sess, benchmark_loader)
    results = {'modelname': config.model.name,
               'runtime': 'onnx_model_cpu',
               'latency_df': onnx_cpu_results,
               }
    bench.write_results(results, projectdir / 'results')


if __name__ == "__main__":
    microbenchmark()
