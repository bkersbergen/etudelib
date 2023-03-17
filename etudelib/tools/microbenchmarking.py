import logging
import warnings
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from omegaconf import OmegaConf
from importlib import import_module

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.deploy.export import *
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
    parser.add_argument("--model", type=str, default="sine", help="Name of the model. E.g. core or gcsan etc")
    parser.add_argument("--qty_interactions", type=int, default=1000,
                        help="Sythetic dataset: Number of user-item interactions to generate.")
    parser.add_argument("--C", type=int, default=50_000,
                        help="Sythetic dataset: Number of distinct items in catalog to generate.")
    parser.add_argument("--t", type=int, default=50,
                        help="Sythetic dataset: Number of timesteps or sequence length of a session as input for a model")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    args = parser.parse_args()
    return args


def microbenchmark(args):
    """Microbenchmarks a session based recommendation based on a provided configuration file."""
    basedir = "../.."
    projectdir = Path(basedir, 'project/benchmark')
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config_path = os.path.join(basedir, f"etudelib/models/{args.model}/config.yaml".lower())
    config = OmegaConf.load(config_path)

    if config.get('project', {}).get("seed") is not None:
        seed_everything(config.project.seed)

    qty_sessions = args.qty_interactions
    batch_size = 32
    device_type = 'cuda'

    config['dataset'] = {}
    config['dataset']['n_items'] = args.C
    config['dataset']['max_seq_length'] = args.t

    logger.info(config)

    train_ds = SyntheticDataset(qty_interactions=args.qty_interactions,
                                qty_sessions=qty_sessions,
                                n_items=args.C,
                                max_seq_length=args.t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    module = import_module(f"etudelib.models.{config.model.name}.lightning_model".lower())
    model = getattr(module, f"{config.model.name}Lightning")(config)

    trainer = Trainer(
        accelerator='gpu' if device_type == 'cuda' else 'cpu',
        devices=1,
        max_epochs=1,
        callbacks=[TQDMProgressBar(refresh_rate=5)],
    )

    # trainer.fit(model, train_loader)

    eager_model = model.get_backbone()

    eager_model = TopKDecorator(eager_model, topk=21)
    eager_model.eval()
    params = []
    for name, parameter in eager_model.named_parameters():
        if parameter.requires_grad:
            # param = parameter.numel()
            params.append([name, parameter.shape])
    for idx, layer in enumerate(params):
        logger.info(f"{idx} {layer}")

    benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    item_seq, session_length, next_item = next(iter(benchmark_loader))
    model_input = (item_seq, session_length)
    eager_model_path = save_eager_model(eager_model, projectdir)
    eager_model = load_eager_model(eager_model_path, device='cpu')

    bench = MicroBenchmark()
    logger.info('-----------------------------------------------------------------------------------------------')
    logger.info(f'BENCHMARK {config.model.name} ON CPU: ${bench.cpu_brand}')

    logger.info(f'{config.model.name} C:{args.C} t:{args.t} eager cpu')
    eager_cpu_results = bench.benchmark_pytorch_predictions(eager_model, benchmark_loader)
    results = {'modelname': config.model.name,
               'runtime': 'eager_model_cpu',
               'latency_df': eager_cpu_results,
               'C': args.C,
               't': args.t,
               }
    bench.write_results(results, projectdir / 'results')

    logger.info(f"{config.model.name} C:{args.C} t:{args.t} jit cpu")
    eager_model = load_eager_model(eager_model_path, device='cpu')
    jit_cpu_model = torch.jit.freeze(torch.jit.trace(eager_model, model_input))
    jit_model_path = save_jit_model(jit_cpu_model, projectdir)
    jit_cpu_model = load_jit_model(jit_model_path, 'cpu')
    jit_cpu_results = bench.benchmark_pytorch_predictions(jit_cpu_model, benchmark_loader)
    results = {'modelname': config.model.name,
               'runtime': 'jit_model_cpu',
               'latency_df': jit_cpu_results,
               'C': args.C,
               't': args.t,
               }
    bench.write_results(results, projectdir / 'results')

    logger.info(f"{config.model.name} C:{args.C} t:{args.t} jitopt cpu")
    eager_model = load_eager_model(eager_model_path, device='cpu')
    jitopt_model = torch.jit.optimize_for_inference(torch.jit.trace(eager_model, model_input))
    jit_model_path = save_jit_model(jitopt_model, projectdir)
    jit_cpu_model = load_jit_model(jit_model_path, 'cpu')
    jitopt_cpu_results = bench.benchmark_pytorch_predictions(jit_cpu_model, benchmark_loader)
    results = {'modelname': config.model.name,
               'runtime': 'jitopt_model_cpu',
               'latency_df': jitopt_cpu_results,
               'C': args.C,
               't': args.t,
               }
    bench.write_results(results, projectdir / 'results')

    logger.info(f"{config.model.name} C:{args.C} t:{args.t} onnx cpu")
    onnx_model_path = save_onnx_model(eager_model, projectdir, model_input)
    ort_sess = load_onnx_session(onnx_model_path, 'cpu')
    onnx_cpu_results = bench.benchmark_onnxed_predictions(ort_sess, benchmark_loader)
    results = {'modelname': config.model.name,
               'runtime': 'onnx_model_cpu',
               'latency_df': onnx_cpu_results,
               'C': args.C,
               't': args.t,
               }
    bench.write_results(results, projectdir / 'results')

    if torch.cuda.is_available():
        logger.info('-----------------------------------------------------------------------------------------------')
        logger.info('BENCHMARK '+config.model.name+' ON GPU: ' + bench.gpu_brand)
        logger.info(f"{config.model.name} C:{args.C} t:{args.t} eager cuda")
        eager_gpu_model = load_eager_model(eager_model_path, device='cuda')
        eager_gpu_results = bench.benchmark_pytorch_predictions(eager_gpu_model, benchmark_loader, device='cuda')
        results = {'modelname': config.model.name,
                   'runtime': 'eager_model_cuda',
                   'latency_df': eager_gpu_results,
                   'C': args.C,
                   't': args.t,
                   }
        bench.write_results(results, projectdir / 'results')
        logger.info('Removing model from GPU')
        eager_gpu_model.to('cpu')
        torch.cuda.empty_cache()
        print(f'CUDA memory used: {int(torch.cuda.memory_allocated(0)/1000)} MB')

        logger.info(f"{config.model.name} C:{args.C} t:{args.t} jit cuda")
        eager_gpu_model = load_eager_model(eager_model_path, device='cuda')
        jit_gpu_model = torch.jit.freeze(torch.jit.trace(eager_gpu_model, (model_input[0].to('cuda'), model_input[1].to('cuda'))))
        eager_gpu_model.to('cpu')
        jit_model_path = save_jit_model(jit_gpu_model, projectdir)
        jit_gpu_model.to('cpu')
        jit_gpu_model = load_jit_model(jit_model_path, device='cuda')
        jitopt_cuda_results = bench.benchmark_pytorch_predictions(jit_gpu_model, benchmark_loader, 'cuda')
        results = {'modelname': config.model.name,
                   'runtime': 'jitopt_model_cuda',
                   'latency_df': jitopt_cuda_results,
                   'C': args.C,
                   't': args.t,
                   }
        bench.write_results(results, projectdir / 'results')
        jit_gpu_model.to('cpu')
        torch.cuda.empty_cache()
        print(f'CUDA memory used: {int(torch.cuda.memory_allocated(0)/1000)} MB')

        logger.info(f"{config.model.name} C:{args.C} t:{args.t} jitopt cuda")
        eager_gpu_model = load_eager_model(eager_model_path, device='cuda')
        jitopt_gpu_model = torch.jit.optimize_for_inference(torch.jit.trace(eager_gpu_model, (model_input[0].to('cuda'), model_input[1].to('cuda'))))
        eager_gpu_model.to('cpu')
        jit_model_path = save_jit_model(jitopt_gpu_model, projectdir)
        jitopt_gpu_model.to('cpu')
        jitopt_gpu_model = load_jit_model(jit_model_path, device='cuda')
        jitopt_cuda_results = bench.benchmark_pytorch_predictions(jitopt_gpu_model, benchmark_loader, 'cuda')
        results = {'modelname': config.model.name,
                   'runtime': 'jitopt_model_cuda',
                   'latency_df': jitopt_cuda_results,
                   'C': args.C,
                   't': args.t,
                   }
        bench.write_results(results, projectdir / 'results')
        jitopt_gpu_model.to('cpu')
        print(f'CUDA memory used: {int(torch.cuda.memory_allocated(0)/1000)} MB')

        logger.info(f"{config.model.name} C:{args.C} t:{args.t} onnx cuda")
        eager_gpu_model = load_eager_model(eager_model_path, device='cuda')
        onnx_model_path = save_onnx_model(eager_gpu_model, projectdir, (model_input[0].to('cuda'), model_input[1].to('cuda')))
        eager_gpu_model.to('cpu')
        ort_sess = load_onnx_session(onnx_model_path, 'cuda')
        onnx_cuda_results = bench.benchmark_onnxed_predictions(ort_sess, benchmark_loader)
        results = {'modelname': config.model.name,
                   'runtime': 'onnx_model_cuda',
                   'latency_df': onnx_cuda_results,
                   'C': args.C,
                   't': args.t,
                   }
        ort_sess.set_providers(['CPUExecutionProvider'])
        torch.cuda.empty_cache()
        bench.write_results(results, projectdir / 'results')
        print(f'CUDA memory used: {int(torch.cuda.memory_allocated(0)/1000)} MB')




if __name__ == "__main__":
    args = get_args()
    args.qty_interactions = 50_000
    for model_name in ['core', 'gcsan', 'gru4rec', 'lightsans', 'narm', 'repeatnet', 'sasrec', 'sine', 'srgnn',
                       'stamp']:
        for C in [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]:
            args.C = C
            args.model = model_name
            for t in [50]:
                args.t = t
                microbenchmark(args)
