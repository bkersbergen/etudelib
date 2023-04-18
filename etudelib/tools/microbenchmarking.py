import warnings
import os
from argparse import ArgumentParser, Namespace
from datetime import date

from omegaconf import OmegaConf
from importlib import import_module

from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from etudelib.data.googlefs.googlefs import upload_to_gcs
from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.deploy.export import *
from etudelib.models.topkdecorator import TopKDecorator
from etudelib.tools.benchmarker.microbenchmarker import MicroBenchmark
from etudelib.utils.loggers import configure_logger
from multiprocessing import Process

import onnxruntime as ort

logger = logging.getLogger(__name__)


def run_benchmark_process(eager_model, new_model_mode, benchmark_loader, device_type, results, projectdir):
    Path(projectdir).mkdir(parents=True, exist_ok=True)
    min_duration_secs = 10
    bench = MicroBenchmark(min_duration_secs=min_duration_secs)
    print('-----------------------------------------------------------------------------------------------')
    print(f'BENCHMARK {results["modelname"]} IN {new_model_mode} MODE ON DEVICE: {device_type} {results["param_source"]}')
    cpu_utilization, used_mem, total_mem = MicroBenchmark.get_metrics_cpu()
    print(f'CPU utilization : {cpu_utilization} %')
    print(f'used_mem: {used_mem} MB')
    print(f'total_mem: {total_mem} MB')
    if device_type != 'cpu':
        gpu_utilization, gpu_mem_used, gpu_memory_total = MicroBenchmark.get_metrics_gpu()
        logger.info(f'CUDA utilization: {gpu_utilization} %')
        logger.info(f'CUDA mem_used: {gpu_mem_used} MB')
        logger.info(f'CUDA memory_total: {gpu_memory_total} MB')

    item_seq, session_length, next_item = next(iter(benchmark_loader))
    model_input = (item_seq, session_length)

    eager_model.to(device_type)
    if new_model_mode == 'eager':
        model = eager_model
        latency_results = bench.benchmark_pytorch_predictions(model, benchmark_loader, device_type)
    elif new_model_mode == 'jit':
        model = torch.jit.freeze(
            torch.jit.trace(eager_model, (model_input[0].to(device_type), model_input[1].to(device_type))))
        latency_results = bench.benchmark_pytorch_predictions(model, benchmark_loader, device_type)
    elif new_model_mode == 'jitopt':
        model = torch.jit.optimize_for_inference(
            torch.jit.trace(eager_model, (model_input[0].to(device_type), model_input[1].to(device_type))))
        latency_results = bench.benchmark_pytorch_predictions(model, benchmark_loader, device_type)
    elif new_model_mode == 'onnx':
        export_path = str(projectdir / "model.pt.onnx")
        print('export_path:' + export_path)
        torch.onnx.export(
            eager_model,
            (model_input[0].to(device_type), model_input[1].to(device_type)),
            export_path,
            input_names=['item_id_list', 'max_seq_length'],  # the model's input names
            output_names=['output'],  # the model's output names
        )
        if device_type == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
            logger.error('ONNX Runtime does not have CUDA support')
            logger.error('Please install onnxruntime-gpu version')
            exit(1)

        if device_type == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        ort_sess = ort.InferenceSession(export_path, providers=providers)
        latency_results = bench.benchmark_onnxed_predictions(ort_sess, benchmark_loader)

    results['runtime'] = '_'.join([new_model_mode, device_type])
    results['latency_df'] = latency_results
    bench.write_results(results, projectdir / 'results')


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="sine", help="Name of the model. E.g. core or gcsan etc")
    parser.add_argument("--qty_interactions", type=int, default=1000,
                        help="Synthetic dataset: Number of user-item interactions to generate.")
    parser.add_argument("--C", type=int, default=50_000,
                        help="Synthetic dataset: Number of distinct items in catalog to generate.")
    parser.add_argument("--t", type=int, default=50,
                        help="Synthetic dataset: Number of timesteps or sequence length of a session as input for a "
                             "model")
    parser.add_argument("--param_source", type=str, default="bolcom",
                        help="Synthetic dataset: using fit parameters from this datasource")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")
    parser.add_argument("--gcs_project_name", type=str, required=False,
                        help="Google Storage Project that contains the bucket. e.g. bolcom-pro-reco-analytics-fcc")
    parser.add_argument("--gcs_bucket_name", type=str, required=False,
                        help="Google Storage Bucket name that contains the directory. e.g. bolcom-pro-reco-analytics-fcc-shared")
    parser.add_argument("--gcs_dir", type=str, required=False,
                        help="Google Storage Directory where to store the results. e.g. bkersbergen_etude")

    args, unknown = parser.parse_known_args()
    return args


def microbenchmark(args):
    """Microbenchmarks a session based recommendation based on a provided configuration file."""
    rootdir = Path(__file__).parent.parent.parent

    projectdir = Path(rootdir, 'projects/microbenchmark')
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config_path = os.path.join(rootdir, f"etudelib/models/{args.model}/config.yaml".lower())
    config = OmegaConf.load(config_path)

    if config.get('project', {}).get("seed") is not None:
        seed_everything(config.project.seed)

    qty_sessions = args.qty_interactions
    batch_size = 32

    config['dataset'] = {}
    config['dataset']['n_items'] = args.C
    config['dataset']['max_seq_length'] = args.t

    logger.info(config)

    train_ds = SyntheticDataset(qty_interactions=args.qty_interactions,
                                qty_sessions=qty_sessions,
                                n_items=args.C,
                                max_seq_length=args.t, param_source=args.param_source)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    module = import_module(f"etudelib.models.{config.model.name}.lightning_model".lower())
    model = getattr(module, f"{config.model.name}Lightning")(config)

    # trainer = Trainer(
    #     accelerator='gpu' if device_type == 'cuda' else 'cpu',
    #     devices=1,
    #     max_epochs=1,
    #     callbacks=[TQDMProgressBar(refresh_rate=5)],
    # )

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
        # logger.info(f"{idx} {layer}")
        pass

    print(args)

    benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    results = {'modelname': config.model.name,
               'C': args.C,
               't': args.t,
               'param_source': args.param_source,
               'config': config,
               'model_architecture': str(eager_model)
               }

    device_types = ['cpu']
    if torch.cuda.is_available():
        device_types.append('cuda')

    for device_type in device_types:
        # Run benchmarks in separate processes to release (CUDA) memory when done
        p = Process(target=run_benchmark_process,
                    args=(eager_model, 'eager', benchmark_loader, device_type, results, projectdir,))
        p.start()
        p.join()

        # p = Process(target=run_benchmark_process,
        #             args=(eager_model, 'jit', benchmark_loader, device_type, results, projectdir,))
        # p.start()
        # p.join()

        p = Process(target=run_benchmark_process,
                    args=(eager_model, 'jitopt', benchmark_loader, device_type, results, projectdir,))
        p.start()
        p.join()

        p = Process(target=run_benchmark_process,
                    args=(eager_model, 'onnx', benchmark_loader, device_type, results, projectdir,))
        p.start()
        p.join()

    if args.gcs_project_name:
        print('Start transferring results to google storage bucket')
        upload_to_gcs(local_dir=projectdir,
                      gcs_project_name=args.gcs_project_name,
                      gcs_bucket_name=args.gcs_bucket_name,
                      gcs_dir=args.gcs_dir + '/' + str(date.today()))
        print('End transferring results to google storage bucket')


if __name__ == "__main__":
    args = get_args()
    args.qty_interactions = 50_000
    args.gcs_project_name = 'bolcom-pro-reco-analytics-fcc'
    args.gcs_bucket_name = 'bolcom-pro-reco-analytics-fcc-shared'
    args.gcs_dir = 'bkersbergen_etude'
    for model_name in ['core']:
        for C in [1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000]:
            args.C = C
            args.model = model_name
            for t in [50]:
                args.t = t
                # for param_source in ['bolcom', 'rsc15']:
                for param_source in ['bolcom']:
                    args.param_source = param_source
                    microbenchmark(args)
