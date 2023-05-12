import math
import os
from importlib import import_module

import yaml
import torch
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.deploy.modelutil import ModelUtil

from etudelib.deploy.export import TorchServeExporter
import subprocess

from etudelib.models.topkdecorator import TopKDecorator


def export_models():
    rootdir = Path(__file__).parent.parent.parent
    # for C in [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000]:
    for C in [1_000_000]:
        t = 50
        param_source = 'bolcom'
        # initializing the synthetic dataset takes very long for a large C value.
        train_ds = SyntheticDataset(qty_interactions=50_000,
                                    qty_sessions=50_000,
                                    n_items=C,
                                    max_seq_length=t, param_source=param_source)
        benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        item_seq, session_length, next_item = next(iter(benchmark_loader))
        model_input = (item_seq, session_length)
        # for model_name in ['core', 'gcsan', 'gru4rec', 'lightsans', 'narm', 'noop', 'repeatnet', 'sasrec', 'sine', 'srgnn',
        #            'stamp']:
        # for model_name in ['noop']:
        for model_name in ['noop', 'sasrec', 'core']:
            output_path = f'{rootdir}/rust/model_store/'
            print(f'export model: model_name={model_name}, C={C}, max_seq_length={t}, param_source={param_source}')
            payload_path, eager_model_path, jitopt_model_path, onnx_model_path = create_model(
                model_name=model_name, C=C, max_seq_length=t, param_source=param_source, model_input=model_input)


            # TorchServeExporter.export_mar_file(eager_model_path, payload_path, output_path)
            # docker_build_push(eager_model_path)
            # TorchServeExporter.export_mar_file(jitopt_model_path, payload_path, output_path)
            # docker_build_push(jitopt_model_path)
            # TorchServeExporter.export_mar_file(onnx_model_path, payload_path, output_path)
            # docker_build_push(onnx_model_path)


def create_model(model_name: str, C: int, max_seq_length:int, param_source: str, model_input):
    device_type = 'cpu'
    rootdir = Path(__file__).parent.parent

    projectdir = Path(rootdir, 'rust/model_store')

    config_path = os.path.join(rootdir, f"etudelib/models/{model_name}/config.yaml".lower())
    config = OmegaConf.load(config_path)

    heuristic_embedding_size = 2 ** math.ceil(math.log2(C ** 0.25))
    if config.get('model').get('embedding_size'):
        config['model']['embedding_size'] = heuristic_embedding_size
    elif config.get('model').get('hidden_size'):
        config['model']['hidden_size'] = heuristic_embedding_size
    print(f'Overwriting item embedding output size to: {heuristic_embedding_size}')

    config['dataset'] = {}
    config['dataset']['n_items'] = C
    config['dataset']['max_seq_length'] = max_seq_length

    module = import_module(f"etudelib.models.{config.model.name}.lightning_model".lower())
    model = getattr(module, f"{config.model.name}Lightning")(config)

    eager_model = model.get_backbone()

    eager_model = TopKDecorator(eager_model, topk=21)
    eager_model.eval()

    base_filename = f'{model_name}_{param_source}_c{C}_t{max_seq_length}'

    payload = {'max_seq_length': max_seq_length,
               'C': C,
               'idx2item': [i for i in range(C)],
               }

    payload_path = str(projectdir / f'{base_filename}_payload.yaml')
    eager_model_path = str(projectdir / f'{base_filename}_eager.pth')
    jitopt_model_path = str(projectdir / f'{base_filename}_jitopt.pth')
    onnx_model_path = str(projectdir / f'{base_filename}_onnx.pth')

    eager_model.to(device_type)

    jit_model = torch.jit.optimize_for_inference(
        torch.jit.trace(eager_model, (model_input[0].to(device_type), model_input[1].to(device_type))))

    conf = OmegaConf.create(payload)
    with open(payload_path, 'w+') as fp:
        OmegaConf.save(config=conf, f=fp)

    torch.save(eager_model, eager_model_path)
    torch.jit.save(jit_model, jitopt_model_path)  # save jitopt model
    torch.onnx.export(
        eager_model,
        (model_input[0].to(device_type), model_input[1].to(device_type)),
        onnx_model_path,
        input_names=['item_id_list', 'max_seq_length'],  # the model's input names
        output_names=['output'],  # the model's output names
    )
    return payload_path, eager_model_path, jitopt_model_path, onnx_model_path


# def docker_build_push(model_path):
#     filename_with_ext = os.path.basename(model_path)  # 'myfile.zip'
#     filename_without_ext, file_ext = os.path.splitext(filename_with_ext)  # ('myfile', '.zip')
#     rootdir = Path(__file__).parent.parent.parent
#     task = ['make', 'model_build', f'MARFILE_WO_EXT={filename_without_ext}']
#     print(rootdir)
#     p = subprocess.Popen(task, cwd=rootdir)
#     p.wait()


if __name__ == '__main__':
    export_models()


