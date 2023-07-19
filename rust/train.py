import math
import os
from importlib import import_module

import torch
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.models.topkdecorator import TopKDecorator

PROJECT_ID="bk47471"

def export_models():
    rootdir = Path(__file__).parent.parent.parent
    BUCKET_BASE_URI='gs://'+PROJECT_ID+'-shared/model_store'
    # for C in [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000]:
    device_types = ['cpu']
    if torch.cuda.is_available():
        device_types.append('cuda')

    for C in [10_000]:
        max_seq_length = 50
        param_source = 'bolcom'
        # initializing the synthetic dataset takes very long for a large C value.
        train_ds = SyntheticDataset(qty_interactions=50_000,
                                    qty_sessions=50_000,
                                    n_items=C,
                                    max_seq_length=max_seq_length, param_source=param_source)
        benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        item_seq, session_length, next_item = next(iter(benchmark_loader))
        model_input = (item_seq, session_length)
        print(model_input)
        # for model_name in ['core', 'gcsan', 'gru4rec', 'lightsans', 'narm', 'noop', 'repeatnet', 'sasrec', 'sine', 'srgnn',
        #            'stamp']:
        # for model_name in ['noop']:


        for model_name in ['gru4rec']:
            print(f'export model: model_name={model_name}, C={C}, max_seq_length={max_seq_length}, param_source={param_source}')
            eager_model, payload = train_model(
                model_name=model_name, C=C, max_seq_length=max_seq_length, param_source=param_source, model_input=model_input)

            # export for both CPU and GPU, because JIT trace hardcodes the device_type in the exported model
            for device_type in device_types:
                # Move model and tensors to the device_type
                eager_model = eager_model.to(device_type)
                model_input = (model_input[0].to(device_type), model_input[1].to(device_type))

                _recommendations = eager_model.forward(*model_input)

                jit_model = torch.jit.optimize_for_inference(torch.jit.trace(eager_model, model_input))

                base_filename = f'{model_name}_{param_source}_c{C}_t{max_seq_length}_{device_type}'

                projectdir = Path(rootdir, 'rust/model_store', base_filename)
                print(projectdir)
                projectdir.mkdir(parents=True, exist_ok=True)

                payload_path = str(projectdir / f'{base_filename}_payload.yaml')
                eager_model_path = str(projectdir / f'{base_filename}_eager.pth')
                jitopt_model_path = str(projectdir / f'{base_filename}_jitopt.pth')
                onnx_model_path = str(projectdir / f'{base_filename}_onnx.pth')

                conf = OmegaConf.create(payload)
                with open(payload_path, 'w+') as fp:
                    OmegaConf.save(config=conf, f=fp)

                torch.save(eager_model, eager_model_path)
                torch.jit.save(jit_model, jitopt_model_path)  # save jitopt model
                torch.onnx.export(
                    eager_model,
                    model_input,
                    onnx_model_path,
                    input_names=['item_id_list', 'max_seq_length'],  # the model's input names
                    output_names=['output'],  # the model's output names
                )

                destination = f'{BUCKET_BASE_URI}/{base_filename}'
                os.system(f'gsutil cp -r {payload_path} {destination}/')
                os.system(f'gsutil cp -r {eager_model_path} {destination}/')
                os.system(f'gsutil cp -r {jitopt_model_path} {destination}/')
                os.system(f'gsutil cp -r {onnx_model_path} {destination}/')


def train_model(model_name: str, C: int, max_seq_length:int, param_source: str, model_input):
    rootdir = Path(__file__).parent.parent

    base_filename = f'{model_name}_{param_source}_c{C}_t{max_seq_length}'

    projectdir = Path(rootdir, 'rust/model_store', base_filename)
    print(projectdir)
    projectdir.mkdir(parents=True, exist_ok=True)
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

    payload = {'max_seq_length': max_seq_length,
               'C': C,
               'idx2item': [i for i in range(C)],
               }

    # payload_path = str(projectdir / f'{base_filename}_payload.yaml')
    # eager_model_path = str(projectdir / f'{base_filename}_eager.pth')
    # jitopt_model_path = str(projectdir / f'{base_filename}_jitopt.pth')
    # onnx_model_path = str(projectdir / f'{base_filename}_onnx.pth')
    #
    # device_type = 'cpu'
    # # eager_model.to(device_type)
    #
    # # jit_model = torch.jit.optimize_for_inference(
    # #     torch.jit.trace(eager_model, (model_input[0].to(device_type), model_input[1].to(device_type))))
    # jit_model = torch.jit.optimize_for_inference(
    #     torch.jit.trace(eager_model, (model_input[0], model_input[1])))
    #
    # conf = OmegaConf.create(payload)
    # with open(payload_path, 'w+') as fp:
    #     OmegaConf.save(config=conf, f=fp)
    #
    # torch.save(eager_model, eager_model_path)
    # torch.jit.save(jit_model, jitopt_model_path)  # save jitopt model
    # torch.onnx.export(
    #     eager_model,
    #     (model_input[0].to(device_type), model_input[1].to(device_type)),
    #     onnx_model_path,
    #     input_names=['item_id_list', 'max_seq_length'],  # the model's input names
    #     output_names=['output'],  # the model's output names
    # )
    return eager_model, payload
    # return payload_path, eager_model_path, jitopt_model_path, onnx_model_path


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


