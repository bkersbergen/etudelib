import math
import os
from importlib import import_module
from pathlib import Path

import torch
from omegaconf import OmegaConf
from etudelib.models.topkdecorator import TopKDecorator


class ModelUtil:

    @staticmethod
    def create_model(model_name: str, C: int, max_seq_length:int, param_source: str, model_input):
        device_type = 'cpu'
        rootdir = Path(__file__).parent.parent.parent

        projectdir = Path(rootdir, 'projects/benchmark')

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

        payload_path = str(projectdir / f'{base_filename}_payload.torch')
        eager_model_path = str(projectdir / f'{base_filename}_eager.pth')
        jitopt_model_path = str(projectdir / f'{base_filename}_jitopt.pth')
        onnx_model_path = str(projectdir / f'{base_filename}_onnx.pth')

        eager_model.to(device_type)

        jit_model = torch.jit.optimize_for_inference(
            torch.jit.trace(eager_model, (model_input[0].to(device_type), model_input[1].to(device_type))))

        torch.save(payload, payload_path)
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
