import os
import sys
from importlib import import_module
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.models.topkdecorator import TopKDecorator
from etudelib.utils.loggers import configure_logger

from timeit import default_timer as timer

sys.path.insert(0, '../etudelib/deploy/inferences/')

import unittest
from etudelib.deploy.inferences.torch_inferencer import TorchInferencer


class Metrics:
    def add_time(self, name, value, idx=None, unit='ms'):
        pass


class Context:
    metrics = Metrics()


class TestingTorchInferencer(unittest.TestCase):

    def create_model(self, model_name: str):
        C = 1000000
        max_seq_length = 50
        dataset_name = 'bolcom'
        device_type = 'cpu'
        rootdir = Path(__file__).parent.parent.parent.parent

        projectdir = Path(rootdir, 'projects/benchmark')
        configure_logger(level='INFO')

        config_path = os.path.join(rootdir, f"etudelib/models/{model_name}/config.yaml".lower())
        config = OmegaConf.load(config_path)

        config['dataset'] = {}
        config['dataset']['n_items'] = C
        config['dataset']['max_seq_length'] = max_seq_length

        train_ds = SyntheticDataset(qty_interactions=50_000,
                                    qty_sessions=50_000,
                                    n_items=C,
                                    max_seq_length=max_seq_length, param_source=dataset_name)
        benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

        module = import_module(f"etudelib.models.{config.model.name}.lightning_model".lower())
        model = getattr(module, f"{config.model.name}Lightning")(config)

        eager_model = model.get_backbone()

        eager_model = TopKDecorator(eager_model, topk=21)
        eager_model.eval()

        base_filename = f'{model_name}_{dataset_name}_{C}_{max_seq_length}'

        payload = {'max_seq_length': max_seq_length,
                   'C': C,
                   'idx2item': [i for i in range(C)]
                   }

        payload_path = str(projectdir / f'{base_filename}.payload.torch')
        eager_model_path = str(projectdir / f'{base_filename}.eager.pth')
        jitopt_model_path = str(projectdir / f'{base_filename}.jitopt.pth')
        onnx_model_path = str(projectdir / f'{base_filename}.onnx.pth')

        eager_model.to(device_type)

        item_seq, session_length, next_item = next(iter(benchmark_loader))
        model_input = (item_seq, session_length)

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

    def test_model_returns_predictions(self):
        payload_path, eager_model_path, jitopt_model_path, onnx_model_path = self.create_model('core')

        sut = TorchInferencer()
        sut.initialize_from_file(onnx_model_path)
        request_data = [{'body': {'instances': [{'context': [8, 5, 12, 300]}], 'parameters': ['somestring']}}]

        result = sut.handle(request_data, context=Context())
        print(result)
        predictions = result[0]['predictions']
        self.assertTrue(len(predictions[0]['items']) > 5)

    def test_naive_inference_benchmark(self):
        payload_path, eager_model_path, jitopt_model_path, onnx_model_path = self.create_model('core')

        sut = TorchInferencer()
        sut.initialize_from_file(onnx_model_path)
        request_data = [{'body': {'instances': [{'context': [8, 5, 12, 300]}], 'parameters': ['somestring']}}]

        qty_warmup = 50
        for i in range(qty_warmup):
            result = sut.handle(request_data, context=Context())[0]
        loops = 1000
        start = timer()
        for i in range(loops):
            result = sut.handle(request_data, context=Context())[0]
        tot_ms = (timer() - start) * 1000
        print('avg prediction ms: {} n={}'.format((tot_ms / loops), loops))
