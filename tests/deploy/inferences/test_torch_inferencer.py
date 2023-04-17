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

from tests.deploy.modelutil import ModelUtil

sys.path.insert(0, '../etudelib/deploy/inferences/')

import unittest
from etudelib.deploy.inferences.torch_inferencer import TorchInferencer


class Metrics:
    def add_time(self, name, value, idx=None, unit='ms'):
        pass


class Context:
    metrics = Metrics()


class TestingTorchInferencer(unittest.TestCase):

    def test_model_returns_predictions(self):
        payload_path, eager_model_path, jitopt_model_path, onnx_model_path = ModelUtil.create_model('core', C=10000)

        sut = TorchInferencer()
        sut.initialize_from_file(onnx_model_path)
        request_data = [{'body': {'instances': [{'context': [8, 5, 12, 300]}], 'parameters': ['somestring']}}]

        result = sut.handle(request_data, context=Context())
        print(result)
        predictions = result[0]['predictions']
        self.assertTrue(len(predictions[0]['items']) > 5)

    def test_naive_inference_benchmark(self):
        payload_path, eager_model_path, jitopt_model_path, onnx_model_path = ModelUtil.create_model('core', C=10000)

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
