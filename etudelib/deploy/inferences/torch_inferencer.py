import logging
import os
import onnxruntime as ort
from importlib import import_module
from pathlib import Path

from timeit import default_timer as timer

import numpy as np
import torch
from omegaconf import OmegaConf
from ts.torch_handler.base_handler import BaseHandler

from etudelib.models.topkdecorator import TopKDecorator
from etudelib.utils.loggers import configure_logger

logger = logging.getLogger(__name__)


class TorchInferencer(BaseHandler):

    def __init__(self):
        self.model_filename = None
        self.C = None
        self.ort_sess = None
        self.max_seq_length = None
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.item2idx = None  # (dict) each key is an item_id and the value is the idx number
        self.idx2item = None  # (list) each position contains an item_id
        self.context = None
        self.runtime = ''

    def initialize(self, context):
        #  load the model
        logger.info('TorchInferencer.initialize(): {}'.format(context))
        properties = context.system_properties
        logger.info("properties: {}".format(properties))
        model_dir = properties.get("model_dir")
        # Read torch serialized file
        serialized_file = context.manifest['model']['serializedFile']
        self.model_filename = serialized_file
        model_path = os.path.join(model_dir, serialized_file)
        self.initialize_from_file(model_path)

    def initialize_from_file(self, model_path):
        logger.info('TorchInferencer.initialize_from_file(): {}'.format(model_path))
        if model_path.endswith('_eager.pth'):
            self.runtime = 'eager'
            filename_without_extension = model_path.split('_eager.pth')[0]
            self.model = torch.load(model_path, map_location=self.device_type).to(self.device_type)
            self.model.eval()
        elif model_path.endswith('_jitopt.pth'):
            self.runtime = 'jitopt'
            filename_without_extension = model_path.split('_jitopt.pth')[0]
            self.model = torch.jit.load(model_path, map_location=self.device_type).to(self.device_type)
            self.model.eval()
        elif model_path.endswith('_onnx.pth'):
            self.runtime = 'onnx'
            filename_without_extension = model_path.split('_onnx.pth')[0]
            if self.device_type == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                sess_options = None
            else:
                providers = ['CPUExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.inter_op_num_threads = 1
                sess_options.intra_op_num_threads = 1
            try:
                self.ort_sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            except ImportError as error:
                logger.warning("Onnx ImportError." + str(error))
        else:
            logger.error("Unknown Model serialization for:", model_path)

        logger.info(f'Runtime: {self.runtime}')
        logger.info(f'Device type: {self.device_type}')
        payload = torch.load(filename_without_extension + '_payload.torch')

        self.max_seq_length = payload.get('max_seq_length')
        self.idx2item = payload.get('idx2item')  # list
        self.C = payload.get('C')  # catalog size
        self.item2idx = dict(zip(self.idx2item, range(len(self.idx2item))))  # dict

    def handle(self, data, context):
        self.context = context
        t0 = timer()
        data_preprocess = self.preprocess(data)
        t1 = timer()
        output = self.inference(data_preprocess)
        t2 = timer()
        output = self.postprocess(output)
        t3 = timer()
        preprocess_time_ms = (t1 - t0) * 1000
        inference_time_ms = (t2 - t1) * 1000
        postprocess_time_ms = (t3 - t2) * 1000
        output = [{'items': output, 'nf': {'preprocess_ms': preprocess_time_ms,
                                           'inference_ms': inference_time_ms,
                                           'postprocess_ms': postprocess_time_ms,
                                           'model': self.model_filename,
                                           'device': self.device_type,
                                           }}]
        return [{
            "predictions": output,
        }]

    def inference(self, data, *args, **kwargs):
        if self.runtime == 'onnx':
            return self.ort_sess.run(None, data)
        else:
            items, lengths = data
            with torch.no_grad():
                return self.model.forward(items, lengths)

    def preprocess(self, data):
        # expected input:
        # VertexAI enforces 'instances' and TorchServe adds 'body' to the request
        # data = [{'body': {'instances': [{'context': [8, 5, 12, 300]}], 'parameters': ['string']}}]
        payload = data[0].get("body")  # body when http POST

        if payload is None:
            error_message = "Preprocess error No Body Payload in request: {}".format(data)
            logger.error(error_message)
            self.context.request_processor[0].report_status(400, error_message)
            return ["{} data: {}".format(error_message, data)]
        try:
            evolving_session_items = payload['instances'][0]['context']
        except:
            error_message = "Preprocess error No instances context: {}".format(data)
            logger.error(error_message)
            self.context.request_processor[0].report_status(400, error_message)
            return ["{} data: {}".format(error_message, data)]
        if evolving_session_items is None:
            error_message = "Preprocess error No evolving_session_items in request: {}".format(data)
            logger.error(error_message)
            self.context.request_processor[0].report_status(400, error_message)
            return ["{} data: {}".format(error_message, data)]
        if len(evolving_session_items) == 0:
            error_message = "Preprocess error Evolving_session_items is empty: {}".format(data)
            logger.error(error_message)
            self.context.request_processor[0].report_status(400, error_message)
            return ["{} data: {}".format(error_message, data)]

        evolving_session_items = evolving_session_items[-self.max_seq_length:]  # use most recent max_seq_length items
        idx_seq = [self.item2idx.get(item_id, 0) for item_id in evolving_session_items]
        padded_tensor = torch.zeros((self.max_seq_length,), dtype=torch.int64, device=self.device_type)
        padded_tensor[:len(idx_seq)] = torch.tensor(idx_seq, dtype=torch.int64, device=self.device_type)

        # convert tensors to batch_size 1
        item_seq = padded_tensor.unsqueeze(0)
        session_length = torch.tensor(len(idx_seq), device=self.device_type).unsqueeze(0)

        if self.runtime == 'onnx':
            key = {}
            # onnx crashes when unused input Tensors are provided to a onnx-session
            input_params = set([node.name for node in self.ort_sess._inputs_meta])
            for onnx_param in input_params:
                if onnx_param == 'item_id_list':
                    key[onnx_param] = item_seq.numpy()
                elif onnx_param == 'max_seq_length':
                    key[onnx_param] = np.array(session_length.numpy(), dtype=np.int64)
            return key
        else:
            return item_seq, session_length

    def postprocess(self, tensor):
        usecase = type(tensor)
        if usecase == torch.return_types.topk:
            indices_batch = tensor.indices.detach().cpu().numpy()
        elif usecase == list:
            # onnx list. First element are the values, Second element are the indexed item ids
            if len(tensor) == 2:
                # Onnx TopK results
                indices_batch = tensor[1]
            else:
                # Onnx results
                indices_batch = tensor[0]
        elif usecase == tuple:
            indices_batch = tensor[1].detach().cpu().numpy()
        else:
            indices_batch = tensor.detach().cpu().numpy()

        reco_item_ids = []
        for idx in indices_batch[0]:
            if idx < len(self.idx2item):
                reco_item_ids.append(self.idx2item[idx])
        return reco_item_ids


if __name__ == '__main__':
    model_name = 'core'
    C = 1000000
    max_seq_length = 50
    dataset_name = 'synthetic'
    rootdir = Path(__file__).parent.parent.parent.parent

    projectdir = Path(rootdir, 'projects/benchmark')
    configure_logger(level='INFO')

    config_path = os.path.join(rootdir, f"etudelib/models/{model_name}/config.yaml".lower())
    config = OmegaConf.load(config_path)

    config['dataset'] = {}
    config['dataset']['n_items'] = C
    config['dataset']['max_seq_length'] = max_seq_length

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
    torch.save(payload, str(projectdir / f'{base_filename}_payload.torch'))

    eager_model_file = str(projectdir / f'{base_filename}_eager.pth')
    torch.save(eager_model, eager_model_file)

    inferencer = TorchInferencer()
    inferencer.initialize_from_file(eager_model_file)
    request_data = [{'body':
        {
            'instances': [{'context': [1, 2, 3]}, {'context': [2, 3, 4]}]
        }
    }]
    preprocessed = inferencer.preprocess(request_data)
    inferenced = inferencer.inference(preprocessed)
    recos = inferencer.postprocess(inferenced)

    recos = inferencer.handle(request_data, context=None)
    print(recos)
