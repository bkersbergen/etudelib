import gc
import os
import pickle
import platform
from datetime import datetime
from timeit import default_timer as timer
import logging

import numpy as np
import pandas as pd
import torch
from cpuinfo import get_cpu_info

logger = logging.getLogger(__name__)


class MicroBenchmark:
    def __init__(self):
        self.cpu_brand = get_cpu_info().get('brand_raw')
        self.platform = platform.platform()
        self.python_version = platform.python_version()
        self.pt_model_file = None
        logger.info('Platform:' + self.platform)
        logger.info('Python:' + self.python_version)
        logger.info('CPU detected:' + self.cpu_brand)
        if torch.cuda.is_available():
            self.gpu_brand = torch.cuda.get_device_name(0)
            logger.info('GPU detected:' + self.gpu_brand)
        elif torch.backends.mps.is_available():
            self.mps_brand = "mps"
            logger.info('MPS detected:' + self.mps_brand)
        else:
            logger.info('No accelerator detected. CPU only tests')
        self.min_duration_secs = 60 + 60  # warmup needed was 40 secs for windows

    @staticmethod
    def printresults(t_records):
        # t_records a list of 'timeit' floats
        # printing the execution time
        for index, exec_time in enumerate(t_records, 1):
            # printing execution time of code in milliseconds
            m_secs = round(exec_time * 10 ** 3, 2)
            logger.info(f"Case {index}: Time Taken: {m_secs}ms")
        m_secs = round((sum(t_records) / len(t_records)) * 10 ** 3, 2)
        logger.info(f"Average Time Taken: {m_secs}ms")

    def benchmark_pytorch_predictions(self, model, dataloader, device='cpu'):
        result = []
        model.to(device)
        model.eval()
        gc.collect()
        with torch.no_grad():
            test_start = timer()
            iterator = iter(dataloader)
            while timer() - test_start < self.min_duration_secs:
                try:
                    item_seq, session_length, next_item = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    item_seq, session_length, next_item = next(iterator)
                item_seq = item_seq.to(device)
                session_length = session_length.to(device)
                start = timer()
                reco_items = MicroBenchmark.get_item_ids(model.forward(item_seq, session_length))
                duration = timer() - start
                result.append([duration * 1000, datetime.now()])
            torch.cuda.empty_cache()
        logger.info(f"Benchmark ran for {int(timer() - test_start)} secs")
        df = pd.DataFrame(result, columns=['LatencyInMs', 'DateTime'])
        return df

    # @staticmethod
    # def pytorch_predict(pytorch_model: Module, item_seq: Tensor, item_seq_len: Tensor, device_type: str):
    #     # Tensors and model must be on same device
    #     # pytorch_model can be eager-cpu, eager-gpu, script-cpu, script-gpu, trace-cpu, trace-gpu
    #     # Tensor 'item_seq' and 'item_seq_len' must be on the same device.
    #     seq_output = pytorch_model.topk_forward(item_seq, item_seq_len)
    #     return seq_output

    def benchmark_onnxed_predictions(self, ort_sess, dataloader):
        # onnx crashes when unused input Tensors are provided to a onnx-session
        input_params = set([node.name for node in ort_sess._inputs_meta])
        gc.collect()
        result = []
        test_start = timer()
        iterator = iter(dataloader)
        while timer() - test_start < self.min_duration_secs:
            try:
                item_seq, session_length, next_item = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                item_seq, session_length, next_item = next(iterator)
            key = {}
            for onnx_param in input_params:
                if onnx_param == 'item_id_list':
                    key[onnx_param] = item_seq.numpy()
                elif onnx_param == 'max_seq_length':
                    key[onnx_param] = np.array(session_length.numpy(), dtype=np.int64)
            start = timer()
            reco_items = MicroBenchmark.get_item_ids(ort_sess.run(None, key))
            duration = timer() - start
            result.append([duration * 1000, datetime.now()])
        logger.info(f"Benchmark ran for {int(timer() - test_start)} secs")
        df = pd.DataFrame(result, columns=['LatencyInMs', 'DateTime'])
        return df

    @staticmethod
    def onnxed_predict(ort_sess, new_inter):
        item_seq = new_inter['item_id_list']
        item_seq_len = new_inter['item_length']
        seq_output = ort_sess.run(None, {'item_id_list': item_seq.numpy(),
                                         'max_seq_length': np.array([item_seq_len], dtype=np.int64)})
        return seq_output

    @staticmethod
    def get_item_ids(tensor):
        usecase = type(tensor)
        if usecase == torch.return_types.topk:
            return tensor.indices.detach().cpu().numpy()
        elif usecase == list:
            # onnx list. First element are the values, Second element are the indexed item ids
            return tensor[1]
        elif usecase == tuple:
            return tensor[1].detach().cpu().numpy()
        else:
            return tensor.detach().cpu().numpy()

    def write_results(self, results, result_path):
        results['benchmark'] = 'single threaded microbenchmark'
        results['device'] = self.cpu_brand
        results['platform'] = self.platform
        results['python_version'] = self.python_version
        results['dt'] = datetime.now()
        mandatory_keys = {'modelname',
                          'benchmark',
                          'runtime',
                          'latency_df',
                          'device',
                          'platform',
                          'python_version',
                          'dt',
                          'C',
                          't',
                          }
        for key in mandatory_keys:
            if key not in results.keys():
                raise Exception("field: '{}' is missing in results".format(key))

        output_file = '{}/{}_{}_{}_{}_results.pickle'.format(result_path, results['modelname'], results['runtime'], results['C'], results['t'])
        os.makedirs(result_path, exist_ok=True)
        with open(output_file, 'wb') as handle:
            pickle.dump(results, handle)

    @staticmethod
    def read_results(result_path):
        with open(result_path, 'rb') as file:
            results = pickle.load(file)
        return results
