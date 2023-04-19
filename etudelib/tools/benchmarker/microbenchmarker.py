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
import psutil

logger = logging.getLogger(__name__)


class MicroBenchmark:
    def __init__(self, min_duration_secs):
        self.cpu_brand = get_cpu_info().get('brand_raw')
        self.cpu_utilization, self.used_mem, self.total_mem = MicroBenchmark.get_metrics_cpu()
        logger.info(f'CPU utilization: {self.cpu_utilization} %')
        logger.info(f'used_mem: {self.used_mem} MB')
        logger.info(f'total_mem: {self.total_mem} MB')
        self.platform = platform.platform()
        self.python_version = platform.python_version()
        logger.info('Platform:' + self.platform)
        logger.info('Python:' + self.python_version)
        logger.info('CPU detected:' + self.cpu_brand)
        if torch.cuda.is_available():
            self.gpu_brand = torch.cuda.get_device_name(0)
            logger.info('GPU detected:' + self.gpu_brand)
            self.gpu_utilization, self.gpu_mem_used, self.gpu_memory_total = MicroBenchmark.get_metrics_gpu()
            logger.info(f'CUDA utilization: {self.gpu_utilization} %')
            logger.info(f'CUDA mem_used: {self.gpu_mem_used} MB')
            logger.info(f'CUDA memory_total: {self.gpu_memory_total} MB')
        elif torch.backends.mps.is_available():
            self.mps_brand = "mps"
            logger.info('MPS detected:' + self.mps_brand)
        else:
            logger.info('No accelerator detected. CPU only tests')
        self.seconds_between_metrics = 5
        self.min_duration_secs = min_duration_secs  # 60 + 60  # warmup needed was 40 secs for windows

    @staticmethod
    def get_metrics_cpu():
        cpu_utilization = psutil.cpu_percent()
        svmem = psutil.virtual_memory()
        used_mem = round(svmem.used / 1_024_000)
        total_mem = round(svmem.total / 1_024_000)
        return cpu_utilization, used_mem, total_mem

    @staticmethod
    def get_metrics_gpu():
        gpu_utilization = torch.cuda.utilization(0)
        gpu_mem_used = round(torch.cuda.memory_allocated(0) / 1_024_000)
        gpu_memory_total = round(torch.cuda.get_device_properties(0).total_memory / 1_024_000)
        return gpu_utilization, gpu_mem_used, gpu_memory_total

    def benchmark_pytorch_predictions(self, model, dataloader, device='cpu'):
        result = []
        model.to(device)
        model.eval()
        output_columns = ['LatencyInMs', 'DateTime', 'CPUUtilization', 'UsedMem']
        if device != 'cpu':
            output_columns = output_columns + ['GPUUtilization', 'GPUMemUsed']
        last_metric_ts = 0
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            test_start = timer()
            iterator = iter(dataloader)
            while timer() - test_start < self.min_duration_secs:
                try:
                    item_seq, session_length, next_item = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    item_seq, session_length, next_item = next(iterator)
                start = timer()
                item_seq = item_seq.to(device)
                session_length = session_length.to(device)
                reco_items = MicroBenchmark.get_item_ids(model.forward(item_seq, session_length))
                duration = timer() - start
                if timer() - last_metric_ts > self.seconds_between_metrics:
                    last_metric_ts = timer()
                    cpu_utilization, used_mem, _ = MicroBenchmark.get_metrics_cpu()
                    gpu_utilization, gpu_mem_used, _ = MicroBenchmark.get_metrics_gpu() if device != 'cpu' else (
                    None, None, None)
                else:
                    cpu_utilization, used_mem, gpu_utilization, gpu_mem_used = None, None, None, None
                row = [duration * 1000, datetime.now(), cpu_utilization, used_mem]
                if device != 'cpu':
                    row = row + [gpu_utilization, gpu_mem_used]
                result.append(row)
        logger.info(f"Benchmark ran for {int(timer() - test_start)} secs")
        df = pd.DataFrame(result, columns=output_columns)
        return df

    def benchmark_onnxed_predictions(self, ort_sess, dataloader):
        result = []
        # onnx crashes when unused input Tensors are provided to a onnx-session
        input_params = set([node.name for node in ort_sess._inputs_meta])
        output_columns = ['LatencyInMs', 'DateTime', 'CPUUtilization', 'UsedMem']
        device = 'cuda' if 'CUDAExecutionProvider' in ort_sess.get_providers() else 'cpu'
        if device != 'cpu':
            output_columns = output_columns + ['GPUUtilization', 'GPUMemUsed']
        iterator = iter(dataloader)
        last_metric_ts = 0
        gc.collect()
        torch.cuda.empty_cache()
        test_start = timer()
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
            if timer() - last_metric_ts > self.seconds_between_metrics:
                last_metric_ts = timer()
                cpu_utilization, used_mem, _ = MicroBenchmark.get_metrics_cpu()
                gpu_utilization, gpu_mem_used, _ = MicroBenchmark.get_metrics_gpu() if device != 'cpu' else (
                    None, None, None)
            else:
                cpu_utilization, used_mem, gpu_utilization, gpu_mem_used = None, None, None, None
            row = [duration * 1000, datetime.now(), cpu_utilization, used_mem]
            if device != 'cpu':
                row = row + [gpu_utilization, gpu_mem_used]
            result.append(row)
        logger.info(f"Benchmark ran for {int(timer() - test_start)} secs")
        df = pd.DataFrame(result, columns=output_columns)
        return df


    @staticmethod
    def get_item_ids(tensor):
        usecase = type(tensor)
        if usecase == torch.return_types.topk:
            return tensor.indices.detach().cpu().numpy()
        elif usecase == list:
            # onnx list. First element are the values, Second element are the indexed item ids
            if len(tensor) == 2:
                # Onnx TopK results
                return tensor[1]
            else:
                # Onnx results
                return tensor[0]
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

        output_filename = f'{result_path}/{results["modelname"]}_{results["runtime"]}_{results["param_source"]}_C{results["C"]}_t{results["t"]}_results.pickle'.lower()
        os.makedirs(result_path, exist_ok=True)
        with open(output_filename, 'wb') as handle:
            pickle.dump(results, handle)


    @staticmethod
    def read_results(result_path):
        with open(result_path, 'rb') as file:
            results = pickle.load(file)
        return results
