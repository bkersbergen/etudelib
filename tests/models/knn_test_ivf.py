#%%
import os
n_threads = 1
os.environ["OMP_NUM_THREADS"] = f"{n_threads}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}" 
os.environ["MKL_NUM_THREADS"] = f"{n_threads}" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_threads}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}" 
import torch
import multiprocessing
import numpy as np
import time
import faiss
import faiss.contrib.torch_utils
from pathlib import Path
from multiprocessing import Process, Manager
from numba import njit
# Set other environment variables
torch.set_num_threads(n_threads)
#%% Parameters 
nq = 1000               # Number of queries to test
nq_warmup = 500
k = 21
seed = 1234
# C = 40_000_000
C = 1_000_000
device_gpu = torch.device(0)
device_cpu = torch.device('cpu')
d = int(2**np.ceil(np.log2(C**(0.25))))
#%% Helper functions
# Torch timer function
def timer_torch_mmtopk(xq, xb, k=21, mode='eager'):
    nq = xq.shape[0]
    results = np.zeros((nq, k))
    timings = np.zeros(nq)
    model = torch_model(xb, k)
    model.eval()
    model.to(xb.device)
    # JITted model
    if mode == 'jitopt':
        model = torch.jit.optimize_for_inference(torch.jit.trace(model, (xq[0].to(xb.device))))
    for i in range(nq):
        query = xq[i]
        if xb.device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = model(query.to(xb.device))
            result = result.to(torch.device('cpu'))
            end.record()
            torch.cuda.synchronize()
            timings[i] = start.elapsed_time(end) / 1000
        else:
            t0 = time.perf_counter()
            result = model(query)
            t1 = time.perf_counter()
            timings[i] = t1 - t0
        results[i] = result.cpu().numpy()

    return results, timings

# Class for torch model
class torch_model(torch.nn.Module):
    def __init__(self, xb, k):
        super().__init__()
        self.lin = torch.nn.Linear(xb.shape[0], xb.shape[1], bias=False)
        self.lin.weight = torch.nn.Parameter(xb)
        self.k = k

    def forward(self, xq):
        dot_product = self.lin(xq)
        top_k = torch.topk(dot_product, k=self.k, dim=0)
        return top_k[1]

# Timer function
def timer_faiss(func, xq, *args, k=21, **kwargs):
    nq = xq.shape[0]
    results = np.zeros((nq, k))
    timings = np.zeros(nq)
    for i in range(nq):
        query = xq[[i]]
        t0 = time.perf_counter()
        result = func(query, *args, **kwargs)
        t1 = time.perf_counter()
        timings[i] = t1 - t0
        results[i] = result[1]

    return results, timings

# Function to quickly calculate recall[1]@k per sample
def recall1(y_true, y_pred, k):
    n_samples = len(y_true)
    corrects = (y_true[:, :1] == y_pred[:, :k]).sum()
    return corrects / n_samples

# Function to evaluate recall@k per sample
@njit(fastmath=True)
def recall(y_true, y_pred, k):
    n_samples = len(y_true)
    recall = np.zeros(n_samples)
    for i in range(n_samples):
        intersection = np.intersect1d(y_true[i, :k], y_pred[i, :k])
        recall[i] = len(intersection) / k

    return np.mean(recall)
# Torch-CPU
def bench_torch(algorithm, C, d, seed, k, results, device, mode='eager'):
    print(f"Running algorithm {algorithm}")
    # Create datasets
    g = torch.Generator()
    g.manual_seed(seed)
    xb = torch.rand((C, d), generator=g, dtype=torch.float32)
    xb /= torch.norm(xb, dim=-1, keepdim=True)
    xb = xb.to(device)
    g.manual_seed(seed + 1)
    xq = torch.rand((nq, d), generator=g, dtype=torch.float32)
    xq /= torch.norm(xq, dim=-1, keepdim=True)
    results[algorithm] = timer_torch_mmtopk(xq, xb, k, mode)

def bench_faiss_exact(algorithm, C, d, seed, k, results, device):
    print(f"Running algorithm {algorithm}")
    # Create datasets
    g = torch.Generator()
    g.manual_seed(seed)
    xb = torch.rand((C, d), generator=g, dtype=torch.float32)
    xb /= torch.norm(xb, dim=-1, keepdim=True)
    xb = xb.to(device)
    g.manual_seed(seed + 1)
    xq = torch.rand((nq, d), generator=g, dtype=torch.float32)
    xq /= torch.norm(xq, dim=-1, keepdim=True)
    #  Create index
    if device.type == 'cuda':
        res = faiss.StandardGpuResources()
        index_faiss = faiss.GpuIndexFlatL2(res, d)
    else:
        index_faiss = faiss.IndexFlatL2(d)  
    
    index_faiss.add(xb)                  
    results[algorithm] = timer_faiss(index_faiss.search, xq, k, k=k)

def bench_faiss_approximate(algorithm, C, d, seed, k, results, filename, device):
    print(f"Running algorithm {algorithm}")
    # Create datasets
    g = torch.Generator()
    g.manual_seed(seed)
    xb = torch.rand((C, d), generator=g, dtype=torch.float32)
    xb /= torch.norm(xb, dim=-1, keepdim=True)
    g.manual_seed(seed + 1)
    xq = torch.rand((nq, d), generator=g, dtype=torch.float32)
    xq /= torch.norm(xq, dim=-1, keepdim=True)
    # Sources:
    #  - https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors
    #  - https://github.com/facebookresearch/faiss/blob/main/benchs/bench_hnsw.py
    current_path = Path(__file__).parent
    file_path = current_path.joinpath(filename)
    if file_path.exists():
        print(" Index exists, loading from file")
        full_index = faiss.read_index(str(file_path))
    else:
        print(" Index does not exist, training...")
        n_threads_max = multiprocessing.cpu_count()
        faiss.omp_set_num_threads(n_threads_max)
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        if C <= 1_000_000:
            index = faiss.index_factory(d, "IVF16384,Flat", faiss.METRIC_INNER_PRODUCT)    
        elif C <= 10_000_000 and C > 1_000_000:
            index = faiss.index_factory(d, "IVF65536,Flat", faiss.METRIC_INNER_PRODUCT)    
        else:
            index = faiss.index_factory(d, "IVF262144,Flat", faiss.METRIC_INNER_PRODUCT)    
        # Extract IVF and train on GPU
        res = faiss.StandardGpuResources()
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(index_ivf.d))
        index_ivf.clustering_index = clustering_index
        t_start = time.perf_counter()
        index.train(xb)
        t_end = time.perf_counter()
        print(f" Index building time: {t_end - t_start:.2f}s")
        # Add datapoints on GPU - sharded
        index = faiss.index_gpu_to_cpu(index)
        # Retrieve free GPU memory on device_id to determine n_batches
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        # IVF: bytes / vector = 4*d+8. C datapoints. Twice necessary (for index + data). 5% buffer
        index_size = 1.05 * 2 * C * (4 * d + 8)
        n_shards = int((index_size + free_mem - 1) // free_mem)
        print(f"Number of index shards: {n_shards}")
        t_start = time.perf_counter()
        for shard in range(n_shards): 
            # Clone index
            index_shard = faiss.clone_index(index)
            i0 = C * shard // n_shards
            i1 = C * (shard + 1) // n_shards
            # Send to GPU and add datapoints
            index_shard = faiss.index_cpu_to_gpu(res, 0, index_shard)
            index_shard.add_with_ids(xb[i0:i1], torch.arange(i0, i1))
            # Send to CPU and merge
            index_shard = faiss.index_gpu_to_cpu(index_shard)
            if shard == 0:
                full_index = faiss.clone_index(index_shard)
            else:
                faiss.merge_into(full_index, index_shard, False)
        t_end = time.perf_counter()
        print(f" Index adding datapoints time: {t_end - t_start:.2f}s")
        
        # Save
        faiss.write_index(full_index, str(file_path))

    faiss.omp_set_num_threads(1)
    if device.type == 'cuda':
        res = faiss.StandardGpuResources()
        full_index = faiss.index_cpu_to_gpu(res, 0, full_index)
        full_index.nprobe = 2048
    else:
        # full_index.nprobe = 4096
        full_index.nprobe = 8192


    results[algorithm] = timer_faiss(full_index.search, xq, k, k=k)
#%% Loop
if __name__ == '__main__':
    print(f"Benchmark for: C={C:,} items, d={d} embedding size and k={k} for top-k.")
    manager = Manager()
    results = manager.dict()
    # Torch CPU matmul+topk
    # algorithm = 'torch-cpu-mmtopk'
    # results[algorithm] = []
    # mode = 'eager'
    # p = Process(target = bench_torch, args = (algorithm, C, d, seed, k, results, device_cpu, mode, ))
    # p.start()
    # p.join()
    # Torch CPU-JITopt matmul+topk
    # algorithm = 'torch-jitopt-cpu-mmtopk'
    # mode = 'jitopt'
    # p = Process(target = bench_torch, args = (algorithm, C, d, seed, k, results, device_cpu, mode, ))
    # p.start()
    # p.join()
    # Torch GPU matmul+topk
    algorithm = 'torch-gpu-mmtopk'
    mode = 'eager'
    p = Process(target = bench_torch, args = (algorithm, C, d, seed, k, results, device_gpu, mode,))
    p.start()
    p.join()
    # Torch GPU-JITopt matmul+topk
    # algorithm = 'torch-jitopt-gpu-mmtopk'
    # mode = 'jitopt'
    # p = Process(target = bench_torch, args = (algorithm, C, d, seed, k, results, device_gpu, mode,))
    # p.start()
    # p.join()
    # FAISS CPU exact
    # algorithm = 'faiss-cpu-exact'
    # p = Process(target = bench_faiss_exact, args = (algorithm, C, d, seed, k, results, device_cpu, ))
    # p.start()
    # p.join()
    # FAISS GPU exact
    # algorithm = 'faiss-gpu-exact'
    # p = Process(target = bench_faiss_exact, args = (algorithm, C, d, seed, k, results, device_gpu, ))
    # p.start()
    # p.join()
    # FAISS CPU approximate
    algorithm = 'faiss-cpu-approx'
    filename = f"faiss_approx_index_C{C}_d{d}.index"
    p = Process(target = bench_faiss_approximate, args = (algorithm, C, d, seed, k, results, filename, device_cpu, ))
    p.start()
    p.join()
    # FAISS GPU approximate
    algorithm = 'faiss-gpu-approx'
    p = Process(target = bench_faiss_approximate, args = (algorithm, C, d, seed, k, results, filename, device_gpu, ))
    p.start()
    p.join()
    #%% Print timing and check accuracy of algorithms
    ground_truth_model = 'torch-gpu-mmtopk'
    ground_truth_result, _ = results[ground_truth_model]
    print(f"Benchmark for: C={C:,} items, d={d} embedding size and k={k} for top-k.")
    print(f"Ground truth model for recall: {ground_truth_model}")
    recall_ks = [1, 3, k]
    for key, values in results.items():
        result, timing = results[key]
        t_avg = timing[nq_warmup:].mean() * 1000
        t_std = timing[nq_warmup:].std() * 1000
        # Calculate recall@k for all samples over all queries
        recalls1 = [recall1(result, ground_truth_result, k) for k in recall_ks]
        recalls = [recall(result, ground_truth_result, k) for k in recall_ks]
        with np.printoptions(precision=3, suppress=True):
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(f"{key:<25}: {t_avg:7.2f} ±{t_std:5.2f} ms | Recall[1]@{recall_ks}: {np.array(recalls1)} | Recall@{recall_ks}: {np.array(recalls)}")