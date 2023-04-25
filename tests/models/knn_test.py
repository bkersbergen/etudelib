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
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from pathlib import Path
from numba import njit
# Set other environment variables
torch.set_num_threads(n_threads)
current_path = Path(__file__).parent
n_threads_max = multiprocessing.cpu_count()
#%% Parameters 
n_iters = 1000               # iterations for timing
n_iters_warmup = 500        
k = 21
nq = 1 # number of queries (batch size)
device_cuda = torch.device(0)
device_cpu = torch.device('cpu')
rng = np.random.default_rng(1234)             # make reproducible
C, d = 10_000_000, 64
# C, d = 100_000, 64
xb = rng.random((C, d), dtype = np.float32)
xb = xb / np.linalg.norm(xb, axis=-1, keepdims=True)
#%% Create queries
xq = rng.random((n_iters, nq, d), dtype = np.float32)
xq = xq / np.linalg.norm(xq, axis=-1, keepdims=True)
#%% Helper functions
# Timer function
def timer(func, xq, *args, n_iters=1000, nq=1, k=21, **kwargs):
    results = np.zeros((n_iters, nq, k))
    timings = np.zeros(n_iters)
    for i in range(n_iters):
        t0 = time.perf_counter()
        result = func(xq[i], *args, **kwargs)
        t1 = time.perf_counter()
        timings[i] = t1 - t0
        results[i] = result[1]

    return results, timings

# Torch timer function
def timer_torch_mmtopk(xq_t, xb_t, n_iters=1000, nq=1, k=21, mode='eager'):
    results = np.zeros((n_iters, nq, k))
    timings = np.zeros(n_iters)
    model = torch_model(xb_t, k)
    model.eval()
    model.to(xb_t.device)
    # JITted and quantized models
    if mode == 'jitopt':
        model = torch.jit.optimize_for_inference(torch.jit.trace(model, (xq_t[0].to(xb_t.device))))
    elif mode == 'quant':
        torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    for i in range(n_iters):
        if xb_t.device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = model(xq_t[i].to(xb_t.device))
            result = result.to(torch.device('cpu'))
            end.record()
            torch.cuda.synchronize()
            timings[i] = start.elapsed_time(end) / 1000
        else:
            t0 = time.perf_counter()
            result = model(xq_t[i])
            t1 = time.perf_counter()
            timings[i] = t1 - t0
        results[i] = result.cpu().numpy()

    return results, timings

# Class for torch model
class torch_model(torch.nn.Module):
    def __init__(self, xb_t, k):
        super().__init__()
        self.lin = torch.nn.Linear(xb_t.shape[0], xb_t.shape[1], bias=False)
        self.lin.weight = torch.nn.Parameter(xb_t)
        self.k = k

    def forward(self, xq_t):
        dot_product = self.lin(xq_t)
        top_k = torch.topk(dot_product, k=self.k, dim=1)
        return top_k[1]

def numpy_model(xq, xb, k):
    dot_product = xb @ xq.T
    top_k = np.argpartition(dot_product, -k, axis=0)[-k:]
    
    return (0, top_k.T)

# Function to quickly calculate recall per sample
@njit(fastmath=True)
def recall(y_true, y_pred, nq, k):
    n_samples = len(y_true)
    recall = 0
    for q in range(nq):
        for i in range(n_samples):
            intersection = np.intersect1d(y_true[i, q, :k], y_pred[i, q, :k])
            recall += len(intersection)

    return recall / (n_samples * nq * k)
#%% Loop
results = {}
# Sklearn-brute force
# algorithm = 'sklearn-brutef'
# print(f"Running algorithm {algorithm}")
# neigh_sklearn = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=1).fit(xb)
# results[algorithm] = timer(neigh_sklearn.kneighbors, xq, n_iters=n_iters, nq=nq, k=k)
# Scipy-KDTree
# algorithm = 'scipy-kdtree'
# print(f"Running algorithm {algorithm}")
# neigh_scipy = KDTree(xb, leafsize=40)
# results[algorithm] = timer(neigh_scipy.query, xq, k, n_iters=n_iters, nq=nq, k=k)
# Torch CPU matmul+topk
algorithm = 'torch-cpu-mmtopk'
print(f"Running algorithm {algorithm}")
xq_t = torch.from_numpy(xq.copy()).contiguous()
xb_t = torch.from_numpy(xb.copy()).contiguous()
results[algorithm] = timer_torch_mmtopk(xq_t, xb_t, n_iters=n_iters, nq=nq, k=k)
# Torch CPU-JITopt matmul+topk
algorithm = 'torch-jitopt-cpu-mmtopk'
print(f"Running algorithm {algorithm}")
results[algorithm] = timer_torch_mmtopk(xq_t, xb_t, n_iters=n_iters, nq=nq, k=k, mode='jitopt')
# Torch CPU-quantized matmul+topk
# algorithm = 'torch-quant-cpu-mmtopk'
# print(f"Running algorithm {algorithm}")
# results[algorithm] = timer_torch_mmtopk(xq_t, xb_t, n_iters=n_iters, nq=nq, k=k, mode='quant')
# Torch GPU matmul+topk
algorithm = 'torch-gpu-mmtopk'
print(f"Running algorithm {algorithm}")
xb_t_cuda = torch.from_numpy(xb.copy()).contiguous().to(device_cuda)
results[algorithm] = timer_torch_mmtopk(xq_t, xb_t_cuda, n_iters=n_iters, nq=nq, k=k)
# Torch GPU-JITopt matmul+topk
algorithm = 'torch-jitopt-gpu-mmtopk'
print(f"Running algorithm {algorithm}")
xb_t_cuda = torch.from_numpy(xb.copy()).contiguous().to(device_cuda)
results[algorithm] = timer_torch_mmtopk(xq_t, xb_t_cuda, n_iters=n_iters, nq=nq, k=k, mode='jitopt')
# Numpy exact
algorithm = 'numpy-mmargpartition'
print(f"Running algorithm {algorithm}")
results[algorithm] = timer(numpy_model, xq, xb, k, n_iters=n_iters, nq=nq, k=k)
# FAISS CPU exact
algorithm = 'faiss-cpu-exact'
print(f"Running algorithm {algorithm}")
faiss.omp_set_num_threads(n_threads_max)
index_faiss = faiss.IndexFlatL2(d)   # build the index
index_faiss.add(xb)                  # add vectors to the index
faiss.omp_set_num_threads(n_threads)
results[algorithm] = timer(index_faiss.search, xq, k, n_iters=n_iters, nq=nq, k=k)
# FAISS GPU exact
algorithm = 'faiss-gpu-exact'
print(f"Running algorithm {algorithm}")
res = faiss.StandardGpuResources()
index_faiss_gpu = faiss.index_cpu_to_gpu(res, 0, index_faiss)
results[algorithm] = timer(index_faiss_gpu.search, xq, k, n_iters=n_iters, nq=nq, k=k)
# FAISS CPU approximate
# Sources:
#  - https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors
#  - https://github.com/facebookresearch/faiss/blob/main/benchs/bench_hnsw.py
algorithm = 'faiss-cpu-approx'
print(f"Running algorithm {algorithm}")
faiss.omp_set_num_threads(n_threads_max)
index_faiss_approximate = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 32)
index_faiss_approximate.hnsw.efSearch = 1024
index_faiss_approximate.hnsw.efConstruction = 80
index_faiss_approximate.train(xb)
index_faiss_approximate.add(xb)                  # add vectors to the index
faiss.omp_set_num_threads(n_threads)
results[algorithm] = timer(index_faiss_approximate.search, xq, k, n_iters=n_iters, nq=nq, k=k)
# FAISS GPU approximate
algorithm = 'faiss-gpu-approx'
print(f"Running algorithm {algorithm}")
index_faiss_approximate_gpu = faiss.index_cpu_to_gpu(res, 0, index_faiss_approximate)
results[algorithm] = timer(index_faiss_approximate_gpu.search, xq, k, n_iters=n_iters, nq=nq, k=k)
#%% Print timing and check accuracy of algorithms
ground_truth_model = 'torch-cpu-mmtopk'
ground_truth_result, _ = results[ground_truth_model]
print(f"Benchmark for: C={C:,} items, d={d} embedding size and k={k} for top-k.")
print(f"Ground truth model for recall: {ground_truth_model}")
recall_ks = [1, 3, k]
for key, values in results.items():
    result, timing = results[key]
    t_avg = timing[n_iters_warmup:].mean() * 1000
    t_std = timing[n_iters_warmup:].std() * 1000
    # Calculate recall@k for all samples over all queries
    recalls = [recall(result, ground_truth_result, nq, k) for k in recall_ks]
    with np.printoptions(precision=3, suppress=True):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print(f"{key:<25}: {t_avg:7.2f} ±{t_std:5.2f} ms | Recall@{recall_ks}: {np.array(recalls)}")