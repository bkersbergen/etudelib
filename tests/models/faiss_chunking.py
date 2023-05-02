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
# C = 20_000_000
C = 40_000_000
device_gpu = torch.device(0)
device_cpu = torch.device('cpu')
d = int(2**np.ceil(np.log2(C**(0.25))))
print(f"Benchmark for: C={C:,} items, d={d} embedding size and k={k} for top-k.")
#%% Create data
g = torch.Generator()
g.manual_seed(seed)
xb = torch.rand((C, d), generator=g, dtype=torch.float32)
xb /= torch.norm(xb, dim=-1, keepdim=True)
g.manual_seed(seed + 1)
xq = torch.rand((nq, d), generator=g, dtype=torch.float32)
xq /= torch.norm(xq, dim=-1, keepdim=True)
#%% Test merging of datasets
n_threads_max = multiprocessing.cpu_count()
faiss.omp_set_num_threads(n_threads_max)
# index = faiss.index_factory(d, "IVF16384,Flat")
index = faiss.index_factory(d, "IVF262144,Flat")    
res = faiss.StandardGpuResources()
index_ivf = faiss.extract_index_ivf(index)
clustering_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(index_ivf.d))
index_ivf.clustering_index = clustering_index
t_start = time.perf_counter()
index.train(xb)
t_end = time.perf_counter()
print(f" Index training time: {t_end - t_start:.2f}s")
#%% Create index
# def add_to_index_sharded(index, xb, gpu_id=0):
#     res = faiss.StandardGpuResources()
#     C = xb.shape[0]
#     d = xb.shape[1]
#     # Retrieve free GPU memory on device_id to determine n_batches
#     free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
#     # IVF bytes / vector = 4*d+8. C datapoints. Twice necessary (for index + data). 10% buffer
#     index_size = 1.1 * 2 * C * (4 * d + 8)
#     n_shards = int((index_size + free_mem - 1) // free_mem)
#     print(f"Number of index shards: {n_shards}")
#     for shard in range(n_shards): 
#         # Clone index
#         index_shard = faiss.clone_index(index)
#         i0 = C * shard // n_shards
#         i1 = C * (shard + 1) // n_shards
#         # Send to GPU
#         index_shard = faiss.index_cpu_to_gpu(res, 0, index_shard)
#         # Add datapoints
#         index_shard.add_with_ids(xb[i0:i1], torch.arange(i0, i1))
#         # Send to CPU and merge
#         index_shard = faiss.index_gpu_to_cpu(index_shard)
#         if shard == 0:
#             full_index = index_shard
#         else:
#             faiss.merge_into(full_index, index_shard, False)

#     return full_index
# full_index = add_to_index_sharded(index, xb)
index = faiss.index_gpu_to_cpu(index)
# Retrieve free GPU memory on device_id to determine n_batches
free_mem, total_mem = torch.cuda.mem_get_info(0)
# IVF bytes / vector = 4*d+8. C datapoints. Twice necessary (for index + data). 10% buffer
index_size = 1.1 * 2 * C * (4 * d + 8)
n_shards = int((index_size + free_mem - 1) // free_mem)
print(f"Number of index shards: {n_shards}")
t_start = time.perf_counter()
for shard in range(n_shards): 
    # Clone index
    index_shard = faiss.clone_index(index)
    i0 = C * shard // n_shards
    i1 = C * (shard + 1) // n_shards
    # Send to GPU
    index_shard = faiss.index_cpu_to_gpu(res, 0, index_shard)
    # Add datapoints
    index_shard.add_with_ids(xb[i0:i1], torch.arange(i0, i1))
    # Send to CPU and merge
    index_shard = faiss.index_gpu_to_cpu(index_shard)
    if shard == 0:
        full_index = faiss.clone_index(index_shard)
    else:
        faiss.merge_into(full_index, index_shard, False)
t_end = time.perf_counter()
print(f" Index adding datapoints time: {t_end - t_start:.2f}s")