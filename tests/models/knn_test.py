#%%
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
import torch
torch.set_num_threads(1)
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import faiss
import matplotlib.pyplot as plt
#%% Parameters 
d = 64                       # hidden dimension
C = 1_000_000               # database size
n_iters = 1000               # iterations for timing
n_iters_warmup = 500        
nq = 1                       # batch size
k = 21
device_cuda = torch.device(0)
device_cpu = torch.device('cpu')
rng = np.random.default_rng(1234)             # make reproducible
# Create database
xb = rng.random((C, d), dtype = np.float32)
xb[:, 0] += np.arange(C) / 1000.
xb = xb / np.linalg.norm(xb, axis=-1, keepdims=True)
# Send to device
xb_t = torch.from_numpy(xb.copy()).contiguous()
xb_t_cuda = xb_t.to(device_cuda)
# Create NN Trees
neigh_sklearn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=1, leaf_size=40).fit(xb)
neigh_scipy = KDTree(xb, leafsize=40)
# FAISS
index = faiss.IndexFlatL2(d)   # build the index
index.add(xb)                  # add vectors to the index
#%% Loop - sklearn
t_sklearn = np.zeros(n_iters)
for i in range(n_iters):
    # Create query
    xq = rng.random((nq, d), dtype = np.float32)
    xq[:, 0] += np.arange(nq) / 1000.
    xq = xq / np.linalg.norm(xq, axis=-1, keepdims=True)
    
    t_start_sklearn = time.perf_counter()
    _, I_sklearn = neigh_sklearn.kneighbors(xq)    
    t_end_sklearn = time.perf_counter()
    t_sklearn[i] = (t_end_sklearn - t_start_sklearn)

print(f"Time sklearn   : {t_sklearn[n_iters_warmup:].mean() * 1000:.2f} ±{t_sklearn[n_iters_warmup:].std() * 1000:.2f} ms")
#%% Loop - scipy
t_scipy = np.zeros(n_iters)
for i in range(n_iters):
    # Create query
    xq = rng.random((nq, d), dtype = np.float32)
    xq[:, 0] += np.arange(nq) / 1000.
    xq = xq / np.linalg.norm(xq, axis=-1, keepdims=True)
    
    t_start_scipy = time.perf_counter()
    _, I_scipy  = neigh_scipy.query(xq, k)    
    t_end_scipy  = time.perf_counter()
    t_scipy[i] = (t_end_scipy - t_start_scipy)

print(f"Time scipy     : {t_scipy[n_iters_warmup:].mean() * 1000:.2f} ±{t_scipy[n_iters_warmup:].std() * 1000:.2f} ms")
#%% Loop - Torch CPU
t_torch_mmtopk_cpu = np.zeros(n_iters)
for i in range(n_iters):
    # Create query
    xq = rng.random((nq, d), dtype = np.float32)
    xq[:, 0] += np.arange(nq) / 1000.
    xq = xq / np.linalg.norm(xq, axis=-1, keepdims=True)
    xq_t = torch.from_numpy(xq.copy()).contiguous()

    t_start_torch_mmtopk_cpu = time.perf_counter()
    dot_product_cpu = xb_t  @ xq_t.T
    _, I_torch_mmtopk_cpu = torch.topk(dot_product_cpu, k, dim=0)
    t_end_torch_mmtopk_cpu  = time.perf_counter()
    t_torch_mmtopk_cpu[i] = (t_end_torch_mmtopk_cpu - t_start_torch_mmtopk_cpu)

print(f"Time torch_cpu : {t_torch_mmtopk_cpu[n_iters_warmup:].mean() * 1000:.2f} ±{t_torch_mmtopk_cpu[n_iters_warmup:].std() * 1000:.2f} ms")
#%% Loop - Torch CUDA
t_torch_mmtopk_cuda = np.zeros(n_iters)
for i in range(n_iters):
    # Create query
    xq = rng.random((nq, d), dtype = np.float32)
    xq[:, 0] += np.arange(nq) / 1000.
    xq = xq / np.linalg.norm(xq, axis=-1, keepdims=True)
    xq_t = torch.from_numpy(xq.copy()).contiguous()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    xq_t_cuda = xq_t.to(device_cuda)
    dot_product = xb_t_cuda @ xq_t_cuda.T
    _, I_torch_mmtopk_cuda = torch.topk(dot_product, k, dim=0)
    I_torch_mmtopk_cuda = I_torch_mmtopk_cuda.to(device_cpu)
    end.record()
    torch.cuda.synchronize()
    t_iter = start.elapsed_time(end)
    t_torch_mmtopk_cuda[i] = t_iter / 1000

print(f"Time torch_cuda: {t_torch_mmtopk_cuda[n_iters_warmup:].mean() * 1000:.2f} ±{t_torch_mmtopk_cuda[n_iters_warmup:].std() * 1000:.2f} ms")
#%% Loop - FAISS CPU
t_faiss_cpu = np.zeros(n_iters)
for i in range(n_iters):
    # Create query
    xq = rng.random((nq, d), dtype = np.float32)
    xq[:, 0] += np.arange(nq) / 1000.
    xq = xq / np.linalg.norm(xq, axis=-1, keepdims=True)

    t_start_faiss_cpu = time.perf_counter()
    _, I_faiss = index.search(xq, k)
    t_end_faiss_cpu  = time.perf_counter()
    t_faiss_cpu[i] = (t_end_faiss_cpu - t_start_faiss_cpu)

print(f"Time faiss_cpu : {t_faiss_cpu[n_iters_warmup:].mean() * 1000:.2f} ±{t_faiss_cpu[n_iters_warmup:].std() * 1000:.2f} ms")