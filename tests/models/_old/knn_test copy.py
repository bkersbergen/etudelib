#%%
import torch
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.spatial import KDTree as KDTreeScipy
import matplotlib.pyplot as plt
# torch.set_num_threads(1)
#%% Follow online example
# https://github.com/facebookresearch/faiss/wiki/Getting-started
d = 64                        # hidden dimension
C = 1_000_000                      # database size
n_iters = 1000               # iterations for timing
nq = 1                       # batch size == sequence length in our problem
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
neigh_scipy = KDTreeScipy(xb, leafsize=40)
#%% Loop
t_sklearn = 0.0
t_scipy = 0.0
t_torch_mmtopk_cpu = 0.0
t_torch_mmtopk_cuda = 0.0
n_iters_warmup = 500
t_sklearn = np.zeros(n_iters)
for i in range(n_iters):
    # Create query
    xq = rng.random((nq, d), dtype = np.float32)
    xq[:, 0] += np.arange(nq) / 1000.
    xq = xq / np.linalg.norm(xq, axis=-1, keepdims=True)
    xq_t = torch.from_numpy(xq.copy())
    
    # Sklearn timing
    t_start_sklearn = time.perf_counter()
    _, I_sklearn = neigh_sklearn.kneighbors(xq)    
    t_end_sklearn = time.perf_counter()
    # t_sklearn += (i >= n_iters_warmup) * (t_end_sklearn - t_start_sklearn) / (n_iters - n_iters_warmup)
    t_sklearn[i] = (t_end_sklearn - t_start_sklearn)

    # Scipy timing
    # t_start_scipy = time.perf_counter()
    # _, I_scipy = neigh_scipy.query(xq, k)
    # t_end_scipy = time.perf_counter()
    # t_scipy += (i >= n_iters_warmup) * (t_end_scipy - t_start_scipy) / (n_iters - n_iters_warmup)
    # assert np.all(np.sort(I_sklearn) == np.sort(I_scipy))

    # Torch matmul+topk CPU timing
    # t0 = time.perf_counter()
    # dot_product_cpu = xb_t @ xq_t.T
    # _, I_torch_mmtopk_cpu = torch.topk(dot_product_cpu, k, dim=0)
    # t1 = time.perf_counter()
    # t_torch_mmtopk_cpu += (i >= n_iters_warmup) * (t1 - t0) / (n_iters - n_iters_warmup)
    # assert np.all(np.sort(I_torch_mmtopk_cpu.numpy().T) == np.sort(I_sklearn))

    # Torch matmul+topk CUDA timing
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    # xq_t_cuda = xq_t.to(device_cuda)
    # dot_product = xb_t_cuda @ xq_t_cuda.T
    # _, I_torch_mmtopk_cuda = torch.topk(dot_product, k, dim=0)
    # I_torch_mmtopk_cuda = I_torch_mmtopk_cuda.to(device_cpu)
    # end.record()
    # torch.cuda.synchronize()
    # t_iter = start.elapsed_time(end)
    # t_torch_mmtopk_cuda += (i >= n_iters_warmup) * t_iter / (n_iters - n_iters_warmup)
    # assert np.all(np.sort(I_torch_mmtopk_cuda.numpy().T) == np.sort(I_sklearn))

# print(f"Time sklearn   : {t_sklearn * 1000:.2f} ms")
# print(f"Time scipy     : {t_scipy * 1000:.2f} ms")
# print(f"Time torch_cpu : {t_torch_mmtopk_cpu:.2f} ms")
# print(f"Time torch_cuda: {t_torch_mmtopk_cuda:.2f} ms")
#%% Loop - Sklearn
n_iters_warmup = 500
t_scipy = 0.0
t_sklearn = np.zeros(n_iters)
t_torch_mmtopk_cpu = np.zeros(n_iters)
t_torch_mmtopk_cuda = np.zeros(n_iters)
t0 = time.perf_counter()
for i in range(n_iters):
    # Create query
    xq = rng.random((nq, d), dtype = np.float32)
    xq[:, 0] += np.arange(nq) / 1000.
    xq = xq / np.linalg.norm(xq, axis=-1, keepdims=True)
    xq_t = torch.from_numpy(xq.copy()).contiguous()
    
    # Sklearn timing
    # t_start_sklearn = time.perf_counter()
    # _, I_sklearn = neigh_sklearn.kneighbors(xq)    
    # t_end_sklearn = time.perf_counter()
    # # t_sklearn += (i >= n_iters_warmup) * (t_end_sklearn - t_start_sklearn) / (n_iters - n_iters_warmup)
    # t_sklearn[i] = (t_end_sklearn - t_start_sklearn)

    # Scipy timing
    # t_start_scipy = time.perf_counter()
    # _, I_scipy = neigh_scipy.query(xq, k)
    # t_end_scipy = time.perf_counter()
    # t_scipy += (i >= n_iters_warmup) * (t_end_scipy - t_start_scipy) / (n_iters - n_iters_warmup)
    # assert np.all(np.sort(I_sklearn) == np.sort(I_scipy))

    # Torch matmul+topk CPU timing
    dot_product_cpu = xb_t @ xq_t.T
    _, I_torch_mmtopk_cpu = torch.topk(dot_product_cpu, k, dim=0)
    # assert np.all(np.sort(I_torch_mmtopk_cpu.numpy().T) == np.sort(I_sklearn))

    # Torch matmul+topk CUDA timing
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    # xq_t_cuda = xq_t.to(device_cuda)
    # dot_product = xb_t_cuda @ xq_t_cuda.T
    # _, I_torch_mmtopk_cuda = torch.topk(dot_product, k, dim=0)
    # I_torch_mmtopk_cuda = I_torch_mmtopk_cuda.to(device_cpu)
    # end.record()
    # torch.cuda.synchronize()
    # t_iter = start.elapsed_time(end)
    # t_torch_mmtopk_cuda[i] = t_iter / 1000
    # assert np.all(np.sort(I_torch_mmtopk_cuda.numpy().T) == np.sort(I_sklearn))

t1 = time.perf_counter()
t_torch_mmtopk_cpu[i] = (t1 - t0)

# print(f"Time sklearn   : {t_sklearn * 1000:.2f} ms")
# print(f"Time scipy     : {t_scipy * 1000:.2f} ms")
# print(f"Time torch_cpu : {t_torch_mmtopk_cpu:.2f} ms")
# print(f"Time torch_cuda: {t_torch_mmtopk_cuda:.2f} ms")