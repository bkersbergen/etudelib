#%%
import torch
import faiss
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
#%% Follow online example
# https://github.com/facebookresearch/faiss/wiki/Getting-started
d = 64                        # hidden dimension
C = 10_000_000                      # database size
n_iters = 100               # iterations for timing
nq = 1                       # batch size == sequence length in our problem
k = 21
device_cuda = torch.device(0)
device_cpu = torch.device('cpu')
rng = np.random.default_rng(1234)             # make reproducible
# Create database
xb = rng.random((C, d), dtype = np.float32)
xb[:, 0] += np.arange(C) / 1000.
faiss.normalize_L2(xb)
xb_t = torch.from_numpy(xb).to(device_cuda)
# FAISS
index = faiss.IndexFlatL2(d)   # build the index
index.add(xb)                  # add vectors to the index
# Sklearn
neigh = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=1).fit(xb)
#%% Loop
t_faiss = 0.0
t_cuda = 0.0
t_sklearn = 0.0
for i in range(n_iters):
    # Create query
    xq = rng.random((nq, d), dtype = np.float32)
    xq[:, 0] += np.arange(nq) / 1000.
    faiss.normalize_L2(xq)
    # FAISS timing
    t0 = time.perf_counter()
    D_faiss, I_faiss = index.search(xq, k)
    t1 = time.perf_counter()
    t_faiss += (t1 - t0) / n_iters
    # Cuda timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    xq_t = torch.from_numpy(xq).to(device_cuda)
    dot_product = xb_t @ xq_t.T
    V_cuda, I_cuda = torch.topk(dot_product, k, dim=0)
    I_cpu = I_cuda.to(device_cpu)
    end.record()
    torch.cuda.synchronize()
    t_cuda_iter = start.elapsed_time(end)
    t_cuda += t_cuda_iter / n_iters
    assert np.all(np.sort(I_cpu.numpy().squeeze(), axis=0) == np.sort(I_faiss.squeeze().T, axis=0))
    # Sklearn timing
    t0 = time.perf_counter()
    D_sklearn, I_sklearn = neigh.kneighbors(xq)    
    t1 = time.perf_counter()
    t_sklearn += (t1 - t0) / n_iters
    assert np.all(np.sort(I_sklearn.T, axis=0) == np.sort(I_cpu.numpy(), axis=0))


print(f"Time faiss: {t_faiss * 1000:.2f} ms")
print(f"Time cuda: {t_cuda:.2f} ms")
print(f"Time sklearn: {t_sklearn * 1000:.2f} ms")