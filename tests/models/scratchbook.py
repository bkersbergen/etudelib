#%%
n_iters = 100
import time
import torch
from faiss.loader import *
t = 0
n_items = 5_000_000
#%%
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_items, embedding_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(n_items, embedding_size)
        self.linear = nn.Linear(embedding_size, embedding_size)

    def forward(self, item_seq):
        x = self.linear(self.embedding(item_seq))

        return x


class TopK(nn.Module):
    def __init__(self, model, topk_method):
        super(TopK, self).__init__()
        self.topk_method = topk_method
        self.model = model

    def forward(self, x):
        if self.topk_method == 'mm+topk':
            out = x @ self.model.embedding.weight.transpose(0, 1)
        else:
            out = x

        return out


n_items = 10_000
embedding_size = 32
t = 50
topk_method = 'mm+topk'
net = TopK(Net(n_items, embedding_size), topk_method)
item_seq = torch.randint(0, n_items, size=(1, t))
out = net(item_seq)

#%%
for i in range(n_iters):
    t0 = time.perf_counter()
    random_ints = torch.randint(high=n_items, size=(1, n_items))
    topk = torch.topk(random_ints ,k=21)
    t1 = time.perf_counter()
    t += (t1 - t0)

print(f"Time is {t * 1000 / n_iters:.2f}ms")
#%%
import faiss
import faiss.contrib.torch_utils
import torch
C, d = 1_000_000, 64
nq = 1000
xb = torch.rand((C, d))
xq = torch.rand((nq, d))
index = faiss.index_factory(d, "IVF16384,Flat", faiss.METRIC_INNER_PRODUCT)    
res = faiss.StandardGpuResources()
index_ivf = faiss.extract_index_ivf(index)
clustering_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(index_ivf.d))
index_ivf.clustering_index = clustering_index
index.train(xb)
index.set_direct_map_type(faiss.DirectMap.Hashtable)
index = faiss.index_cpu_to_gpu(res, 0, index)
index.add_with_ids(xb, torch.arange(C))
index = faiss.index_gpu_to_cpu(index)
# index.set_direct_map_type(faiss.DirectMap.Hashtable)
index.reconstruct(0)
#%%
import time
batch_size = 1
sequence_length = 50
n_iters = 1000
item_seq = torch.randint(high=C, size=(batch_size, sequence_length))
embedding = torch.nn.Embedding(C, d)
embedding.weight.data = xb

t_loop, t_emb = 0, 0
for i in range(n_iters):
    t0 = time.perf_counter()
    result = torch.zeros((batch_size, sequence_length, d))
    for batch in range(batch_size):
        for item in range(sequence_length):
            result[batch, item] = index.reconstruct(item_seq[batch, item].item())

    t1 = time.perf_counter()
    t_loop += (t1 - t0)
    t0 = time.perf_counter()
    result_emb = embedding(item_seq)
    t1 = time.perf_counter()
    t_emb += (t1 - t0)
    assert torch.allclose(result, result_emb)



print(f"Time loop:      {t_loop * 1e6 / n_iters:.2f} micros")
print(f"Time embedding:  {t_emb * 1e6 / n_iters:.2f} micros")
#%% Simple K-Means
import numpy as np
from numba import njit
k = 10
n_vectors = 10_000_000
d = 64
seed = 1
n_iters = 1000
max_abs_error = 1e-3
rng = np.random.default_rng(seed=seed)
X = rng.random((n_vectors, d), dtype=np.float32)
#%%
# Randomly assign points to a cluster and initialize clusters by taking mean of values
@njit(fastmath=True)
def kmeans(X, k, seed=0):
    d = X.shape[1]
    seeded_rng = np.random.seed(seed)
    labels = np.random.randint(0, k, size=(n_vectors))
    clusters = np.zeros((d, k), dtype=X.dtype)
    for cluster in range(k):
        index = labels == cluster
        Xcluster = X[index]
        clusters[:, cluster] = np.sum(Xcluster, axis=0) / len(Xcluster)

    # K-means
    clusters_new = clusters
    for i in range(n_iters):
        distances = np.sum((np.expand_dims(X, -1) - np.expand_dims(clusters, 0))**2, axis=1)
        labels = np.argmin(distances, axis=1)
        for cluster in range(k):
            index = labels == cluster
            Xcluster = X[index]
            clusters_new[:, cluster] = np.sum(Xcluster, axis=0) / len(Xcluster)
        abs_difference = np.abs(clusters_new - clusters)
        # Check
        # print(f"Iter: {i}, diff: {np.sum(abs_difference)}")
        # Early stopping
        if np.all(abs_difference <= max_abs_error):
            break
        # Assign to clusters
        clusters = clusters_new
    
    return clusters, labels
#%%
clusters, labels = kmeans(X, k, seed=seed)
#%% 
import numpy as np
seed = 0
rng = np.random.default_rng(seed)
a = -10**6
b = 10**6
m = rng.integers(1, 10)
n = rng.integers(1, 10)
arr1 = np.sort((b - a) * rng.random(m) + a)
arr2 = np.sort((b - a) * rng.random(n) + a)

gt = np.median(np.sort(np.concatenate((arr1, arr2))))
med1 = np.median(arr1)
med2 = np.median(arr2)
I1 = np.searchsorted(arr1, med2)
I2 = np.searchsorted(arr2, med1)
#%%
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        integer = 0
        i = 0
        while i < len(s):
            if s[i] == 'I':
                if i + 1 < len(s):
                    if s[i + 1] == 'V':
                        integer += 4
                        i += 2
                    elif s[i + 1] == 'X':
                        integer += 9
                        i += 2
                    else:
                        integer += 1
                        i += 1
                else:
                    integer += 1
                    i += 1                
            elif s[i] == 'V':
                integer += 5
                i += 1

            elif s[i] == 'X':
                if i + 1 < len(s):
                    if s[i + 1] == 'L':
                        integer += 40
                        i += 2
                    elif s[i + 1] == 'C':
                        integer += 90
                        i += 2
                    else:
                        integer += 10
                        i += 1
                else:
                    integer += 10
                    i += 1     

            elif s[i] == 'L':
                integer += 50
                i += 1

            elif s[i] == 'C':
                if i + 1 < len(s):
                    if s[i + 1] == 'D':
                        integer += 400
                        i += 2
                    elif s[i + 1] == 'M':
                        integer += 900
                        i += 2
                    else:
                        integer += 100
                        i += 1
                else:
                    integer += 100
                    i += 1  

            elif s[i] == 'D':
                integer += 500                    
                i += 1

            elif s[i] == 'M':
                integer += 1000
                i += 1


        return integer   
    
roman = 'IV'
test = Solution().romanToInt(roman)
#%%
nums1 = [1, 2]
nums2 = [3, 4]
total = nums1 + nums2

total.sort()
total_length = len(nums1) + len(nums2)
median = (total[total_length // 2 - 1] + total[int(total_length / 2)]) / 2
#%%
dividend = 10
divisor = 2

quotient = dividend
i = 1
while i < divisor:
    quotient >>= 1
    i <<= 1