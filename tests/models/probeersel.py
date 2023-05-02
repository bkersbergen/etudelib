#%%
import numpy as np
import faiss
faiss.get_num_gpus()
C, d = 1_000_000, 64
index = faiss.index_factory(d, "IVF16384,Flat")
xt = faiss.rand((C, d))
index_ivf = faiss.extract_index_ivf(index)
clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
index_ivf.clustering_index = clustering_index
index.train(xt)

index.add(xt)

