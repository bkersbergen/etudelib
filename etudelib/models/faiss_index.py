import torch
import faiss
import faiss.contrib.torch_utils
import multiprocessing
from pathlib import Path
#%% 
class faiss_index(object):
    def __init__(self, embedding_matrix, faiss_index_dir):
        super(faiss_index, self).__init__()
        # Sources:
        #  - https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors
        #  - https://github.com/facebookresearch/faiss/blob/main/benchs/bench_hnsw.py
        self.C, self.d = embedding_matrix.shape[0], embedding_matrix.shape[1]
        filename = f"faiss_C{self.C}_d{self.d}.index"
        current_path = Path(__file__).parent
        filepath = current_path.joinpath(faiss_index_dir).joinpath(filename)
        if filepath.exists():
            print(" Index exists, loading from file")
            full_index = faiss.read_index(str(filepath))
            assert full_index.d == self.d
            assert full_index.ntotal == self.C
        else:
            print(" Index does not exist, training...")
            n_threads_max = multiprocessing.cpu_count()
            faiss.omp_set_num_threads(n_threads_max)
            # Get index config and train index
            index = self.get_index()
            full_index = self.train_index(index, embedding_matrix)          
            # Save
            faiss.write_index(full_index, str(filepath))
        
        faiss.omp_set_num_threads(1)      
        self.index = full_index

    def send_to_device(self, device):
        if device == 'cuda':
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        if self.C > 100_000:
            self.index.nprobe = 4096
            if device == 'cuda':
                self.index.nprobe = 2048

    def search(self, query, k):
        return self.index.search(query, k)

    def get_index(self):
        if self.C <= 100_000:
            index = faiss.IndexFlatIP(self.d)
        elif self.C > 100_000:    
            if self.C <= 1_000_000 and self.C > 100_000:
                index = faiss.index_factory(self.d, "IVF16384,Flat", faiss.METRIC_INNER_PRODUCT)    
            elif self.C <= 10_000_000 and self.C > 1_000_000:
                index = faiss.index_factory(self.d, "IVF65536,Flat", faiss.METRIC_INNER_PRODUCT)    
            else:
                index = faiss.index_factory(self.d, "IVF262144,Flat", faiss.METRIC_INNER_PRODUCT)    

        return index

    def train_index(self, index, embedding_matrix):
        if self.C <= 100_000:
            index.add(embedding_matrix)
            full_index = index
        else:
            n_gpus = torch.cuda.device_count()
            if n_gpus > 0:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                # Extract IVF and cluster
                index_ivf = faiss.extract_index_ivf(index)
                index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
                clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(index_ivf.d), co=co)
                index_ivf.clustering_index = clustering_index
                index.train(embedding_matrix)
                # Make sure index is on CPU
                index = faiss.index_gpu_to_cpu(index)
                # Retrieve minimum free GPU memory to determine n_batches
                free_mem = float('inf')
                for i in range(torch.cuda.device_count()):
                    free_mem = min(free_mem, torch.cuda.mem_get_info(i)[0])
                free_mem *= n_gpus
                # IVF: bytes / vector = 4*d+8. C datapoints. Twice necessary (for index + data). 20% buffer
                index_size = (4 * self.d + 8) * self.C * 2 * 1.2
                n_shards = int((index_size + free_mem - 1) // free_mem)
                print(f"Number of index shards: {n_shards}")
                for shard in range(n_shards): 
                    # Clone index
                    index_shard = faiss.clone_index(index)
                    i0 = self.C * shard // n_shards
                    i1 = self.C * (shard + 1) // n_shards
                    # Send to GPU and add datapoints
                    index_shard = faiss.index_cpu_to_all_gpus(index_shard, co=co)
                    index_shard.add_with_ids(embedding_matrix[i0:i1], torch.arange(i0, i1))
                    # Send to CPU and merge
                    index_shard = faiss.index_gpu_to_cpu(index_shard)
                    if shard == 0:
                        full_index = faiss.clone_index(index_shard)
                    else:
                        faiss.merge_into(full_index, index_shard, False)                
            else:
                index.train(embedding_matrix)
                index.add_with_ids(embedding_matrix, torch.arange(self.C))
                full_index = index
        
        return full_index
