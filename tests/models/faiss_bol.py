import os
import time
import faiss
import numpy as np
import function_utils
import gc

# Follow the setting from: https://gist.github.com/mdouze/46d6bbbaabca0b9778fca37ed2bcccf6
# with which only the IVF (clustering) step is done via GPU, the rest is done via CPU
def train_index_gpu(index, embeddings, num_feats=128):
    print('Extracting IVF index', flush=True)
    index_ivf = faiss.extract_index_ivf(index)
    print('Extracted IVF index', flush=True)
    print('Converting index cpu to all gpu with Flat L2 and no of features is {}'.format(num_feats), flush=True)
    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(num_feats))
    print('Converted index cpu to all gpu', flush=True)
    index_ivf.clustering_index = clustering_index
    print('Training data of size {} with GPUs'.format(embeddings.shape), flush=True)
    # training with GPU
    index.train(embeddings)
    print('Finished Training with GPUs', flush=True)


# Reference: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
def get_index_config(nsamples, using_gpu=False):
    if nsamples < 10000:
        index_config = {'criteria': 0, 'num_clusters_IVF': 0, 'num_neighbor_HNSW': 0}
    elif nsamples < 1000000:
        index_config = {'criteria': 1, 'num_clusters_IVF': int(2 * np.sqrt(nsamples)), 'num_neighbor_HNSW': 128}
    elif nsamples < 10000000:
        if using_gpu is True:
            index_config = {'criteria': 1, 'num_clusters_IVF': 32768, 'num_neighbor_HNSW': 128}
        else:
            # CPU cannot handle clustering with a huge number of clusters
            index_config = {'criteria': 1, 'num_clusters_IVF': int(2 * np.sqrt(nsamples)), 'num_neighbor_HNSW': 128}
    else:
        index_config = {'criteria': 3, 'num_clusters_IVF': 262144, 'num_neighbor_HNSW': 128}
    return index_config


def build_index_faiss(embeddings, global_id_list, indexing_criteria=1, noriginal_feats=512,
                      nreduced_feats=128, nclusters_IVF=262144, nneighbor_HNSW=32, using_gpu=False):
    start_time = time.time()
    nsamples = embeddings.shape[0]

    if indexing_criteria == 0:
        # Use the faiss default option with IndexFlatL2
        indexing_type = "IDMap,Flat"
        index = faiss.index_factory(noriginal_feats, indexing_type)
    elif indexing_criteria == 1:
        indexing_type = 'IVF' + str(nclusters_IVF) + '_HNSW' + str(nneighbor_HNSW) + ',Flat'
        print('Indexing using {} indexing type with {} original features  for a faster search'.format(indexing_type, noriginal_feats), flush=True)
        index = faiss.index_factory(noriginal_feats, indexing_type)

        if using_gpu is False:
            print("Using CPU option for small and medium categories", flush=True)
            index.train(embeddings)
        else:
            print("Training using GPU", flush=True)
            train_index_gpu(index, embeddings, noriginal_feats)

            # For associating between embeddings and global_ids
            index_ivf = faiss.extract_index_ivf(index)
            index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
    elif indexing_criteria == 2:
        indexing_type = 'IVF' + str(nclusters_IVF) + '_HNSW' + str(nneighbor_HNSW) + ',PQ64'
        print('Indexing using {} indexing type with {} original features  for a faster search'.format(indexing_type, noriginal_feats), flush=True)
        index = faiss.index_factory(noriginal_feats, indexing_type)
        train_index_gpu(index, embeddings, noriginal_feats)

        # For associating between embeddings and global_ids
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
    elif indexing_criteria == 3:
        indexing_type = 'PCA' + str(nreduced_feats) + ',IVF' + \
                        str(nclusters_IVF) + '_HNSW' + str(nneighbor_HNSW) + ',Flat'
        print('Indexing using {} indexing type with {} original features  for a faster search'.format(indexing_type, noriginal_feats),
              flush=True)
        index = faiss.index_factory(noriginal_feats, indexing_type)
        train_index_gpu(index, embeddings, nreduced_feats)

        # For associating between embeddings and global_ids
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
    else:
        # Use the faiss default option with IndexFlatL2
        indexing_type = "IDMap,Flat"
        index = faiss.index_factory(noriginal_feats, indexing_type)

    #print('Adding vectors to index', flush=True)
    index.add_with_ids(embeddings, global_id_list)
    #index.add(embeddings)

    #print('Finished adding vectors to index', flush=True)
    #print('ntotal value of index: {}'.format(index.ntotal), flush=True)
    #print('Number of samples {} and number of features {}'.format(nsamples, noriginal_feats), flush=True)
    print('Time needed to process with IVF index option is {} seconds'.format(time.time() - start_time), flush=True)
    return index


def indexing_per_category(all_embeddings, all_global_id_list, ind, category_order, config,
                                  num_original_feats=512, num_reduced_feats=128, nclusters_IVF=262144,
                                  nneighbor_HNSW=32, num_probes=16, efSearch=64, kneighbor=80, csv_batch_size=5000,
                                  using_gpu=False):

    embeddings = all_embeddings[ind]
    global_id_list = all_global_id_list[ind]

    index = build_index_faiss(embeddings, global_id_list, config,
                              num_original_feats, num_reduced_feats, nclusters_IVF, nneighbor_HNSW, using_gpu)

    print('Start searching on index with batch size {}, probe {}, and efSearch {}'.format(csv_batch_size,
                                                                                          num_probes, efSearch), flush=True)

    params, dummy_1, dummy_2, num_wrong_trigger_ids = create_recommendation_parallel(index, embeddings, csv_batch_size,
                                                                                     kneighbor, num_probes, efSearch,
                                                                                     global_id_list, category_order, using_gpu)

    print('Finish searching on index', flush=True)
    del index; del embeddings; gc.collect()  # memory clean up
    return params, num_wrong_trigger_ids

def search_similarity_for_ids(index, ids_of_embeddings, kneighbor=200, nprobe=1):
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
    faiss.ParameterSpace().set_index_parameter(index, "efSearch",64)
    max_dist = 0
    print(index.nprobe, flush=True)

    params = []
    embeddings_to_search = find_embeddings_for_ids(index, ids_of_embeddings)
    print(embeddings_to_search)
    D, I = index.search(embeddings_to_search, kneighbor)
    max_dist = max(max_dist, np.amax(D))

    csv_name = 'reco_set_chunk_nprobes_{}_{}_in_{}.csv'.format(nprobe, 1, 1)
    params.append((D, I, csv_name))
    return params, max_dist

def find_embeddings_for_ids(index, ids_of_embeddings):
    return np.asarray(list(map(lambda i: find_embedding_for_id(index, i), ids_of_embeddings)))

def find_embedding_for_id(index, id_of_embedding):
    embedding = index.reconstruct(int(id_of_embedding))
    # print(id_of_embedding, flush=True)
    # print(embedding, flush=True)
    return embedding

def create_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def detect_wrong_trigger_ids(true_trigger_ids, estimated_trigger_ids):
    if(len(true_trigger_ids) != len(estimated_trigger_ids)):
        print('Lengths of the two vectors are not equal', flush=True)
        return None

    ids = np.where(true_trigger_ids != estimated_trigger_ids)[0]
    wrong_trigger_ids = np.stack((true_trigger_ids[ids], estimated_trigger_ids[ids]), axis=1)
    return wrong_trigger_ids

def insert_true_trigger_ids(D, I , true_trigger_ids):
    d = D[:, 0]
    d[:] = 0.0
    D = np.insert(D, 0, d, axis=1)
    I = np.insert(I, 0, true_trigger_ids, axis=1)
    return D, I

def create_recommendation_parallel(index, embeddings, csv_batch_size=5000, kneighbor=200, nprobes=1, efSearch=1, list_global_id=[], category_order=0, using_gpu=False):
    max_dist = 0
    num_selected_wrong_trigger_ids = 5

    # Setting parameters in faiss for querying in case the indexing is not flat indexing type
    if not type(index) == faiss.swigfaiss.IndexIDMap:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobes)
        faiss.ParameterSpace().set_index_parameter(index, "efSearch", efSearch)

    params = []
    nsamples = embeddings.shape[0]
    all_chunks = list(range(nsamples))
    chunks = list(create_chunks(all_chunks, csv_batch_size))
    nchunks = len(chunks)
    #print('Number of chunks is {}'.format(nchunks), flush=True)
    #print('Searching similar items for all triggers', flush=True)

    for chunk_id, chunk in enumerate(chunks):
        D, I = index.search(embeddings[chunk], kneighbor)
        max_dist = max(max_dist, np.amax(D))  # Find the max value of distance matrix D
        if using_gpu is False:
            csv_name = 'reco_set_cpu_chunk_{}_in_{}_category_{}.csv'.format(chunk_id, nchunks, category_order)
        else:
            csv_name = 'reco_set_gpu_chunk_{}_in_{}_category_{}.csv'.format(chunk_id, nchunks, category_order)

        true_trigger_ids = list_global_id[chunk]
        estimated_trigger_ids = I[:, 0]
        if chunk_id == 0:
            wrong_trigger_ids_examples = detect_wrong_trigger_ids(true_trigger_ids, estimated_trigger_ids)
            num_wrong_trigger_ids = len(wrong_trigger_ids_examples)
        else:
            wrong_trigger_ids = detect_wrong_trigger_ids(true_trigger_ids, estimated_trigger_ids)
            wrong_trigger_ids_examples = np.concatenate((wrong_trigger_ids_examples,
                                                     wrong_trigger_ids[:num_selected_wrong_trigger_ids, :]), axis=0)
            num_wrong_trigger_ids += len(wrong_trigger_ids)

        D, I = insert_true_trigger_ids(D, I, true_trigger_ids)

        params.append((D, I, csv_name))

    #print('Searched similar items for all triggers...Done', flush=True)

    return params, max_dist, wrong_trigger_ids_examples, num_wrong_trigger_ids

def save_index_faiss(index, project_bucket_dir, indexing_dir, indexing_name, gpu=False):
    if gpu:
        print('Converting index from gpu to cpu..', flush=True)
        cpu_index = faiss.index_gpu_to_cpu(index)
        print('Finished converting index from gpu to cpu..', flush=True)
    else:
        cpu_index = index

    print('Start saving index to disk...', flush=True)
    start_time = time.time()

    faiss.write_index(cpu_index, indexing_name)
    print('Index is saved to disk. Took {} seconds'.format(time.time() - start_time), flush=True)

    print("Index file size: {} MB".format(
        round(os.path.getsize(indexing_name) / float(1024 ** 2), 2)), flush=True)

    print("List of files in the current directory:", flush=True)
    print(os.listdir())

    function_utils.upload_file(project_bucket_dir, indexing_dir, indexing_name)

    # delete indexing file from the local machine
    function_utils.delete_file(indexing_name)

def train_on_subset_and_add_in_batches(dataset, indexing_criteria=2, nlist=100, m=8):
    """
    Trains on a subset of the record set. The current implementation trains on the first batch
    and adds the remaining records in batches.
    This approach consumes less memory than loading the whole recordset and training/addining them

    Args:
    - dataset: batch generator of embeddings
    - indexing_criteria: indexing_criteria is recommended for large samples
    - nlist: number of clusters (coarse quantization)
    - m: number of subvectors
    Returns:
     the final index
    """

    global num_feats
    global embedding_model_name
    start_time = time.time()

    # using multiple gpus
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True

    nsamples = 0
    quantizer = faiss.IndexHNSWFlat(num_feats, 32) #faiss.IndexFlatL2(num_feats) #faiss.IndexHNSWFlat(d, 32)
    if indexing_criteria == 1:
        # Faster search
        index = faiss.IndexIVFFlat(quantizer, num_feats, nlist)
        index = faiss.index_cpu_to_all_gpus(index, co)

        e_train = next(dataset)[embedding_model_name] # one batch has at least 2**24 samples

        index.train(e_train)
        index.add(e_train)
        nsamples += e_train.shape[0]
        del e_train # free memory
    elif indexing_criteria == 2:
        # Consume lower memory. # 8 specifies that each sub-vector is encoded as 8 bits
        index = faiss.IndexIVFPQ(quantizer, num_feats, nlist, m, 8)
        index = faiss.index_cpu_to_all_gpus(index, co)

        # train index with the first batch (this is bad if first batch is not a random sample set)
        # if N batches are needed for training, then we should be sure that N*batch_size*num_feats*4 bytes fits in RAM
        # ge_train = tuple([next(dataset)[embedding_model_name] for _ in range(N)])
        # e_train = np.concatenate(ge_train, axis=0)
        e_train = next(dataset)[embedding_model_name] # one batch has at least 2**24 samples
        index.train(e_train)
        index.add(e_train)
        nsamples += e_train.shape[0]
        del e_train # free memory
    else:
        # Use the faiss default option with IndexFlatL2
        index = quantizer

    # incrementally adds vectors in batches
    for batch in dataset:
        e_b = batch[embedding_model_name]
        index.add(e_b)
        nsamples += e_b.shape[0]

    print('ntotal value of index: {}'.format(index.ntotal), flush=True)
    print('Number of samples {} and number of features {}'.format(nsamples, num_feats), flush=True)
    print('Time needed to process with IVF index option is {} seconds'.format(time.time() - start_time), flush=True)
    return index
