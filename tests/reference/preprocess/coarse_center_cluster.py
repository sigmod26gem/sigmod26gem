import faiss
import numpy as np
import torch
from tqdm import tqdm

dataset_name = "msmarco"
file_num = 354
target_cluster_num = 40960

code_dir = f"gem_data/{dataset_name}/docdata"
center_dir = f"gem_data/{dataset_name}/cdata"
tmp_dir = f"gem_data/{dataset_name}/tmp"

doc_codes = []
doc_lens = []
for i in tqdm(range(file_num)):
    doc_codes.append(np.load(f'{code_dir}/doc_codes_{i}.npy'))
    doc_lens.append(np.load(f'{code_dir}/doclens{i}.npy'))
doc_codes = np.concatenate(doc_codes, axis=0)
doc_lens = np.concatenate(doc_lens, axis=0)


pre_off = 0
cluster2doc = {}
for i in tqdm(range(len(doc_lens))):
    cur_doc_code = doc_codes[pre_off:pre_off + doc_lens[i]]
    cur_doc_code = set(cur_doc_code.tolist())
    for c in cur_doc_code:
        if c not in cluster2doc:
            cluster2doc[c] = []
        cluster2doc[c].append(i)
    pre_off += doc_lens[i]
print(len(cluster2doc))
original_clusters = cluster2doc


data = np.load(f'{center_dir}/centroids.npy').astype(np.float32)
num_vectors = data.shape[0]
vector_dim = data.shape[1]

res = faiss.StandardGpuResources()  
gpu_kmeans = faiss.Kmeans(vector_dim, target_cluster_num, niter=1000, gpu=True, spherical=True, verbose=True)

gpu_kmeans.train(data)
distances, labels = gpu_kmeans.index.search(data, 1)  
centroids = gpu_kmeans.centroids  


label2cluster = {}
for i in range(labels.shape[0]):
    if labels[i][0] not in label2cluster:
        label2cluster[labels[i][0]] = [i]
    else:
        label2cluster[labels[i][0]].append(i)

label2doc = {}
for i in range(len(label2cluster)):
    if i not in label2doc:
        label2doc[i] = []
    for j in label2cluster[i]: 
        label2doc[i].extend(original_clusters[j])
    label2doc[i] = sorted(list(set(label2doc[i])))

write_datas = []
for i in range(len(label2doc)):
    write_datas.append(' '.join([str(_) for _ in label2doc[i]]) + '\n')

with open(f'{tmp_dir}/init_cluster_info.txt', 'w') as f:
    f.writelines(write_datas)
np.save(f'{center_dir}/coarse_centroids.npy', centroids)
np.save(f'{tmp_dir}/coarse_cluster_labels.npy', labels)




