import numpy as np
from tqdm import tqdm


dataset_name = "msmarco"
file_num = 354
coarse_cluster_num = 40960

code_dir = f"gem_data/{dataset_name}/docdata"
center_dir = f"gem_data/{dataset_name}/cdata"
tmp_dir = f"gem_data/{dataset_name}/tmp"

cluster_num = coarse_cluster_num
cluster2docs = {}
with open(f'{tmp_dir}/init_cluster_info.txt', 'r') as f:
    for i, line in enumerate(f):
        cluster2docs[i] = {}
        all_data = line.strip().split()
        for _ in all_data:
            cluster2docs[i][int(_)] = 1
cluster_labels = np.load(f'{tmp_dir}/coarse_cluster_labels.npy')

doc_codes = []
doc_lens = []
for i in tqdm(range(file_num)):
    doc_codes.append(np.load(f'{code_dir}/doc_codes_{i}.npy'))
    doc_lens.append(np.load(f'{code_dir}/doclens{i}.npy'))
doc_codes = np.concatenate(doc_codes, axis=0)
doc_lens = np.concatenate(doc_lens, axis=0)
new_doc_codes = cluster_labels[doc_codes].squeeze()

def top_k_tfidf(doc, word_count, k):
    doc_id = [_[0] for _ in doc]
    weights = [_[1] for _ in doc]
    unique_words, counts = np.unique(doc_id, return_counts=True)
    tf = counts / len(doc_id)
    total_docs = len(doc_lens)
    idf = np.array([np.log(total_docs / (word_count[w])) for w in unique_words])
    tfidf = tf * idf
    topk_indices = np.argsort(-tfidf)[:k]
    return unique_words[topk_indices], tfidf[topk_indices], tf[topk_indices], idf[topk_indices]
def top_k_tfidf_now(doc, word_count, k):
    doc_id = doc
    unique_words, counts = np.unique(doc_id, return_counts=True)
    tf = counts / len(doc_id)
    total_docs = len(doc_lens)
    idf = np.array([np.log(total_docs / (word_count[w])) for w in unique_words])
    tfidf = tf * idf
    topk_indices = np.argsort(-tfidf)[:k]
    return unique_words[topk_indices], tfidf[topk_indices], tf[topk_indices], idf[topk_indices]

cluster_count = [len(cluster2docs[_]) for _ in cluster2docs]

pre = 0
tfidf_topk = 20
new_cluster2doc = {}
for di in tqdm(range(len(doc_lens))):
    new_codes = new_doc_codes[pre: pre + doc_lens[di]]
    filterlist, tfidf_score, tf_score, idf_score = top_k_tfidf_now(new_codes, cluster_count, tfidf_topk)
    filterlist = filterlist.tolist()
    # if use_adaptive:
    #   xfeatures = [len(new_codes)]
    #   for i in range(10):
    #       if i < len(filterlist):
    #           xfeatures.extend([round(tfidf_score[i], 4), len(cluster2docs[filterlist[i]])])
    #       else:
    #           xfeatures.extend([0, 0])
    #   X = np.array([xfeatures])
    #   y_pred = classifier.predict(X)
    #   filterlist = filterlist[:y_pred[0]]
    for cid in filterlist:
        if cid not in new_cluster2doc:
            new_cluster2doc[cid] = []
        new_cluster2doc[cid].append(di)
    pre += doc_lens[di]
    
for i in range(cluster_num):
    if i not in new_cluster2doc:
        new_cluster2doc[i] = [new_cluster2doc[i]]
print(len(new_cluster2doc))


write_datas = []
for i in range(cluster_num):
    write_datas.append(' '.join([str(_) for _ in sorted(new_cluster2doc[i], reverse=True)]) + '\n')
with open(f'{center_dir}/coarse_cluster_info.txt', 'w') as f:
    f.writelines(write_datas)