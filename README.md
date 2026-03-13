# GEM

This is the C++ implementation of paper "GEM: A Native Graph-based Index for Multi-Vector Retrieval".

## Dataset

**Marco_embedding**: 

Download datasets from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ytianbc_connect_ust_hk/EuaO6KmFlR5JpeWUUKLH6ccBEpFudB8yEJYBGnPRpX-G3g?e=zyZvsf).
Pls see https://microsoft.github.io/msmarco/Datasets for more details about datasets - MS MARCO (MicroSoft MAchine Reading COmprehension).
We follow https://github.com/ThirdAIResearch/Dessert to generate `doclensxx.npy` and `encodingx_float16.npy` (For end-to-end testing, we can skip these two intermediate files and directly make modifications on [colBERT/PLAID](https://github.com/stanford-futuredata/ColBERT)).

The dataset contains a total of 8.8 million passages. Their embeddings are stored across 354 separate `.npy` files, with each embedding having a dimension of 128. Since the number of embeddings per passage varies, we use `doclens` files to store the number of embeddings for each passage. This allows us to correctly split the corresponding `encoding` files.

For example, in `doclens1.npy`, there are 25,000 integers, such as `[63, 52, 63, ...]`. This means `encoding1_float16.npy` contains embeddings for 25,000 passages. The first 63 rows (63 * 128) correspond to passage 1, the next 52 rows correspond to passage 2, and so on.



**Lotte_embedding**: 

Download datasets from [here](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md).
Pls see https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md for more details about datasets - LoTTE.
We follow the similar procedure to produce embeddings for LoTTE dataset.

**OKVQA_embedding**: 

Download datasets from [here](https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md).
Pls see https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md for details about datasets - OKVQA.
We use [PreFLMR_ViT-L](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L) to generate embeddings for Queries and Documents.


**EVQA_embedding**: 

Download datasets from [here](https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md).
Pls see https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md for details about datasets - EVQA.
We use [PreFLMR_ViT-L](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L) to generate embeddings for Queries and Documents.

## Experiments

### How to run

```
cd hnswlib/
mkdir build
cd build
cmake ..
make
./example_vecset_search_gem
```

test on single cpu:

```
taskset -c 0 ./example_vecset_search_gem
```

please change the value of `int dataset` variable to test different datasets.

```
0 - msmarco
1 - lotte
2 - okvqa
3 - evqa
```

And put the data in `gem_data` folder for different datasets, like this:

```
gem_data
| -- msmarco
|    | -- qdata
|         | -- qembs.npy
|         | -- qrels.csv
|    | -- cdata
|         | -- centroids.npy
|         | -- coarse_cluster_info.txt
|         | -- coarse_centroids.npy
|    | -- docdata
|         | -- encoding_0_float16.npy
|         | -- doc_codes_0.npy
|         | -- doclens0.npy
|         | -- ....
| -- lotte
| ...
```

Here we give the example data and index results for evqa dataset for test (Except for the doc embedding and query embedding because they are too large).