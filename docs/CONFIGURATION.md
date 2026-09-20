# Configuration

`gem_run` accepts one INI path. Unknown keys, duplicate keys, and invalid numbers are rejected. The C++ library accepts typed options directly and does not read environment variables or change global BLAS settings.

## Run and Data

| Key | Default | Meaning |
|---|---|---|
| `run.action` | `search` | `build`, `repair`, or `search` |
| `data.vectors` | Required | Document NPY path; `{shard}` is replaced by a file number |
| `data.lengths` | Required | Integer NPY containing tokens per document |
| `data.codes` | Required | Integer NPY containing fine codes per token |
| `data.shards` | 1 | Number of files, numbered from 0 |
| `data.fine_centroids` | Required | Fine centroid matrix |
| `data.graph_centroids` | Required | Graph centroid matrix |
| `data.clusters` | Required | One cluster per line; space-separated document IDs; empty lines allowed |
| `index.path` | Required | Graph input, or graph output for build |
| `index.output` | Required for repair | Separate repaired-graph output |
| `query.vectors` | Required for search | Query NPY |
| `query.lengths` | Required for 2D queries | Tokens per query; optional for fixed-length 3D queries |
| `query.qrels` | Empty | Two columns: zero-based query ID and document ID |
| `query.limit` | 0 | Query count; 0 uses all queries |

NPY files use version 1, little-endian values, and row-major order. Embeddings are FP16 or FP32; lengths and codes are 32-bit or 64-bit integers. Two-dimensional embeddings concatenate variable-length sets. Three-dimensional queries use `[query, token, dimension]`. Empty documents, invalid codes, and inconsistent dimensions or lengths are rejected.

## Build and Search

| Key | Default | Meaning |
|---|---:|---|
| `build.m` | 24 | HNSW neighbor parameter |
| `build.ef_construction` | 80 | Construction search width; at least m |
| `build.seed` | 100 | HNSW random seed |
| `build.distance_budget_bytes` | 4294967296 | Dense centroid matrix budget; size is `4 * fine_centers^2` bytes |
| `search.nprobe` | 4 | Graph clusters selected per query token |
| `search.ef` | 4000 | Graph search width |
| `search.rerank_k` | 512 | Original-vector rerank candidates |
| `search.k` | 100 | Returned documents |

Require `ef >= rerank_k >= k > 0` and `nprobe <= graph_clusters`. Dimensions and centroid counts come from the input arrays.

## Batch Evaluation

| Key | Default | Meaning |
|---|---:|---|
| `runtime.workers` | 1 | Concurrent query workers |
| `runtime.inner_threads` | 1 | OpenBLAS/OpenMP inner threads; normally 1 for concurrent queries |
| `benchmark.warmup` | 1 | Warmup passes over the query batch |
| `benchmark.repeats` | 3 | Measured passes |
| `benchmark.output` | Empty | Last-pass TSV: query, rank, document, distance |

Recall averages the fraction of relevant documents found per labeled query. Queries without labels are excluded. Without qrels, recall is null.

## Preprocessing

`python/encode.py` reads a separate `[encoding]` INI, illustrated in `configs/encoding.ini`.

| Key | Default | Meaning |
|---|---|---|
| `encoding.codes` | Required | Per-token fine-code NPY path, with optional `{shard}` |
| `encoding.lengths` | Required | Per-document token counts, with optional `{shard}` |
| `encoding.fine_centroids` | Required | Input fine centroid matrix |
| `encoding.output` | Required | Directory for coarse centroids, labels, and memberships |
| `encoding.shards` | 1 | Number of code/length file pairs |
| `encoding.graph_clusters` | 1024 | Coarse codebook size |
| `encoding.top_r` | 20 | Maximum memberships retained per document |
| `encoding.iterations` | 1000 | Faiss K-Means iterations |
| `encoding.gpu` | true | Use GPU-enabled Faiss; false selects Faiss CPU training |

Paths are relative to the configuration file. Cluster count, top-r, shard count, and iteration count must be positive. Output files must not already exist.
