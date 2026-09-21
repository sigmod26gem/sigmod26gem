# GEM

This is the C++ implementation of "GEM: A Native Graph-based Index for Multi-Vector Retrieval". It provides a library and an INI-driven executable for index construction, search, and evaluation.

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

## Build

Requirements: a C++17 compiler, CMake 3.18 or newer, Eigen3, OpenBLAS with CBLAS headers, OpenMP, and zlib. On Debian/Ubuntu, the development packages are:

```bash
sudo apt-get install build-essential cmake libeigen3-dev libopenblas-dev zlib1g-dev
```

Run from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

Add `-DGEM_NATIVE_ARCH=ON` when configuring to optimize for the build machine's CPU. Use `-DBUILD_TESTING=OFF` to build only the library and application.

## Input Data

Set file paths in [configs/evqa.ini](configs/evqa.ini). Paths are relative to the INI file. `{shard}` expands to IDs from zero through `data.shards - 1`; filenames can follow any convention matching those patterns.

| Input | Format |
|---|---|
| `data.vectors` | FP16 or FP32 NPY, concatenated document vectors, shape `[tokens, dimension]` |
| `data.lengths` | Integer NPY, one vector count per document |
| `data.codes` | Integer NPY, one fine-centroid ID per document vector |
| `data.fine_centroids` | FP16 or FP32 NPY, shape `[fine_centroids, dimension]` |
| `data.graph_centroids` | FP16 or FP32 NPY, shape `[graph_clusters, dimension]` |
| `data.clusters` | Text, one line per graph cluster, containing space-separated document IDs; empty lines represent empty clusters |
| `query.vectors` | FP16 or FP32 NPY, concatenated query vectors |
| `query.lengths` | Integer NPY, one vector count per query |
| `query.qrels` | Optional text file with two columns: `query_id document_id` |

The loader accepts row-major, little-endian NPY v1/v2/v3 files and 32/64-bit integer arrays. Query lengths may be omitted for fixed-length arrays with shape `[queries, tokens_per_query, dimension]`. Document IDs follow document order across input shards; query IDs follow query order. Both start at zero. Each document must belong to at least one graph cluster.

Embeddings are loaded into FP32 memory. A saved graph must be loaded with the same corpus, codebooks, and document ordering used to construct it. The repository includes partial EVQA inputs and an example graph; obtain the remaining embeddings and codes before running a full search.

## Run

### Search and Evaluation

Edit the data and index paths in `configs/evqa.ini`, then run:

```bash
./build/gem_run configs/evqa.ini
```

The example uses these search settings:

```ini
[search]
nprobe = 4
ef = 4000
rerank_k = 512
k = 100

[runtime]
workers = 1
inner_threads = 1

[benchmark]
warmup = 1
repeats = 3
```

`nprobe` selects graph clusters per query vector, `ef` controls graph search breadth, `rerank_k` selects candidates for FP32 MaxSim reranking, and `k` is the result count. Require `ef >= rerank_k >= k > 0` and `nprobe <= graph_clusters`.

`runtime.workers` runs concurrent queries, each with its own workspace. `runtime.inner_threads` controls OpenMP and OpenBLAS parallelism; keep it at one when testing query-level concurrency. Set `query.limit` to a positive number for a subset, or zero for all queries.

Each measured pass reports QPS, mean query latency in milliseconds, and macro Recall@k averaged over queries with labels. Without `query.qrels`, recall is `null`. Loading is timed separately. An optional `benchmark.output` path writes TSV rows containing query ID, zero-based rank, document ID, and distance. Smaller distances rank first.

### Index Construction

Copy the configuration to keep search and build settings separate:

```bash
cp configs/evqa.ini configs/evqa-build.ini
mkdir -p indexes
```

In `configs/evqa-build.ini`, set:

```ini
[run]
action = build

[index]
path = ../indexes/evqa.bin

[build]
m = 24
ef_construction = 80
seed = 100
distance_budget_bytes = 4294967296
```

Then run:

```bash
./build/gem_run configs/evqa-build.ini
```

The build action constructs the graph, runs reachability repair, and saves it. It refuses to overwrite an existing output graph. Create the output directory beforehand. To query this graph, set `index.path` in the search configuration to the same file.

The builder uses the original cluster-wise HNSW insertion with qEMD distances. Its dense fine-centroid distance matrix needs `4 * fine_centroids^2` bytes, in addition to the corpus and graph. `build.distance_budget_bytes` limits that matrix alone; it is not a total process memory limit. Graph insertion is currently serial.

To repair an existing graph, use `run.action = repair`, keep `index.path` as the input, and specify a new `index.output` path. Search loads the saved graph without automatically repairing it.

Unknown configuration keys and duplicate keys are rejected. All runtime settings are read from the INI file.

## Preprocessing

[python/encode.py](python/encode.py) trains coarse centroids from existing fine centroids and generates document memberships using TF-IDF. It requires fine centroids, token fine codes, and document lengths as input; fine-codebook training and token encoding are upstream steps.

Install NumPy and Faiss in your Python environment, edit [configs/encoding.ini](configs/encoding.ini), then run:

```bash
python3 python/encode.py configs/encoding.ini
```

`graph_clusters` sets the coarse-codebook size, `top_r` controls document cluster assignment, and `iterations` sets the training iterations. The example uses `gpu = true`, which requires GPU-enabled Faiss; set `gpu = false` for CPU training.

The output directory receives `coarse_centroids.npy`, `coarse_cluster_labels.npy`, `init_cluster_info.txt`, and `coarse_cluster_info.txt`. Point `data.graph_centroids` and `data.clusters` in the build configuration to the generated files. Existing output files are rejected.

## C++ Library

Link against the CMake target `gem::gem` and include `<gem/index.h>`. `Index::build` and `Index::load` take ownership of an `EncodedCorpus`; `<gem/io.h>` provides the NPY loader. For example, given populated `gem::CorpusFiles files`:

```cpp
#include <gem/index.h>
#include <gem/io.h>

auto index = gem::Index::load("indexes/evqa.bin", gem::load_corpus(files));
auto queries = gem::load_queries("queries.npy", "query_lengths.npy");
gem::QueryWorkspace workspace;
gem::SearchOptions options;
std::vector<gem::Result> results;
index.search(queries.at(0), options, workspace, results);
```

For construction, use `Index::build(corpus, options)`, followed by `repair()` and `save(path)`. Concurrent searches share an index and use separate workspaces and result vectors. Build, load, repair, and destruction require exclusive access.

Workspaces retain score buffers, traversal heaps, and visited arrays between queries. Destroy a workspace to release its memory. The graph scorer uses a shared, first-occurrence-unique fine-code layout; qEMD construction keeps the original token multiplicities. FP32 scoring, entry order, and GEM's round-robin traversal remain unchanged.

`load_corpus` checks array formats and shard dimensions. `Index::build/load` validates corpus-wide invariants once before use; standalone consumers of `load_corpus` can call `EncodedCorpus::validate()` explicitly.

## Tests

CTest covers data/configuration validation, corrupt graph and NPY inputs, failed output writes, workspace reuse, distance kernels, build/save/load/search, and concurrent queries. The preprocessing test is enabled when Python and NumPy are available.

For a real-data comparison against the original GEM search entry, supply a complete EVQA corpus and its matching graph:

```bash
python3 tests/compare_evqa.py build /absolute/path/to/evqa \
    /absolute/path/to/results 128 /absolute/path/to/index/0.bin
```

This compares document IDs, ranks, and scores for the reference entry and library with one and eight workers. The reference uses dimension 128 and nprobe 4. Its graph filename must be `0.bin`; the library itself accepts any filename.

## Code Layout

```text
include/gem/   Public API, data views, options, results, query workspace
src/encoding/  Encoded-corpus and assignment validation
src/build/     Cluster graph construction and finalization
src/search/    Routing, traversal, reranking, query workspace
src/distance/  qEMD, MaxSim, numerical kernels
src/graph/     GEM-modified HNSW kernel
src/io/        Data loading and graph serialization
apps/          INI entry and batch evaluation
python/        Coarse clustering and document assignment
configs/       Example configurations
tests/         Unit tests and reference comparisons
```
