# GEM Library

## Build

Dependencies: C++17, CMake 3.18+, Eigen3, OpenBLAS, OpenMP, and zlib. AMX is not required.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
./build/gem_run configs/evqa.ini
```

`GEM_NATIVE_ARCH=ON` enables the current CPU's instruction set. The default uses the compiler's portable CPU target. Relative paths are resolved from the INI file's directory.

## API

Use `add_subdirectory` and link `gem::gem`. Provide normalized document vectors, token offsets, fine codes, fine and graph centroids, and document-to-cluster assignments.

```cpp
#include <gem/index.h>
#include <gem/io.h>

auto corpus = gem::load_corpus(files);
auto index = gem::Index::build(std::move(corpus), build_options);
index.repair();
index.save("graph.bin");

gem::QueryWorkspace workspace;
std::vector<gem::Result> results;
index.search(query, search_options, workspace, results);
```

`Index::load(graph_path, std::move(corpus))` loads an existing graph. Index owns the corpus. `VectorSetView` borrows the caller's query memory, which must remain valid until search returns.

Concurrent callers share a read-only Index and provide separate workspaces and result vectors. Build, load, and repair require exclusive access. Each workspace retains buffers for route scores, fine scores, masks, MaxSim maxima, and rerank pair scores. HNSW candidate queues remain query-local allocations.

`Result.distance = 1 - mean_i(max_j(dot(q_i, d_j)))`; smaller values are better. Graph scoring uses GEM's FP32 lookup table, followed by FP32 original-vector reranking.

Build preserves cluster order, per-document HNSW insertion, and qEMD transport weights including code multiplicities. The library builder currently executes serially. The transport solver is unchanged; its input matrix and histograms are reused.

## Preprocessing

```bash
python3 python/encode.py configs/encoding.ini
```

This starts with a fine codebook and per-token fine codes, trains the coarse codebook with Faiss, and performs GEM's TF-IDF assignment. It preserves the upstream normalized-centroid training and top-r selection formulas. NumPy and Faiss are required; GPU execution also requires a GPU-enabled Faiss installation. Fine-codebook training and token encoding are not implemented by these upstream scripts.

The output contains `coarse_centroids.npy`, `coarse_cluster_labels.npy`, `init_cluster_info.txt`, and `coarse_cluster_info.txt`. Set `data.graph_centroids` and `data.clusters` in the build/search INI to the generated files. Existing outputs are not overwritten.

## Index Files

The original HNSW binary format is retained. It stores platform-dependent descriptors and has no corpus fingerprint. Use the graph and encoded corpus from the same dataset version. Load validates document count and labels, then rebinds document addresses.

Load does not repair the graph. Use `run.action=repair` and a distinct `index.output` to repair an existing index. The build executable repairs before saving. Both actions reject existing output files.

## Tests

CTest covers distance kernels, qEMD strides and multiplicities, build/repair/save/load, concurrent queries with different search parameters, data loading, and strict configuration parsing. The Python assignment test is included when NumPy is available.

```bash
python3 tests/compare_evqa.py /absolute/build /absolute/evqa /absolute/results 128 /absolute/graph/0.bin
```

Use 0 instead of 128 to test all queries. The graph argument is optional and defaults to the bundled EVQA graph. That graph contains 51,472 documents; another public EVQA corpus has 51,462 documents and requires its matching graph.

The comparison runs the original search entry, the library with one worker, and the library with eight workers. Each uses one warmup and three measured passes. It compares every returned document, rank, and score, and checks for duplicate results. The reference entry requires nprobe=4 and 128-dimensional embeddings. Its original `main` is not executed.

Library QPS measures the whole query batch; mean latency measures each query's execution. Both exclude loading. See [ARCHITECTURE.md](ARCHITECTURE.md), [CONFIGURATION.md](CONFIGURATION.md), and [VALIDATION.md](VALIDATION.md).
