# Code Organization

```text
include/gem/      Public API, data views, options, results, workspace handle
src/index.cpp    API orchestration and corpus ownership
src/encoding/    Encoded-corpus and cluster-assignment validation
src/build/       Cluster insertion and graph finalization
src/search/      Routing, fine scoring, traversal, reranking, query scratch
src/distance/    FP32 dot products, MaxSim, qEMD, original distance helpers
src/graph/       GEM-modified HNSW kernel and its private adapter
src/io/          NPY loading and graph serialization
apps/            INI parsing, executable, batch evaluation
python/          Coarse-codebook training and TF-IDF document assignment
configs/         Build/search and preprocessing configurations
tests/           Functional, concurrency, and equivalence tests
tests/reference/ Original examples and experimental sources
docs/            API, configuration, architecture, validation
```

## Read the Main Paths

- Search: `src/index.cpp` calls `search/routing.cpp`, `search/traversal.cpp`, and `search/rerank.cpp`. Numerical kernels live in `distance/`.
- Build: `build/cluster_graph.cpp` prepares centroid distances and inserts each cluster's documents using qEMD. `build/finalize.cpp` contains the original reachability repair.
- Storage: `io/numpy.cpp` reads embeddings and codes. `io/graph_index.cpp` loads or saves the graph and rebinds its descriptors.
- Encoding: `python/gem_preprocess/pipeline.py` trains coarse centroids from fine centroids. `assignment.py` generates memberships using the original TF-IDF formula. `src/encoding/` validates these inputs when the library takes ownership.

## Ownership and Dependencies

Index owns EncodedCorpus and the private Graph adapter. Graph owns the HNSW allocation and descriptors; the descriptors borrow Index-owned arrays. QueryWorkspace owns mutable query scratch. SearchOptions is supplied per call, including ef and nprobe.

Public headers expose no HNSW, Eigen, BLAS, or transport-solver types. Runtime sources do not include test or reference sources. The reference executable is a test target and is excluded when `BUILD_TESTING=OFF`.

`src/graph/hnswalg.h` holds the HNSW template's state and lifecycle. Its method bodies are grouped in `src/graph/detail/*.inl`: layer and cluster searches, multi-entry search, neighbor selection, edge operations, serialization, labels, insertion/update, query entry points, and repair. These fragments are included inside the template class; their method order and bodies are retained. This is separate from the compiled `Graph` adapter in `graph/index.cpp`.

`distance/otlib` and `io/cnpy` contain the upstream numerical and NPY dependencies. Their original source attribution is retained. HNSW's license is in `src/graph/LICENSE`.

## Original Entry

The root README is unchanged. `hnswlib/CMakeLists.txt` is a small compatibility entry for its build command, forwarding to the root CMake project and enabling `example_vecset_search_gem`. It contains no index implementation. That executable uses the original example in `tests/reference/hnswlib/examples/cpp/`, linked against the relocated kernel.

Other historical examples, Python bindings, and the glass experiments are retained under `tests/reference/` for source comparison. Their old nested build and packaging files are not supported entry points. Use the root CMake project and `gem_run` for the library path.
