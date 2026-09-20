# Validation

## Runtime Refactor, 2026-09-20

`60a2fae` introduces the library, query-owned workspaces, and INI entry. `da2a6b9` removes duplicate results caused by clusters sharing the same entry document.

Functional tests compare FP32 graph and rerank scores against the original Eigen expressions with a `1e-6` tolerance. They cover qEMD codebook strides and code multiplicities; small-corpus build/repair/save/load/search; eight concurrent workers with different nprobe, ef, and rerank_k; empty clusters; duplicate entries; invalid data; memory budgets; and strict configuration parsing.

These tests passed in native Release, portable Release, and AddressSanitizer/UndefinedBehaviorSanitizer Debug builds.

### EVQA Equivalence

The input is the authors' public 51,462-document EVQA corpus and its matching `index1024_all_24_80/0.bin`, with dimension 128, 32,768 fine centroids, and 1,024 graph clusters. Parameters are nprobe=4, ef=4000, rerank_k=512, and k=100. No additional graph repair is run after load.

The first 128 queries are run with one warmup and three measured passes. The reference invokes the original example's search method; the library invokes `Index::search`. Both use the branch's shared HNSW fixes.

| Check | Library, 1 worker | Library, 8 workers |
|---|---:|---:|
| Top-100 results compared | 12,800 | 12,800 |
| Document or rank mismatches | 0 | 0 |
| Maximum TSV score difference | 0 | 0 |
| Duplicate result documents | 0 | 0 |
| Recall@100 | 71.0938% | 71.0938% |

Before the shared-entry fix, both entries also matched, including six duplicate documents. The fixed version was checked separately.

### Initial Performance Check

The first refactor comparison used identical inputs, search options, and native compilation. Medians of three passes were:

| Entry | Query workers | QPS | Mean query latency |
|---|---:|---:|---:|
| Original entry | 1 | 4.43 | 225.85 ms |
| Library | 1 | 4.43 | 225.82 ms |
| Library | 8 | 29.21 | 267.91 ms |

Single-worker performance was unchanged within measurement variation. These measurements precede the shared-entry fix and have slightly different timing boundaries between the original and new harnesses. A separate construction job was active during subsequent validation; those timings are excluded from performance comparisons.

### Remaining Coverage

Real-data equivalence currently covers 128 queries. Full EVQA and isolated final-commit throughput measurements remain necessary before merge. Build has small-corpus functional coverage, without full-corpus build performance results. New centroid caches, scoring layouts, and SIMD kernels require separate distance, ranking, and performance comparisons.

Commands are in [LIBRARY.md](LIBRARY.md); configuration fields are in [CONFIGURATION.md](CONFIGURATION.md).

## Module Layout Refactor, 2026-09-20

The implementation is divided into encoding, build, search, distance, graph, and IO modules. Public types are split into standalone headers; `Index` retains ownership and orchestration. HNSW template methods are grouped into private implementation fragments. Expanding those fragments reproduces the preceding commit's method bodies, excluding whitespace and the C++17 inline declaration of the shared output mutex.

Validation after the move:

- Native Release: all three CTest tests passed.
- Portable Debug with AddressSanitizer and UndefinedBehaviorSanitizer: all three CTest tests passed, including leak detection.
- Portable Release: the library, `gem_run`, and the original README's example compiled through the compatibility CMake entry with `BUILD_TESTING=OFF`.
- EVQA: the same 128-query comparison passed for the reference entry and library with one and eight workers. Each run used one warmup and three measured passes; all returned 12,800 top-100 rows with matching document IDs, ranks, and output scores. Recall@100 remained 71.0938%.
- The library's single-worker TSV was also byte-identical to the result saved before the module move, avoiding reliance solely on the two entries sharing the relocated kernel.
- Python assignment tests matched the original TF-IDF formula and ordering for top-r values 1, 3, and 20. Full codebook training was not rerun.
- The root README remains byte-identical to upstream main.

Real-data validation still covers 128 queries; this change does not establish full-corpus build or throughput improvements.
