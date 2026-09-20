# 配置参数

`gem_run` 只接收一个 INI 路径。未知字段、重复字段和无效数值会报错。参数可直接通过 C++ 结构体传入 library，library 不读取环境变量或修改 BLAS 的全局线程设置。

## 运行与数据

| 字段 | 默认值 | 含义 |
|---|---|---|
| `run.action` | `search` | `build`、`repair` 或 `search` |
| `data.vectors` | 必填 | 文档向量 NPY 路径，可包含 `{shard}` |
| `data.lengths` | 必填 | 每篇文档的 token 数，整数 NPY |
| `data.codes` | 必填 | 每个 token 的 fine-code ID，整数 NPY |
| `data.shards` | 1 | 文件分片数，编号从 0 开始 |
| `data.fine_centroids` | 必填 | fine centroid 矩阵 |
| `data.graph_centroids` | 必填 | graph centroid 矩阵 |
| `data.clusters` | 必填 | 每行一个 cluster，空格分隔的文档编号；允许空行 |
| `index.path` | 必填 | 加载或构建保存的图文件 |
| `index.output` | repair 必填 | 修复图的独立输出路径 |
| `query.vectors` | search 必填 | 查询向量 NPY |
| `query.lengths` | 2D 查询必填 | 每条查询的 token 数；3D 等长查询可省略 |
| `query.qrels` | 空 | 两列文本：从 0 开始的 query_id、document_id |
| `query.limit` | 0 | 测试查询数，0 为全部 |

NPY 使用 v1、little-endian、row-major。向量为 FP16/FP32，lengths 和 codes 为 32/64-bit 整数。二维向量按各对象长度拼接；三维查询采用 `[query, token, dimension]`。空文档、越界 code、不匹配的维度或长度会报错。

## 构建与查询

| 字段 | 默认值 | 含义 |
|---|---|---|
| `build.m` | 24 | HNSW 邻居参数 |
| `build.ef_construction` | 80 | 构建搜索宽度，至少为 m |
| `build.seed` | 100 | HNSW 随机种子 |
| `build.distance_budget_bytes` | 4294967296 | dense centroid matrix 的容量预算，实际占用为 `4 * fine_centers^2` 字节 |
| `search.nprobe` | 4 | 每个 query token 选择的 graph clusters 数量 |
| `search.ef` | 4000 | 图搜索宽度 |
| `search.rerank_k` | 512 | 原始向量重排候选数 |
| `search.k` | 100 | 返回文档数 |

要求 `ef >= rerank_k >= k > 0`，`nprobe <= graph_clusters`。centroid 数量与维度从输入文件读取。

## 批量测试

| 字段 | 默认值 | 含义 |
|---|---|---|
| `runtime.workers` | 1 | 并发处理查询的线程数 |
| `runtime.inner_threads` | 1 | OpenBLAS/OpenMP 内层线程数；并发查询通常取 1 |
| `benchmark.warmup` | 1 | 全查询批次预热次数 |
| `benchmark.repeats` | 3 | 全查询批次测量次数 |
| `benchmark.output` | 空 | 最后一遍的结果 TSV：query、rank、document、distance |

Recall 是有标注查询上的平均召回比例，每条查询以其全部相关文档数为分母。没有 qrels 时输出 null。
