# GEM library

## 编译与运行

依赖 C++17、CMake、Eigen3、OpenBLAS、OpenMP 和 zlib。AMX 不在依赖中。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
./build/gem_run configs/evqa.ini
```

默认使用编译器的通用 CPU 目标。`-DGEM_NATIVE_ARCH=ON` 为当前 CPU 编译，生成的二进制需要相同的指令集支持。配置中的路径相对于 INI 文件所在目录解析。

## 代码入口

| 目录 | 职责 |
|---|---|
| `include/gem/index.h` | 数据类型、构建和查询参数、Index、QueryWorkspace |
| `include/gem/io.h` | NPY 输入与 cluster 文件加载 |
| `src/index.cpp` | 编码数据与原有 HNSW 内核之间的连接、构图、查询与补边 |
| `src/search/` | 查询 workspace、FP32 查表评分、原始向量 MaxSim |
| `src/build/` | qEMD cost matrix 与 histogram 缓冲区复用 |
| `src/io/` | 统一数据读取与格式检查 |
| `apps/` | INI 解析、运行入口、批量测试与 recall 计算 |
| `tests/` | 距离、构建、持久化、并发、配置与数据加载测试 |
| `hnswlib/` | 原有 GEM 图内核及依赖；旧 example 用于对照 |

## API

在 CMake 中添加本项目并链接 `gem::gem`。调用者提供归一化的向量、每篇文档的 token offsets、fine codes、两级 centroids 和 cluster membership。

```cpp
auto corpus = gem::load_corpus(files);
auto index = gem::Index::build(std::move(corpus), build_options);
index.repair();
index.save("graph.bin");

gem::QueryWorkspace workspace;
std::vector<gem::Result> results;
index.search(query, search_options, workspace, results);
```

`Index::load(graph_path, std::move(corpus))` 加载已有图并绑定文档数据。Index 持有 corpus 的所有权。`VectorSetView` 借用调用者的查询内存，查询返回前保持有效。

每个并发调用者使用独立的 QueryWorkspace 和结果 vector。Index 的图、文档与 centroids 在查询期间共享只读。workspace 保留上一轮的分配容量，包括入口评分、fine-score table、cluster mask、MaxSim maxima 和重排 pair scores。HNSW 内部的候选队列暂时仍由单次查询创建。

`Result.distance = 1 - mean_i(max_j(dot(q_i, d_j)))`，越小越相似。保留 GEM 的 FP32 查表、原始向量重排和 HNSW 搜索策略。

构建使用原有 cluster 内逐文档插入和 qEMD，保留 code 出现次数对应的 transport 权重。第一版 library 的构建顺序为串行，方便对照与排查。qEMD solver 本身未修改，只复用其输入矩阵和 histogram。

## 索引读写

当前沿用作者的 HNSW 文件格式，支持仓库中的 `0.bin`。该格式依赖原有布局，尚未增加 corpus 指纹和跨平台格式版本；调用者需要使用与图匹配的编码数据。加载时检查文档数量与 label，重新绑定向量地址。

`load()` 不修改图。旧索引需要补边时，先用 `run.action=repair` 与独立的 `index.output` 生成新文件。构建工具在保存前完成补边。工具拒绝覆盖已有的构建或修复输出。

## 验证

`gem_tests` 覆盖非默认维度、空 cluster、qEMD 步长和重复 code、save/load、八线程结果一致性；`gem_io_config_tests` 覆盖数据与配置错误。

EVQA 对照命令：

```bash
python3 tests/compare_evqa.py /absolute/path/to/build /absolute/path/to/evqa /absolute/path/to/results 128
```

最后一个数字是查询数，0 表示全部查询。其后可增加匹配该 corpus 的图文件路径；默认使用仓库自带的图。注意仓库图包含 51,472 篇文档，另一份公开 EVQA 数据包含 51,462 篇，必须使用对应版本的图。脚本按顺序测试旧入口、新入口单线程和八线程，各预热一遍、测量三遍；逐项比较 top-k 文档编号、排名和分数。对照入口固定 nprobe=4、128 维、单查询线程。未执行原 example 的 main，也不加载其训练查询。

基准 QPS 为整个查询批次的吞吐，mean_ms 为单条查询执行时间的平均值，两者均排除数据加载。输出结果保存在指定目录，INI 和 JSON 汇总一起保留。
