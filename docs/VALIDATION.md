# 重构验证

## 2026-09-20

本次包含两个代码提交：`60a2fae` 引入 library、查询 workspace 和 INI 入口；`da2a6b9` 修复多个 graph clusters 共用入口文档时返回重复结果的问题。

### 功能测试

- FP32 graph scorer、原始向量 MaxSim 与旧 Eigen 表达式比较，误差阈值为 `1e-6`。
- qEMD 检查实际 codebook 步长及重复 code 的 transport 权重。
- 小规模数据完成 build、repair、save、load 与 search。
- 八个线程共享一个 Index，各自持有 workspace，混合使用不同 nprobe、ef 和 rerank_k，结果与逐条串行查询一致。
- 检查空 cluster、重复入口、非法 code、容量预算、NPY 类型与形状、未知和重复 INI 字段。
- Release native、Release portable，以及开启 AddressSanitizer/UndefinedBehaviorSanitizer 的 Debug 构建运行同一组 CTest。

### EVQA 结果一致性

使用作者公开的 51,462 篇文档版本和匹配的 `index1024_all_24_80/0.bin`。向量维度为 128，fine centroids 为 32,768，graph clusters 为 1,024。搜索参数为 nprobe=4、ef=4000、rerank_k=512、k=100；本次加载后未额外 repair。

选取前 128 条查询，每个入口预热一遍、测量三遍。对照入口调用旧 example 的 search，新入口调用 `Index::search`，两者共享当前分支的 HNSW 内核修复。

| 检查 | 单线程 library | 八线程 library |
|---|---:|---:|
| 比较的 top-100 结果数 | 12,800 | 12,800 |
| 文档编号与排名不同的结果数 | 0 | 0 |
| TSV 输出分数最大误差 | 0 | 0 |
| 重复结果文档数 | 0 | 0 |
| Recall@100 | 71.0938% | 71.0938% |

修复重复入口前，library 与旧入口的结果同样逐项一致，但包含 6 个重复文档结果。重复入口修复在两个入口中共同生效，修复后的排序单独完成上述比较。

### 性能观察

结构重构的第一轮测量采用相同输入、搜索参数和 native 编译选项，三次测量的中位数如下：

| 入口 | 查询线程数 | QPS | 平均查询延迟 |
|---|---:|---:|---:|
| 旧入口 | 1 | 4.43 | 225.85 ms |
| library | 1 | 4.43 | 225.82 ms |
| library | 8 | 29.21 | 267.91 ms |

单线程性能基本持平。该轮对应重复入口修复前的版本；旧入口与 library 的计时边界也有少量差别。修复后的测试期间，机器上同时有另一个多线程构图任务，吞吐数字未用于前后性能比较。

### 后续测试

当前真实数据对照覆盖 128 条查询。合并前继续运行全量 EVQA，并在空闲机器上重测最终提交的吞吐。构建目前完成小规模功能测试，尚未进行全量构图性能比较。新的 centroid cache、评分布局和 SIMD kernel 分别完成距离、排名和性能验证后，再单独提交。

命令和数据格式见 [LIBRARY.md](LIBRARY.md)；完整配置字段见 [CONFIGURATION.md](CONFIGURATION.md)。
