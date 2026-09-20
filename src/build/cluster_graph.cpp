#include "graph/index_impl.h"
#include "distance/kernels.h"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gem::detail {
Graph Graph::build(const EncodedCorpus& corpus, const BuildOptions& options) {
    if (options.m < 2 || options.m > 10000 || options.ef_construction < options.m)
        throw std::invalid_argument("invalid HNSW build options");
    auto impl = std::make_unique<Impl>(corpus);
    const auto centers = corpus.fine_centroids.size() / corpus.documents.dimension;
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (centers > limit / centers || centers * centers > limit / sizeof(float))
        throw std::overflow_error("workspace size overflow");
    if (centers * centers * sizeof(float) > options.distance_budget_bytes)
        throw std::invalid_argument("dense centroid matrix exceeds build.distance_budget_bytes");
    std::vector<float> distances(centers * centers);
    dot_table(corpus.fine_centroids.data(), centers, corpus.fine_centroids.data(), centers,
              corpus.documents.dimension, distances.data());
    impl->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        impl->space.get(), impl->documents.size() + 1, options.m, options.ef_construction, options.seed);
    impl->graph->fstdistfuncClusterEMD = [centers](const vectorset* a, const vectorset* b, const float* table) {
        vectorset left(nullptr, nullptr, 0, 0), right(nullptr, nullptr, 0, 0);
        std::memcpy(&left, a, sizeof(left));
        std::memcpy(&right, b, sizeof(right));
        return qemd_distance(left.codes, left.vecnum, right.codes, right.vecnum, table, centers);
    };
    for (const auto& cluster : corpus.clusters) {
        if (cluster.empty()) continue;
        impl->graph->entry_map.clear();
        for (auto id : cluster)
            impl->graph->addClusterPointEntry(&impl->documents[id], distances.data(), id, cluster.front());
    }
    impl->prepare_search(corpus);
    return Graph(std::move(impl));
}
}  // namespace gem::detail
