#pragma once
#include "index.h"
#include "hnswlib.h"

namespace gem::detail {
struct ScoreQuery : vectorset {
    float* maxima;
    ScoreQuery(float* scores, std::size_t centers, std::size_t tokens, float* scratch)
        : vectorset(scores, nullptr, centers, tokens), maxima(scratch) {}
};

struct Graph::Impl {
    std::vector<vectorset> documents;
    std::vector<std::vector<std::size_t>> internal_clusters;
    std::unique_ptr<hnswlib::L2VSSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;
    explicit Impl(const EncodedCorpus& corpus);
    void prepare_search(const EncodedCorpus& corpus);
};
}  // namespace gem::detail
