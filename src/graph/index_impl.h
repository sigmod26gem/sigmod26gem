#pragma once
#include "index.h"
#include "hnswlib.h"
#include "distance/kernels.h"

namespace gem::detail {
struct ScoreQuery : vectorset {
    float* maxima;
    const char* graph_records;
    std::size_t record_stride;
    const std::size_t* code_offsets;
    const int* unique_codes;
    ScoreQuery(float* scores, std::size_t centers, std::size_t tokens, float* scratch,
               const char* records, std::size_t stride, const std::size_t* offsets, const int* codes)
        : vectorset(scores, nullptr, centers, tokens), maxima(scratch), graph_records(records),
          record_stride(stride), code_offsets(offsets), unique_codes(codes) {}
    float score_internal(std::size_t id) const {
        const auto begin = code_offsets[id], end = code_offsets[id + 1];
        return code_distance(data, vecnum, dim, unique_codes + begin, end - begin, maxima);
    }
};

struct Graph::Impl {
    std::vector<vectorset> documents;
    std::vector<std::vector<std::size_t>> internal_clusters;
    std::vector<std::size_t> unique_code_offsets;
    std::vector<int> unique_codes;
    std::unique_ptr<hnswlib::L2VSSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;
    explicit Impl(const EncodedCorpus& corpus);
    void prepare_search(const EncodedCorpus& corpus);
};
}  // namespace gem::detail
