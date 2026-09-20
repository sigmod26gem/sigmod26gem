#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gem {
struct VectorSetView {
    const float* data;
    std::size_t count;
    std::size_t dimension;
};

struct MultiVectors {
    std::size_t dimension = 0;
    std::vector<float> values;
    std::vector<std::uint64_t> offsets;
    std::size_t size() const { return offsets.empty() ? 0 : offsets.size() - 1; }
    VectorSetView at(std::size_t id) const;
    void validate() const;
};

struct EncodedCorpus {
    MultiVectors documents;
    std::vector<int> codes;
    std::vector<float> fine_centroids;
    std::vector<float> graph_centroids;
    std::vector<std::vector<std::size_t>> clusters;
    void validate() const;
};
}  // namespace gem
