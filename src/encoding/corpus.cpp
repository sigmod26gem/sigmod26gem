#include "gem/data.h"
#include "assignment.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gem {
VectorSetView MultiVectors::at(std::size_t id) const {
    if (id >= size()) throw std::out_of_range("document/query id out of range");
    return {values.data() + offsets[id] * dimension,
            static_cast<std::size_t>(offsets[id + 1] - offsets[id]), dimension};
}

void MultiVectors::validate() const {
    if (!dimension || dimension > INT32_MAX || offsets.size() < 2 || offsets[0] != 0 ||
        values.size() % dimension || offsets.back() != values.size() / dimension)
        throw std::invalid_argument("invalid vector dimensions or offsets");
    for (std::size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] <= offsets[i - 1] || offsets[i] - offsets[i - 1] > INT32_MAX)
            throw std::invalid_argument("empty or oversized vector set");
    for (float x : values)
        if (!std::isfinite(x)) throw std::invalid_argument("non-finite embedding");
}

void EncodedCorpus::validate() const {
    documents.validate();
    const auto d = documents.dimension;
    const auto n = fine_centroids.size() / d;
    if (documents.size() >= INT32_MAX || !n || n > INT32_MAX ||
        fine_centroids.size() % d || graph_centroids.size() % d ||
        clusters.empty() || clusters.size() > INT32_MAX ||
        graph_centroids.size() / d != clusters.size() ||
        codes.size() != documents.offsets.back())
        throw std::invalid_argument("inconsistent encoded corpus sizes");
    for (int code : codes)
        if (code < 0 || static_cast<std::size_t>(code) >= n)
            throw std::invalid_argument("fine code out of range");
    for (const auto* centers : {&fine_centroids, &graph_centroids})
        for (float x : *centers)
            if (!std::isfinite(x)) throw std::invalid_argument("non-finite centroid");
    detail::validate_assignment(clusters, documents.size());
}
}  // namespace gem
