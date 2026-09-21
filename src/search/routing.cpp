#include "pipeline.h"
#include "distance/kernels.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace gem::detail {
namespace {
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("workspace size overflow");
    return a * b;
}
}  // namespace

void validate_query(VectorSetView query, const EncodedCorpus& corpus, const SearchOptions& options) {
    if (!query.data || !query.count || query.count > INT32_MAX ||
        query.dimension != corpus.documents.dimension)
        throw std::invalid_argument("invalid query shape");
    if (!options.k || !options.nprobe || options.nprobe > corpus.clusters.size() ||
        options.rerank_k < options.k || options.ef < options.rerank_k)
        throw std::invalid_argument("require nprobe <= clusters and ef >= rerank_k >= k > 0");
    for (std::size_t i = 0; i < product(query.count, query.dimension); ++i)
        if (!std::isfinite(query.data[i])) throw std::invalid_argument("non-finite query");
}

void route_query(VectorSetView query, const EncodedCorpus& corpus, const SearchOptions& options,
                 const std::vector<std::vector<std::size_t>>& internal_clusters,
                 std::size_t capacity, SearchScratch& w) {
    const auto nc = corpus.clusters.size();
    w.route_scores.resize(product(query.count, nc));
    w.route_order.resize(nc);
    w.allowed.assign(capacity, false);
    w.entries.clear();
    // Bucket reuse would make GEM's entry order depend on preceding queries.
    std::unordered_set<int> selected_clusters;
    dot_table(query.data, query.count, corpus.graph_centroids.data(), nc,
              query.dimension, w.route_scores.data());
    for (std::size_t token = 0; token < query.count; ++token) {
        for (std::size_t c = 0; c < nc; ++c)
            w.route_order[c] = {w.route_scores[token * nc + c], static_cast<int>(c)};
        std::partial_sort(w.route_order.begin(), w.route_order.begin() + options.nprobe, w.route_order.end(),
                          [](auto a, auto b) { return a.first > b.first; });
        for (std::size_t j = 0; j < options.nprobe; ++j) selected_clusters.insert(w.route_order[j].second);
    }
    for (auto cluster : selected_clusters) {
        if (corpus.clusters[cluster].empty()) continue;
        w.entries.push_back(corpus.clusters[cluster].front());
        for (auto id : internal_clusters[cluster]) w.allowed[id] = true;
    }
}

void score_fine_centers(VectorSetView query, const EncodedCorpus& corpus, SearchScratch& w) {
    const auto nf = corpus.fine_centroids.size() / query.dimension;
    w.fine_scores.resize(product(query.count, nf));
    w.maxima.resize(query.count);
    dot_table(corpus.fine_centroids.data(), nf, query.data, query.count,
              query.dimension, w.fine_scores.data());
}
}  // namespace gem::detail
