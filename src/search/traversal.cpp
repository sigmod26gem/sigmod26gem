#include "graph/index_impl.h"

namespace gem::detail {
void Graph::traverse(float* scores, std::size_t centers, std::size_t query_tokens,
                     float* maxima, std::size_t ef,
                     const std::vector<std::size_t>& entries, const std::vector<bool>& allowed,
                     std::vector<std::pair<float, std::size_t>>& candidates) const {
    ScoreQuery query(scores, centers, query_tokens, maxima);
    auto found = impl_->graph->searchKnnClusterEntries(&query, ef, entries, nullptr, &allowed, ef);
    candidates.clear();
    while (!found.empty()) {
        candidates.push_back(found.top());
        found.pop();
    }
}
}  // namespace gem::detail
