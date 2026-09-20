#include "graph/index_impl.h"

namespace gem::detail {
void Graph::traverse(float* scores, std::size_t centers, std::size_t query_tokens,
                     float* maxima, std::size_t ef,
                     const std::vector<std::size_t>& entries, const std::vector<bool>& allowed,
                     std::vector<std::pair<float, std::size_t>>& candidates,
                     hnswlib::EntrySearchScratch<float>& scratch) const {
    const auto& graph = *impl_->graph;
    ScoreQuery query(scores, centers, query_tokens, maxima,
                     graph.data_level0_memory_, graph.size_data_per_element_,
                     impl_->unique_code_offsets.data(), impl_->unique_codes.data());
    if (graph.num_deleted_)
        graph.searchBaseLayerClusterEntriesInto<false>(entries, &query, ef, scratch, nullptr, nullptr, &allowed);
    else
        graph.searchBaseLayerClusterEntriesInto<true>(entries, &query, ef, scratch, nullptr, nullptr, &allowed);
    // Keep GEM's two heap orderings, including its equal-score tie behavior.
    auto& found = scratch.results;
    while (!scratch.top.empty()) {
        const auto result = scratch.top.top();
        found.emplace(result.first, graph.getExternalLabel(result.second));
        scratch.top.pop();
    }
    candidates.clear();
    while (!found.empty()) {
        candidates.push_back(found.top());
        found.pop();
    }
}
}  // namespace gem::detail
