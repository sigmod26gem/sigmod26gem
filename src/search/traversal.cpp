#include "graph/index_impl.h"
#include <algorithm>
#include <functional>

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
    const auto score = [&query](hnswlib::tableint id) { return query.score_internal(id); };
    if (graph.num_deleted_)
        graph.searchBaseLayerClusterEntriesScored<false>(entries, ef, scratch, score, nullptr, nullptr, &allowed);
    else
        graph.searchBaseLayerClusterEntriesScored<true>(entries, ef, scratch, score, nullptr, nullptr, &allowed);
    // Match the old external-label max-heap order, including equal scores.
    candidates.clear();
    candidates.reserve(scratch.top.size());
    for (const auto& result : scratch.top.values())
        candidates.emplace_back(result.first, graph.getExternalLabel(result.second));
    scratch.top.clear();
    std::sort(candidates.begin(), candidates.end(), std::greater<>());
}
}  // namespace gem::detail
