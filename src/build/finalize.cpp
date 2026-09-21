#include "graph/index_impl.h"

namespace gem::detail {
void Graph::repair(const EncodedCorpus& corpus) {
    auto& graph = *impl_->graph;
    for (std::size_t i = 0; i < corpus.clusters.size(); ++i) {
        const auto& cluster = corpus.clusters[i];
        if (cluster.empty()) continue;
        graph.entry_map.clear();
        for (auto id : cluster) graph.entry_map[graph.label_lookup_.at(id)] = i;
        auto reachable = graph.searchNodesForFix(cluster.front(), i);
        std::size_t cursor = 0;
        for (std::size_t j = 1; j < cluster.size(); ++j) {
            const auto id = graph.label_lookup_.at(cluster[j]);
            if (graph.entry_map[id] == i + 1) continue;
            while (cursor < reachable.size()) {
                const auto from = reachable[cursor].first;
                if (graph.canAddEdgeinter(from)) {
                    graph.mutuallyConnectTwoInterElement(from, id);
                    if (graph.canAddEdgeinter(id)) graph.mutuallyConnectTwoInterElement(id, from);
                    break;
                }
                ++cursor;
            }
            auto more = graph.searchNodesForFix(cluster[j], i);
            reachable.insert(reachable.end(), more.begin(), more.end());
        }
    }
}
}  // namespace gem::detail
