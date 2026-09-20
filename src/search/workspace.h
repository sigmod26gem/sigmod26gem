#pragma once
#include "gem/workspace.h"
#include "graph/search_scratch.h"
#include <vector>
#include <utility>

namespace gem::detail {
struct SearchScratch {
    hnswlib::EntrySearchScratch<float> traversal;
    std::vector<float> route_scores, fine_scores, maxima, pair_scores;
    std::vector<std::pair<float, int>> route_order;
    std::vector<std::pair<float, std::size_t>> candidates;
    std::vector<std::size_t> entries;
    std::vector<bool> allowed;
};
}  // namespace gem::detail

namespace gem {
struct QueryWorkspace::Impl : detail::SearchScratch {};
}  // namespace gem
