#pragma once
#include "gem/index.h"
#include <utility>

namespace gem {
struct QueryWorkspace::Impl {
    std::vector<float> route_scores, fine_scores, maxima, pair_scores;
    std::vector<std::pair<float, int>> route_order;
    std::vector<std::pair<float, std::size_t>> candidates;
    std::vector<std::size_t> entries;
    std::vector<bool> allowed;
};
}  // namespace gem
