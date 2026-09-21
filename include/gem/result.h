#pragma once

#include <cstddef>

namespace gem {
struct Result {
    std::size_t id;
    // Original GEM distance: 1 - mean over query-token maximum inner products.
    float distance;
};
}  // namespace gem
