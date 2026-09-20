#pragma once

#include <cstddef>

namespace gem {
struct BuildOptions {
    std::size_t m = 24;
    std::size_t ef_construction = 80;
    std::size_t seed = 100;
    // The original GEM builder materializes a dense centroid matrix.
    std::size_t distance_budget_bytes = std::size_t{4} << 30;
};

struct SearchOptions {
    std::size_t nprobe = 4;
    std::size_t ef = 4000;
    std::size_t rerank_k = 512;
    std::size_t k = 100;
};
}  // namespace gem
