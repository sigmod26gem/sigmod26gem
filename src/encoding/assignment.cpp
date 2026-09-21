#include "assignment.h"
#include <algorithm>
#include <stdexcept>

namespace gem::detail {
void validate_assignment(const std::vector<std::vector<std::size_t>>& clusters,
                         std::size_t document_count) {
    std::vector<bool> covered(document_count, false);
    for (const auto& cluster : clusters) {
        auto sorted = cluster;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::invalid_argument("duplicate document in graph cluster");
        for (auto id : cluster) {
            if (id >= covered.size()) throw std::invalid_argument("cluster document out of range");
            covered[id] = true;
        }
    }
    if (std::find(covered.begin(), covered.end(), false) != covered.end())
        throw std::invalid_argument("document has no graph-cluster assignment");
}
}  // namespace gem::detail
