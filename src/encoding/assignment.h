#pragma once
#include <cstddef>
#include <vector>

namespace gem::detail {
void validate_assignment(const std::vector<std::vector<std::size_t>>& clusters,
                         std::size_t document_count);
}  // namespace gem::detail
