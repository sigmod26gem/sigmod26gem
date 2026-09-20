#include "search/kernels.h"
#include "EMD.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace gem::detail {
float qemd_distance(const int* a, std::size_t n, const int* b, std::size_t m,
                    const float* center_scores, std::size_t fine_centers) {
    struct Scratch { std::vector<double> costs, left, right; };
    // Scratch is thread-private; codebook size and matrix are explicit inputs.
    thread_local Scratch scratch;
    if (!n || !m || n > INT32_MAX || m > INT32_MAX || n > scratch.costs.max_size() / m)
        throw std::invalid_argument("invalid qEMD dimensions");
    scratch.costs.resize(n * m);
    scratch.left.assign(n, 1.0 / n);
    scratch.right.assign(m, 1.0 / m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            scratch.costs[i * m + j] = 1.0 - center_scores[std::size_t(a[i]) * fine_centers + b[j]];
    return EMD_wrap_self(n, m, scratch.left.data(), scratch.right.data(), scratch.costs.data(), 1000);
}
}  // namespace gem::detail
