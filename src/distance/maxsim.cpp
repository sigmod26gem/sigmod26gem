#include "kernels.h"
#include <Eigen/Dense>
#include <algorithm>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace gem::detail {
float code_distance(const float* table, std::size_t query_tokens,
                    std::size_t fine_centers, const int* codes,
                    std::size_t tokens, float* maxima) {
    Eigen::Map<const Eigen::MatrixXf> scores(table, query_tokens, fine_centers);
    Eigen::Map<Eigen::VectorXf> best(maxima, query_tokens);
#if defined(__AVX512F__)
    // Keep maxima in registers across codes; retain GEM's final reduction.
    std::size_t q = 0;
    for (; q + 128 <= query_tokens; q += 128) {
        __m512 acc[8];
        for (auto& value : acc) value = _mm512_set1_ps(-9.0f);
        for (std::size_t j = 0; j < tokens; ++j) {
            const float* row = table + std::size_t(codes[j]) * query_tokens + q;
            for (int block = 0; block < 8; ++block)
                acc[block] = _mm512_max_ps(acc[block], _mm512_loadu_ps(row + 16 * block));
        }
        for (int block = 0; block < 8; ++block)
            _mm512_storeu_ps(maxima + q + 16 * block, acc[block]);
    }
    for (; q + 16 <= query_tokens; q += 16) {
        __m512 acc = _mm512_set1_ps(-9.0f);
        for (std::size_t j = 0; j < tokens; ++j)
            acc = _mm512_max_ps(acc, _mm512_loadu_ps(table + std::size_t(codes[j]) * query_tokens + q));
        _mm512_storeu_ps(maxima + q, acc);
    }
    for (; q < query_tokens; ++q) {
        float value = -9.0f;
        for (std::size_t j = 0; j < tokens; ++j)
            value = std::max(value, table[std::size_t(codes[j]) * query_tokens + q]);
        maxima[q] = value;
    }
#else
    best.setConstant(-9.0f);
    for (std::size_t j = 0; j < tokens; ++j) best = best.cwiseMax(scores.col(codes[j]));
#endif
    return (1.0f - best.array()).sum() / query_tokens;
}

}  // namespace gem::detail
