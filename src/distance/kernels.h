#pragma once
#include "gem/data.h"
#include <vector>

namespace gem::detail {
void dot_table(const float* a, std::size_t n, const float* b, std::size_t m,
               std::size_t dimension, float* output);
float code_distance(const float* table, std::size_t query_tokens,
                    std::size_t fine_centers, const int* codes,
                    std::size_t tokens, float* maxima);
float rerank_distance(VectorSetView query, VectorSetView document,
                      std::vector<float>& pair_scores);
float qemd_distance(const int* a, std::size_t n, const int* b, std::size_t m,
                    const float* center_scores, std::size_t fine_centers);
}  // namespace gem::detail
