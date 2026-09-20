#include "kernels.h"
#include <Eigen/Dense>
#include <cblas.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gem::detail {
void dot_table(const float* a, std::size_t n, const float* b, std::size_t m,
               std::size_t dimension, float* output) {
    if (n > INT32_MAX || m > INT32_MAX || dimension > INT32_MAX)
        throw std::overflow_error("BLAS dimensions exceed int32");
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, dimension,
                1.0f, a, dimension, b, dimension, 0.0f, output, m);
}

float code_distance(const float* table, std::size_t query_tokens,
                    std::size_t fine_centers, const int* codes,
                    std::size_t tokens, float* maxima) {
    Eigen::Map<const Eigen::MatrixXf> scores(table, query_tokens, fine_centers);
    Eigen::Map<Eigen::VectorXf> best(maxima, query_tokens);
    best.setConstant(-9.0f);
    for (std::size_t j = 0; j < tokens; ++j) best = best.cwiseMax(scores.col(codes[j]));
    return (1.0f - best.array()).sum() / query_tokens;
}

float rerank_distance(VectorSetView query, VectorSetView document,
                      std::vector<float>& pair_scores) {
    if (query.count > pair_scores.max_size() / document.count)
        throw std::overflow_error("rerank workspace too large");
    const auto required = query.count * document.count;
    if (pair_scores.size() < required) pair_scores.resize(required);
    using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMatrix> q(query.data, query.count, query.dimension);
    Eigen::Map<const RowMatrix> d(document.data, document.count, document.dimension);
    Eigen::Map<Eigen::MatrixXf> scores(pair_scores.data(), query.count, document.count);
    scores.noalias() = q * d.transpose();
    return 1.0f - scores.rowwise().maxCoeff().sum() / query.count;
}
}  // namespace gem::detail
