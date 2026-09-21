#include "kernels.h"
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <stdexcept>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace gem::detail {
#if defined(__AVX512F__)
namespace {
template<std::size_t QueryBlocks>
void score_block(const float* packed, VectorSetView document, float* maxima) {
    constexpr std::size_t lanes = 16, doc_block = 12;
    const auto dim = document.dimension;
    __m512 best[QueryBlocks];
    for (auto& value : best) value = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    std::size_t j = 0;
    for (; j + doc_block <= document.count; j += doc_block) {
        __m512 scores[doc_block][QueryBlocks];
        #pragma GCC unroll 12
        for (std::size_t r = 0; r < doc_block; ++r)
            for (auto& score : scores[r]) score = _mm512_setzero_ps();
        const float* doc = document.data + j * dim;
        for (std::size_t k = 0; k < dim; ++k) {
            __m512 q[QueryBlocks];
            for (std::size_t b = 0; b < QueryBlocks; ++b)
                q[b] = _mm512_loadu_ps(packed + (b * dim + k) * lanes);
            #pragma GCC unroll 12
            for (std::size_t r = 0; r < doc_block; ++r) {
                const __m512 value = _mm512_set1_ps(doc[r * dim + k]);
                for (std::size_t b = 0; b < QueryBlocks; ++b)
                    scores[r][b] = _mm512_fmadd_ps(q[b], value, scores[r][b]);
            }
        }
        // Consume the token-pair tile in registers; only maxima reach memory.
        #pragma GCC unroll 12
        for (std::size_t r = 0; r < doc_block; ++r)
            for (std::size_t b = 0; b < QueryBlocks; ++b)
                best[b] = _mm512_max_ps(best[b], scores[r][b]);
    }
    for (; j < document.count; ++j) {
        __m512 scores[QueryBlocks];
        for (auto& score : scores) score = _mm512_setzero_ps();
        const float* doc = document.data + j * dim;
        for (std::size_t k = 0; k < dim; ++k) {
            const __m512 value = _mm512_set1_ps(doc[k]);
            for (std::size_t b = 0; b < QueryBlocks; ++b)
                scores[b] = _mm512_fmadd_ps(_mm512_loadu_ps(packed + (b * dim + k) * lanes),
                                          value, scores[b]);
        }
        for (std::size_t b = 0; b < QueryBlocks; ++b)
            best[b] = _mm512_max_ps(best[b], scores[b]);
    }
    for (std::size_t b = 0; b < QueryBlocks; ++b)
        _mm512_storeu_ps(maxima + b * lanes, best[b]);
}
}  // namespace
#endif

void prepare_rerank_query(VectorSetView query, RerankWorkspace& w) {
    if (!query.data || !query.count || !query.dimension)
        throw std::invalid_argument("rerank requires a nonempty query");
    w.query = {};
#if defined(__AVX512F__)
    constexpr std::size_t lanes = 16;
    if (query.count > w.packed_query.max_size() - (lanes - 1))
        throw std::overflow_error("rerank query too large");
    const auto padded = (query.count + lanes - 1) / lanes * lanes;
    if (query.dimension > w.packed_query.max_size() / padded)
        throw std::overflow_error("rerank query workspace too large");
    w.packed_query.resize(padded * query.dimension);
    w.scores.resize(padded);
    for (std::size_t qb = 0; qb < query.count; qb += lanes) {
        const auto count = std::min(lanes, query.count - qb);
        float* output = w.packed_query.data() + qb * query.dimension;
        for (std::size_t k = 0; k < query.dimension; ++k) {
            for (std::size_t lane = 0; lane < count; ++lane)
                output[k * lanes + lane] = query.data[(qb + lane) * query.dimension + k];
            std::fill(output + k * lanes + count, output + (k + 1) * lanes, 0.0f);
        }
    }
#endif
    w.query = query;
}

float rerank_distance(VectorSetView document, RerankWorkspace& w) {
    const auto query = w.query;
    if (!query.count || !document.count || !document.data || document.dimension != query.dimension)
        throw std::invalid_argument("rerank requires prepared query and matching document dimensions");
#if defined(__AVX512F__)
    for (std::size_t qb = 0; qb < query.count; qb += 32) {
        const float* packed = w.packed_query.data() + qb * query.dimension;
        if (query.count - qb > 16)
            score_block<2>(packed, document, w.scores.data() + qb);
        else
            score_block<1>(packed, document, w.scores.data() + qb);
    }
    // Preserve GEM's final reduction order.
    Eigen::Map<const Eigen::MatrixXf> maxima(w.scores.data(), query.count, 1);
    return 1.0f - maxima.rowwise().maxCoeff().sum() / query.count;
#else
    if (query.count > w.scores.max_size() / document.count)
        throw std::overflow_error("rerank workspace too large");
    const auto required = query.count * document.count;
    if (w.scores.size() < required) w.scores.resize(required);
    using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMatrix> q(query.data, query.count, query.dimension);
    Eigen::Map<const RowMatrix> d(document.data, document.count, document.dimension);
    Eigen::Map<Eigen::MatrixXf> scores(w.scores.data(), query.count, document.count);
    scores.noalias() = q * d.transpose();
    return 1.0f - scores.rowwise().maxCoeff().sum() / query.count;
#endif
}
}  // namespace gem::detail
