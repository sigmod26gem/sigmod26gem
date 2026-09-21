#include "distance/kernels.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {
float max_rerank_error = 0.0f;
void check_rerank(float actual, float expected) {
    max_rerank_error = std::max(max_rerank_error, std::abs(actual - expected));
    // SIMD and Eigen GEMV can accumulate a dot product in different orders.
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > 2e-6f * std::max(1.0f, std::abs(expected)))
        throw std::runtime_error("FP32 rerank score exceeds rounding tolerance");
}

float reference_code(const float* table, std::size_t nq, std::size_t nc,
                     const int* codes, std::size_t len, float* maxima) {
    Eigen::Map<const Eigen::MatrixXf> scores(table, nq, nc);
    Eigen::Map<Eigen::VectorXf> best(maxima, nq);
    best.setConstant(-9.0f);
    for (std::size_t j = 0; j < len; ++j) best = best.cwiseMax(scores.col(codes[j]));
    return (1.0f - best.array()).sum() / nq;
}

float reference_rerank(gem::VectorSetView query, gem::VectorSetView doc,
                       std::vector<float>& workspace) {
    using Row = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    workspace.resize(query.count * doc.count);
    Eigen::Map<const Row> q(query.data, query.count, query.dimension);
    Eigen::Map<const Row> d(doc.data, doc.count, doc.dimension);
    Eigen::Map<Eigen::MatrixXf> scores(workspace.data(), query.count, doc.count);
    scores.noalias() = q * d.transpose();
    return 1.0f - scores.rowwise().maxCoeff().sum() / query.count;
}

void equivalence() {
    std::mt19937 gen(91);
    std::uniform_real_distribution<float> uniform(-12, 1);
    for (std::size_t nq : {1, 7, 15, 16, 17, 31, 32, 33, 37, 64, 127, 128, 129,
                          255, 256, 257, 319, 320, 321, 513}) {
        const std::size_t nc = 257;
        std::vector<float> table(nq * nc), actual(nq), expected(nq);
        for (float& x : table) x = uniform(gen);
        for (std::size_t len : {0, 1, 3, 31, 128, 181, 512}) {
            std::vector<int> codes(len);
            for (auto& c : codes) c = gen() % nc;
            const auto a = gem::detail::code_distance(table.data(), nq, nc, codes.data(), len, actual.data());
            const auto b = reference_code(table.data(), nq, nc, codes.data(), len, expected.data());
            if (a != b || actual != expected) throw std::runtime_error("graph score differs");
        }
        for (std::size_t nd : {1, 7, 11, 12, 13, 23, 24, 25, 33, 63, 64, 65,
                              127, 128, 129, 180, 181, 513}) {
            const std::size_t dim = nq % 2 ? 13 : 128;
            std::vector<float> q(nq * dim), d(nd * dim), b;
            gem::detail::RerankWorkspace a;
            for (float& x : q) x = uniform(gen) / 12;
            for (float& x : d) x = uniform(gen) / 12;
            const gem::VectorSetView qv{q.data(), nq, dim}, dv{d.data(), nd, dim};
            gem::detail::prepare_rerank_query(qv, a);
            const auto actual_score = gem::detail::rerank_distance(dv, a);
            const auto expected_score = reference_rerank(qv, dv, b);
            check_rerank(actual_score, expected_score);
        }
    }
    std::cout << "exact graph scores and FP32 rerank equivalence across block boundaries passed\n";
}

void rerank_dimensions() {
    std::mt19937 gen(95);
    std::uniform_real_distribution<float> uniform(-1, 1);
    gem::detail::RerankWorkspace w;
    std::vector<float> reference;
    for (std::size_t dim : {1, 31, 64, 129, 1024}) {
        for (std::size_t nq : {17, 32, 33}) {
            std::vector<float> q(nq * dim), d(181 * dim);
            for (auto& x : q) x = uniform(gen) / std::sqrt(float(dim));
            for (auto& x : d) x = uniform(gen) / std::sqrt(float(dim));
            const gem::VectorSetView query{q.data(), nq, dim};
            gem::detail::prepare_rerank_query(query, w);
            for (std::size_t len : {11, 12, 13, 25, 181}) {
                const gem::VectorSetView doc{d.data(), len, dim};
                check_rerank(gem::detail::rerank_distance(doc, w),
                             reference_rerank(query, doc, reference));
            }
        }
    }
    const float data[] = {1, -1};
    bool rejected = false;
    try { gem::detail::prepare_rerank_query({data, 0, 2}, w); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("empty rerank query accepted");
    gem::detail::prepare_rerank_query({data, 1, 2}, w);
    rejected = false;
    try { gem::detail::rerank_distance({data, 2, 1}, w); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("mismatched rerank dimensions accepted");
    std::cout << "rerank dimensions and reused query shapes passed; max_abs_error="
              << max_rerank_error << '\n';
}

void rerank_workspace_reuse() {
    constexpr std::size_t nq = 37, dim = 19;
    std::mt19937 gen(94);
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<float> q(nq * dim), d(4097 * dim), expected;
    gem::detail::RerankWorkspace actual;
    for (auto& x : q) x = uniform(gen);
    for (auto& x : d) x = uniform(gen);
    const gem::VectorSetView query{q.data(), nq, dim};
    gem::detail::prepare_rerank_query(query, actual);
    float* storage = nullptr;
    for (std::size_t len : {4097, 1, 65, 7, 1025, 64, 4097}) {
        const gem::VectorSetView doc{d.data(), len, dim};
        const auto a = gem::detail::rerank_distance(doc, actual);
        const auto b = reference_rerank(query, doc, expected);
        check_rerank(a, b);
#if defined(__AVX512F__)
        const auto padded = (nq + 15) / 16 * 16;
        if (actual.scores.size() != padded || actual.packed_query.size() != padded * dim)
            throw std::runtime_error("unbounded rerank workspace");
#endif
        if (storage && storage != actual.scores.data()) throw std::runtime_error("rerank workspace reallocated");
        storage = actual.scores.data();
    }
    std::fill(q.begin(), q.end(), 1.0f);
    std::fill(d.begin(), d.end(), -1.0f);
    gem::detail::prepare_rerank_query(query, actual);
    const gem::VectorSetView negative_doc{d.data(), 129, dim};
    if (gem::detail::rerank_distance(negative_doc, actual) !=
        reference_rerank(query, negative_doc, expected))
        throw std::runtime_error("all-negative rerank score differs");
    std::cout << "bounded rerank workspace reuse and negative scores passed\n";
}

template<class Fn> double time_calls(Fn fn, int repeats) {
    volatile float sink = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) sink = sink + fn(i);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repeats;
}

void benchmark() {
    std::mt19937 gen(93);
    std::uniform_real_distribution<float> uniform(-1, 1);
    constexpr std::size_t nq = 320, nc = 32768, len = 180, docs = 128, dim = 128;
    std::vector<float> table(nq * nc), maxima(nq), q(nq * dim), d(len * dim), a;
    gem::detail::RerankWorkspace b;
    std::vector<int> codes(docs * len);
    for (auto& x : table) x = uniform(gen);
    for (auto& x : q) x = uniform(gen);
    for (auto& x : d) x = uniform(gen);
    for (auto& c : codes) c = gen() % nc;
    for (int repeat = 0; repeat < 5; ++repeat) {
        const auto graph_ref = time_calls([&](int i) {
            return reference_code(table.data(), nq, nc, codes.data() + (i % docs) * len, len, maxima.data());
        }, 2048);
        const auto graph_new = time_calls([&](int i) {
            return gem::detail::code_distance(table.data(), nq, nc, codes.data() + (i % docs) * len, len, maxima.data());
        }, 2048);
        const gem::VectorSetView qv{q.data(), nq, dim}, dv{d.data(), len, dim};
        gem::detail::prepare_rerank_query(qv, b);
        const auto rerank_ref = time_calls([&](int) { return reference_rerank(qv, dv, a); }, 1024);
        const auto rerank_new = time_calls([&](int) { return gem::detail::rerank_distance(dv, b); }, 1024);
        std::cout << "{\"graph_reference_ms\":" << graph_ref << ",\"graph_ms\":" << graph_new
                  << ",\"rerank_reference_ms\":" << rerank_ref << ",\"rerank_ms\":" << rerank_new << "}\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        equivalence();
        rerank_workspace_reuse();
        rerank_dimensions();
        if (argc == 2 && std::string(argv[1]) == "--benchmark") benchmark();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
