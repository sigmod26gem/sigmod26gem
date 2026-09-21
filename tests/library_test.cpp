#include "gem/index.h"
#include "distance/kernels.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unistd.h>

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

gem::EncodedCorpus corpus() {
    gem::EncodedCorpus c;
    c.documents.dimension = 8;
    c.documents.offsets.push_back(0);
    std::mt19937 gen(42);
    std::normal_distribution<float> normal;
    for (int i = 0; i < 24; ++i) {
        float norm = 0;
        for (int j = 0; j < 8; ++j) {
            const float x = normal(gen);
            c.fine_centroids.push_back(x);
            norm += x * x;
        }
        for (int j = 0; j < 8; ++j) c.fine_centroids[i * 8 + j] /= std::sqrt(norm);
    }
    c.graph_centroids.assign(c.fine_centroids.begin(), c.fine_centroids.begin() + 24);
    c.clusters.resize(3);  // Include an empty cluster.
    for (std::size_t id = 0; id < 48; ++id) {
        for (std::size_t t = 0; t < 1 + id % 5; ++t) {
            const int code = (id * 3 + t / 2) % 24;  // Repeated codes retain qEMD mass.
            c.codes.push_back(code);
            c.documents.values.insert(c.documents.values.end(), c.fine_centroids.begin() + code * 8,
                                       c.fine_centroids.begin() + (code + 1) * 8);
        }
        c.documents.offsets.push_back(c.codes.size());
        c.clusters[id % 2].push_back(id);
    }
    c.clusters[1].insert(c.clusters[1].begin(), 0);  // Two clusters share an entry.
    return c;
}

void kernels() {
    std::mt19937 gen(8);
    std::uniform_real_distribution<float> uniform(-1, 1);
    for (std::size_t n : {1, 7, 32, 37, 320}) {
        std::vector<float> table(n * 24), scratch(n);
        for (auto& x : table) x = uniform(gen);
        const std::vector<int> codes = {3, 20, 3, 0, 23, 12};
        Eigen::Map<const Eigen::MatrixXf> matrix(table.data(), n, 24);
        Eigen::VectorXf maxima = Eigen::VectorXf::Constant(n, -9);
        for (int c : codes) maxima = maxima.cwiseMax(matrix.col(c));
        const float expected = (1.0f - maxima.array()).sum() / n;
        check(std::abs(expected - gem::detail::code_distance(table.data(), n, 24,
                     codes.data(), codes.size(), scratch.data())) < 1e-6f, "graph scorer mismatch");
        const std::vector<int> unique = {3, 20, 0, 23, 12};
        check(expected == gem::detail::code_distance(table.data(), n, 24,
                     unique.data(), unique.size(), scratch.data()), "code dedup changed MaxSim");
    }
    auto c = corpus();
    gem::detail::RerankWorkspace scratch;
    for (std::size_t i = 0; i < 48; ++i) {
        auto a = c.documents.at(i), b = c.documents.at(47 - i);
        using Row = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const Row> am(a.data, a.count, a.dimension), bm(b.data, b.count, b.dimension);
        Eigen::MatrixXf scores = am * bm.transpose();
        const float expected = 1 - scores.rowwise().maxCoeff().sum() / a.count;
        gem::detail::prepare_rerank_query(a, scratch);
        check(std::abs(expected - gem::detail::rerank_distance(b, scratch)) < 1e-6f,
              "rerank scorer mismatch");
    }
    const float center_scores[] = {1, 0, 0, 1};
    const int a[] = {0, 1}, b[] = {1, 0}, d[] = {0, 0};
    check(std::abs(gem::detail::qemd_distance(a, 2, b, 2, center_scores, 2)) < 1e-6f, "qEMD identity");
    check(std::abs(gem::detail::qemd_distance(a, 2, d, 2, center_scores, 2) - .5f) < 1e-6f,
          "qEMD multiplicity/stride");
}

bool equal(const std::vector<gem::Result>& a, const std::vector<gem::Result>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].id != b[i].id || a[i].distance != b[i].distance) return false;
    return true;
}

int main() {
    try {
        kernels();
        auto input = corpus();
        gem::BuildOptions build;
        build.m = 8;
        build.ef_construction = 24;
        auto index = gem::Index::build(input, build);
        index.repair();
        const auto file = std::filesystem::temp_directory_path() / ("gem-test-" + std::to_string(getpid()) + ".bin");
        index.save(file.string());
        auto loaded = gem::Index::load(file.string(), input);
        std::filesystem::remove(file);
        gem::SearchOptions options;
        options.nprobe = 3;
        options.ef = 48;
        options.rerank_k = 24;
        options.k = 10;
        gem::QueryWorkspace w1, w2;
        std::vector<std::vector<gem::Result>> expected(48);
        for (std::size_t i = 0; i < 48; ++i) {
            std::vector<gem::Result> actual;
            index.search(input.documents.at(i), options, w1, expected[i]);
            loaded.search(input.documents.at(i), options, w2, actual);
            check(equal(actual, expected[i]), "save/load mismatch");
            std::vector<std::size_t> ids;
            for (auto result : actual) ids.push_back(result.id);
            std::sort(ids.begin(), ids.end());
            check(std::adjacent_find(ids.begin(), ids.end()) == ids.end(), "duplicate result document");
        }
        std::vector<gem::SearchOptions> variants(3, options);
        variants[0].nprobe = 1;
        variants[0].ef = 16;
        variants[0].rerank_k = 12;
        variants[1].nprobe = 2;
        variants[1].ef = 32;
        std::vector<std::vector<std::vector<gem::Result>>> mixed(3,
            std::vector<std::vector<gem::Result>>(48));
        for (std::size_t v = 0; v < variants.size(); ++v)
            for (std::size_t q = 0; q < 48; ++q)
                loaded.search(input.documents.at(q), variants[v], w1, mixed[v][q]);
        std::vector<std::future<void>> tasks;
        for (int worker = 0; worker < 8; ++worker) tasks.push_back(std::async(std::launch::async, [&, worker] {
            gem::QueryWorkspace workspace;
            std::vector<gem::Result> actual;
            for (int repeat = 0; repeat < 4; ++repeat)
                for (std::size_t i = 0; i < 48; ++i) {
                    const auto v = (i + worker + repeat) % variants.size();
                    loaded.search(input.documents.at(i), variants[v], workspace, actual);
                    check(equal(actual, mixed[v][i]), "concurrent mask/ef/workspace mismatch");
                }
        }));
        for (auto& task : tasks) task.get();
        auto bad = corpus();
        bad.codes[0] = 1000;
        bool rejected = false;
        try { bad.validate(); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "out-of-range code accepted");
        rejected = false;
        build.distance_budget_bytes = 1;
        try { auto unused = gem::Index::build(input, build); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "distance budget ignored");
        std::cout << "kernels, qEMD, build, repair, save/load, 8-thread equivalence and validation passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
