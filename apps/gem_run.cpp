#include "benchmark.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <cblas.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: gem_run config.ini\n";
        return 2;
    }
    try {
        const auto config = gem::app::read_config(argv[1]);
        omp_set_num_threads(config.inner_threads);
        omp_set_max_active_levels(1);
        openblas_set_num_threads(config.inner_threads);
        const auto begin = std::chrono::steady_clock::now();
        auto corpus = gem::load_corpus(config.data);
        std::cout << "documents=" << corpus.documents.size() << " dimension=" << corpus.documents.dimension
                  << " fine_centers=" << corpus.fine_centroids.size() / corpus.documents.dimension
                  << " graph_clusters=" << corpus.clusters.size() << '\n';
        if (config.action == "build" || config.action == "repair") {
            const auto destination = config.action == "build" ? config.index : config.index_output;
            if (std::filesystem::exists(destination)) throw std::runtime_error("output graph already exists: " + destination);
            auto index = config.action == "build" ? gem::Index::build(std::move(corpus), config.build)
                                                    : gem::Index::load(config.index, std::move(corpus));
            index.repair();
            index.save(destination);
            std::cout << "saved=" << destination << " elapsed_seconds="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() << '\n';
        } else {
            auto index = gem::Index::load(config.index, std::move(corpus));
            auto queries = gem::load_queries(config.query_vectors, config.query_lengths);
            std::cout << "load_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
                      << " inner_threads=" << config.inner_threads << '\n';
            gem::app::benchmark(index, queries, config);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "GEM: " << e.what() << '\n';
        return 1;
    }
}
