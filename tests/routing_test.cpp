#include "search/pipeline.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Original full-table routing, including partial_sort ties and hash insertion order.
void reference(gem::VectorSetView query, const gem::EncodedCorpus& corpus,
               const gem::SearchOptions& options,
               const std::vector<std::vector<std::size_t>>& internal_clusters,
               std::size_t capacity, gem::detail::SearchScratch& w) {
    const auto nc = corpus.clusters.size();
    w.route_scores.resize(query.count * nc);
    w.route_order.resize(nc);
    w.allowed.assign(capacity, false);
    w.entries.clear();
    std::unordered_set<int> selected_clusters;
    gem::detail::dot_table(query.data, query.count, corpus.graph_centroids.data(), nc,
                          query.dimension, w.route_scores.data());
    for (std::size_t token = 0; token < query.count; ++token) {
        for (std::size_t c = 0; c < nc; ++c)
            w.route_order[c] = {w.route_scores[token * nc + c], static_cast<int>(c)};
        std::partial_sort(w.route_order.begin(), w.route_order.begin() + options.nprobe, w.route_order.end(),
                          [](auto a, auto b) { return a.first > b.first; });
        for (std::size_t j = 0; j < options.nprobe; ++j) selected_clusters.insert(w.route_order[j].second);
    }
    for (auto cluster : selected_clusters) {
        if (corpus.clusters[cluster].empty()) continue;
        w.entries.push_back(corpus.clusters[cluster].front());
        for (auto id : internal_clusters[cluster]) w.allowed[id] = true;
    }
}

void routing() {
    std::mt19937 random(79);
    std::uniform_real_distribution<float> uniform(-1, 1);
    for (std::size_t dimension : {4, 17, 128}) {
        for (std::size_t nc : {7, 67, 1024}) {
            gem::EncodedCorpus corpus;
            corpus.documents.dimension = dimension;
            corpus.graph_centroids.resize(nc * dimension);
            corpus.clusters.resize(nc);
            std::vector<std::vector<std::size_t>> internal(nc);
            for (auto& value : corpus.graph_centroids) value = uniform(random);
            for (std::size_t c = 0; c < nc; ++c) {
                // Empty clusters and a nonidentity external/internal ID mapping.
                if (c % 13) {
                    corpus.clusters[c] = {nc - c - 1};
                    internal[c] = {c, (c + 1) % nc};
                }
                if (c && c % 3 == 0)
                    std::copy_n(corpus.graph_centroids.data(), dimension,
                                corpus.graph_centroids.data() + c * dimension);
            }
            gem::detail::SearchScratch expected, actual;
            std::vector<float> query(321 * dimension);
            for (auto& value : query) value = uniform(random);
            // A zero query token forces a complete tie between all centroids.
            std::fill_n(query.data(), dimension, 0.0f);
            for (std::size_t n : {1, 31, 32, 33, 63, 65, 320, 321, 7}) {
                for (std::size_t nprobe : {std::size_t{1}, std::size_t{4}, nc}) {
                    gem::SearchOptions options;
                    options.nprobe = nprobe;
                    const gem::VectorSetView view{query.data(), n, dimension};
                    reference(view, corpus, options, internal, nc, expected);
                    gem::detail::route_query(view, corpus, options, internal, nc, actual);
                    if (actual.entries != expected.entries || actual.allowed != expected.allowed)
                        std::cerr << "routing mismatch: dimension=" << dimension << " clusters=" << nc
                                  << " tokens=" << n << " nprobe=" << nprobe << '\n';
                    check(actual.entries == expected.entries, "routing changed entry order");
                    check(actual.allowed == expected.allowed, "routing changed the document mask");
                    for (std::size_t j = 0; j < nprobe; ++j)
                        check(actual.route_order[j].second == expected.route_order[j].second,
                              "routing changed final-token top-k ties");
                }
            }
        }
    }
}
}  // namespace

int main() {
    try {
        routing();
        std::cout << "routing matches full-table entries, masks and top-k ties\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
