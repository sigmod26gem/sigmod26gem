#include "gem/index.h"
#include "search/kernels.h"
#include "search/workspace.h"
#include <atomic>
#include <cmath>
#include <functional>
#include <stdexcept>
#include "hnswlib/hnswlib.h"

namespace gem {
namespace {
struct ScoreQuery : vectorset {
    float* maxima;
    ScoreQuery(float* scores, std::size_t centers, std::size_t tokens, float* scratch)
        : vectorset(scores, nullptr, centers, tokens), maxima(scratch) {}
};

float score_document(const vectorset* query, const vectorset* doc, int) {
    const auto& q = static_cast<const ScoreQuery&>(*query);
    vectorset aligned(nullptr, nullptr, 0, 0);
    // Legacy HNSW stores the descriptor at a potentially unaligned byte offset.
    std::memcpy(&aligned, doc, sizeof(aligned));
    return detail::code_distance(q.data, q.vecnum, q.dim, aligned.codes, aligned.vecnum, q.maxima);
}

std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("workspace size overflow");
    return a * b;
}
}  // namespace

struct Index::Impl {
    EncodedCorpus corpus;
    std::vector<vectorset> documents;
    std::vector<std::vector<std::size_t>> internal_clusters;
    std::unique_ptr<hnswlib::L2VSSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;

    explicit Impl(EncodedCorpus input) : corpus(std::move(input)) {
        corpus.validate();
        documents.reserve(corpus.documents.size());
        for (std::size_t id = 0; id < corpus.documents.size(); ++id) {
            const auto offset = corpus.documents.offsets[id];
            const auto view = corpus.documents.at(id);
            documents.emplace_back(corpus.documents.values.data() + offset * view.dimension,
                                   corpus.codes.data() + offset, view.dimension, view.count);
        }
        space.reset(new hnswlib::L2VSSpace(corpus.documents.dimension));
    }

    void prepare_search() {
        if (graph->cur_element_count != documents.size())
            throw std::invalid_argument("graph and corpus document counts differ");
        for (std::size_t id = 0; id < documents.size(); ++id) {
            if (graph->label_lookup_.find(id) == graph->label_lookup_.end())
                throw std::invalid_argument("graph labels must be corpus document ids");
            graph->loadDataAddress(&documents[id], id);
        }
        internal_clusters.resize(corpus.clusters.size());
        for (std::size_t i = 0; i < corpus.clusters.size(); ++i)
            for (auto id : corpus.clusters[i])
                internal_clusters[i].push_back(graph->label_lookup_.at(id));
        graph->fstdistfuncCluster = score_document;
    }
};

Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Index::~Index() = default;
Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;
std::size_t Index::size() const { return impl_->documents.size(); }

Index Index::build(EncodedCorpus corpus, const BuildOptions& options) {
    if (options.m < 2 || options.m > 10000 || options.ef_construction < options.m)
        throw std::invalid_argument("invalid HNSW build options");
    auto impl = std::make_unique<Impl>(std::move(corpus));
    const auto centers = impl->corpus.fine_centroids.size() / impl->corpus.documents.dimension;
    const auto elements = product(centers, centers);
    if (product(elements, sizeof(float)) > options.distance_budget_bytes)
        throw std::invalid_argument("dense centroid matrix exceeds build.distance_budget_bytes");
    std::vector<float> distances(elements);
    detail::dot_table(impl->corpus.fine_centroids.data(), centers,
                      impl->corpus.fine_centroids.data(), centers,
                      impl->corpus.documents.dimension, distances.data());
    impl->graph.reset(new hnswlib::HierarchicalNSW<float>(impl->space.get(), impl->documents.size() + 1,
                        options.m, options.ef_construction, options.seed));
    impl->graph->fstdistfuncClusterEMD = [centers](const vectorset* a, const vectorset* b, const float* table) {
        vectorset left(nullptr, nullptr, 0, 0), right(nullptr, nullptr, 0, 0);
        std::memcpy(&left, a, sizeof(left));
        std::memcpy(&right, b, sizeof(right));
        return detail::qemd_distance(left.codes, left.vecnum, right.codes, right.vecnum, table, centers);
    };
    // Preserve cluster and insertion order. Parallel build is validated separately.
    for (const auto& cluster : impl->corpus.clusters) {
        if (cluster.empty()) continue;
        impl->graph->entry_map.clear();
        for (auto id : cluster)
            impl->graph->addClusterPointEntry(&impl->documents[id], distances.data(), id, cluster.front());
    }
    impl->prepare_search();
    return Index(std::move(impl));
}

Index Index::load(const std::string& graph_file, EncodedCorpus corpus) {
    auto impl = std::make_unique<Impl>(std::move(corpus));
    impl->graph.reset(new hnswlib::HierarchicalNSW<float>(impl->space.get(), graph_file, false,
                                                       impl->documents.size() + 1));
    impl->prepare_search();
    return Index(std::move(impl));
}

void Index::save(const std::string& graph_file) const {
    impl_->graph->saveIndex(graph_file);
}

void Index::repair() {
    auto& graph = *impl_->graph;
    for (std::size_t i = 0; i < impl_->corpus.clusters.size(); ++i) {
        const auto& cluster = impl_->corpus.clusters[i];
        if (cluster.empty()) continue;
        graph.entry_map.clear();
        for (auto id : cluster) graph.entry_map[graph.label_lookup_.at(id)] = i;
        auto reachable = graph.searchNodesForFix(cluster.front(), i);
        std::size_t cursor = 0;
        for (std::size_t j = 1; j < cluster.size(); ++j) {
            const auto id = graph.label_lookup_.at(cluster[j]);
            if (graph.entry_map[id] == i + 1) continue;
            while (cursor < reachable.size()) {
                const auto from = reachable[cursor].first;
                if (graph.canAddEdgeinter(from)) {
                    graph.mutuallyConnectTwoInterElement(from, id);
                    if (graph.canAddEdgeinter(id)) graph.mutuallyConnectTwoInterElement(id, from);
                    break;
                }
                ++cursor;
            }
            auto more = graph.searchNodesForFix(cluster[j], i);
            reachable.insert(reachable.end(), more.begin(), more.end());
        }
    }
}

void Index::search(VectorSetView query, const SearchOptions& options,
                   QueryWorkspace& workspace, std::vector<Result>& results) const {
    const auto& corpus = impl_->corpus;
    if (!query.data || !query.count || query.count > INT32_MAX ||
        query.dimension != corpus.documents.dimension)
        throw std::invalid_argument("invalid query shape");
    if (!options.k || !options.nprobe || options.nprobe > corpus.clusters.size() ||
        options.rerank_k < options.k || options.ef < options.rerank_k)
        throw std::invalid_argument("require nprobe <= clusters and ef >= rerank_k >= k > 0");
    for (std::size_t i = 0; i < product(query.count, query.dimension); ++i)
        if (!std::isfinite(query.data[i])) throw std::invalid_argument("non-finite query");
    if (!workspace.impl_) workspace.impl_ = std::make_unique<QueryWorkspace::Impl>();
    auto& w = *workspace.impl_;
    const auto nc = corpus.clusters.size();
    const auto nf = corpus.fine_centroids.size() / query.dimension;
    w.route_scores.resize(product(query.count, nc));
    w.fine_scores.resize(product(query.count, nf));
    w.maxima.resize(query.count);
    w.route_order.resize(nc);
    w.allowed.assign(impl_->graph->max_elements_, false);
    // GEM consumes hash iteration order as entry order. Reusing the bucket count
    // would make traversal depend on previous queries, so keep this set local.
    std::unordered_set<int> selected_clusters;
    w.entries.clear();
    w.candidates.clear();
    results.clear();
    detail::dot_table(query.data, query.count, corpus.graph_centroids.data(), nc,
                      query.dimension, w.route_scores.data());
    for (std::size_t token = 0; token < query.count; ++token) {
        for (std::size_t c = 0; c < nc; ++c) w.route_order[c] = {w.route_scores[token * nc + c], static_cast<int>(c)};
        std::partial_sort(w.route_order.begin(), w.route_order.begin() + options.nprobe, w.route_order.end(),
                          [](auto a, auto b) { return a.first > b.first; });
        for (std::size_t j = 0; j < options.nprobe; ++j) selected_clusters.insert(w.route_order[j].second);
    }
    detail::dot_table(corpus.fine_centroids.data(), nf, query.data, query.count,
                      query.dimension, w.fine_scores.data());
    for (auto cluster : selected_clusters) {
        if (corpus.clusters[cluster].empty()) continue;
        w.entries.push_back(corpus.clusters[cluster].front());
        for (auto id : impl_->internal_clusters[cluster]) w.allowed[id] = true;
    }
    ScoreQuery scores(w.fine_scores.data(), nf, query.count, w.maxima.data());
    auto candidates = impl_->graph->searchKnnClusterEntries(&scores, options.ef, w.entries,
                                                          nullptr, &w.allowed, options.ef);
    while (!candidates.empty()) {
        w.candidates.push_back(candidates.top());
        candidates.pop();
    }
    const auto count = std::min(options.rerank_k, w.candidates.size());
    std::partial_sort(w.candidates.begin(), w.candidates.begin() + count, w.candidates.end(),
                      [](auto a, auto b) { return a.first < b.first; });
    results.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto id = w.candidates[i].second;
        results.push_back({id, detail::rerank_distance(query, corpus.documents.at(id), w.pair_scores)});
    }
    std::sort(results.begin(), results.end(), [](auto a, auto b) { return a.distance < b.distance; });
    if (results.size() > options.k) results.resize(options.k);
}
}  // namespace gem
