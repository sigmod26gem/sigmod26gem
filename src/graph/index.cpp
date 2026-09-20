#include "index_impl.h"
#include "distance/kernels.h"
#include <cstring>
#include <stdexcept>

namespace gem::detail {
namespace {
float score_document(const vectorset* query, const vectorset* doc, int) {
    const auto& q = static_cast<const ScoreQuery&>(*query);
    vectorset aligned(nullptr, nullptr, 0, 0);
    // Legacy HNSW descriptors may reside at unaligned byte offsets.
    std::memcpy(&aligned, doc, sizeof(aligned));
    return code_distance(q.data, q.vecnum, q.dim, aligned.codes, aligned.vecnum, q.maxima);
}
}  // namespace

Graph::Impl::Impl(const EncodedCorpus& corpus) {
    documents.reserve(corpus.documents.size());
    for (std::size_t id = 0; id < corpus.documents.size(); ++id) {
        const auto offset = corpus.documents.offsets[id];
        const auto view = corpus.documents.at(id);
        documents.emplace_back(const_cast<float*>(view.data),
                               const_cast<int*>(corpus.codes.data() + offset),
                               view.dimension, view.count);
    }
    space = std::make_unique<hnswlib::L2VSSpace>(corpus.documents.dimension);
}

void Graph::Impl::prepare_search(const EncodedCorpus& corpus) {
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

Graph::Graph(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Graph::~Graph() = default;
Graph::Graph(Graph&&) noexcept = default;
Graph& Graph::operator=(Graph&&) noexcept = default;
std::size_t Graph::capacity() const { return impl_->graph->max_elements_; }
const std::vector<std::vector<std::size_t>>& Graph::clusters() const { return impl_->internal_clusters; }
}  // namespace gem::detail
