#include "gem/index.h"
#include "graph/index.h"
#include "search/pipeline.h"

namespace gem {
struct Index::Impl {
    EncodedCorpus corpus;
    std::unique_ptr<detail::Graph> graph;
    explicit Impl(EncodedCorpus input) : corpus(std::move(input)) { corpus.validate(); }
};

Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Index::~Index() = default;
Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;
std::size_t Index::size() const { return impl_->corpus.documents.size(); }

Index Index::build(EncodedCorpus corpus, const BuildOptions& options) {
    auto impl = std::make_unique<Impl>(std::move(corpus));
    impl->graph = std::make_unique<detail::Graph>(detail::Graph::build(impl->corpus, options));
    return Index(std::move(impl));
}

Index Index::load(const std::string& path, EncodedCorpus corpus) {
    auto impl = std::make_unique<Impl>(std::move(corpus));
    impl->graph = std::make_unique<detail::Graph>(detail::Graph::load(path, impl->corpus));
    return Index(std::move(impl));
}

void Index::save(const std::string& path) const { impl_->graph->save(path); }
void Index::repair() { impl_->graph->repair(impl_->corpus); }

void Index::search(VectorSetView query, const SearchOptions& options,
                   QueryWorkspace& workspace, std::vector<Result>& results) const {
    const auto& corpus = impl_->corpus;
    detail::validate_query(query, corpus, options);
    if (!workspace.impl_) workspace.impl_ = std::make_unique<QueryWorkspace::Impl>();
    auto& scratch = *workspace.impl_;
    detail::route_query(query, corpus, options, impl_->graph->clusters(), impl_->graph->capacity(), scratch);
    detail::score_fine_centers(query, corpus, scratch);
    impl_->graph->traverse(scratch.fine_scores.data(), corpus.fine_centroids.size() / query.dimension,
                           query.count, scratch.maxima.data(), options.ef,
                           scratch.entries, scratch.allowed, scratch.candidates);
    detail::rerank_candidates(query, corpus, options, scratch, results);
}
}  // namespace gem
