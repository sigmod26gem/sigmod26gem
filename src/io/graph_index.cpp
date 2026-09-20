#include "graph/index_impl.h"

namespace gem::detail {
Graph Graph::load(const std::string& path, const EncodedCorpus& corpus) {
    auto impl = std::make_unique<Impl>(corpus);
    impl->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        impl->space.get(), path, false, corpus.documents.size() + 1);
    impl->prepare_search(corpus);
    return Graph(std::move(impl));
}
void Graph::save(const std::string& path) const { impl_->graph->saveIndex(path); }
}  // namespace gem::detail
