#include "pipeline.h"
#include "distance/kernels.h"
#include <algorithm>

namespace gem::detail {
void rerank_candidates(VectorSetView query, const EncodedCorpus& corpus, const SearchOptions& options,
                       SearchScratch& w, std::vector<Result>& results) {
    const auto count = std::min(options.rerank_k, w.candidates.size());
    std::partial_sort(w.candidates.begin(), w.candidates.begin() + count, w.candidates.end(),
                      [](auto a, auto b) { return a.first < b.first; });
    results.clear();
    results.reserve(count);
    if (count) prepare_rerank_query(query, w.rerank);
    for (std::size_t i = 0; i < count; ++i) {
        const auto id = w.candidates[i].second;
        results.push_back({id, rerank_distance(corpus.documents.at(id), w.rerank)});
    }
    std::sort(results.begin(), results.end(), [](auto a, auto b) { return a.distance < b.distance; });
    if (results.size() > options.k) results.resize(options.k);
}
}  // namespace gem::detail
