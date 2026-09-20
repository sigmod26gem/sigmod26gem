#pragma once
#include "gem/data.h"
#include "gem/options.h"
#include "gem/result.h"
#include "workspace.h"

namespace gem::detail {
void validate_query(VectorSetView query, const EncodedCorpus& corpus, const SearchOptions& options);
void route_query(VectorSetView query, const EncodedCorpus& corpus, const SearchOptions& options,
                 const std::vector<std::vector<std::size_t>>& internal_clusters,
                 std::size_t capacity, SearchScratch& scratch);
void score_fine_centers(VectorSetView query, const EncodedCorpus& corpus, SearchScratch& scratch);
void rerank_candidates(VectorSetView query, const EncodedCorpus& corpus, const SearchOptions& options,
                       SearchScratch& scratch, std::vector<Result>& results);
}  // namespace gem::detail
