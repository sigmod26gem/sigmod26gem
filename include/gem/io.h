#pragma once
#include "gem/index.h"

namespace gem {
struct CorpusFiles {
    std::string vectors, lengths, codes;
    std::string fine_centroids, graph_centroids, clusters;
    std::size_t shards = 1;
};
EncodedCorpus load_corpus(const CorpusFiles& files);
MultiVectors load_queries(const std::string& vectors, const std::string& lengths = "");
}  // namespace gem
