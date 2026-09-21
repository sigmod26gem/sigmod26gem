#pragma once
#include "gem/data.h"
#include <string>

namespace gem {
struct CorpusFiles {
    std::string vectors, lengths, codes;
    std::string fine_centroids, graph_centroids, clusters;
    std::size_t shards = 1;
};
// Reads encoded arrays. Index::build/load validates corpus-wide invariants;
// standalone consumers can call EncodedCorpus::validate().
EncodedCorpus load_corpus(const CorpusFiles& files);
MultiVectors load_queries(const std::string& vectors, const std::string& lengths = "");
}  // namespace gem
