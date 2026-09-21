#pragma once
#include "gem/io.h"
#include "gem/options.h"

namespace gem::app {
struct RunConfig {
    CorpusFiles data;
    BuildOptions build;
    SearchOptions search;
    std::string action = "search", index, index_output, query_vectors, query_lengths, qrels, output;
    std::size_t workers = 1, inner_threads = 1, warmup = 1, repeats = 3, limit = 0;
};
RunConfig read_config(const std::string& filename);
}  // namespace gem::app
