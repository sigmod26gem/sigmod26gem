#pragma once
#include "gem/data.h"
#include "gem/options.h"
#include "search_scratch.h"
#include <memory>
#include <string>
#include <utility>

namespace gem::detail {
// Owns the graph; document descriptors borrow the Index-owned corpus.
class Graph {
 public:
    static Graph build(const EncodedCorpus& corpus, const BuildOptions& options);
    static Graph load(const std::string& path, const EncodedCorpus& corpus);
    void save(const std::string& path) const;
    void repair(const EncodedCorpus& corpus);
    void traverse(float* scores, std::size_t centers, std::size_t query_tokens,
                  float* maxima, std::size_t ef,
                  const std::vector<std::size_t>& entries, const std::vector<bool>& allowed,
                  std::vector<std::pair<float, std::size_t>>& candidates,
                  hnswlib::EntrySearchScratch<float>& scratch) const;
    std::size_t capacity() const;
    const std::vector<std::vector<std::size_t>>& clusters() const;
    ~Graph();
    Graph(Graph&&) noexcept;
    Graph& operator=(Graph&&) noexcept;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
 private:
    struct Impl;
    explicit Graph(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}  // namespace gem::detail
