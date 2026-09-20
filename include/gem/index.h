#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gem {

struct VectorSetView {
    const float* data;
    std::size_t count;
    std::size_t dimension;
};

struct MultiVectors {
    std::size_t dimension = 0;
    std::vector<float> values;
    std::vector<std::uint64_t> offsets;
    std::size_t size() const { return offsets.empty() ? 0 : offsets.size() - 1; }
    VectorSetView at(std::size_t id) const;
    void validate() const;
};

struct EncodedCorpus {
    MultiVectors documents;
    std::vector<int> codes;
    std::vector<float> fine_centroids;
    std::vector<float> graph_centroids;
    std::vector<std::vector<std::size_t>> clusters;
    void validate() const;
};

struct BuildOptions {
    std::size_t m = 24;
    std::size_t ef_construction = 80;
    std::size_t seed = 100;
    // The original GEM builder materializes a dense centroid matrix.
    std::size_t distance_budget_bytes = std::size_t{4} << 30;
};

struct SearchOptions {
    std::size_t nprobe = 4;
    std::size_t ef = 4000;
    std::size_t rerank_k = 512;
    std::size_t k = 100;
};

struct Result {
    std::size_t id;
    // Original GEM distance: 1 - mean over query-token maximum inner products.
    float distance;
};

class QueryWorkspace {
 public:
    QueryWorkspace();
    ~QueryWorkspace();
    QueryWorkspace(QueryWorkspace&&) noexcept;
    QueryWorkspace& operator=(QueryWorkspace&&) noexcept;
    QueryWorkspace(const QueryWorkspace&) = delete;
    QueryWorkspace& operator=(const QueryWorkspace&) = delete;
 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Index;
};

// Build/load own their corpus; search is read-only. Each concurrent caller owns
// a separate workspace and output vector. Build/load/repair are exclusive.
class Index {
 public:
    static Index build(EncodedCorpus corpus, const BuildOptions& options = {});
    static Index load(const std::string& graph_file, EncodedCorpus corpus);
    void save(const std::string& graph_file) const;
    void repair();
    void search(VectorSetView query, const SearchOptions& options,
                QueryWorkspace& workspace, std::vector<Result>& results) const;
    std::size_t size() const;
    ~Index();
    Index(Index&&) noexcept;
    Index& operator=(Index&&) noexcept;
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
 private:
    struct Impl;
    explicit Index(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}  // namespace gem
