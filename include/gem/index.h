#pragma once

#include "gem/data.h"
#include "gem/options.h"
#include "gem/result.h"
#include "gem/workspace.h"
#include <memory>
#include <string>
#include <vector>

namespace gem {

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
