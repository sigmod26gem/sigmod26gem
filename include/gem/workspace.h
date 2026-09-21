#pragma once

#include <memory>

namespace gem {
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
}  // namespace gem
