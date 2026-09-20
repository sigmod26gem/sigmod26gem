#include "workspace.h"
namespace gem {
QueryWorkspace::QueryWorkspace() : impl_(new Impl) {}
QueryWorkspace::~QueryWorkspace() = default;
QueryWorkspace::QueryWorkspace(QueryWorkspace&&) noexcept = default;
QueryWorkspace& QueryWorkspace::operator=(QueryWorkspace&&) noexcept = default;
}  // namespace gem
