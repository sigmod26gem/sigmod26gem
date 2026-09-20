#include "config.h"
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace gem::app {
namespace {
std::string trim(std::string s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    return begin == std::string::npos ? "" : s.substr(begin, s.find_last_not_of(" \t\r\n") - begin + 1);
}
}  // namespace
RunConfig read_config(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("cannot open config: " + filename);
    RunConfig c;
    std::map<std::string, std::string*> strings = {
        {"run.action", &c.action}, {"index.path", &c.index}, {"index.output", &c.index_output},
        {"data.vectors", &c.data.vectors}, {"data.lengths", &c.data.lengths}, {"data.codes", &c.data.codes},
        {"data.fine_centroids", &c.data.fine_centroids}, {"data.graph_centroids", &c.data.graph_centroids},
        {"data.clusters", &c.data.clusters}, {"query.vectors", &c.query_vectors},
        {"query.lengths", &c.query_lengths}, {"query.qrels", &c.qrels}, {"benchmark.output", &c.output}};
    std::map<std::string, std::size_t*> numbers = {
        {"data.shards", &c.data.shards}, {"build.m", &c.build.m}, {"build.ef_construction", &c.build.ef_construction},
        {"build.seed", &c.build.seed}, {"build.distance_budget_bytes", &c.build.distance_budget_bytes},
        {"search.nprobe", &c.search.nprobe}, {"search.ef", &c.search.ef}, {"search.rerank_k", &c.search.rerank_k},
        {"search.k", &c.search.k}, {"runtime.workers", &c.workers}, {"runtime.inner_threads", &c.inner_threads},
        {"benchmark.warmup", &c.warmup}, {"benchmark.repeats", &c.repeats}, {"query.limit", &c.limit}};
    const std::set<std::string> sections = {"run", "index", "data", "query", "build", "search", "runtime", "benchmark"};
    std::set<std::string> seen;
    std::string line, section;
    std::size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        if (line[0] == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            if (!sections.count(section)) throw std::runtime_error("unknown section: " + section);
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos || section.empty()) throw std::runtime_error("invalid config line " + std::to_string(lineno));
        const auto key = section + "." + trim(line.substr(0, eq));
        const auto value = trim(line.substr(eq + 1));
        if (!seen.insert(key).second) throw std::runtime_error("duplicate key: " + key);
        if (strings.count(key)) *strings.at(key) = value;
        else if (numbers.count(key)) {
            std::size_t parsed = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
                throw std::runtime_error("invalid nonnegative integer: " + key);
            *numbers.at(key) = parsed;
        } else throw std::runtime_error("unknown key: " + key);
    }
    for (auto& item : strings) {
        if (item.first == "run.action" || item.second->empty()) continue;
        const auto path = std::filesystem::path(*item.second);
        *item.second = (path.is_absolute() ? path : std::filesystem::absolute(filename).parent_path() / path).lexically_normal().string();
    }
    if (c.action != "build" && c.action != "search" && c.action != "repair")
        throw std::runtime_error("run.action must be build, search or repair");
    for (const auto& key : {"data.vectors", "data.lengths", "data.codes", "data.fine_centroids", "data.graph_centroids", "data.clusters", "index.path"})
        if (strings.at(key)->empty()) throw std::runtime_error(std::string("missing key: ") + key);
    if (!c.data.shards || !c.workers || !c.inner_threads || !c.repeats || c.workers > INT32_MAX || c.inner_threads > INT32_MAX || c.warmup > SIZE_MAX - c.repeats)
        throw std::runtime_error("invalid shard, worker, thread or repetition count");
    if (c.action == "search" && c.query_vectors.empty()) throw std::runtime_error("query.vectors is required");
    if (c.action == "repair" && (c.index_output.empty() || c.index_output == c.index))
        throw std::runtime_error("repair requires a separate index.output graph path");
    return c;
}
}  // namespace gem::app
