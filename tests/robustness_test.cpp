#include "benchmark.h"
#include "gem/io.h"
#include "graph/search_scratch.h"
#include "graph/visited_list_pool.h"
#include "cnpy.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    check(rejected, "invalid input accepted");
}
gem::EncodedCorpus corpus() {
    gem::EncodedCorpus c;
    c.documents.dimension = 4;
    c.documents.offsets = {0};
    c.fine_centroids = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    c.graph_centroids = {1,0,0,0};
    c.clusters.resize(1);
    for (std::size_t id = 0; id < 16; ++id) {
        const auto code = id % 4;
        c.documents.values.insert(c.documents.values.end(), c.fine_centroids.begin() + code * 4,
                                  c.fine_centroids.begin() + (code + 1) * 4);
        c.documents.offsets.push_back(id + 1);
        c.codes.push_back(code);
        c.clusters[0].push_back(id);
    }
    return c;
}
void npy_file(const fs::path& path, unsigned char version, std::string shape = "1, 1, 4",
              bool fortran = false, bool payload = true) {
    std::string header = "{'descr': '<f4', 'fortran_order': " + std::string(fortran ? "True" : "False") +
                         ", 'shape': (" + shape + "), }";
    header.resize(511, ' ');
    header += '\n';
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY", 6);
    out.put(version); out.put(0);
    for (int byte = 0; byte < (version == 1 ? 2 : 4); ++byte) out.put((header.size() >> (8 * byte)) & 255);
    out.write(header.data(), header.size());
    const float data[] = {1, 2, 3, 4};
    if (payload) out.write(reinterpret_cast<const char*>(data), sizeof(data));
}
std::size_t descriptors() {
    return std::distance(fs::directory_iterator("/proc/self/fd"), fs::directory_iterator{});
}
void numpy(const fs::path& root) {
    const auto path = root / "array.npy";
    for (unsigned char version : {1, 2, 3}) {
        npy_file(path, version);
        const auto array = gem::load_queries(path.string());
        check(array.size() == 1 && array.dimension == 4 && array.values == std::vector<float>({1,2,3,4}),
              "long NPY header corrupted payload");
    }
    npy_file(path, 1, "1, 1, 4", true);
    rejects([&] { gem::load_queries(path.string()); });
    npy_file(path, 1, "18446744073709551615, 4");
    rejects([&] { gem::load_queries(path.string()); });
    npy_file(path, 1, "1, -1, 4");
    rejects([&] { gem::load_queries(path.string()); });
    npy_file(path, 1, "1, 1, 4", false, false);
    const auto before = fs::exists("/proc/self/fd") ? descriptors() : 0;
    for (int repeat = 0; repeat < 64; ++repeat) rejects([&] { cnpy::npy_load(path.string()); });
    if (before) check(before == descriptors(), "NPY exception leaked a file descriptor");
    fs::resize_file(path, 6);
    rejects([&] { gem::load_queries(path.string()); });
}

struct GraphHeader {
    std::size_t offset0, capacity, count, stride, label_offset, data_offset;
    int max_level;
    unsigned int entry;
    std::size_t max_m, max_m0, m;
    double mult;
    std::size_t ef;
};

void graph(const fs::path& root, const gem::Index& index, const gem::EncodedCorpus& c) {
    const auto path = root / "graph.bin";
    index.save(path.string());
    std::ifstream in(path, std::ios::binary);
    const std::vector<char> original{std::istreambuf_iterator<char>(in), {}};
    GraphHeader header;
    std::memcpy(&header, original.data(), sizeof(header));
    check(header.count == c.documents.size(), "unexpected graph header layout");
    auto write = [&](const std::vector<char>& bytes) {
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), bytes.size());
    };
    auto corrupt = [&](std::size_t offset, auto value) {
        auto bytes = original;
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
        write(bytes);
        rejects([&] { gem::Index::load(path.string(), c); });
    };
    corrupt(offsetof(GraphHeader, data_offset), std::size_t(1));
    corrupt(offsetof(GraphHeader, capacity), std::size_t(0));
    corrupt(offsetof(GraphHeader, count), std::numeric_limits<std::size_t>::max());
    corrupt(offsetof(GraphHeader, entry), static_cast<unsigned int>(header.count));
    corrupt(offsetof(GraphHeader, mult), std::numeric_limits<double>::quiet_NaN());
    corrupt(sizeof(header), static_cast<unsigned short>(header.max_m0 + 1));
    corrupt(sizeof(header) + sizeof(unsigned int), std::numeric_limits<unsigned int>::max());
    std::size_t first_label;
    std::memcpy(&first_label, original.data() + sizeof(header) + header.label_offset, sizeof(first_label));
    corrupt(sizeof(header) + header.stride + header.label_offset, first_label);
    corrupt(sizeof(header) + header.count * header.stride, 1U);
    for (std::size_t bytes : {std::size_t(4), sizeof(header) - 1, original.size() - 1}) {
        write(std::vector<char>(original.begin(), original.begin() + bytes));
        rejects([&] { gem::Index::load(path.string(), c); });
    }
    write(original);
    auto loaded = gem::Index::load(path.string(), c);
    check(loaded.size() == c.documents.size(), "valid graph rejected");
}

void benchmark(const fs::path& root, const gem::Index& index, const gem::EncodedCorpus& c) {
    gem::app::RunConfig config;
    config.search.nprobe = 1;
    config.search.ef = 16;
    config.search.rerank_k = 16;
    config.search.k = 4;
    config.limit = 1;
    config.warmup = 0;
    config.repeats = 1;
    config.qrels = (root / "qrels.tsv").string();
    for (const auto line : {"0 99999\n", "-1 0\n", "0 -1\n", "16 0\n", "0 0 extra\n"}) {
        { std::ofstream out(config.qrels); out << line; }
        rejects([&] { gem::app::benchmark(index, c.documents, config); });
    }
    { std::ofstream out(config.qrels); out << "0 0\n15 15\n"; }
    gem::app::benchmark(index, c.documents, config);  // Valid qrels outside query.limit.
    config.warmup = SIZE_MAX;
    rejects([&] { gem::app::benchmark(index, c.documents, config); });
    config.warmup = 0;
    if (fs::exists("/dev/full")) {
        config.output = "/dev/full";
        rejects([&] { gem::app::benchmark(index, c.documents, config); });
    }
}

void workspace() {
    hnswlib::VisitedListPool pool(0, 10);
    hnswlib::VisitedList* borrowed = nullptr;
    try {
        auto lease = pool.acquire();
        borrowed = lease.get();
        throw std::runtime_error("simulated distance failure");
    } catch (const std::runtime_error&) {}
    auto reused = pool.acquire();
    check(reused.get() == borrowed, "visited list not returned after an exception");
    hnswlib::EntrySearchScratch<float> scratch;
    scratch.reset(10, 3);
    for (int i = 0; i < 256; ++i) scratch.frontiers[0].emplace(float(i), i);
    const auto capacity = scratch.frontiers[0].capacity();
    scratch.reset(10, 1);
    scratch.reset(10, 3);
    check(scratch.frontiers[0].empty() && scratch.frontiers[0].capacity() == capacity,
          "frontier allocation was discarded between queries");
    scratch.visited.test_and_set(0);
    scratch.reset(10, 1);
    check(!scratch.visited.contains(0), "visited bits retained between queries");
}
}  // namespace

int main() {
    const auto root = fs::temp_directory_path() / ("gem-robustness-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        numpy(root);
        workspace();
        auto input = corpus();
        gem::BuildOptions build;
        build.m = 4;
        build.ef_construction = 16;
        auto index = gem::Index::build(input, build);
        graph(root, index, input);
        benchmark(root, index, input);
        fs::remove_all(root);
        std::cout << "corrupt input, failed output, benchmark validation and workspace reuse passed\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
