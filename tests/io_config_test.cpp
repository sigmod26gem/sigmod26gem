#include "config.h"
#include "gem/io.h"
#include "cnpy.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

template<class F> void rejects(F action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    if (!failed) throw std::runtime_error("invalid input was accepted");
}

int main() {
    const auto root = std::filesystem::temp_directory_path() / ("gem-io-test-" + std::to_string(getpid()));
    std::filesystem::create_directories(root);
    try {
        const float values[] = {1, 0, 0, 1, 1, 0};
        const std::int64_t lengths[] = {2, 1};
        cnpy::npy_save((root / "vectors.npy").string(), values, {3, 2});
        cnpy::npy_save((root / "lengths.npy").string(), lengths, {2});
        auto vectors = gem::load_queries((root / "vectors.npy").string(), (root / "lengths.npy").string());
        if (vectors.size() != 2 || vectors.at(0).count != 2 || vectors.at(1).data[0] != 1)
            throw std::runtime_error("ragged NPY load mismatch");
        rejects([&] { gem::load_queries((root / "lengths.npy").string(), (root / "lengths.npy").string()); });
        cnpy::npy_save((root / "wrong_dtype.npy").string(), lengths, {1, 2});
        rejects([&] { gem::load_queries((root / "wrong_dtype.npy").string(), (root / "lengths.npy").string()); });
        cnpy::npy_save((root / "bad_lengths.npy").string(), lengths, {1});
        rejects([&] { gem::load_queries((root / "vectors.npy").string(), (root / "bad_lengths.npy").string()); });
        const std::string valid = "[data]\nvectors=v.npy\nlengths=l.npy\ncodes=c.npy\nfine_centroids=f.npy\ngraph_centroids=g.npy\nclusters=c.txt\n[index]\npath=graph.bin\n[query]\nvectors=q.npy\n";
        auto write = [&](const std::string& text) { std::ofstream(root / "test.ini") << text; };
        write(valid);
        auto config = gem::app::read_config((root / "test.ini").string());
        if (config.data.vectors != (root / "v.npy").string() || config.workers != 1)
            throw std::runtime_error("config defaults/relative paths mismatch");
        for (const std::string suffix : {"[runtime]\nworkers=2\nworkers=3\n", "[search]\neff=10\n",
                  "[runtime]\nworkers=-1\n", "[runtime]\nworkers=3x\n", "[unknown]\n", "[runtime]\nworkers=0\n"}) {
            write(valid + suffix);
            rejects([&] { gem::app::read_config((root / "test.ini").string()); });
        }
        std::filesystem::remove_all(root);
        std::cout << "NPY dtype/shape/offsets and strict INI tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
