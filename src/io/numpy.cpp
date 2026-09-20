#include "gem/io.h"
#include "cnpy.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gem {
namespace {
struct Array { cnpy::NpyArray data; std::string dtype; };
Array read(const std::string& path) {
    auto array = cnpy::npy_load(path);
    if (array.fortran_order) throw std::runtime_error("row-major NPY required: " + path);
    const auto dtype = array.dtype;
    return {std::move(array), dtype};
}

float half(std::uint16_t x) {
    const int sign = x >> 15, exponent = (x >> 10) & 31, fraction = x & 1023;
    if (exponent == 31) throw std::runtime_error("non-finite FP16 value");
    const float value = exponent == 0 ? std::ldexp(float(fraction), -24)
                                     : std::ldexp(float(fraction + 1024), exponent - 25);
    return sign ? -value : value;
}

void floats(const Array& a, std::vector<float>& out) {
    if (a.dtype != "<f2" && a.dtype != "<f4") throw std::runtime_error("expected FP16 or FP32 NPY");
    const auto offset = out.size();
    if (a.data.num_vals > out.max_size() - offset) throw std::overflow_error("vector data too large");
    out.resize(offset + a.data.num_vals);
    if (a.dtype == "<f4") {
        if (a.data.num_vals) std::memcpy(out.data() + offset, a.data.data<float>(), a.data.num_bytes());
    } else {
        for (std::size_t i = 0; i < a.data.num_vals; ++i) out[offset + i] = half(a.data.data<std::uint16_t>()[i]);
    }
}

std::vector<std::uint64_t> integers(const std::string& path) {
    auto a = read(path);
    if (a.data.shape.size() != 1) throw std::runtime_error("expected one-dimensional integer NPY: " + path);
    std::vector<std::uint64_t> result(a.data.num_vals);
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (a.dtype == "<i4") {
            auto v = a.data.data<std::int32_t>()[i];
            if (v < 0) throw std::runtime_error("negative integer: " + path);
            result[i] = v;
        } else if (a.dtype == "<i8") {
            auto v = a.data.data<std::int64_t>()[i];
            if (v < 0) throw std::runtime_error("negative integer: " + path);
            result[i] = v;
        } else if (a.dtype == "<u4") result[i] = a.data.data<std::uint32_t>()[i];
        else if (a.dtype == "<u8") result[i] = a.data.data<std::uint64_t>()[i];
        else throw std::runtime_error("expected 32/64-bit integer NPY: " + path);
    }
    return result;
}

std::string shard(std::string path, std::size_t id, std::size_t count) {
    const auto pos = path.find("{shard}");
    if (pos != std::string::npos) path.replace(pos, 7, std::to_string(id));
    else if (count != 1) throw std::invalid_argument("multi-shard path must contain {shard}");
    return path;
}

void append_vectors(const std::string& path, const std::string& lens_path, MultiVectors& out) {
    auto a = read(path);
    const auto& shape = a.data.shape;
    if (shape.size() != 2 && shape.size() != 3) throw std::runtime_error("vectors must be a 2D or 3D NPY");
    const auto d = shape.back();
    if (!d || (out.dimension && out.dimension != d)) throw std::runtime_error("embedding dimension mismatch");
    out.dimension = d;
    std::vector<std::uint64_t> lengths;
    if (!lens_path.empty()) lengths = integers(lens_path);
    else if (shape.size() == 3) lengths.assign(shape[0], shape[1]);
    else throw std::runtime_error("2D vectors require lengths");
    std::uint64_t sum = 0;
    for (auto length : lengths) {
        if (!length || length > INT32_MAX || sum > std::numeric_limits<std::uint64_t>::max() - length)
            throw std::runtime_error("invalid vector-set length");
        sum += length;
    }
    if (sum != a.data.num_vals / d) throw std::runtime_error("sum(lengths) differs from vector count");
    if (out.offsets.empty()) out.offsets.push_back(0);
    for (auto length : lengths) out.offsets.push_back(out.offsets.back() + length);
    floats(a, out.values);
}
}  // namespace

MultiVectors load_queries(const std::string& vectors, const std::string& lengths) {
    MultiVectors result;
    append_vectors(vectors, lengths, result);
    result.validate();
    return result;
}

EncodedCorpus load_corpus(const CorpusFiles& files) {
    if (!files.shards) throw std::invalid_argument("data.shards must be positive");
    EncodedCorpus result;
    for (std::size_t i = 0; i < files.shards; ++i) {
        append_vectors(shard(files.vectors, i, files.shards), shard(files.lengths, i, files.shards), result.documents);
        const auto values = integers(shard(files.codes, i, files.shards));
        const auto begin = result.codes.size();
        result.codes.resize(begin + values.size());
        for (std::size_t j = 0; j < values.size(); ++j) {
            if (values[j] > INT32_MAX) throw std::runtime_error("code exceeds int32");
            result.codes[begin + j] = values[j];
        }
        if (result.codes.size() != result.documents.offsets.back())
            throw std::runtime_error("shard token/code counts differ");
    }
    for (const auto& pair : {std::make_pair(files.fine_centroids, &result.fine_centroids),
                            std::make_pair(files.graph_centroids, &result.graph_centroids)}) {
        auto a = read(pair.first);
        if (a.data.shape.size() != 2 || a.data.shape[1] != result.documents.dimension)
            throw std::runtime_error("centroid dimension mismatch");
        floats(a, *pair.second);
    }
    std::ifstream in(files.clusters);
    if (!in) throw std::runtime_error("cannot open cluster file: " + files.clusters);
    std::string line;
    while (std::getline(in, line)) {
        result.clusters.emplace_back();
        std::istringstream row(line);
        std::int64_t id;
        while (row >> id) {
            if (id < 0) throw std::runtime_error("negative cluster document id");
            result.clusters.back().push_back(id);
        }
        if (!row.eof()) throw std::runtime_error("invalid cluster document id");
    }
    if (in.bad()) throw std::runtime_error("cannot read cluster file: " + files.clusters);
    return result;
}
}  // namespace gem
