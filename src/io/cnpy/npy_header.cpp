#include "npy_header.h"
#include <charconv>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>

namespace cnpy::detail {
namespace {
std::size_t number(const std::string& text) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("invalid NPY integer");
    return value;
}
void read_exact(FILE* file, void* buffer, std::size_t bytes) {
    if (std::fread(buffer, 1, bytes, file) != bytes)
        throw std::runtime_error("truncated NPY header");
}
}  // namespace

std::size_t remaining_bytes(FILE* file) {
    const auto pos = ftello(file);
    if (pos < 0 || fseeko(file, 0, SEEK_END)) throw std::runtime_error("cannot seek NPY file");
    const auto end = ftello(file);
    if (end < pos || fseeko(file, pos, SEEK_SET)) throw std::runtime_error("cannot size NPY file");
    return static_cast<std::size_t>(end - pos);
}

std::size_t array_bytes(const std::vector<std::size_t>& shape, std::size_t word_size) {
    if (!word_size) throw std::runtime_error("invalid NPY item size");
    std::size_t count = 1;
    for (auto dimension : shape) {
        if (dimension && count > std::numeric_limits<std::size_t>::max() / dimension)
            throw std::runtime_error("NPY shape overflows address space");
        count *= dimension;
    }
    if (count > std::numeric_limits<std::size_t>::max() / word_size)
        throw std::runtime_error("NPY payload overflows address space");
    return count * word_size;
}

NpyHeader read_header(FILE* file) {
    unsigned char prefix[8], length[4]{};
    read_exact(file, prefix, sizeof(prefix));
    if (std::memcmp(prefix, "\x93NUMPY", 6) || prefix[6] < 1 || prefix[6] > 3 || prefix[7] != 0)
        throw std::runtime_error("invalid NPY magic or version");
    read_exact(file, length, prefix[6] == 1 ? 2 : 4);
    const std::size_t header_size = std::size_t(length[0]) | (std::size_t(length[1]) << 8) |
        (std::size_t(length[2]) << 16) | (std::size_t(length[3]) << 24);
    if (!header_size || header_size > remaining_bytes(file))
        throw std::runtime_error("truncated NPY header");
    std::string header(header_size, '\0');
    read_exact(file, header.data(), header.size());
    if (header.back() != '\n') throw std::runtime_error("invalid NPY header terminator");
    std::smatch match;
    NpyHeader result;
    if (!std::regex_search(header, match, std::regex("['\"]descr['\"]\\s*:\\s*['\"]([<|][biufc][0-9]+)['\"]")))
        throw std::runtime_error("little-endian numeric NPY required");
    result.dtype = match[1];
    result.word_size = number(result.dtype.substr(2));
    if (!result.word_size || (result.dtype[0] == '|' && result.word_size != 1))
        throw std::runtime_error("invalid NPY byte order or item size");
    if (!std::regex_search(header, match, std::regex("['\"]fortran_order['\"]\\s*:\\s*(True|False)")))
        throw std::runtime_error("missing NPY array order");
    result.fortran_order = match[1] == "True";
    if (!std::regex_search(header, match, std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)")))
        throw std::runtime_error("missing NPY shape");
    const std::string shape = match[1];
    if (!std::regex_match(shape, std::regex("\\s*([0-9]+\\s*(,\\s*[0-9]+\\s*)*,?\\s*)?")))
        throw std::runtime_error("invalid NPY shape");
    const std::regex integer("[0-9]+");
    for (auto it = std::sregex_iterator(shape.begin(), shape.end(), integer); it != std::sregex_iterator(); ++it)
        result.shape.push_back(number(it->str()));
    array_bytes(result.shape, result.word_size);
    return result;
}
}  // namespace cnpy::detail
