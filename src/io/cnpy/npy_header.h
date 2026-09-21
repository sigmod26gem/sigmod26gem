#pragma once
#include <cstdio>
#include <string>
#include <vector>

namespace cnpy::detail {
struct NpyHeader {
    std::vector<std::size_t> shape;
    std::string dtype;
    std::size_t word_size;
    bool fortran_order;
};
NpyHeader read_header(FILE* file);
std::size_t array_bytes(const std::vector<std::size_t>& shape, std::size_t word_size);
std::size_t remaining_bytes(FILE* file);
}  // namespace cnpy::detail
