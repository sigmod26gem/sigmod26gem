#include "kernels.h"
#include <cblas.h>
#include <cstdint>
#include <stdexcept>

namespace gem::detail {
void dot_table(const float* a, std::size_t n, const float* b, std::size_t m,
               std::size_t dimension, float* output) {
    if (n > INT32_MAX || m > INT32_MAX || dimension > INT32_MAX)
        throw std::overflow_error("BLAS dimensions exceed int32");
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, dimension,
                1.0f, a, dimension, b, dimension, 0.0f, output, m);
}
}  // namespace gem::detail
