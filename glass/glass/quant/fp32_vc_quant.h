#pragma once

#include "../common.h"
#include "../memory.h"
#include "../simd/distance.h"

namespace glass {

template <Metric metric, int DIM = 0> struct FP32VCQuantizer {
  using data_type = float;
  constexpr static int kAlign = 16;
  int d, d_align, code_num = 0;
  int* code_size = nullptr;
  int* code_index = nullptr;
  char *codes = nullptr;

  FP32VCQuantizer() = default;

  explicit FP32VCQuantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(nullptr) {
      }

  ~FP32VCQuantizer() { free(codes); free(code_size); free(code_index); }

  void train(const float *data, const int * p_num, int64_t n) {
    for (int i = 0; i < n; i++){
      code_num += p_num[i];
    }
    code_size = (int *)alloc2M(n * 4);
    std::memcpy(code_size, p_num, n * 4);
    code_index = (int *)alloc2M(n * 4);
    code_index[0] = 0;
    for(int i = 1; i < n; i++){
      code_index[i] = code_index[i - 1] + code_size[i - 1];
    }
    codes = (char *)alloc2M(n * code_num * 4);
    for (int64_t i = 0; i < n; ++i) {
      encode(data + code_index[i] * d, get_data(i), i);
    }
  }

  void encode(const float *from, char *to, int idx) { std::memcpy(to, from, d * 4 * code_size[idx]); }

  char *get_data(int u) const { return codes + code_index[u] * d * 4; }

  template <typename Pool>
  void reorder(const Pool &pool, const float *, int *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  template <int DALIGN = do_align(DIM, kAlign), int VEC_NUM = 0> struct Computer {
    using dist_type = float;
    constexpr static auto dist_func = L2SqrVC;
    const FP32VCQuantizer &quant;
    float *q = nullptr;
    int q_num = 0;
    Computer(const FP32VCQuantizer &quant, const float *query, const int _q_num)
        : quant(quant), q_num(_q_num), q((float *)alloc64B(quant.d * q_num * 4)) {
      std::memcpy(q, query, quant.d * q_num * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, q_num, (data_type *)quant.get_data(u), code_size[u], quant.d);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query, const int q_num) const {
    return Computer<0, 0>(*this, query, q_num);
  }
};

} // namespace glass
