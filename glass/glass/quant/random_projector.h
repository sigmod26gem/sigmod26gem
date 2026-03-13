#pragma once

#include "../common.h"
#include "../memory.h"
#include "fp32_quant.h"
#include "../simd/distance.h"
#include <random>

namespace glass {

template <Metric metric, int d_reduced = 64,
          typename Reorderer = FP32Quantizer<metric>, int DIM = 0>
struct FP32RPQuantizer {
  using data_type = float;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;
  float *rndAs = nullptr;

  Reorderer reorderer;

  FP32RPQuantizer() = default;

  explicit FP32RPQuantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_reduced * 4),
        reorderer(dim) {}

  ~FP32RPQuantizer() { free(codes); free(rndAs); }

  void train(const float *data, int64_t n) {
    rndAs = (float *)alloc2M(d * d_reduced * 4);
    std::mt19937 rng(int(0));
    std::normal_distribution<float> nd;
    for (int i = 0; i < d_reduced; i++) {
      for (int j = 0; j < d; j++) {
        rndAs[i * d + j] = nd(rng);
      }
    }
    codes = (char *)alloc2M(n * code_size);
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
    reorderer.train(data, n);
  }

  void encode(const float *from, char *to) const { 
    float* to_float = (float* ) to;
    for(int i = 0; i < d_reduced; i++) {
      to_float[i] = IP(from, rndAs + i * d, d);
    }  
  }

  char *get_data(int u) const { return codes + u * code_size; }


  template <typename Pool>
  void reorder(const Pool &pool, const float *q, int *dst, int k) const {
    int cap = pool.capacity();
    auto computer = reorderer.get_computer(q);
    searcher::MaxHeap<typename Reorderer::template Computer<0>::dist_type> heap(
        k);
    for (int i = 0; i < cap; ++i) {
      if (i + 1 < cap) {
        computer.prefetch(pool.id(i + 1), 1);
      }
      int id = pool.id(i);
      float dist = computer(id);
      heap.push(id, dist);
    }
    for (int i = 0; i < k; ++i) {
      dst[i] = heap.pop();
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = float;
    constexpr static auto dist_func = metric == Metric::L2 ? L2Sqr : IP;
    const FP32RPQuantizer &quant;
    float *q = nullptr;
    Computer(const FP32RPQuantizer &quant, const float *query)
        : quant(quant), q((float *)alloc64B(d_reduced * 4)) {
      quant.encode(query, (char* )q);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), d_reduced);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

}