#pragma once

#include "../common.h"
#include "../memory.h"
#include "../neighbor.h"
#include "fp32_quant.h"
#include "../simd/distance.h"

#include <cmath>

namespace glass {

template <Metric metric, typename Reorderer = FP32Quantizer<metric>,
          int DIM = 0>
struct SQ8UQuantizer {
  using data_type = uint8_t;
  constexpr static int kAlign = 16;
  float mx = -HUGE_VALF, mi = HUGE_VALF, dif;
  int d, d_align;
  int64_t code_size;
  data_type *codes = nullptr;

  Reorderer reorderer;

  SQ8UQuantizer() = default;

  explicit SQ8UQuantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
        reorderer(dim) {}

  ~SQ8UQuantizer() { free(codes); }

  void train(const float *data, int n) {
    for (int64_t i = 0; i < n * d; ++i) {
      mx = std::max(mx, data[i]);
      mi = std::min(mi, data[i]);
    }
    dif = mx - mi;
    codes = (data_type *)alloc2M(n * code_size);
    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
    reorderer.train(data, n);
  }

  char *get_data(int u) const { return (char *)codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (int j = 0; j < d; ++j) {
      float x = (from[j] - mi) / dif;
      if (x < 0.0) {
        x = 0.0;
      }
      if (x > 0.999) {
        x = 0.999;
      }
      uint8_t y = x * 255;
      to[j] = y;
    }
  }

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
    using dist_type = int32_t;
    constexpr static auto dist_func = L2SqrSQ8;
    const SQ8UQuantizer &quant;
    uint8_t *q;
    Computer(const SQ8UQuantizer &quant, const float *query)
        : quant(quant), q((uint8_t *)alloc64B(quant.d_align * 4)) {
      quant.encode(query, (char *)q);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace glass
