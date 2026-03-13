#pragma once

// #include "../common.h"
// #include "../memory.h"
// #include "fp32_quant.h"
// #include "../simd/distance.h"
// #include <faiss/VectorTransform.h>
// #include <random>
// #include <arm_neon.h>

// namespace glass {

// template <Metric metric, int d_reduced = 64,
//           typename Reorderer = FP32Quantizer<metric>, int DIM = 0>
// struct SQ8RPQuantizerFP16 {
//   using data_type = float;
//   constexpr static int kAlign = 16;
//   int d, d_align;
//   int64_t code_size;
//   char *codes = nullptr;
//   faiss::PCAMatrix *pca;
//   float16_t *trans = nullptr;
//   float16_t mx = -HUGE_VALF, mi = HUGE_VALF, dif;
//   Reorderer reorderer;

//   SQ8RPQuantizerFP16() = default;

//   explicit SQ8RPQuantizerFP16(int dim)
//       : d(dim), d_align(do_align(dim, kAlign)), code_size(d_reduced),
//         reorderer(dim) {}

//   ~SQ8RPQuantizerFP16() { free(codes); delete pca; free(trans); }

//   void train(const float *data, int64_t n) {
//     pca = new faiss::PCAMatrix(d, d_reduced);
//     pca->train(n, data);
//     float* temp_trans = pca->A.data();
//     trans = (float16_t *)alloc2M(d * d_reduced * sizeof(float16_t));
//     for (int i = 0; i < d * d_reduced; ++i)
//       trans[i] = temp_trans[i];
//     float16_t *temp_codes = (float16_t *)alloc2M(n * d_reduced * sizeof(float16_t));
//     float16_t *temp_data = (float16_t *)alloc2M(n * d * sizeof(float16_t));
// #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < d; ++j) {
//         temp_data[i * d + j] = data[i * d + j];
//       }
//       encode_rp(temp_data + i * d, temp_codes + i * d_reduced);
//     }
// #pragma omp parallel for reduction(max: mx) reduction(min: mi)
//     for (int i = 0; i < n * d_reduced; ++i) {
//       mx = std::max(mx, temp_codes[i]);
//       mi = std::min(mi, temp_codes[i]);
//     }
//     dif = mx - mi;
//     codes = (char *)alloc2M(n * code_size);
// #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < n; ++i) {
//       encode_sq(temp_codes + i * d_reduced, get_data(i));
//     }
//     free(temp_codes);
//     free(temp_data);
//     reorderer.train(data, n);
//   }

//   void encode_rp(const float16_t *from, float16_t *to) const { 
//     for (int i = 0; i < d_reduced; ++i) {
//       to[i] = IPFP16(from, trans + i * d, d);
//     }
//   }

//   void encode_sq(const float16_t* from, char *to) const {
//     for (int j = 0; j < d_reduced; ++j) {
//       float16_t x = (from[j] - mi) / dif;
//       if (x < 0.0) {
//         x = 0.0;
//       }
//       if (x > 0.999) {
//         x = 0.999;
//       }
//       uint8_t y = x * 255;
//       to[j] = y;
//     }
//   }

//   void encode(const float* from, char* to) const {
//     float16_t* dim_reduced = (float16_t*)alloc64B(d_reduced * sizeof(float16_t));
//     float16_t* temp_data = (float16_t*)alloc64B(d * sizeof(float16_t));
//     for(int j = 0; j < d; j++)
//       temp_data[j] = from[j];
//     encode_rp(temp_data, dim_reduced);
//     encode_sq(dim_reduced, to);
//     free(dim_reduced);
//     free(temp_data);
//   }

//   char *get_data(int u) const { return codes + u * code_size; }


//   template <typename Pool>
//   void reorder(const Pool &pool, const float *q, int *dst, int k) const {
//     int cap = pool.capacity();
//     auto computer = reorderer.get_computer(q);
//     searcher::MaxHeap<typename Reorderer::template Computer<0>::dist_type> heap(
//         k);
//     for (int i = 0; i < cap; ++i) {
//       if (i + 1 < cap) {
//         computer.prefetch(pool.id(i + 1), 1);
//       }
//       int id = pool.id(i);
//       float dist = computer(id);
//       heap.push(id, dist);
//     }
//     for (int i = 0; i < k; ++i) {
//       dst[i] = heap.pop();
//     }
//   }

//   template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
//     using dist_type = int32_t;
//     constexpr static auto dist_func = L2SqrSQ8;
//     const SQ8RPQuantizerFP16 &quant;
//     uint8_t *q = nullptr;
//     Computer(const SQ8RPQuantizerFP16 &quant, const float *query)
//         : quant(quant), q((uint8_t *)alloc64B(d_reduced)) {
//       quant.encode(query, (char* )q);
//     }
//     ~Computer() { free(q); }
//     dist_type operator()(int u) const {
//       return dist_func(q, (uint8_t *)quant.get_data(u), d_reduced);
//     }
//     void prefetch(int u, int lines) const {
//       mem_prefetch(quant.get_data(u), lines);
//     }
//   };

//   auto get_computer(const float *query) const {
//     return Computer<0>(*this, query);
//   }
// };

// }