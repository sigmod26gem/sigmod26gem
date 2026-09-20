#pragma once

#include <cstdint>
#include <cstdio>
#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "../common.h"
#include "avx2.h"
#include "avx512.h"

namespace glass {

template <typename T1, typename T2, typename U, typename... Params>
using Dist = U (*)(const T1 *, const T2 *, int, Params...);

GLASS_INLINE inline void prefetch_L1(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T0);
#else
  __builtin_prefetch(address, 0, 3);
#endif
}

GLASS_INLINE inline void prefetch_L2(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T1);
#else
  __builtin_prefetch(address, 0, 2);
#endif
}

GLASS_INLINE inline void prefetch_L3(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T2);
#else
  __builtin_prefetch(address, 0, 1);
#endif
}

inline void mem_prefetch(char *ptr, const int num_lines) {
  switch (num_lines) {
  default:
    [[fallthrough]];
  case 28:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 27:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 26:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 25:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 24:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 23:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 22:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 21:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 20:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 19:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 18:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 17:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 16:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 15:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 14:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 13:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 12:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 11:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 10:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 9:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 8:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 7:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 6:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 5:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 4:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 3:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 2:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 1:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 0:
    break;
  }
}

FAST_BEGIN
inline float L2SqrRef(const float *x, const float *y, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sum;
}
FAST_END

FAST_BEGIN
inline float IPRef(const float *x, const float *y, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
}
FAST_END

inline float L2Sqr(const float *x, const float *y, int d) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm512_loadu_ps(x);
    x += 16;
    auto yy = _mm512_loadu_ps(y);
    y += 16;
    auto t = _mm512_sub_ps(xx, yy);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(t, t));
  }
  return reduce_add_f32x16(sum);
#elif defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm256_loadu_ps(x);
    x += 8;
    auto yy = _mm256_loadu_ps(y);
    y += 8;
    auto t = _mm256_sub_ps(xx, yy);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
  }
  return reduce_add_f32x8(sum);
#elif defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0);
  for (int32_t i = 0; i < d; i += 4) {
    auto xx = vld1q_f32(x + i);
    auto yy = vld1q_f32(y + i);
    auto t = vsubq_f32(xx, yy);
    sum = vmlaq_f32(sum, t, t);
  }
  return vaddvq_f32(sum);
#else
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sum;
#endif
}

inline float IP(const float *x, const float *y, int d) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm512_loadu_ps(x);
    x += 16;
    auto yy = _mm512_loadu_ps(y);
    y += 16;
    sum = _mm512_add_ps(sum, _mm512_mul_ps(xx, yy));
  }
  return -reduce_add_f32x16(sum);
#elif defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm256_loadu_ps(x);
    x += 8;
    auto yy = _mm256_loadu_ps(y);
    y += 8;
    sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
  }
  return -reduce_add_f32x8(sum);
#elif defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0);
  for (int32_t i = 0; i < d; i += 4) {
    auto xx = vld1q_f32(x + i);
    auto yy = vld1q_f32(y + i);
    sum = vmlaq_f32(sum, xx, yy);
  }
  return vaddvq_f32(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * y[i];
  }
  return -sum;
#endif
}

#if defined(__ARM_NEON)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")

inline float16_t IPFP16(const float16_t *x, const float16_t *y, int d) {
  float16x8_t sum = vdupq_n_f16(0);
  for (int32_t i = 0; i < d; i += 8) {
    auto xx = vld1q_f16(x + i);
    auto yy = vld1q_f16(y + i);
    auto prod = vmulq_f16(xx, yy);
    sum = vaddq_f16(sum, prod);
  }
  sum = vpaddq_f16(sum, sum);
  sum = vpaddq_f16(sum, sum);
  sum = vpaddq_f16(sum, sum);
  return vget_lane_f16(vget_low_f16(sum), 0);
}
#endif

inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  __m512 dot5 = _mm512_set1_ps(0.5f);
  __m512 const_255 = _mm512_set1_ps(255.0f);
  for (int i = 0; i < d; i += 16) {
    auto zz = _mm_loadu_epi8(y + i);
    auto zzz = _mm512_cvtepu8_epi32(zz);
    auto yy = _mm512_cvtepi32_ps(zzz);
    yy = _mm512_add_ps(yy, dot5);
    auto mi512 = _mm512_loadu_ps(mi + i);
    auto dif512 = _mm512_loadu_ps(dif + i);
    yy = _mm512_mul_ps(yy, dif512);
    yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
    auto xx = _mm512_loadu_ps(x + i);
    auto d = _mm512_sub_ps(_mm512_mul_ps(xx, const_255), yy);
    sum = _mm512_fmadd_ps(d, d, sum);
  }
  return reduce_add_f32x16(sum);
// #elif defined(__ARM_NEON)
//   float32x4_t sum = vdupq_n_f32(0.0f);
//   static const float32x4_t dot5 = vdupq_n_f32(0.5f);
//   static const float32x4_t const_255 = vdupq_n_f32(255.0f);

//   for (int i = 0; i < d; i += 8) {
//     uint16x8_t yy = vmovl_u8(vld1_u8(y + i));
//     float32x4_t yy1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(yy)));
//     float32x4_t yy2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(yy)));
//     yy1 = vaddq_f32(yy1, dot5);
//     yy2 = vaddq_f32(yy2, dot5);
//     float32x4_t mi1 = vld1q_f32(mi + i);
//     float32x4_t mi2 = vld1q_f32(mi + i + 4);
//     float32x4_t dif1 = vld1q_f32(dif + i);
//     float32x4_t dif2 = vld1q_f32(dif + i + 4);
//     yy1 = vmulq_f32(yy1, dif1);
//     yy2 = vmulq_f32(yy2, dif2);
//     yy1 = vaddq_f32(yy1, vmulq_f32(mi1, const_255));
//     yy2 = vaddq_f32(yy2, vmulq_f32(mi2, const_255));
//     float32x4_t xx1 = vld1q_f32(x + i);
//     float32x4_t xx2 = vld1q_f32(x + i + 4);
//     float32x4_t d1 = vsubq_f32(vmulq_f32(xx1, const_255), yy1);
//     float32x4_t d2 = vsubq_f32(vmulq_f32(xx2, const_255), yy2);
//     sum = vmlaq_f32(sum, d1, d1);
//     sum = vmlaq_f32(sum, d2, d2);
//   }
//   return vaddvq_f32(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = (y[i] + 0.5f);
    yy = yy * dif[i] + mi[i] * 255.0f;
    float dif_val = x[i] * 255.0f - yy;
    sum += dif_val * dif_val;
  }
  return sum;
#endif
}


inline int32_t L2SqrSQ8(const uint8_t* x, const uint8_t* y, int d) {
#if defined(__ARM_NEON)
  uint32x4_t sum = vdupq_n_u32(0);
  for (int i = 0; i < d; i += 16) {
    uint8x16_t xx = vld1q_u8(x + i);
    uint8x16_t yy = vld1q_u8(y + i);
    uint8x16_t diff = vabdq_u8(xx, yy);
    uint8x8_t diff1 = vget_low_u8(diff);
    uint8x8_t diff2 = vget_high_u8(diff);
    uint32x4_t diff1_sq_sum = vpaddlq_u16(vmull_u8(diff1, diff1));
    uint32x4_t diff2_sq_sum = vpaddlq_u16(vmull_u8(diff2, diff2));
    sum = vaddq_u32(sum, diff1_sq_sum);
    sum = vaddq_u32(sum, diff2_sq_sum);
  }
  return vaddvq_u32(sum);
#else
  int32_t sum = 0;
  for (int i = 0; i < d; ++i) {
    int32_t diff = x[i] - y[i];
    sum += diff * diff;
  }
  return sum;
#endif
}


inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {

#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  __m512 dot5 = _mm512_set1_ps(0.5f);
  __m512 const_255 = _mm512_set1_ps(255.0f);
  for (int i = 0; i < d; i += 16) {
    auto zz = _mm_loadu_epi8(y + i);
    auto zzz = _mm512_cvtepu8_epi32(zz);
    auto yy = _mm512_cvtepi32_ps(zzz);
    yy = _mm512_add_ps(yy, dot5);
    auto mi512 = _mm512_loadu_ps(mi + i);
    auto dif512 = _mm512_loadu_ps(dif + i);
    yy = _mm512_mul_ps(yy, dif512);
    yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
    auto xx = _mm512_loadu_ps(x + i);
    sum = _mm512_fmadd_ps(xx, yy, sum);
  }
  return -reduce_add_f32x16(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = y[i] + 0.5f;
    yy = yy * dif[i] + mi[i] * 255.0f;
    sum += x[i] * yy;
  }
  return -sum;
#endif
}

inline int32_t L2SqrSQ4(const uint8_t *x, const uint8_t *y, int d) {
#if defined(__AVX2__)
  __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
  __m256i mask = _mm256_set1_epi8(0xf);
  for (int i = 0; i < d; i += 64) {
    auto xx = _mm256_loadu_si256((__m256i *)(x + i / 2));
    auto yy = _mm256_loadu_si256((__m256i *)(y + i / 2));
    auto xx1 = _mm256_and_si256(xx, mask);
    auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
    auto yy1 = _mm256_and_si256(yy, mask);
    auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
    auto d1 = _mm256_sub_epi8(xx1, yy1);
    auto d2 = _mm256_sub_epi8(xx2, yy2);
    d1 = _mm256_abs_epi8(d1);
    d2 = _mm256_abs_epi8(d2);
    sum1 = _mm256_add_epi16(sum1, _mm256_maddubs_epi16(d1, d1));
    sum2 = _mm256_add_epi16(sum2, _mm256_maddubs_epi16(d2, d2));
  }
  sum1 = _mm256_add_epi32(sum1, sum2);
  return reduce_add_i16x16(sum1);
#elif defined(__ARM_NEON)
  uint32x4_t sum = vdupq_n_u32(0);
  static const uint8x16_t mask = vdupq_n_u8(0xf);
  for (int i = 0; i < d; i += 32) {
    uint8x16_t xx = vld1q_u8(x + i / 2);
    uint8x16_t yy = vld1q_u8(y + i / 2);
    uint8x16_t xx1 = vandq_u8(xx, mask);
    uint8x16_t xx2 = vandq_u8(vshrq_n_u8(xx, 4), mask);
    uint8x16_t yy1 = vandq_u8(yy, mask);
    uint8x16_t yy2 = vandq_u8(vshrq_n_u8(yy, 4), mask);
    uint8x16_t d1 = vabdq_u8(xx1, yy1);
    uint8x16_t d2 = vabdq_u8(xx2, yy2);
    uint16x8_t d1_sq = vpaddlq_u8(vmulq_u8(d1, d1));
    uint16x8_t d2_sq = vpaddlq_u8(vmulq_u8(d2, d2));
    uint16x8_t d_sq = vaddq_u16(d1_sq, d2_sq);
    sum = vaddq_u32(sum, vpaddlq_u16(d_sq));
  }
  return vaddvq_u32(sum);
#else
  int32_t sum = 0;
  for (int i = 0; i < d; i += 2) {
    int32_t xx = x[i / 2] & 15;
    int32_t yy = y[i / 2] & 15;
    sum += (xx - yy) * (xx - yy);

    xx = x[i / 2] >> 4 & 15;
    yy = y[i / 2] >> 4 & 15;
    sum += (xx - yy) * (xx - yy);
  }
  return sum;
#endif
}

inline float L2SqrVC(const float* p, size_t p_size, const float* q, size_t q_size, size_t dim) {
    float sum = 0.0f;

    // Iterate over each vector in q
    for (size_t i = 0; i < q_size; ++i) {
        const float* vec_q = q + i * dim; // Pointer to the i-th vector in q
        float maxDist = 999999.9f; // Start with 0 to find maximum distance

        // Find the maximum L2 distance to any vector in p
        for (size_t j = 0; j < p_size; ++j) {
            const float* vec_p = p + j * dim; // Pointer to the j-th vector in p
            maxDist = std::min(maxDist, L2Sqr(vec_q, vec_p, dim));
        }

        sum += maxDist;
    }

    // Return the average maximum distance from each vector in q to any vector in p
    return sum / q_size;
}

inline float IPVC(const float* p, size_t p_size, const float* q, size_t q_size, size_t dim) {
    float sum = 0.0f;

    // Iterate over each vector in q
    for (size_t i = 0; i < q_size; ++i) {
        const float* vec_q = q + i * dim; // Pointer to the i-th vector in q
        float maxDist = 0.0f; // Start with 0 to find maximum distance

        // Find the maximum L2 distance to any vector in p
        for (size_t j = 0; j < p_size; ++j) {
            const float* vec_p = p + j * dim; // Pointer to the j-th vector in p
            maxDist = std::max(maxDist, IP(vec_q, vec_p, dim));
        }

        sum += maxDist;
    }

    // Return the average maximum distance from each vector in q to any vector in p
    return sum / q_size;
}

} // namespace glass
