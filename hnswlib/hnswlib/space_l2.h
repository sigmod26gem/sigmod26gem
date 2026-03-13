#pragma once
#include "hnswlib.h"
#include "vectorset.h"
#include<algorithm>
#include <vector>
#include <cmath>
#include <omp.h> 
#include <Eigen/Dense>
#include <cassert>
#include <cblas.h>
#include "../otlib/EMD.h"
#include <chrono>

constexpr int NUM_CLUSTER_CALC = 262144;

inline std::atomic<int> l2_sqr_call_count(0);
inline std::atomic<int> l2_vec_call_count(0);

using namespace Eigen;

namespace hnswlib {

static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return res;
}

static float
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

#if defined(USE_AVX)

// Favor using AVX if available.
static float
InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128 v1, v2;
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return sum;
}

static float
InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif


#if defined(USE_AVX512)

static float
InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    size_t loop = qty16 / 4;
    
    while (loop--) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v3 = _mm512_loadu_ps(pVect1);
        __m512 v4 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v5 = _mm512_loadu_ps(pVect1);
        __m512 v6 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v7 = _mm512_loadu_ps(pVect1);
        __m512 v8 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        sum512 = _mm512_fmadd_ps(v3, v4, sum512);
        sum512 = _mm512_fmadd_ps(v5, v6, sum512);
        sum512 = _mm512_fmadd_ps(v7, v8, sum512);
    }

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
    }

    float sum = _mm512_reduce_add_ps(sum512);
    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_AVX)

static float
InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
static DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

static float
InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float
InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif

class InnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    InnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        }
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

~InnerProductSpace() {}
};


static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // l2_sqr_call_count.fetch_add(1, std::memory_order_relaxed); 
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

static float
MyInnerProduct(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += (*pVect1) * (*pVect2); // 计算内积
        pVect1++;
        pVect2++;
    }
    return res;
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // l2_sqr_call_count.fetch_add(1, std::memory_order_relaxed); 
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // l2_sqr_call_count.fetch_add(1, std::memory_order_relaxed); 
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

static float
L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // l2_sqr_call_count.fetch_add(1, std::memory_order_relaxed); 
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

static float
L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // l2_sqr_call_count.fetch_add(1, std::memory_order_relaxed); 
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif


#if defined(USE_SSE)
static float
L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);


    size_t qty4 = qty >> 2;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

class L2VSSpace : public SpaceInterface<float>{
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
public:
    L2VSSpace(size_t dim){
        fstdistfunc_ = L2Sqr;
        dim_ = dim;
        data_size_ = sizeof(float*) + sizeof(int) * 2;
    }

    size_t get_data_size(){
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2VSSpace(){}
};

class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};

static int
L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int res = 0;
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;

    qty = qty >> 2;
    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    int res = 0;
    unsigned char* a = (unsigned char*)pVect1;
    unsigned char* b = (unsigned char*)pVect2;

    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}



static float L2SqrVecCF(const vectorset* q, const vectorset* p, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    size_t q_vecnum = q->vecnum;
    size_t p_vecnum = p->vecnum ;
    // #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < q_vecnum; ++i) {
        const float* vec_q = q->data + i * (level + 1) * q->dim;
        float maxDist = 99999.9f;
        // #pragma omp simd reduction(min:maxDist)
        for (size_t j = 0; j < p_vecnum; ++j) {
            const float* vec_p = p->data + j * p->dim;
            __builtin_prefetch(vec_p + p->dim, 0, 1);  // 预取下一行数据
            float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
            maxDist = std::min(maxDist, dist);
        }
        sum1 += maxDist;
    }
    return sum1;
}

float max_inner_product_sum(const float* A, const float* B, int n, int m, int d) {
    // **使用 Eigen::Map<> 映射 A 和 B，不复制数据**
    // Eigen::Map<const Eigen::MatrixXf> A_mat(A, n, d);
    // Eigen::Map<const Eigen::MatrixXf> B_mat(B, m, d);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_mat(A, n, d);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_mat(B, m, d);
    Eigen::MatrixXf C = A_mat * B_mat.transpose();
    return C.rowwise().maxCoeff().sum();
}

static float L2SqrVecEigenCF(const vectorset* q, const vectorset* p, int level) {
    float sum1 = max_inner_product_sum(q->data, p->data, q->vecnum, p->vecnum, q->dim);
    return 1 - sum1 / q->vecnum;
}


void fast_dot_product_blas(int n, int d, int m, float* A, float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, m, d,
                1.0f, A, d,
                B, d,
                0.0f, C, m);
}

static float L2SqrVecBlasCF(const vectorset* q, const vectorset* p, float* C, int level) {
    float sum1 = 0.0f;
    size_t n = q->vecnum;
    size_t m = p->vecnum;
    // std::vector<float> C(n * m);
    fast_dot_product_blas(n, q->dim, m, q->data, p->data, C);
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < n; ++i) {
        float maxDist = -9.0f;
        for (size_t j = 0; j < m; ++j) {
            maxDist = std::max(maxDist, C[i * m + j]);
        }
        sum1 += 1 - maxDist;
    }
    return sum1;
}


static float L2SqrVecGetDistance(const float* q, const float* p, float* C, int n, int m, int d) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    // std::cout << q[0] << std::endl;
    // std::cout << p[0] << std::endl;
    // std::cout << C[0] << std::endl;
    // std::cout << n << ' ' << m << ' ' << d << std::endl;
    // #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        const float* vec_q = q + i * d;
        for (size_t j = 0; j < m; ++j) {
            const float* vec_p = p + j * d;
            C[i * m + j] = L2Sqrfunc_(vec_q, vec_p, &d);
        }
    }
    return 0;
}

static float L2SqrVecBlasDistance(const vectorset* q, const vectorset* p, float* C, int n, int m, int d) {
    float sum1 = 0.0f;
    std::vector<float> C2(n * m);
    fast_dot_product_blas(n, d, m, q->data, p->data, C);
    return 0;
}

// static float L2SqrVecEMD(const vectorset* q, const vectorset* p, int level) {
//     float sum1 = 0.0f;
//     float sum2 = 0.0f;
//     level = 0;
//     // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
//     float (*L2Sqrfunc_)(const void*, const void*, const void*);
//     #if defined(USE_AVX512)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
//     #elif defined(USE_AVX)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
//     #else 
//     L2Sqrfunc_ = L2Sqr;
//     #endif
//     size_t q_vecnum = q->vecnum;
//     size_t p_vecnum = p->vecnum ;

//     std::vector<float> aw(q_vecnum);
//     std::vector<float> av(q_vecnum);
//     std::vector<float> bw(p_vecnum);
//     std::vector<float> bv(p_vecnum);
//     for (int i = 0; i < q_vecnum; i++) {
//         const float* vec_q = q->data;
//         av[i] = vec_q[i];
//         aw[i] = 1.0;
//     }
//     for (int i = 0; i < p_vecnum; i++) {
//         const float* vec_p = p->data;
//         bv[i] = vec_p[i];
//         bw[i] = 1.0;
//     }
//     double dist = wasserstein(av,aw,bv,bw);
//     return dist;
// }

float hungarian_algorithm(const std::vector<std::vector<float>>& cost) {
    size_t n = cost.size();
    size_t m = cost[0].size();

    // Labels for the rows and columns
    std::vector<float> u(n, 0), v(m, 0);

    // Matching from rows to columns
    std::vector<int> p(m, -1);
    std::vector<int> way(m, -1);

    // Main loop for the Hungarian algorithm
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> min_v(m, std::numeric_limits<float>::infinity());
        std::vector<bool> used(m, false);

        int j0 = -1;
        p[0] = i;

        while (true) {
            j0 = -1;
            for (size_t j = 0; j < m; ++j) {
                if (!used[j]) {
                    float cur = cost[p[j]] [j] - u[p[j]] - v[j];
                    if (cur < min_v[j]) {
                        min_v[j] = cur;
                        way[j] = p[j];
                        if (min_v[j] < min_v[j0]) {
                            j0 = j;
                        }
                    }
                }
            }

            for (size_t j = 0; j < m; ++j) {
                if (used[j]) {
                    u[p[j]] += min_v[j];
                    v[j] -= min_v[j];
                } else {
                    min_v[j] -= min_v[j0];
                }
            }
            
            if (j0 == -1) {
                break;
            }
            p[j0] = i;
        }
    }

    float total_cost = 0;
    for (size_t j = 0; j < m; ++j) {
        total_cost += cost[p[j]] [j];
    }
    return total_cost;
}

static float L2SqrVecEMD(const vectorset* q, const vectorset* p, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    size_t n = q->vecnum;
    size_t m = p->vecnum;
    std::vector<double> dist_flat(n * m);
    // std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(m));
    for (size_t i = 0; i < n; ++i) {
        const float* vec_q = q->data + i * q->dim;
        for (size_t j = 0; j < m; ++j) {
            const float* vec_p = p->data + j * p->dim;
            float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
            dist_flat[i * m + j] = (double)dist;
        }
    }

    std::vector<double> a_hist(n, 1.0 / n);
    std::vector<double> b_hist(m, 1.0 / m);
    // 计算EMD（匹配后的最小搬运成本）
    float emd = EMD_wrap_self(n, m, a_hist.data(), b_hist.data(), dist_flat.data(), 1000);
    // std::cout<< emd << std::endl;
    return (float)emd;
}


static float L2SqrVecChamfer(const vectorset* q, const vectorset* p, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    size_t n = q->vecnum;
    size_t m = p->vecnum;
    std::vector<float> dist_flat(n * m);
    // std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(m));
    for (size_t i = 0; i < n; ++i) {
        const float* vec_q = q->data + i * q->dim;
        for (size_t j = 0; j < m; ++j) {
            const float* vec_p = p->data + j * p->dim;
            float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
            dist_flat[i * m + j] = dist;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        float mindist = 2.0f;
        for (size_t j = 0; j < m; ++j) {
            mindist = std::min(dist_flat[i * m + j], mindist);
        }
        sum1 += mindist;
    }

    return sum1 / n;
}


static float L2SqrVecClusterEMD(const vectorset* q, const vectorset* p, const float* cluster_dis) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 

    size_t n = q->vecnum;
    size_t m = p->vecnum ;
    std::vector<double> dist_flat(n * m);
    // std::cout << n << " " << m << std::endl;

    // #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        long long icode = (long long)q->codes[i] * NUM_CLUSTER_CALC;
        // std::cout << i << " " << icode << " "  << std::flush;
        for (size_t j = 0; j < m; ++j) {
            dist_flat[i * m + j] = (double)1.0f - cluster_dis[icode + p->codes[j]];
        }
    }
    // std::cout << dist_flat[0] << " " << dist_flat[n * m - 1] << std::endl;

    std::vector<double> a_hist(n, 1.0 / n);
    std::vector<double> b_hist(m, 1.0 / m);
    // 计算EMD（匹配后的最小搬运成本）
    float emd = EMD_wrap_self(n, m, a_hist.data(), b_hist.data(), dist_flat.data(), 1000);
    // std::cout<< emd << std::endl;
    return emd;
}

static float L2SqrVecClusterChamfer(const vectorset* q, const vectorset* p, const float* cluster_dis) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 

    size_t n = q->vecnum;
    size_t m = p->vecnum ;

    // #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        long long icode = (long long)q->codes[i] * NUM_CLUSTER_CALC;
        // std::cout << i << " " << icode << " "  << std::flush;
        float mindist = 2.0f;
        for (size_t j = 0; j < m; ++j) {
            mindist = std::min(mindist, 1.0f - cluster_dis[icode + p->codes[j]]);
        }
        sum1 += mindist;
    }

    // for (size_t i = 0; i < m; ++i) {
    //     long long icode = (long long)p->codes[i] * NUM_CLUSTER_CALC;
    //     // std::cout << i << " " << icode << " "  << std::flush;
    //     float mindist = 2.0f;
    //     for (size_t j = 0; j < n; ++j) {
    //         mindist = std::min(mindist, 1.0f - cluster_dis[icode + q->codes[j]]);
    //     }
    //     sum2 += mindist;
    // }
    // return sum1 / n + sum2 / m;
    return sum1 / n;
}

float compute_emd(const std::vector<float>& a, const std::vector<float>& b, 
                  const std::vector<float>& C, int n, int m) {
    std::vector<float> F(n * m, 0);  // 传输矩阵
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, 1.0f, a.data(), m, b.data(), m, 0.0f, F.data(), m);

    float emd = 0.0f;
    for (int i = 0; i < n * m; ++i) {
        emd += F[i] * C[i];  // 计算 EMD 值
    }
    return emd;
}

static float L2SqrVecEMDBlas(const vectorset* q, const vectorset* p, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    size_t n = q->vecnum;
    size_t m = p->vecnum ;
    std::vector<float> dist_flat(n * m);
    // std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(m));
    for (size_t i = 0; i < n; ++i) {
        const float* vec_q = q->data + i * q->dim;
        for (size_t j = 0; j < m; ++j) {
            const float* vec_p = p->data + j * p->dim;
            float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
            dist_flat[i * m + j] = dist;
        }
    }

    std::vector<float> a_hist(n, 1.0 / n);
    std::vector<float> b_hist(m, 1.0 / m);
    // 计算EMD（匹配后的最小搬运成本）
    float emd = compute_emd(a_hist, b_hist, dist_flat, n, m);
    // double emd = EMD_wrap_self(n, m, a_hist.data(), b_hist.data(), dist_flat.data(), 1000);
    // std::cout<< emd << std::endl;
    return emd;
}



static float L2SqrVecSet(const vectorset* q, const vectorset* p, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    size_t q_vecnum = q->vecnum;
    size_t p_vecnum = p->vecnum ;
    // 使用随机数引擎打乱序列
    std::vector<std::vector<float>> dist_matrix(q_vecnum, std::vector<float>(p_vecnum));
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < q_vecnum; ++i) {
        const float* vec_q = q->data + i * (level + 1) * q->dim;
        float maxDist = 99999.9f;
        for (size_t j = 0; j < p_vecnum; ++j) {
            const float* vec_p = p->data + j * p->dim;
            float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
            dist_matrix[i][j] = dist;
            maxDist = std::min(maxDist, dist);
        }
        sum1 += maxDist;
    }

    // //#pragma omp parallel for num_threads(4) reduction(+:sum2)
    // #pragma omp simd reduction(+:sum2)
    // for (size_t i = 0; i < p_vecnum; ++i) {
    //     float maxDist = 99999.9f;
    //     for (size_t j = 0; j < q_vecnum; ++j) {
    //         float dist = dist_matrix[j][i];
    //         maxDist = std::min(maxDist, dist);
    //     }
    //     sum2 += maxDist;
    // }        
    return sum1 / q_vecnum;
}

float compute_with_eigen(const float* data, const int* codes, int n, int d, int m) {
    Eigen::Map<const Eigen::MatrixXf> data_mat(data, n, d);
    Eigen::VectorXf maxDist = Eigen::VectorXf::Constant(n, -9.0f);

    // 遍历所有 center_id，找到最大值
    for (size_t j = 0; j < m; ++j) {
        maxDist = maxDist.cwiseMax(data_mat.col(codes[j])); // Eigen 自动向量化
    }
    // std::cout << "eigen" << std::endl;
    // for (int i = 0; i < n; i ++) {
        // std::cout << i << " " << maxDist[i] << " ";
    // }
    // std::cout << std::endl;
    // 计算 sum1
    return (1.0f - maxDist.array()).sum();
}

// static float L2SqrCluster4Search(const vectorset* q, const vectorset* p, int level) {
//     //l2_sqr_call_count.fetch_add(1, std::memory_order_relaxed);
//     // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed);
//     float sum1 = 0.0f;
//     float sum2 = 0.0f;
//     level = 0;
//     size_t q_vecnum = q->vecnum;
//     size_t p_vecnum = p->vecnum;
//     // for (int i = 0; i < 32; i ++) {
//     //     for (int j = 0; j < 500; j ++) {
//     //         std::cout << i << " " << j << " " << *(q->data + i * 500 + j) << std::endl;
//     //     }
//     // }
//     //std::cout << q->vecnum << " " << q->dim << " " << p->vecnum << std::endl;
//     // sum1 = compute_with_eigen(q->data, p->codes, q->vecnum, 262144, p->vecnum);
    
//     #pragma omp simd reduction(+:sum1)
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         float maxDist = -9.9f;
//         for (size_t j = 0; j < p_vecnum; ++j) {
//             int center_id =  *(p->codes + j);
//             maxDist = std::max(maxDist, *(q->data + i * 262144 + center_id));
//         }
//         sum1 += 1 - maxDist;
//     }
//     return sum1 / q->vecnum;
// }


static float L2SqrCluster4Search(const vectorset* q, const vectorset* p, int level) {
    l2_sqr_call_count.fetch_add(1, std::memory_order_relaxed);
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    level = 0;
    size_t q_vecnum = q->vecnum;
    size_t p_vecnum = p->vecnum;
    sum1 = compute_with_eigen(q->data, p->codes, q->vecnum, NUM_CLUSTER_CALC, p->vecnum);
    return sum1 / q->vecnum;
}

static float L2SqrClusterAVX4Search(const vectorset* q, const vectorset* p, int level) {
    // l2_sqr_call_count.fetch_add(1, std::memory_order_relaxed);
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    level = 0;
    size_t q_vecnum = q->vecnum;
    size_t p_vecnum = p->vecnum;
    // sum1 = compute_with_eigen(q->data, p->codes, q->vecnum, 262144, p->vecnum);
    // std::cout << "AVX" << std::endl;
    #pragma omp simd reduction(+:sum1)
    for (int i = 0; i < q->vecnum; ++i) {
        __m512 max_val = _mm512_set1_ps(std::numeric_limits<float>::lowest());

        for (size_t j = 0; j < p_vecnum; j += 16) {  // 每次处理 16 个列索引
            int valid_count = std::min((size_t)16, p_vecnum - j);
            // 读取索引
            alignas(64) int index_buffer[16] = {0};
            for (int k = 0; k < valid_count; ++k) {
                index_buffer[k] = p->codes[j + k];
            }
            __m512i indices = _mm512_load_si512(index_buffer);
            // 按索引读取数据
            __m512 values = _mm512_i32gather_ps(indices, &(q->data[i * NUM_CLUSTER_CALC]), sizeof(float));
            max_val = _mm512_max_ps(max_val, values);
        }

        // 计算最终最大值
        alignas(64) float max_values[16];
        _mm512_store_ps(max_values, max_val);
        sum1 += *std::max_element(max_values, max_values + 16);
        // std::cout << i << " " << *std::max_element(max_values, max_values + 16) << " ";
    }
    // std::cout << std::endl;
    return 1.0f - sum1 / q->vecnum;
}

static float L2SqrVecSet4Search(const vectorset* q, const vectorset* p, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif

    size_t p_vecnum = p->vecnum;
    std::vector<std::vector<float>> dist_matrix(q->vecnum, std::vector<float>(p_vecnum));
    //#pragma omp parallel for num_threads(4) reduction(+:sum1)
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < q->vecnum; ++i) {
        const float* vec_q = q->data + i * q->dim;
        float maxDist = 99999.9f;
        for (size_t j = 0; j < p_vecnum; ++j) {
            const float* vec_p = p->data + j * p->dim;
            float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
            dist_matrix[i][j] = dist;
            maxDist = std::min(maxDist, dist);
        }
        sum1 += maxDist;
    }

    //#pragma omp parallel for num_threads(4) reduction(+:sum2)
    // #pragma omp simd reduction(+:sum2)
    // for (size_t i = 0; i < p_vecnum; ++i) {
    //     float maxDist = 99999.9f;
    //     for (size_t j = 0; j < q->vecnum; ++j) {
    //         float dist = dist_matrix[j][i];
    //         maxDist = std::min(maxDist, dist);
    //     }
    //     sum2 += maxDist;
    // }

    return sum1 / q->vecnum;
}


static float L2SqrVecSetMap(const vectorset* a, const vectorset* b, const vectorset* c, const uint8_t* old_map_ab, const uint8_t* old_map_bc, uint8_t* new_map, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    size_t fineEdgeMaxlen = 0;
    size_t a_vecnum = std::min(a->vecnum, (size_t)fineEdgeMaxlen);
    size_t b_vecnum = std::min(b->vecnum, (size_t)fineEdgeMaxlen);
    size_t c_vecnum = std::min(c->vecnum, (size_t)fineEdgeMaxlen);
    // std::vector<std::vector<float>> dist_matrix(a_vecnum, std::vector<float>(c_vecnum));

    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        const float* vec_a = a->data + i * a->dim;
        const float* vec_c = c->data + old_map_bc[old_map_ab[i]] * c->dim;
        float dist = L2Sqrfunc_(vec_a, vec_c, &a->dim);
        new_map[i] = old_map_bc[old_map_ab[i]];
        sum1 += dist;
    }

    // #pragma omp simd reduction(+:sum2)
    // for (size_t i = 0; i < c_vecnum; ++i) {
    //     const float* vec_c = c->data + i * c->dim;
    //     const float* vec_a = a->data + old_map_ab[fineEdgeMaxlen + old_map_bc[fineEdgeMaxlen + i]] * a->dim;
    //     float dist = L2Sqrfunc_(vec_c, vec_a, &c->dim);
    //     new_map[i + fineEdgeMaxlen] = old_map_ab[old_map_bc[i + fineEdgeMaxlen] + fineEdgeMaxlen];
    //     sum2 += dist;
    // }
    return sum1 / a_vecnum;
}

// for only top1
static float L2SqrVecSetInitEMD(const vectorset* a, const vectorset* b, uint8_t* new_map, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 

    uint8_t fineEdgeMaxlen = 0;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    size_t a_vecnum = (size_t) std::min(a->vecnum, (size_t)fineEdgeMaxlen);
    size_t b_vecnum = (size_t) std::min(b->vecnum, (size_t)fineEdgeMaxlen);

    std::vector<double> dist_flat(a_vecnum * b_vecnum);
    // std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(m));
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        const float* vec_q = a->data + i * a->dim;
        for (size_t j = 0; j < b_vecnum; ++j) {
            const float* vec_p = b->data + j * b->dim;
            float dist = L2Sqrfunc_(vec_q, vec_p, &a->dim);
            dist_flat[i * b_vecnum + j] = (double)dist;
        }
    }

    std::vector<double> a_hist(a_vecnum, 1.0 / a_vecnum);
    std::vector<double> b_hist(b_vecnum, 1.0 / b_vecnum);
    // 计算EMD（匹配后的最小搬运成本）
    float emd = EMD_wrap_self(a_vecnum, b_vecnum, a_hist.data(), b_hist.data(), dist_flat.data(), 1000);

    #pragma omp simd reduction(+:sum1)        
    for (size_t i = 0; i < a_vecnum; ++i) {
        double maxDist = 99999.9f;
        for (size_t j = 0; j < b_vecnum; ++j) {
            if (dist_flat[i * b_vecnum + j] < maxDist) {
                new_map[i] = j;
                maxDist = dist_flat[i * b_vecnum + j];
            }
        }
    }
    return (float)emd;
}

// for only top1
static float L2SqrVecSetInit(const vectorset* a, const vectorset* b, uint8_t* new_map, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 

    uint8_t fineEdgeMaxlen = 0;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    uint8_t a_vecnum = (uint8_t) std::min(a->vecnum, (size_t)fineEdgeMaxlen);
    uint8_t b_vecnum = (uint8_t) std::min(b->vecnum, (size_t)fineEdgeMaxlen);

    std::vector<std::vector<float>> dist_matrix(a_vecnum, std::vector<float>(b_vecnum));
    //#pragma omp parallel for num_threads(4) reduction(+:sum1)
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        const float* vec_a = a->data + i * a->dim;
        for (size_t j = 0; j < b_vecnum; ++j) {
            const float* vec_b = b->data + j * b->dim;
            float dist = L2Sqrfunc_(vec_a, vec_b, &b->dim);
            dist_matrix[i][j] = dist;
        }
    }

    #pragma omp simd reduction(+:sum1)        
    for (uint8_t i = 0; i < a_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < b_vecnum; ++j) {
            if (dist_matrix[i][j] < maxDist) {
                new_map[i] = j;
                maxDist = dist_matrix[i][j];
            }
        }
        sum1 += maxDist;
    }
    // #pragma omp simd reduction(+:sum2)   
    // for (uint8_t i = 0; i < b_vecnum; ++i) {
    //     float maxDist = 99999.9f;
    //     for (uint8_t j = 0; j < a_vecnum; ++j) {
    //         if (dist_matrix[j][i] < maxDist) {
    //             new_map[i + fineEdgeMaxlen] = j;
    //             maxDist = dist_matrix[j][i];
    //         }
    //     }
    //     sum2 += maxDist;
    // }
    return sum1 / a_vecnum;
}

static std::pair<float, float> L2SqrVecSetInitReturn2(const vectorset* a, const vectorset* b, uint8_t* new_map, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 

    uint8_t fineEdgeMaxlen = 0;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    uint8_t a_vecnum = (uint8_t) std::min(a->vecnum, (size_t)fineEdgeMaxlen);
    uint8_t b_vecnum = (uint8_t) std::min(b->vecnum, (size_t)fineEdgeMaxlen);

    std::vector<std::vector<float>> dist_matrix(a_vecnum, std::vector<float>(b_vecnum));
    //#pragma omp parallel for num_threads(4) reduction(+:sum1)
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        const float* vec_a = a->data + i * a->dim;
        for (size_t j = 0; j < b_vecnum; ++j) {
            const float* vec_b = b->data + j * b->dim;
            float dist = L2Sqrfunc_(vec_a, vec_b, &b->dim);
            dist_matrix[i][j] = dist;
        }
    }

    #pragma omp simd reduction(+:sum1)        
    for (uint8_t i = 0; i < a_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < b_vecnum; ++j) {
            if (dist_matrix[i][j] < maxDist) {
                new_map[i] = j;
                maxDist = dist_matrix[i][j];
            }
        }
        sum1 += maxDist;
    }
    // #pragma omp simd reduction(+:sum2)   
    // for (uint8_t i = 0; i < b_vecnum; ++i) {
    //     float maxDist = 99999.9f;
    //     for (uint8_t j = 0; j < a_vecnum; ++j) {
    //         if (dist_matrix[j][i] < maxDist) {
    //             new_map[i + fineEdgeMaxlen] = j;
    //             maxDist = dist_matrix[j][i];
    //         }
    //     }
    //     sum2 += maxDist;
    // }
    return std::make_pair(sum1 / a_vecnum, sum1);
}



// for top1
static float L2SqrVecSetInitPreCalc(const vectorset* a, const vectorset* b, uint8_t* new_map, std::vector<std::vector<float>>& dist_matrix, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 

    uint8_t fineEdgeMaxlen = 0;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    uint8_t a_vecnum = (uint8_t) std::min(a->vecnum, (size_t)fineEdgeMaxlen);
    uint8_t b_vecnum = (uint8_t) std::min(b->vecnum, (size_t)fineEdgeMaxlen);

    // std::vector<std::vector<float>> dist_matrix(a_vecnum, std::vector<float>(b_vecnum));
    //#pragma omp parallel for num_threads(4) reduction(+:sum1)
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        const float* vec_a = a->data + i * a->dim;
        for (size_t j = 0; j < b_vecnum; ++j) {
            if (dist_matrix[i][j] > 900.0) {
                const float* vec_b = b->data + j * b->dim;
                float dist = L2Sqrfunc_(vec_a, vec_b, &b->dim);
                dist_matrix[i][j] = dist;
            }
        }
    }

    #pragma omp simd reduction(+:sum1)        
    for (uint8_t i = 0; i < a_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < b_vecnum; ++j) {
            if (dist_matrix[i][j] < maxDist) {
                new_map[i] = j;
                maxDist = dist_matrix[i][j];
            }
        }
        sum1 += maxDist;
    }
    // #pragma omp simd reduction(+:sum2)   
    // for (uint8_t i = 0; i < b_vecnum; ++i) {
    //     float maxDist = 99999.9f;
    //     for (uint8_t j = 0; j < a_vecnum; ++j) {
    //         if (dist_matrix[j][i] < maxDist) {
    //             new_map[i + fineEdgeMaxlen] = j;
    //             maxDist = dist_matrix[j][i];
    //         }
    //     }
    //     sum2 += maxDist;
    // }
    return sum1 / a_vecnum;
}

static std::pair<float, float> L2SqrVecSetInitPreCalcReturn2(const vectorset* a, const vectorset* b, uint8_t* new_map, std::vector<std::vector<float>>& dist_matrix, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 

    uint8_t fineEdgeMaxlen = 0;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    uint8_t a_vecnum = (uint8_t) std::min(a->vecnum, (size_t)fineEdgeMaxlen);
    uint8_t b_vecnum = (uint8_t) std::min(b->vecnum, (size_t)fineEdgeMaxlen);

    // std::vector<std::vector<float>> dist_matrix(a_vecnum, std::vector<float>(b_vecnum));
    //#pragma omp parallel for num_threads(4) reduction(+:sum1)
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        const float* vec_a = a->data + i * a->dim;
        for (size_t j = 0; j < b_vecnum; ++j) {
            if (dist_matrix[i][j] > 900.0) {
                const float* vec_b = b->data + j * b->dim;
                float dist = L2Sqrfunc_(vec_a, vec_b, &b->dim);
                dist_matrix[i][j] = dist;
            }
        }
    }

    #pragma omp simd reduction(+:sum1)        
    for (uint8_t i = 0; i < a_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < b_vecnum; ++j) {
            if (dist_matrix[i][j] < maxDist) {
                new_map[i] = j;
                maxDist = dist_matrix[i][j];
            }
        }
        sum1 += maxDist;
    }
    // #pragma omp simd reduction(+:sum2)   
    // for (uint8_t i = 0; i < b_vecnum; ++i) {
    //     float maxDist = 99999.9f;
    //     for (uint8_t j = 0; j < a_vecnum; ++j) {
    //         if (dist_matrix[j][i] < maxDist) {
    //             new_map[i + fineEdgeMaxlen] = j;
    //             maxDist = dist_matrix[j][i];
    //         }
    //     }
    //     sum2 += maxDist;
    // }
    return std::make_pair(sum1 / a_vecnum, sum1 / a_vecnum);
}

static float L2SqrVecSetMapCalc(const vectorset* a, const vectorset* b, const vectorset* c, const uint8_t* old_map_ab, const uint8_t* old_map_bc,  std::vector<std::vector<float>>& dist_matrix, int level) {
    // l2_vec_call_count.fetch_add(1, std::memory_order_relaxed); 
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = InnerProductDistanceSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = InnerProductDistance;
    #endif
    uint8_t fineEdgeMaxlen = 0;
    uint8_t a_vecnum = (uint8_t) std::min(a->vecnum, (size_t)fineEdgeMaxlen);
    uint8_t b_vecnum = (uint8_t) std::min(b->vecnum, (size_t)fineEdgeMaxlen);
    uint8_t c_vecnum = (uint8_t) std::min(c->vecnum, (size_t)fineEdgeMaxlen);
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        for (size_t j = 0; j < c_vecnum; ++j) {
            dist_matrix[i][j] = 9999.0;
        }
    }
    
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        size_t c_ind = (size_t) old_map_bc[old_map_ab[i]];
        if (dist_matrix[i][c_ind] > 9000.0) {
            const float* vec_a = a->data + i * a->dim;
            const float* vec_c = c->data + c_ind * c->dim;
            dist_matrix[i][c_ind] = L2Sqrfunc_(vec_a, vec_c, &a->dim);
        } 
    }

    // #pragma omp simd reduction(+:sum2)
    // for (size_t i = 0; i < c_vecnum; ++i) {
    //     size_t a_ind = (size_t) old_map_ab[old_map_bc[i + fineEdgeMaxlen] + fineEdgeMaxlen];
    //     if (dist_matrix[a_ind][i] > 9000.0) {
    //         const float* vec_a = a->data + a_ind * a->dim;
    //         const float* vec_c = c->data + i * c->dim;
    //         dist_matrix[a_ind][i] = L2Sqrfunc_(vec_a, vec_c, &a->dim);
    //     } 
    // }

    #pragma omp simd reduction(+:sum1)
    for (uint8_t i = 0; i < a_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < c_vecnum; ++j) {
            maxDist = std::min(maxDist, dist_matrix[i][j]);
        }
        sum1 += maxDist;
    }

    // #pragma omp simd reduction(+:sum2)
    // for (uint8_t i = 0; i < c_vecnum; ++i) {
    //     float maxDist = 99999.9f;
    //     for (uint8_t j = 0; j < a_vecnum; ++j) {
    //         maxDist = std::min(maxDist, dist_matrix[j][i]);
    //     }
    //     sum2 += maxDist;
    // }

    return sum1 / a_vecnum;
}



class L2SpaceI : public SpaceInterface<int> {
    DISTFUNC<int> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2SpaceI(size_t dim) {
        if (dim % 4 == 0) {
            fstdistfunc_ = L2SqrI4x;
        } else {
            fstdistfunc_ = L2SqrI;
        }
        dim_ = dim;
        data_size_ = dim * sizeof(unsigned char);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceI() {}
};
}  // namespace hnswlib
