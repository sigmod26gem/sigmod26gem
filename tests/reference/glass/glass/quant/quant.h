#pragma once

#include <string>
#include <unordered_map>

#include "fp32_quant.h"
#include "sq4_quant.h"
#include "sq8_quant.h"
#include "sq8_unified_quant.h"
#include "fp32_vc_quant.h"
// #include "random_projector.h"
// #include "sq8_random_projector.h"
// #include "sq4_random_projector.h"
// #include "sq8_random_projector_with_fp16.h"

namespace glass {

enum class QuantizerType { FP32, SQ8, SQ4, SQ8U };

inline std::unordered_map<int, QuantizerType> quantizer_map;

inline int quantizer_map_init = [] {
  quantizer_map[0] = QuantizerType::FP32;
  quantizer_map[1] = QuantizerType::SQ8;
  quantizer_map[2] = QuantizerType::SQ4;
  quantizer_map[3] = QuantizerType::SQ8U;
  return 42;
}();

} // namespace glass
