#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vortex {
enum class Metric : std::uint32_t { L2 = 0, InnerProduct = 1, Cosine = 2 };
enum class DType : std::uint32_t { FP32 = 0, FP16 = 1, INT8 = 2 };
enum class Kernel { Auto, Scalar };
bool avx2_available();
float scalar_l2(const float *a, const float *b, std::size_t n);
float scalar_dot(const float *a, const float *b, std::size_t n);
float distance(const float *a, const float *b, std::size_t n, Metric metric, Kernel kernel);
std::vector<float> prepare(std::span<const float> input, std::size_t dimension, Metric metric);
std::uint16_t to_half(float value);
float from_half(std::uint16_t bits);
} // namespace vortex
