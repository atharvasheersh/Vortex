#include "vortex/distance.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define VORTEX_X86 1
#endif

namespace vortex {
bool avx2_available() {
#if defined(VORTEX_X86) && (defined(__GNUC__) || defined(__clang__))
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
#elif defined(VORTEX_X86) && defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7)
        return false;
    __cpuidex(regs, 1, 0);
    if (!(regs[2] & (1 << 27)) || !(regs[2] & (1 << 28)))
        return false;
    if ((_xgetbv(0) & 6) != 6)
        return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}
float scalar_l2(const float *a, const float *b, std::size_t n) {
    float sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
float scalar_dot(const float *a, const float *b, std::size_t n) {
    float sum = 0;
    for (std::size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}
#if defined(VORTEX_X86) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) static float simd(const float *a, const float *b, std::size_t n, bool l2) {
    __m256 sums = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        auto x = _mm256_loadu_ps(a + i), y = _mm256_loadu_ps(b + i);
        auto d = _mm256_sub_ps(x, y);
        sums = _mm256_add_ps(sums, l2 ? _mm256_mul_ps(d, d) : _mm256_mul_ps(x, y));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sums);
    float sum = 0;
    for (float lane : lanes)
        sum += lane;
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += l2 ? d * d : a[i] * b[i];
    }
    return sum;
}
#endif
const char *active_kernel(Kernel kernel) {
#if defined(VORTEX_X86) && (defined(__GNUC__) || defined(__clang__))
    if (kernel == Kernel::Auto && avx2_available())
        return "avx2";
#else
    (void)kernel;
#endif
    return "scalar";
}
namespace {
float int8_scalar(const float *a, const std::int8_t *codes, float scale, std::size_t n, bool l2) {
    float sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float value = scale * codes[i];
        if (!std::isfinite(value))
            throw std::runtime_error("non-finite snapshot vector");
        const float delta = a[i] - value;
        sum += l2 ? delta * delta : a[i] * value;
    }
    return sum;
}
#if defined(VORTEX_X86) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) float int8_simd(const float *a, const std::int8_t *codes, float scale,
                                              std::size_t n, bool l2) {
    auto sums = _mm256_setzero_ps();
    const auto scales = _mm256_set1_ps(scale);
    const auto maximum = _mm256_set1_ps(std::numeric_limits<float>::max());
    const auto sign_mask = _mm256_set1_ps(-0.0f);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const auto packed = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(codes + i));
        const auto values = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed)), scales);
        if (_mm256_movemask_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign_mask, values), maximum, _CMP_LE_OQ)) != 255)
            throw std::runtime_error("non-finite snapshot vector");
        const auto query = _mm256_loadu_ps(a + i);
        const auto delta = _mm256_sub_ps(query, values);
        sums = _mm256_add_ps(sums, l2 ? _mm256_mul_ps(delta, delta) : _mm256_mul_ps(query, values));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sums);
    float sum = 0;
    for (auto lane : lanes) sum += lane;
    // Preserve the same accumulation order as the ordinary AVX2 distance path.
    for (; i < n; ++i) {
        const float value = scale * codes[i];
        if (!std::isfinite(value)) throw std::runtime_error("non-finite snapshot vector");
        const float delta = a[i] - value;
        sum += l2 ? delta * delta : a[i] * value;
    }
    return sum;
}
#endif
}
float distance_int8(const float *query, const std::int8_t *codes, float scale, std::size_t n,
                    Metric metric, Kernel kernel) {
    if (!std::isfinite(scale) || scale <= 0)
        throw std::runtime_error("invalid INT8 scale");
    float value;
#if defined(VORTEX_X86) && (defined(__GNUC__) || defined(__clang__))
    if (kernel == Kernel::Auto && avx2_available())
        value = int8_simd(query, codes, scale, n, metric == Metric::L2);
    else
#else
    (void)kernel;
#endif
        value = int8_scalar(query, codes, scale, n, metric == Metric::L2);
    if (!std::isfinite(value)) throw std::overflow_error("distance exceeded float32 range");
    return metric == Metric::L2 ? value : -value;
}
float distance(const float *a, const float *b, std::size_t n, Metric metric, Kernel kernel) {
    float value;
#if defined(VORTEX_X86) && (defined(__GNUC__) || defined(__clang__))
    if (kernel == Kernel::Auto && avx2_available())
        value = simd(a, b, n, metric == Metric::L2);
    else
#else
    (void)kernel;
#endif
        value = metric == Metric::L2 ? scalar_l2(a, b, n) : scalar_dot(a, b, n);
    if (!std::isfinite(value))
        throw std::overflow_error("distance exceeded float32 range");
    return metric == Metric::L2 ? value : -value;
}
std::vector<float> prepare(std::span<const float> input, std::size_t dimension, Metric metric) {
    if (input.size() != dimension)
        throw std::invalid_argument("dimension mismatch");
    std::vector<float> out(input.begin(), input.end());
    double norm = 0;
    for (float v : out) {
        if (!std::isfinite(v))
            throw std::invalid_argument("vectors must be finite");
        norm += static_cast<double>(v) * v;
    }
    if (metric == Metric::Cosine) {
        if (norm == 0)
            throw std::invalid_argument("zero vectors are invalid for cosine");
        norm = std::sqrt(norm);
        for (auto &v : out)
            v = static_cast<float>(v / norm);
    }
    return out;
}
std::uint16_t to_half(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
    auto magnitude = bits & 0x7fffffff;
    if (magnitude > 0x477fe000u)
        throw std::invalid_argument("value exceeds finite FP16 range");
    if (magnitude < 0x33000000u)
        return sign;
    int exponent = static_cast<int>(magnitude >> 23) - 127;
    std::uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    if (exponent < -14) {
        auto shift = static_cast<unsigned>(-exponent - 1);
        auto rounded = mantissa >> shift;
        auto mask = (std::uint32_t{1} << shift) - 1;
        auto remainder = mantissa & mask;
        auto midpoint = std::uint32_t{1} << (shift - 1);
        if (remainder > midpoint || (remainder == midpoint && (rounded & 1)))
            ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    auto rounded = magnitude + 0xfffu + ((magnitude >> 13) & 1u);
    return static_cast<std::uint16_t>(sign | ((rounded - 0x38000000u) >> 13));
}
float from_half(std::uint16_t bits) {
    const float sign = (bits & 0x8000) ? -1.0f : 1.0f;
    auto exponent = (bits >> 10) & 31;
    auto mantissa = bits & 1023;
    if (exponent == 0)
        return sign * std::ldexp(static_cast<float>(mantissa), -24);
    if (exponent == 31)
        return mantissa ? std::numeric_limits<float>::quiet_NaN()
                        : sign * std::numeric_limits<float>::infinity();
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exponent - 15);
}
} // namespace vortex
