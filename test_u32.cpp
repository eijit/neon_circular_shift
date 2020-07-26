#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <cstring>

#include <arm_neon.h>

#include <vector>
#include <random>
#include <chrono>

#include "neon_circular_shift.h"
#include "test_common.h"

static uint32_t shift_l_circular_n_u32(uint32_t v, int n)
{
  const auto tmp1 = (v >> (32 - n));
  const auto tmp2 = (v << n);
  const auto ret = (tmp1 | tmp2);
  return ret;
}

template<int n>
static void test_pure_c(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; ++i) {
    const auto ret = shift_l_circular_n_u32(s[i], n);
    d[i] = ret;
  }
}

template<int n>
static void test_neon(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 2) {
    const auto v = vld1_u32(s + i);
    const auto ret = vshlc_n_u32<n>(v);
    vst1_u32(d + i, ret);
  }
}

template<int n>
static void test_neon_q(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 4) {
    const auto v = vld1q_u32(s + i);
    const auto ret = vshlcq_n_u32<n>(v);
    vst1q_u32(d + i, ret);
  }
}

template<int n>
static uint32x2_t vshlc_slow_n_u32(uint32x2_t v)
{
  const auto tmp0 = vshr_n_u32(v, 32 - n);
  const auto tmp1 = vshl_n_u32(v, n);
  const auto ret = vorr_u32(tmp0, tmp1);
  return ret;
}

template<int n>
static void test_neon_slow(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 2) {
    const auto v = vld1_u32(s + i);
    const auto ret = vshlc_slow_n_u32<n>(v);
    vst1_u32(d + i, ret);
  }
}

template<int n>
static uint32x4_t vshlcq_slow_n_u32(uint32x4_t v)
{
  const auto tmp0 = vshrq_n_u32(v, 32 - n);
  const auto tmp1 = vshlq_n_u32(v, n);
  const auto ret = vorrq_u32(tmp0, tmp1);
  return ret;
}

template<int n>
static void test_neon_slow_q(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 4) {
    const auto v = vld1q_u32(s + i);
    const auto ret = vshlcq_slow_n_u32<n>(v);
    vst1q_u32(d + i, ret);
  }
}

template<int n>
static void perf_pure_c(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  const int m = (n + static_cast<int>(src[0])) % 32;
  for (size_t i = 0; i < buf_len; ++i) {
    auto ret = shift_l_circular_n_u32(s[i], (n + 0) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 1) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 2) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 3) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 4) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 5) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 6) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 7) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 8) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 9) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 10) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 11) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 12) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 13) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 14) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 15) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 16) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 17) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 18) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 19) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 20) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 21) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 22) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 23) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 24) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 25) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 26) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 27) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 28) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 29) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 30) % 32);
    ret = shift_l_circular_n_u32(ret, (m + 31) % 32);
    d[i] = ret;
  }
}

template<int n>
static void perf_copy(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; ++i) {
    d[i] = s[i];
  }
}

#define PERF_NEON(func, src, dst, buf_len, n, stride, ld, st) \
{ \
  const auto s = src.data(); \
  auto d = dst->data(); \
  for (size_t i = 0; i < buf_len; i += stride) { \
    auto ret = ld(s + i); \
    ret = func<(n + 0) % 32>(ret); \
    ret = func<(n + 1) % 32>(ret); \
    ret = func<(n + 2) % 32>(ret); \
    ret = func<(n + 3) % 32>(ret); \
    ret = func<(n + 4) % 32>(ret); \
    ret = func<(n + 5) % 32>(ret); \
    ret = func<(n + 6) % 32>(ret); \
    ret = func<(n + 7) % 32>(ret); \
    ret = func<(n + 8) % 32>(ret); \
    ret = func<(n + 9) % 32>(ret); \
    ret = func<(n + 10) % 32>(ret); \
    ret = func<(n + 11) % 32>(ret); \
    ret = func<(n + 12) % 32>(ret); \
    ret = func<(n + 13) % 32>(ret); \
    ret = func<(n + 14) % 32>(ret); \
    ret = func<(n + 15) % 32>(ret); \
    ret = func<(n + 16) % 32>(ret); \
    ret = func<(n + 17) % 32>(ret); \
    ret = func<(n + 18) % 32>(ret); \
    ret = func<(n + 19) % 32>(ret); \
    ret = func<(n + 20) % 32>(ret); \
    ret = func<(n + 21) % 32>(ret); \
    ret = func<(n + 22) % 32>(ret); \
    ret = func<(n + 23) % 32>(ret); \
    ret = func<(n + 24) % 32>(ret); \
    ret = func<(n + 25) % 32>(ret); \
    ret = func<(n + 26) % 32>(ret); \
    ret = func<(n + 27) % 32>(ret); \
    ret = func<(n + 28) % 32>(ret); \
    ret = func<(n + 29) % 32>(ret); \
    ret = func<(n + 30) % 32>(ret); \
    ret = func<(n + 31) % 32>(ret); \
    st(d + i, ret); \
  } \
}

template<int n>
static void perf_neon(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlc_n_u32, src, dst, buf_len, n, 2, vld1_u32, vst1_u32);
}

template<int n>
static void perf_neon_slow(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlc_slow_n_u32, src, dst, buf_len, n, 2, vld1_u32, vst1_u32);
}

template<int n>
static void perf_neon_q(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlcq_n_u32, src, dst, buf_len, n, 4, vld1q_u32, vst1q_u32);
}

template<int n>
static void perf_neon_slow_q(const std::vector<uint32_t>& src, std::vector<uint32_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlcq_slow_n_u32, src, dst, buf_len, n, 4, vld1q_u32, vst1q_u32);
}

#define GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, n) \
  ref_func<n>(src, &dst1, buf_len); \
  test_func<n>(src, &dst2, buf_len); \
  validate(dst1, dst2, buf_len); \

#define GEN_TEST(ref_func, test_func, src, dst1, dst2, buf_len, validate) \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 1); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 2); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 3); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 4); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 5); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 6); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 7); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 8); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 9); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 10); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 11); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 12); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 13); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 14); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 15); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 16); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 17); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 18); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 19); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 20); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 21); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 22); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 23); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 24); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 25); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 26); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 27); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 28); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 29); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 30); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 31);

#define GEN_PERF(func, src, dst, buf_len) \
  func<1>(src, dst, buf_len); \
  func<2>(src, dst, buf_len); \
  func<3>(src, dst, buf_len); \
  func<4>(src, dst, buf_len); \
  func<5>(src, dst, buf_len); \
  func<6>(src, dst, buf_len); \
  func<7>(src, dst, buf_len); \
  func<8>(src, dst, buf_len); \
  func<9>(src, dst, buf_len); \
  func<10>(src, dst, buf_len); \
  func<11>(src, dst, buf_len); \
  func<12>(src, dst, buf_len); \
  func<13>(src, dst, buf_len); \
  func<14>(src, dst, buf_len); \
  func<15>(src, dst, buf_len); \
  func<16>(src, dst, buf_len); \
  func<17>(src, dst, buf_len); \
  func<18>(src, dst, buf_len); \
  func<19>(src, dst, buf_len); \
  func<20>(src, dst, buf_len); \
  func<21>(src, dst, buf_len); \
  func<22>(src, dst, buf_len); \
  func<23>(src, dst, buf_len); \
  func<24>(src, dst, buf_len); \
  func<25>(src, dst, buf_len); \
  func<26>(src, dst, buf_len); \
  func<27>(src, dst, buf_len); \
  func<28>(src, dst, buf_len); \
  func<29>(src, dst, buf_len); \
  func<30>(src, dst, buf_len); \
  func<31>(src, dst, buf_len);

void test_u32(void)
{
  static const size_t kBufLen = 1024*1024;
  std::vector<uint32_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = mt();
  }

  GEN_TEST(test_pure_c, test_neon,      src, dst1, dst2, kBufLen, validate);
  GEN_TEST(test_pure_c, test_neon_slow, src, dst1, dst2, kBufLen, validate);
}

void perf_u32(void)
{
  static const size_t kBufLen = 1024*1024;
  std::vector<uint32_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = mt();
  }

  const size_t kLoop = 5;
  const auto a_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kLoop; ++i) {
    GEN_PERF(perf_copy, src, &dst1, kBufLen);
  }
  const auto a_end = std::chrono::high_resolution_clock::now();

  const auto c_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kLoop; ++i) {
    GEN_PERF(perf_pure_c, src, &dst1, kBufLen);
  }
  const auto c_end = std::chrono::high_resolution_clock::now();

  const auto nf_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kLoop; ++i) {
    GEN_PERF(perf_neon, src, &dst1, kBufLen);
  }
  const auto nf_end = std::chrono::high_resolution_clock::now();

  const auto ns_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kLoop; ++i) {
    GEN_PERF(perf_neon_slow, src, &dst1, kBufLen);
  }
  const auto ns_end = std::chrono::high_resolution_clock::now();

  const auto a_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(a_end - a_begin);
  const auto c_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(c_end - c_begin);
  const auto nf_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(nf_end - nf_begin);
  const auto ns_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(ns_end - ns_begin);

  printf("%s copy  : %" PRIu64 "\n", __FUNCTION__, a_elapsed.count());
  printf("%s pure c: %" PRIu64 "\n", __FUNCTION__, c_elapsed.count());
  printf("%s neon f: %" PRIu64 "\n", __FUNCTION__, nf_elapsed.count());
  printf("%s neon s: %" PRIu64 "\n", __FUNCTION__, ns_elapsed.count());
}

void test_q_u32(void)
{
  static const size_t kBufLen = 1024*1024;
  std::vector<uint32_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = mt();
  }

  GEN_TEST(test_pure_c, test_neon_q,      src, dst1, dst2, kBufLen, validate);
  GEN_TEST(test_pure_c, test_neon_slow_q, src, dst1, dst2, kBufLen, validate);
}

void perf_q_u32(void)
{
  static const size_t kBufLen = 1024*1024;
  std::vector<uint32_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = mt();
  }

  const size_t kLoop = 5;
  const auto a_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kLoop; ++i) {
    GEN_PERF(perf_copy, src, &dst1, kBufLen);
  }
  const auto a_end = std::chrono::high_resolution_clock::now();

  const auto c_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kLoop; ++i) {
    GEN_PERF(perf_pure_c, src, &dst1, kBufLen);
  }
  const auto c_end = std::chrono::high_resolution_clock::now();

  const auto nf_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kLoop; ++i) {
    GEN_PERF(perf_neon_q, src, &dst1, kBufLen);
  }
  const auto nf_end = std::chrono::high_resolution_clock::now();

  const auto ns_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kLoop; ++i) {
    GEN_PERF(perf_neon_slow_q, src, &dst1, kBufLen);
  }
  const auto ns_end = std::chrono::high_resolution_clock::now();

  const auto a_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(a_end - a_begin);
  const auto c_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(c_end - c_begin);
  const auto nf_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(nf_end - nf_begin);
  const auto ns_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(ns_end - ns_begin);

  printf("%s copy  : %" PRIu64 "\n", __FUNCTION__, a_elapsed.count());
  printf("%s pure c: %" PRIu64 "\n", __FUNCTION__, c_elapsed.count());
  printf("%s neon f: %" PRIu64 "\n", __FUNCTION__, nf_elapsed.count());
  printf("%s neon s: %" PRIu64 "\n", __FUNCTION__, ns_elapsed.count());
}

