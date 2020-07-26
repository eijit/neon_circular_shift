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

static uint16_t shift_l_circular_n_u16(uint16_t v, int n)
{
  const auto tmp1 = (v >> (16 - n));
  const auto tmp2 = (v << n);
  const auto ret = (tmp1 | tmp2);
  return ret;
}

template<int n>
static void test_pure_c(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; ++i) {
    const auto ret = shift_l_circular_n_u16(s[i], n);
    d[i] = ret;
  }
}

template<int n>
static void test_neon(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 4) {
    const auto v = vld1_u16(s + i);
    const auto ret = vshlc_n_u16<n>(v);
    vst1_u16(d + i, ret);
  }
}

template<int n>
static void test_neon_q(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 8) {
    const auto v = vld1q_u16(s + i);
    const auto ret = vshlcq_n_u16<n>(v);
    vst1q_u16(d + i, ret);
  }
}

template<int n>
static uint16x4_t vshlc_slow_n_u16(uint16x4_t v)
{
  const auto tmp0 = vshr_n_u16(v, 16 - n);
  const auto tmp1 = vshl_n_u16(v, n);
  const auto ret = vorr_u16(tmp0, tmp1);
  return ret;
}

template<int n>
static void test_neon_slow(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 4) {
    const auto v = vld1_u16(s + i);
    const auto ret = vshlc_slow_n_u16<n>(v);
    vst1_u16(d + i, ret);
  }
}

template<int n>
static uint16x8_t vshlcq_slow_n_u16(uint16x8_t v)
{
  const auto tmp0 = vshrq_n_u16(v, 16 - n);
  const auto tmp1 = vshlq_n_u16(v, n);
  const auto ret = vorrq_u16(tmp0, tmp1);
  return ret;
}

template<int n>
static void test_neon_slow_q(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 8) {
    const auto v = vld1q_u16(s + i);
    const auto ret = vshlcq_slow_n_u16<n>(v);
    vst1q_u16(d + i, ret);
  }
}

template<int n>
static void perf_pure_c(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  const int m = (n + static_cast<int>(src[0])) % 16;
  for (size_t i = 0; i < buf_len; ++i) {
    auto ret = shift_l_circular_n_u16(s[i], (n + 0) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 1) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 2) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 3) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 4) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 5) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 6) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 7) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 8) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 9) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 10) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 11) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 12) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 13) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 14) % 16);
    ret = shift_l_circular_n_u16(ret, (m + 15) % 16);
    d[i] = ret;
  }
}

template<int n>
static void perf_copy(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
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
    ret = func<(n + 0) % 16>(ret); \
    ret = func<(n + 1) % 16>(ret); \
    ret = func<(n + 2) % 16>(ret); \
    ret = func<(n + 3) % 16>(ret); \
    ret = func<(n + 4) % 16>(ret); \
    ret = func<(n + 5) % 16>(ret); \
    ret = func<(n + 6) % 16>(ret); \
    ret = func<(n + 7) % 16>(ret); \
    ret = func<(n + 8) % 16>(ret); \
    ret = func<(n + 9) % 16>(ret); \
    ret = func<(n + 10) % 16>(ret); \
    ret = func<(n + 11) % 16>(ret); \
    ret = func<(n + 12) % 16>(ret); \
    ret = func<(n + 13) % 16>(ret); \
    ret = func<(n + 14) % 16>(ret); \
    ret = func<(n + 15) % 16>(ret); \
    st(d + i, ret); \
  } \
}

template<int n>
static void perf_neon(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlc_n_u16, src, dst, buf_len, n, 4, vld1_u16, vst1_u16);
}

template<int n>
static void perf_neon_slow(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlc_slow_n_u16, src, dst, buf_len, n, 4, vld1_u16, vst1_u16);
}

template<int n>
static void perf_neon_q(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlcq_n_u16, src, dst, buf_len, n, 8, vld1q_u16, vst1q_u16);
}

template<int n>
static void perf_neon_slow_q(const std::vector<uint16_t>& src, std::vector<uint16_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlcq_slow_n_u16, src, dst, buf_len, n, 8, vld1q_u16, vst1q_u16);
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

void test_u16(void)
{
  static const size_t kBufLen = 65536;
  std::vector<uint16_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = static_cast<uint16_t>(i);
  }

  GEN_TEST(test_pure_c, test_neon,      src, dst1, dst2, kBufLen, validate);
  GEN_TEST(test_pure_c, test_neon_slow, src, dst1, dst2, kBufLen, validate);
}

void perf_u16(void)
{
  static const size_t kBufLen = 2*1024*1024;
  std::vector<uint16_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = mt();
  }

  const size_t kLoop = 10;
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

void test_q_u16(void)
{
  static const size_t kBufLen = 65536;
  std::vector<uint16_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = static_cast<uint16_t>(i);
  }

  GEN_TEST(test_pure_c, test_neon_q,      src, dst1, dst2, kBufLen, validate);
  GEN_TEST(test_pure_c, test_neon_slow_q, src, dst1, dst2, kBufLen, validate);
}

void perf_q_u16(void)
{
  static const size_t kBufLen = 2*1024*1024;
  std::vector<uint16_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = mt();
  }

  const size_t kLoop = 10;
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

