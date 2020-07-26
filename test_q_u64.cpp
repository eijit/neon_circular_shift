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

static uint64_t shift_l_circular_n_u64(uint64_t v, int n)
{
  const auto tmp1 = (v >> (64 - n));
  const auto tmp2 = (v << n);
  const auto ret = (tmp1 | tmp2);
  return ret;
}

template<int n>
static void test_pure_c(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; ++i) {
    const auto ret = shift_l_circular_n_u64(s[i], n);
    d[i] = ret;
  }
}

template<int n>
static void test_neon(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 1) {
    const auto v = vld1_u64(s + i);
    const auto ret = vshlc_n_u64<n>(v);
    vst1_u64(d + i, ret);
  }
}

template<int n>
static void test_neon_q(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 2) {
    const auto v = vld1q_u64(s + i);
    const auto ret = vshlcq_n_u64<n>(v);
    vst1q_u64(d + i, ret);
  }
}

template<int n>
static uint64x1_t vshlc_slow_n_u64(uint64x1_t v)
{
  const auto tmp0 = vshr_n_u64(v, 64 - n);
  const auto tmp1 = vshl_n_u64(v, n);
  const auto ret = vorr_u64(tmp0, tmp1);
  return ret;
}

template<int n>
static void test_neon_slow(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 1) {
    const auto v = vld1_u64(s + i);
    const auto ret = vshlc_slow_n_u64<n>(v);
    vst1_u64(d + i, ret);
  }
}

template<int n>
static uint64x2_t vshlcq_slow_n_u64(uint64x2_t v)
{
  const auto tmp0 = vshrq_n_u64(v, 64 - n);
  const auto tmp1 = vshlq_n_u64(v, n);
  const auto ret = vorrq_u64(tmp0, tmp1);
  return ret;
}

template<int n>
static void test_neon_slow_q(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  for (size_t i = 0; i < buf_len; i += 2) {
    const auto v = vld1q_u64(s + i);
    const auto ret = vshlcq_slow_n_u64<n>(v);
    vst1q_u64(d + i, ret);
  }
}

template<int n>
static void perf_pure_c(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  const auto s = src.data();
  auto d = dst->data();
  const int m = (n + static_cast<int>(src[0])) % 64;
  for (size_t i = 0; i < buf_len; ++i) {
    auto ret = shift_l_circular_n_u64(s[i], (n + 0) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 1) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 2) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 3) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 4) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 5) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 6) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 7) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 8) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 9) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 10) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 11) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 12) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 13) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 14) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 15) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 16) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 17) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 18) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 19) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 20) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 21) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 22) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 23) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 24) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 25) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 26) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 27) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 28) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 29) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 30) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 31) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 32) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 33) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 34) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 35) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 36) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 37) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 38) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 39) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 40) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 41) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 42) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 43) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 44) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 45) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 46) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 47) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 48) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 49) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 50) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 51) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 52) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 53) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 54) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 55) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 56) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 57) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 58) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 59) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 60) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 61) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 62) % 64);
    ret = shift_l_circular_n_u64(ret, (m + 63) % 64);
    d[i] = ret;
  }
}

template<int n>
static void perf_copy(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
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
  for (size_t i = 0; i < buf_len; i += 1) { \
    auto ret = ld(s + i); \
    ret = func<(n + 0) % 64>(ret); \
    ret = func<(n + 1) % 64>(ret); \
    ret = func<(n + 2) % 64>(ret); \
    ret = func<(n + 3) % 64>(ret); \
    ret = func<(n + 4) % 64>(ret); \
    ret = func<(n + 5) % 64>(ret); \
    ret = func<(n + 6) % 64>(ret); \
    ret = func<(n + 7) % 64>(ret); \
    ret = func<(n + 8) % 64>(ret); \
    ret = func<(n + 9) % 64>(ret); \
    ret = func<(n + 10) % 64>(ret); \
    ret = func<(n + 11) % 64>(ret); \
    ret = func<(n + 12) % 64>(ret); \
    ret = func<(n + 13) % 64>(ret); \
    ret = func<(n + 14) % 64>(ret); \
    ret = func<(n + 15) % 64>(ret); \
    ret = func<(n + 16) % 64>(ret); \
    ret = func<(n + 17) % 64>(ret); \
    ret = func<(n + 18) % 64>(ret); \
    ret = func<(n + 19) % 64>(ret); \
    ret = func<(n + 20) % 64>(ret); \
    ret = func<(n + 21) % 64>(ret); \
    ret = func<(n + 22) % 64>(ret); \
    ret = func<(n + 23) % 64>(ret); \
    ret = func<(n + 24) % 64>(ret); \
    ret = func<(n + 25) % 64>(ret); \
    ret = func<(n + 26) % 64>(ret); \
    ret = func<(n + 27) % 64>(ret); \
    ret = func<(n + 28) % 64>(ret); \
    ret = func<(n + 29) % 64>(ret); \
    ret = func<(n + 30) % 64>(ret); \
    ret = func<(n + 31) % 64>(ret); \
    ret = func<(n + 32) % 64>(ret); \
    ret = func<(n + 33) % 64>(ret); \
    ret = func<(n + 34) % 64>(ret); \
    ret = func<(n + 35) % 64>(ret); \
    ret = func<(n + 36) % 64>(ret); \
    ret = func<(n + 37) % 64>(ret); \
    ret = func<(n + 38) % 64>(ret); \
    ret = func<(n + 39) % 64>(ret); \
    ret = func<(n + 40) % 64>(ret); \
    ret = func<(n + 41) % 64>(ret); \
    ret = func<(n + 42) % 64>(ret); \
    ret = func<(n + 43) % 64>(ret); \
    ret = func<(n + 44) % 64>(ret); \
    ret = func<(n + 45) % 64>(ret); \
    ret = func<(n + 46) % 64>(ret); \
    ret = func<(n + 47) % 64>(ret); \
    ret = func<(n + 48) % 64>(ret); \
    ret = func<(n + 49) % 64>(ret); \
    ret = func<(n + 50) % 64>(ret); \
    ret = func<(n + 51) % 64>(ret); \
    ret = func<(n + 52) % 64>(ret); \
    ret = func<(n + 53) % 64>(ret); \
    ret = func<(n + 54) % 64>(ret); \
    ret = func<(n + 55) % 64>(ret); \
    ret = func<(n + 56) % 64>(ret); \
    ret = func<(n + 57) % 64>(ret); \
    ret = func<(n + 58) % 64>(ret); \
    ret = func<(n + 59) % 64>(ret); \
    ret = func<(n + 60) % 64>(ret); \
    ret = func<(n + 61) % 64>(ret); \
    ret = func<(n + 62) % 64>(ret); \
    ret = func<(n + 63) % 64>(ret); \
    st(d + i, ret); \
  } \
}

template<int n>
static void perf_neon(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlc_n_u64, src, dst, buf_len, n, 1, vld1_u64, vst1_u64);
}

template<int n>
static void perf_neon_slow(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlc_slow_n_u64, src, dst, buf_len, n, 1, vld1_u64, vst1_u64);
}

template<int n>
static void perf_neon_q(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlcq_n_u64, src, dst, buf_len, n, 2, vld1q_u64, vst1q_u64);
}

template<int n>
static void perf_neon_slow_q(const std::vector<uint64_t>& src, std::vector<uint64_t>* dst, size_t buf_len)
{
  PERF_NEON(vshlcq_slow_n_u64, src, dst, buf_len, n, 2, vld1q_u64, vst1q_u64);
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
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 31); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 32); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 33); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 34); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 35); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 36); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 37); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 38); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 39); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 40); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 41); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 42); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 43); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 44); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 45); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 46); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 47); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 48); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 49); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 50); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 51); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 52); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 53); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 54); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 55); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 56); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 57); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 58); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 59); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 60); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 61); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 62); \
  GEN_TEST_SUB(ref_func, test_func, src, dst1, dst2, buf_len, validate, 63);

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
  func<31>(src, dst, buf_len); \
  func<32>(src, dst, buf_len); \
  func<33>(src, dst, buf_len); \
  func<34>(src, dst, buf_len); \
  func<35>(src, dst, buf_len); \
  func<36>(src, dst, buf_len); \
  func<37>(src, dst, buf_len); \
  func<38>(src, dst, buf_len); \
  func<39>(src, dst, buf_len); \
  func<40>(src, dst, buf_len); \
  func<41>(src, dst, buf_len); \
  func<42>(src, dst, buf_len); \
  func<43>(src, dst, buf_len); \
  func<44>(src, dst, buf_len); \
  func<45>(src, dst, buf_len); \
  func<46>(src, dst, buf_len); \
  func<47>(src, dst, buf_len); \
  func<48>(src, dst, buf_len); \
  func<49>(src, dst, buf_len); \
  func<50>(src, dst, buf_len); \
  func<51>(src, dst, buf_len); \
  func<52>(src, dst, buf_len); \
  func<53>(src, dst, buf_len); \
  func<54>(src, dst, buf_len); \
  func<55>(src, dst, buf_len); \
  func<56>(src, dst, buf_len); \
  func<57>(src, dst, buf_len); \
  func<58>(src, dst, buf_len); \
  func<59>(src, dst, buf_len); \
  func<60>(src, dst, buf_len); \
  func<61>(src, dst, buf_len); \
  func<62>(src, dst, buf_len); \
  func<63>(src, dst, buf_len);

void test_u64(void)
{
  static const size_t kBufLen = 100*1024;
  std::vector<uint64_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = static_cast<uint64_t>(mt())*static_cast<uint64_t>(mt());
  }

  GEN_TEST(test_pure_c, test_neon,      src, dst1, dst2, kBufLen, validate);
  GEN_TEST(test_pure_c, test_neon_slow, src, dst1, dst2, kBufLen, validate);
}

void perf_u64(void)
{
  static const size_t kBufLen = 100*1024;
  std::vector<uint64_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = static_cast<uint64_t>(mt())*static_cast<uint64_t>(mt());
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

void test_q_u64(void)
{
  static const size_t kBufLen = 100*1024;
  std::vector<uint64_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = static_cast<uint64_t>(mt())*static_cast<uint64_t>(mt());
  }

  GEN_TEST(test_pure_c, test_neon_q,      src, dst1, dst2, kBufLen, validate);
  GEN_TEST(test_pure_c, test_neon_slow_q, src, dst1, dst2, kBufLen, validate);
}

void perf_q_u64(void)
{
  static const size_t kBufLen = 100*1024;
  std::vector<uint64_t> src(kBufLen), dst1(kBufLen), dst2(kBufLen);
  std::mt19937 mt(1000);
  for (size_t i = 0; i < kBufLen; ++i) {
    src[i] = static_cast<uint64_t>(mt())*static_cast<uint64_t>(mt());
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

