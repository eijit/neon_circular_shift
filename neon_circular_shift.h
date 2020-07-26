#ifndef NEON_CIRCULAR_SHIFT_H
#define NEON_CIRCULAR_SHIFT_H

#include <arm_neon.h>

template<int n>
uint8x8_t vshlc_n_u8(uint8x8_t v)
{
  const auto tmp = vshr_n_u8(v, 8 - n);
  const auto ret = vsli_n_u8(tmp, v, n);
  return ret;
}

template<int n>
uint16x4_t vshlc_n_u16(uint16x4_t v)
{
  if (n == 8) {
    const auto tmp = vreinterpret_u8_u16(v);
    const auto ret = vrev16_u8(tmp);
    return vreinterpret_u16_u8(ret);
  }
  const auto tmp = vshr_n_u16(v, 16 - n);
  const auto ret = vsli_n_u16(tmp, v, n);
  return ret;
}

template<int n>
uint32x2_t vshlc_n_u32(uint32x2_t v)
{
  if (n == 16) {
    const auto tmp = vreinterpret_u16_u32(v);
    const auto ret = vrev32_u16(tmp);
    return vreinterpret_u32_u16(ret);
  }
  const auto tmp = vshr_n_u32(v, 32 - n);
  const auto ret = vsli_n_u32(tmp, v, n);
  return ret;
}

template<int n>
uint64x1_t vshlc_n_u64(uint64x1_t v)
{
  if (n == 32) {
    const auto tmp = vreinterpret_u32_u64(v);
    const auto ret = vrev64_u32(tmp);
    return vreinterpret_u64_u32(ret);
  }
  const auto tmp = vshr_n_u64(v, 64 - n);
  const auto ret = vsli_n_u64(tmp, v, n);
  return ret;
}

template<int n>
uint8x16_t vshlcq_n_u8(uint8x16_t v)
{
  const auto tmp = vshrq_n_u8(v, 8 - n);
  const auto ret = vsliq_n_u8(tmp, v, n);
  return ret;
}

template<int n>
uint16x8_t vshlcq_n_u16(uint16x8_t v)
{
  if (n == 8) {
    const auto tmp = vreinterpretq_u8_u16(v);
    const auto ret = vrev16q_u8(tmp);
    return vreinterpretq_u16_u8(ret);
  }
  const auto tmp = vshrq_n_u16(v, 16 - n);
  const auto ret = vsliq_n_u16(tmp, v, n);
  return ret;
}

template<int n>
uint32x4_t vshlcq_n_u32(uint32x4_t v)
{
  if (n == 16) {
    const auto tmp = vreinterpretq_u16_u32(v);
    const auto ret = vrev32q_u16(tmp);
    return vreinterpretq_u32_u16(ret);
  }
  const auto tmp = vshrq_n_u32(v, 32 - n);
  const auto ret = vsliq_n_u32(tmp, v, n);
  return ret;
}

template<int n>
uint64x2_t vshlcq_n_u64(uint64x2_t v)
{
  if (n == 32) {
    const auto tmp = vreinterpretq_u32_u64(v);
    const auto ret = vrev64q_u32(tmp);
    return vreinterpretq_u64_u32(ret);
  }
  const auto tmp = vshrq_n_u64(v, 64 - n);
  const auto ret = vsliq_n_u64(tmp, v, n);
  return ret;
}

#endif /* NEON_CIRCULAR_SHIFT_H */

