# NEON circular shift

## Explanation of the code

This is an optimized implementation of circular (left) shift with NEON.

Usually, circular shift is implemented in three instructions (>>, <<, |) as follows.

```cpp
// pure c
static uint32_t shift_l_circular_n_u32(uint32_t v, int n)
{
  const auto tmp1 = (v >> (32 - n));
  const auto tmp2 = (v << n);
  const auto ret = (tmp1 | tmp2);
  return ret;
}

// naive NEON
template<int n>
static uint32x4_t vshlcq_slow_n_u32(uint32x4_t v)
{
  const auto tmp0 = vshrq_n_u32(v, 32 - n);
  const auto tmp1 = vshlq_n_u32(v, n);
  const auto ret = vorrq_u32(tmp0, tmp1);
  return ret;
}
```

The above NEON implementation is already faster than the pure C implementation.

My implmenetation uses VSLI instruction and reduces one instruction from naitve NEON inmplementation. Additionally, my implementation uses VREV instruction and reduces two instructions from naitve NEON inmplementation when the shift value is the half of the bit length of the integer.

```cpp
// optimized NEON
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
```

## Test results

The following is the test result on Raspberry Pi3 Model B.

```text
perf_u8 copy  : 474
perf_u8 pure c: 5496 
perf_u8 neon f: 1695
perf_u8 neon s: 1755
perf_u16 copy  : 501
perf_u16 pure c: 7052
perf_u16 neon f: 3472
perf_u16 neon s: 3541
perf_u32 copy  : 521
perf_u32 pure c: 9562
perf_u32 neon f: 6859
perf_u32 neon s: 6930
perf_u64 copy  : 181
perf_u64 pure c: 20941
perf_u64 neon f: 5305
perf_u64 neon s: 5334
perf_q_u8 copy  : 472
perf_q_u8 pure c: 5493
perf_q_u8 neon f: 944
perf_q_u8 neon s: 1183
perf_q_u16 copy  : 499
perf_q_u16 pure c: 7053
perf_q_u16 neon f: 1782
perf_q_u16 neon s: 2339
perf_q_u32 copy  : 519
perf_q_u32 pure c: 9560
perf_q_u32 neon f: 3563
perf_q_u32 neon s: 4653
perf_q_u64 copy  : 178
perf_q_u64 pure c: 20940
perf_q_u64 neon f: 5370
perf_q_u64 neon s: 7093
```

||pure C/NEON naive|pure C/NEON|
|:---|---:|---:|
|uint8|3.92|4.11|
|uint8 q|7.06|10.64|
|uint16|2.15|2.20|
|uint16 q|3.56|5.11|
|uint32|1.41|1.43|
|uint32 q|2.19|2.97|
|uint64|4.03|4.05|
|uint64 q|3.00|4.00|

My implementation is slightly faster than the naive NEON implemenation. :D
