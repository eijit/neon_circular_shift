#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <cstring>

#include <arm_neon.h>

#include <vector>
#include <random>
#include <chrono>

#include "neon_circular_shift.h"

void test_u8();
void perf_u8();
void test_u16();
void perf_u16();
void test_u32();
void perf_u32();
void test_u64();
void perf_u64();

void test_q_u8();
void perf_q_u8();
void test_q_u16();
void perf_q_u16();
void test_q_u32();
void perf_q_u32();
void test_q_u64();
void perf_q_u64();

int main(const int argc, const char* argv[])
{
  test_u8();
  perf_u8();
  test_u16();
  perf_u16();
  test_u32();
  perf_u32();
  test_u64();
  perf_u64();

  test_q_u8();
  perf_q_u8();
  test_q_u16();
  perf_q_u16();
  test_q_u32();
  perf_q_u32();
  test_q_u64();
  perf_q_u64();

  return 0;
}

