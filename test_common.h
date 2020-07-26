#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <vector>

template <typename T>
void validate(const std::vector<T>& buf1, const std::vector<T>& buf2, size_t buf_len)
{
  int count = 0;
  for (size_t i = 0; i < buf_len; ++i) {
    if (buf1[i] != buf2[i]) {
      if (count < 10) {
        printf("[%zu] dst1(%" PRIu64 ") != dst2(%" PRIu64 ")\n", i, static_cast<uint64_t>(buf1[i]),  static_cast<uint64_t>(buf2[i]));
      }
      ++count;
    }
  }
}

