cmake_minimum_required(VERSION 3.1)

set(MY_APP neon_circular_shift)

project(${MY_APP})

set(MY_APP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fno-rtti")

add_definitions(-O3 -Wall -DNDEBUG)
add_definitions(-march=armv7-a -mfpu=neon)

add_executable(${MY_APP}
  "${MY_APP_DIR}/main.cpp"
  "${MY_APP_DIR}/test_u8.cpp"
  "${MY_APP_DIR}/test_u16.cpp"
  "${MY_APP_DIR}/test_u32.cpp"
  "${MY_APP_DIR}/test_u64.cpp"
)

#target_include_directories(${MY_APP})

target_link_libraries(${MY_APP})

