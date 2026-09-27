cmake_minimum_required(VERSION 3.28)
project(bit_jev_cpu LANGUAGES C CXX)

# 只编译本项目所需的 CPU 推理库与命令行程序。
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# 固定使用工作区中的 bitnet.cpp 分支，包含 I2_S 算子和 BitNet 图。
set(BITNET_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../../learning/bitnet/3rdparty/llama.cpp")
add_subdirectory("${BITNET_CPP_DIR}" "${CMAKE_BINARY_DIR}/bitnet_cpp")

# 原生程序读取已编码题目与 float32 指针头，主干推理由 llama 库承担。
add_executable(bit-jev-cpu main.cpp)
target_compile_features(bit-jev-cpu PRIVATE cxx_std_17)
target_include_directories(bit-jev-cpu PRIVATE "${BITNET_CPP_DIR}/include" "${BITNET_CPP_DIR}/vendor")
target_link_libraries(bit-jev-cpu PRIVATE llama)
