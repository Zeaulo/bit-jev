cmake_minimum_required(VERSION 3.28)
project(bit_jev_native LANGUAGES C CXX)

# 只编译本项目所需的 GGUF 推理库与命令行程序；后端由调用方选择。
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# 源码构建默认读取工作区依赖；pip 缓存构建显式传入固定版本的路径。
if(NOT DEFINED BITNET_CPP_DIR)
    set(BITNET_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../../learning/bitnet/3rdparty/llama.cpp")
endif()
add_subdirectory("${BITNET_CPP_DIR}" "${CMAKE_BINARY_DIR}/bitnet_cpp")

# 原生程序读取已编码题目与 float32 指针头，主干推理由 llama 库承担。
add_executable(bit-jev-cpu main.cpp)
target_compile_features(bit-jev-cpu PRIVATE cxx_std_17)
# 设备请求必须与编译出的 GPU 后端一致，避免外部 binary 参数静默换后端。
if(GGML_CUDA)
    target_compile_definitions(bit-jev-cpu PRIVATE BIT_JEV_BUILT_CUDA=1)
elseif(GGML_VULKAN)
    target_compile_definitions(bit-jev-cpu PRIVATE BIT_JEV_BUILT_VULKAN=1)
endif()
target_include_directories(bit-jev-cpu PRIVATE "${BITNET_CPP_DIR}/include" "${BITNET_CPP_DIR}/vendor")
target_link_libraries(bit-jev-cpu PRIVATE llama)
