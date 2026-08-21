# qwen38-3090-cuda-server — pure CUDA/C++ inference server for Qwen3.8-27B on RTX 3090.
#
# sm_86 ONLY. This project specializes hard for GA102 and does not attempt to be
# portable across architectures; see reports/PHASE_0.md for the measured hardware
# facts every kernel is designed against.
cmake_minimum_required(VERSION 3.24)
project(qwen38_3090_cuda_server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CUDA host code stays at C++17: nlohmann/json 3.12's std::ranges specialisation
# does not compile under nvcc with -std=c++20, and nothing in the kernels needs
# C++20. Host-only translation units still build as C++20.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --default-stream per-thread is load-bearing for CUDA graph capture of the decode
# step (gemma-cuda-hybrid OPTIMIZATION_LOG cycle [2]); do not drop it.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------- microbench
# Phase 0 hardware probe. Buildable on its own: it has no project dependencies.
add_executable(microbench bench/microbench.cu)
target_link_libraries(microbench PRIVATE CUDA::cudart)

# ---------------------------------------------------------------- tokenizer
add_library(qwen_tokenizer STATIC src/tokenizer/bpe.cpp src/tokenizer/chat_template.cpp)
target_include_directories(qwen_tokenizer PUBLIC src third_party)

# ---------------------------------------------------------------- config + loader
add_library(qwen_core STATIC
  src/config/model_shape.cpp
  src/loader/safetensors.cpp)
target_include_directories(qwen_core PUBLIC src third_party)

# ---------------------------------------------------------------- gates
# Every gate is a standalone executable returning 0 on pass. `ctest` runs them
# all; each also runs on its own for debugging.
enable_testing()

add_executable(gate_nfc tests/gate_nfc.cpp)
target_include_directories(gate_nfc PRIVATE src third_party)
add_test(NAME nfc COMMAND gate_nfc ${CMAKE_SOURCE_DIR}/tests/fixtures/nfc_cases.jsonl)

add_executable(gate_tokenizer tests/gate_tokenizer.cpp)
target_link_libraries(gate_tokenizer PRIVATE qwen_tokenizer)
add_test(NAME tokenizer COMMAND gate_tokenizer
         $ENV{QWEN38_MODEL_DIR}/tokenizer.json
         ${CMAKE_SOURCE_DIR}/tests/fixtures/tokenizer/corpus.jsonl
         ${CMAKE_SOURCE_DIR}/tests/fixtures/tokenizer/meta.json)

add_executable(gate_chat_template tests/gate_chat_template.cpp)
target_link_libraries(gate_chat_template PRIVATE qwen_tokenizer)
add_test(NAME chat_template COMMAND gate_chat_template
         ${CMAKE_SOURCE_DIR}/tests/fixtures/chat/fixtures.jsonl
         $ENV{QWEN38_MODEL_DIR}/tokenizer.json)

add_executable(gate_config tests/gate_config.cpp)
target_link_libraries(gate_config PRIVATE qwen_core)
add_test(NAME config COMMAND gate_config $ENV{QWEN38_MODEL_DIR}/config.json)

add_executable(gate_dequant tests/gate_dequant.cpp)
target_link_libraries(gate_dequant PRIVATE qwen_core)
add_test(NAME dequant COMMAND gate_dequant $ENV{QWEN38_MODEL_DIR}
         ${CMAKE_SOURCE_DIR}/tests/fixtures/dequant)

add_library(qwen_kernels STATIC src/kernels/gemv_w4a16.cu src/kernels/gemm_w4a16.cu
                                src/kernels/gdn.cu)
target_include_directories(qwen_kernels PUBLIC src third_party)
target_link_libraries(qwen_kernels PUBLIC CUDA::cublas)
set_target_properties(qwen_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gate_gemv tests/gate_gemv.cu)
target_link_libraries(gate_gemv PRIVATE qwen_core qwen_kernels)
add_test(NAME gemv_numerics COMMAND gate_gemv $ENV{QWEN38_MODEL_DIR})

add_executable(gate_gemm tests/gate_gemm.cu)
target_link_libraries(gate_gemm PRIVATE qwen_core qwen_kernels)
add_test(NAME gemm_prefill COMMAND gate_gemm $ENV{QWEN38_MODEL_DIR})

add_executable(gate_gdn tests/gate_gdn.cu)
target_link_libraries(gate_gdn PRIVATE qwen_core qwen_kernels)
add_test(NAME gdn COMMAND gate_gdn ${CMAKE_SOURCE_DIR}/tests/fixtures/gdn
         $ENV{QWEN38_BF16_DIR})

add_executable(bench_gdn bench/bench_gdn.cu)
target_link_libraries(bench_gdn PRIVATE qwen_kernels)

add_executable(bench_decode_gemv bench/bench_decode_gemv.cu)
target_link_libraries(bench_decode_gemv PRIVATE qwen_core qwen_kernels)
add_test(NAME gemv_bandwidth COMMAND bench_decode_gemv $ENV{QWEN38_MODEL_DIR})

# ---------------------------------------------------------------- server
# Sources are added per phase as each gate passes.
#   Phase 1 (in progress): src/config, src/loader, src/tokenizer/chat_template
