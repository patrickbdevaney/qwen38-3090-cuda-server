# qwen38-3090-cuda-server — pure CUDA/C++ inference server for Qwen3.8-27B on RTX 3090.
#
# sm_86 ONLY. This project specializes hard for GA102 and does not attempt to be
# portable across architectures; see reports/PHASE_0.md for the measured hardware
# facts every kernel is designed against.
cmake_minimum_required(VERSION 3.24)
project(qwen38_3090_cuda_server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
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

# ---------------------------------------------------------------- server
# Sources are added per phase as each gate passes. Nothing here yet: Phase 0 is
# recon only, and the directive forbids writing kernels before Gate 0 passes.
# Gate 0 has now passed (reports/PHASE_0.md); Phase 1 adds:
#   src/config, src/loader, src/tokenizer
