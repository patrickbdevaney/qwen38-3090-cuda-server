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

# Checkpoint paths. $ENV{} is evaluated at CONFIGURE time, so a reconfigure in a
# shell without these exported would silently give every gate an empty path.
# Cache variables keep them once set.
set(QWEN38_MODEL_DIR "${QWEN38_MODEL_DIR}" CACHE PATH "W4A16 checkpoint directory")
set(QWEN38_BF16_DIR  "${QWEN38_BF16_DIR}"  CACHE PATH "BF16 reference checkpoint (gates only)")
set(QWEN38_DRAFT_DIR "${QWEN38_DRAFT_DIR}" CACHE PATH "DFlash2 drafter checkpoint")
if(NOT QWEN38_MODEL_DIR)
  message(WARNING "QWEN38_MODEL_DIR is unset; model-dependent gates will fail. "
                  "Configure with -DQWEN38_MODEL_DIR=/path/to/checkpoint")
endif()

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
         ${QWEN38_MODEL_DIR}/tokenizer.json
         ${CMAKE_SOURCE_DIR}/tests/fixtures/tokenizer/corpus.jsonl
         ${CMAKE_SOURCE_DIR}/tests/fixtures/tokenizer/meta.json)

add_executable(gate_chat_template tests/gate_chat_template.cpp)
target_link_libraries(gate_chat_template PRIVATE qwen_tokenizer)
add_test(NAME chat_template COMMAND gate_chat_template
         ${CMAKE_SOURCE_DIR}/tests/fixtures/chat/fixtures.jsonl
         ${QWEN38_MODEL_DIR}/tokenizer.json)

add_executable(gate_config tests/gate_config.cpp)
target_link_libraries(gate_config PRIVATE qwen_core)
add_test(NAME config COMMAND gate_config ${QWEN38_MODEL_DIR}/config.json)

add_executable(gate_dequant tests/gate_dequant.cpp)
target_link_libraries(gate_dequant PRIVATE qwen_core)
add_test(NAME dequant COMMAND gate_dequant ${QWEN38_MODEL_DIR}
         ${CMAKE_SOURCE_DIR}/tests/fixtures/dequant)

add_library(qwen_kernels STATIC src/kernels/gemv_w4a16.cu src/kernels/gemm_w4a16.cu
                                src/kernels/gdn.cu src/kernels/attn.cu
                                src/kernels/elementwise.cu src/kernels/sampling.cu)
target_include_directories(qwen_kernels PUBLIC src third_party)
target_link_libraries(qwen_kernels PUBLIC CUDA::cublas)
set_target_properties(qwen_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gate_gemv tests/gate_gemv.cu)
target_link_libraries(gate_gemv PRIVATE qwen_core qwen_kernels)
add_test(NAME gemv_numerics COMMAND gate_gemv ${QWEN38_MODEL_DIR})

add_executable(gate_gemm tests/gate_gemm.cu)
target_link_libraries(gate_gemm PRIVATE qwen_core qwen_kernels)
add_test(NAME gemm_prefill COMMAND gate_gemm ${QWEN38_MODEL_DIR})

add_executable(gate_gdn tests/gate_gdn.cu)
target_link_libraries(gate_gdn PRIVATE qwen_core qwen_kernels)
add_test(NAME gdn COMMAND gate_gdn ${CMAKE_SOURCE_DIR}/tests/fixtures/gdn
         ${QWEN38_BF16_DIR})

add_executable(bench_gdn bench/bench_gdn.cu)
target_link_libraries(bench_gdn PRIVATE qwen_kernels)

add_executable(gate_attn tests/gate_attn.cu)
target_link_libraries(gate_attn PRIVATE qwen_core qwen_kernels)
add_test(NAME attn COMMAND gate_attn ${CMAKE_SOURCE_DIR}/tests/fixtures/attn
         ${QWEN38_BF16_DIR})

add_executable(bench_attn bench/bench_attn.cu)
target_link_libraries(bench_attn PRIVATE qwen_kernels)

add_executable(bench_decode_gemv bench/bench_decode_gemv.cu)
target_link_libraries(bench_decode_gemv PRIVATE qwen_core qwen_kernels)
add_test(NAME gemv_bandwidth COMMAND bench_decode_gemv ${QWEN38_MODEL_DIR})

# ---------------------------------------------------------------- model
add_library(qwen_model STATIC src/model/model.cu)
target_link_libraries(qwen_model PUBLIC qwen_core qwen_kernels)

add_executable(gate_forward tests/gate_forward.cu)
target_link_libraries(gate_forward PRIVATE qwen_model)
add_test(NAME forward COMMAND gate_forward ${QWEN38_MODEL_DIR}
         ${CMAKE_SOURCE_DIR}/tests/fixtures/reference 8 1 128)

add_executable(gate_graph tests/gate_graph.cu)
target_link_libraries(gate_graph PRIVATE qwen_model)
add_test(NAME cuda_graph COMMAND gate_graph ${QWEN38_MODEL_DIR})

add_executable(bench_decode bench/bench_decode.cu)
target_link_libraries(bench_decode PRIVATE qwen_model)

add_executable(debug_layers tests/debug_layers.cu)
target_link_libraries(debug_layers PRIVATE qwen_model)

# ---------------------------------------------------------------- gguf
add_library(qwen_gguf STATIC src/gguf/gguf.cpp)
target_include_directories(qwen_gguf PUBLIC src third_party)

add_executable(gguf_inspect tools/gguf_inspect.cpp)
target_link_libraries(gguf_inspect PRIVATE qwen_gguf)

add_library(qwen_gguf_dequant STATIC src/gguf/dequant.cu)
target_link_libraries(qwen_gguf_dequant PUBLIC qwen_gguf)

# Oracle: links llama.cpp's own ggml so the goldens are ggml's, not a
# reimplementation of ggml. Only built when LLAMA_CPP_DIR points at a build.
set(LLAMA_CPP_DIR "${LLAMA_CPP_DIR}" CACHE PATH "llama.cpp source+build, for the GGUF oracle")
if(LLAMA_CPP_DIR AND EXISTS "${LLAMA_CPP_DIR}/ggml/include/ggml.h")
  add_executable(gguf_dequant_ref tools/gguf_dequant_ref.cpp)
  target_include_directories(gguf_dequant_ref PRIVATE ${LLAMA_CPP_DIR}/ggml/include)
  target_link_directories(gguf_dequant_ref PRIVATE ${LLAMA_CPP_DIR}/build/bin)
  target_link_libraries(gguf_dequant_ref PRIVATE qwen_gguf ggml ggml-base)
  set_target_properties(gguf_dequant_ref PROPERTIES
                        BUILD_RPATH ${LLAMA_CPP_DIR}/build/bin)
endif()

add_executable(gate_gguf tests/gate_gguf.cu)
target_link_libraries(gate_gguf PRIVATE qwen_gguf_dequant)
set(QWEN38_GGUF "${QWEN38_GGUF}" CACHE FILEPATH "UD-IQ4_XS GGUF checkpoint")
if(QWEN38_GGUF)
  add_test(NAME gguf_dequant COMMAND gate_gguf ${QWEN38_GGUF}
           ${CMAKE_SOURCE_DIR}/tests/fixtures/gguf)
endif()

# ---------------------------------------------------------------- vision
add_library(qwen_vision STATIC src/vision/vit.cu src/vision/image.cpp src/vision/mm.cpp)
target_link_libraries(qwen_vision PUBLIC qwen_core CUDA::cublas)
target_include_directories(qwen_vision PUBLIC src third_party)

add_executable(gate_vision tests/gate_vision.cu)
target_link_libraries(gate_vision PRIVATE qwen_vision)
add_test(NAME vision COMMAND gate_vision ${QWEN38_MODEL_DIR}
         ${CMAKE_SOURCE_DIR}/tests/fixtures/vision)

# ---------------------------------------------------------------- prefix cache
add_library(qwen_cache STATIC src/cache/prefix.cu)
target_link_libraries(qwen_cache PUBLIC qwen_model)

add_executable(probe_chunk tests/probe_chunk.cu)
target_link_libraries(probe_chunk PRIVATE qwen_model)

add_executable(gate_prefix tests/gate_prefix.cu)
target_link_libraries(gate_prefix PRIVATE qwen_cache)
add_test(NAME prefix_cache COMMAND gate_prefix ${QWEN38_MODEL_DIR} 3000 24 24 512)

# ---------------------------------------------------------------- drafter
add_library(qwen_draft STATIC src/draft/dflash.cu)
target_link_libraries(qwen_draft PUBLIC qwen_model)

add_executable(gate_dflash tests/gate_dflash.cu)
target_link_libraries(gate_dflash PRIVATE qwen_draft)
add_test(NAME dflash COMMAND gate_dflash
         ${QWEN38_DRAFT_DIR}
         ${CMAKE_SOURCE_DIR}/tests/fixtures/dflash
         ${QWEN38_BF16_DIR})

# ---------------------------------------------------------------- spec
add_library(qwen_spec STATIC src/spec/spec.cu)
target_link_libraries(qwen_spec PUBLIC qwen_model qwen_draft)

add_executable(gate_spec tests/gate_spec.cu)
target_link_libraries(gate_spec PRIVATE qwen_spec)
add_test(NAME spec_lossless COMMAND gate_spec ${QWEN38_MODEL_DIR} 7 96 ${QWEN38_DRAFT_DIR})

add_executable(bench_block bench/bench_block.cu)
target_link_libraries(bench_block PRIVATE qwen_model)

add_executable(bench_verify bench/bench_verify.cu)
target_link_libraries(bench_verify PRIVATE qwen_core qwen_kernels)

add_executable(bench_dflash bench/bench_dflash.cu)
target_link_libraries(bench_dflash PRIVATE qwen_spec)

add_executable(bench_spec bench/bench_spec.cu)
target_link_libraries(bench_spec PRIVATE qwen_spec)

# ---------------------------------------------------------------- server
add_library(qwen_server STATIC
  src/server/server.cpp
  src/server/reasoning_parser.cpp)
target_link_libraries(qwen_server PUBLIC qwen_model qwen_tokenizer qwen_spec qwen_cache qwen_vision)
target_include_directories(qwen_server PUBLIC src third_party)

add_executable(cuda_server src/main.cpp)
target_link_libraries(cuda_server PRIVATE qwen_server)

add_executable(cuda_chat src/clients/tui.cpp)
target_include_directories(cuda_chat PRIVATE src third_party)
target_link_libraries(cuda_chat PRIVATE Threads::Threads)

add_executable(gate_reasoning tests/gate_reasoning.cpp)
target_include_directories(gate_reasoning PRIVATE src third_party)
target_sources(gate_reasoning PRIVATE src/server/reasoning_parser.cpp)
add_test(NAME reasoning_parser COMMAND gate_reasoning)
