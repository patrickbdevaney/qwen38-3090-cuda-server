#include "gemm_w4a16.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace qwen {
namespace {

#define CB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS %s:%d status %d\n",__FILE__,__LINE__,int(s)); abort(); } } while(0)

// [out/32][in/8][32] interleaved int4  ->  row-major [nrows][in] bf16.
// One warp per output row; lanes walk the row so the loads stay coalesced in
// both the source (stride 32 words) and the destination (contiguous).
__global__ void k_dequant_rows(__nv_bfloat16* __restrict__ dst,
                               const uint32_t* __restrict__ qw,
                               const __nv_bfloat16* __restrict__ sc,
                               const uint8_t* __restrict__ zp,
                               int row0, int nrows, int in_f, int G, int group) {
  const int r = blockIdx.x * blockDim.y + threadIdx.y;
  if (r >= nrows) return;
  const int o = row0 + r;
  const int b = o >> 5, l = o & 31;
  const int words = in_f >> 3;

  for (int t = threadIdx.x; t < words; t += blockDim.x) {
    const uint32_t w = qw[(size_t(b) * words + t) * 32 + l];
    const int g = (t * 8) / group;
    const float s  = __bfloat162float(sc[(size_t(b) * G + g) * 32 + l]);
    const float z  = float(zp[(size_t(b) * G + g) * 32 + l]);
    __nv_bfloat16* d = dst + size_t(r) * in_f + t * 8;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
      const float q = float((w >> (4 * p)) & 0xFu);
      d[p] = __float2bfloat16((q - z) * s);
    }
  }
}

}  // namespace

void gemm_workspace_alloc(GemmWorkspace& ws, size_t bytes) {
  ws.bytes = bytes;
  cudaMalloc(&ws.wbuf, bytes);
  CB(cublasCreate(&ws.cublas));
  // FP32 accumulation is the default; TF32 must not silently downgrade bf16.
  CB(cublasSetMathMode(ws.cublas, CUBLAS_DEFAULT_MATH));
}

void gemm_workspace_free(GemmWorkspace& ws) {
  if (ws.wbuf) cudaFree(ws.wbuf);
  if (ws.cublas) cublasDestroy(ws.cublas);
  ws = GemmWorkspace{};
}

void w4a16_dequant_rows(__nv_bfloat16* dst, const W4A16Weights& w,
                        int row0, int nrows, cudaStream_t st) {
  dim3 blk(32, 8);
  k_dequant_rows<<<(nrows + 7) / 8, blk, 0, st>>>(
      dst, w.qweight, w.scale, w.zp, row0, nrows, w.in_f, w.num_groups, w.group_size);
}

void gemm_w4a16(__nv_bfloat16* C, const W4A16Weights& w, const __nv_bfloat16* A,
                int M, GemmWorkspace& ws, cudaStream_t st) {
  const int K = w.in_f, N = w.out_f;
  int tile = int(ws.bytes / (size_t(K) * sizeof(__nv_bfloat16)));
  tile = (tile / 32) * 32;                       // keep the 32-row interleave intact
  if (tile <= 0) { fprintf(stderr, "gemm: workspace too small for in_f=%d\n", K); abort(); }
  if (tile > N) tile = N;

  CB(cublasSetStream(ws.cublas, st));
  const float alpha = 1.f, beta = 0.f;

  for (int n0 = 0; n0 < N; n0 += tile) {
    const int nn = (n0 + tile <= N) ? tile : (N - n0);
    w4a16_dequant_rows(ws.wbuf, w, n0, nn, st);
    // Row-major C[M,N] = A[M,K] @ W[N,K]^T. In cuBLAS column-major that is
    // C_cm[N,M] = W_cm[K,N]^T @ A_cm[K,M].
    CB(cublasGemmEx(ws.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    nn, M, K,
                    &alpha,
                    ws.wbuf, CUDA_R_16BF, K,
                    A,       CUDA_R_16BF, K,
                    &beta,
                    C + n0,  CUDA_R_16BF, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

}  // namespace qwen
