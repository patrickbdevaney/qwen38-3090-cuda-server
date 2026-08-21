// Attention cost against the 16.5 ms/token weight-traffic budget. 16 of 64
// layers, but KV traffic grows with context: 32 KiB/token across all 16 layers
// means 4 GiB read per decoded token at 128K.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "../src/kernels/attn.cuh"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

int main() {
  qwen::AttnDims D;
  const int NL = 16, NQ = D.num_q_heads, NKV = D.num_kv_heads, HD = D.head_dim;
  printf("decode: KV read per token = %d layers x %d kvh x %d x 2 = %d KiB/token of ctx\n",
         NL, NKV, HD, NL * NKV * HD * 2 / 1024);
  for (int ctx : {4096, 32768, 65536, 131072}) {
    __nv_bfloat16 *q, *o; uint8_t *kc, *vc; float* ws;
    CK(cudaMalloc(&q, size_t(NQ)*HD*2)); CK(cudaMemset(q, 0x3c, size_t(NQ)*HD*2));
    CK(cudaMalloc(&o, size_t(NQ)*HD*2));
    CK(cudaMalloc(&kc, size_t(ctx)*NKV*HD)); CK(cudaMemset(kc, 0x38, size_t(ctx)*NKV*HD));
    CK(cudaMalloc(&vc, size_t(ctx)*NKV*HD)); CK(cudaMemset(vc, 0x38, size_t(ctx)*NKV*HD));
    const int sp = qwen::attn_decode_splits(ctx);
    CK(cudaMalloc(&ws, qwen::attn_decode_workspace_bytes(D, sp)));
    for (int i = 0; i < 5; ++i) qwen::attn_decode(o, q, kc, vc, nullptr, nullptr, ctx, ctx, D, ws, sp);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int IT = 30;
    cudaEventRecord(a);
    for (int i = 0; i < IT; ++i) qwen::attn_decode(o, q, kc, vc, nullptr, nullptr, ctx, ctx, D, ws, sp);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=IT;
    const double bytes = 2.0*ctx*NKV*HD;
    printf("ctx %7d  layer %7.3f ms  x%2d = %7.2f ms/token   KV BW %6.1f GB/s  splits %d\n",
           ctx, ms, NL, ms*NL, bytes/(ms*1e-3)/1e9, sp);
    cudaFree(q); cudaFree(o); cudaFree(kc); cudaFree(vc); cudaFree(ws);
    cudaEventDestroy(a); cudaEventDestroy(b);
  }
  printf("\nbudget: weights 16.5 ms + GDN 0.76 ms per token (measured)\n");
  return 0;
}
