// GatedDeltaNet cost, decode and prefill. 48 of 64 layers run this, so it is
// pure overhead against the 14.6 ms/token weight-traffic budget.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../src/kernels/gdn.cuh"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

int main(int argc, char** argv) {
  qwen::GdnDims D;
  const int NL = 48;
  const int CD = D.conv_dim(), VD = D.val_dim(), NV = D.num_v_heads;

  for (int T : {1, 8, 512, 4096}) {
    __nv_bfloat16 *qkv, *out, *z, *w, *cw, *al, *dt;
    float *state, *g, *beta, *cs;
    CK(cudaMalloc(&qkv, size_t(T)*CD*2));  CK(cudaMemset(qkv, 0x3c, size_t(T)*CD*2));
    CK(cudaMalloc(&out, size_t(T)*VD*2));
    CK(cudaMalloc(&z,   size_t(T)*VD*2));  CK(cudaMemset(z, 0x3c, size_t(T)*VD*2));
    CK(cudaMalloc(&w,   size_t(D.head_v)*2)); CK(cudaMemset(w, 0x3c, size_t(D.head_v)*2));
    CK(cudaMalloc(&cw,  size_t(CD)*D.conv_k*2)); CK(cudaMemset(cw, 0x3c, size_t(CD)*D.conv_k*2));
    CK(cudaMalloc(&al,  size_t(NV)*2)); CK(cudaMemset(al, 0, size_t(NV)*2));
    CK(cudaMalloc(&dt,  size_t(NV)*2)); CK(cudaMemset(dt, 0, size_t(NV)*2));
    CK(cudaMalloc(&state, size_t(NV)*D.head_k*D.head_v*4)); CK(cudaMemset(state, 0, size_t(NV)*D.head_k*D.head_v*4));
    CK(cudaMalloc(&cs, size_t(CD)*(D.conv_k-1)*4)); CK(cudaMemset(cs, 0, size_t(CD)*(D.conv_k-1)*4));
    CK(cudaMalloc(&g,    size_t(T)*NV*4)); CK(cudaMemset(g, 0, size_t(T)*NV*4));
    CK(cudaMalloc(&beta, size_t(T)*NV*4)); CK(cudaMemset(beta, 0, size_t(T)*NV*4));

    auto run = [&]{
      qwen::gdn_conv(qkv, cs, qkv, cw, CD, D.conv_k, T, true);
      qwen::gdn_scan(out, state, qkv, g, beta, D, T);
      qwen::gdn_norm_gate(z, out, z, w, T, NV, D.head_v, D.rms_eps);
    };
    for (int i = 0; i < 3; ++i) run();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int IT = T > 512 ? 3 : 30;
    cudaEventRecord(a);
    for (int i = 0; i < IT; ++i) run();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=IT;

    // scan-only, to separate state traffic from the conv
    for (int i = 0; i < 3; ++i) qwen::gdn_scan(out, state, qkv, g, beta, D, T);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < IT; ++i) qwen::gdn_scan(out, state, qkv, g, beta, D, T);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float sms=0; cudaEventElapsedTime(&sms,a,b); sms/=IT;

    const double state_bytes = double(NV)*D.head_k*D.head_v*4*2;   // one read + one write
    printf("T=%-5d  layer %7.3f ms (scan %7.3f)  x%d layers = %8.2f ms   "
           "scan state BW %6.1f GB/s\n",
           T, ms, sms, NL, ms*NL, state_bytes/(sms*1e-3)/1e9);
    cudaFree(qkv); cudaFree(out); cudaFree(z); cudaFree(w); cudaFree(cw);
    cudaFree(al); cudaFree(dt); cudaFree(state); cudaFree(cs); cudaFree(g); cudaFree(beta);
    cudaEventDestroy(a); cudaEventDestroy(b);
  }
  printf("\nbudget context: decode weight traffic is 16.5 ms/token measured (Phase 2)\n");
  return 0;
}
