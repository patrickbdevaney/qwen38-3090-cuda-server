// Phase 0 hardware microbenchmarks for RTX 3090 (GA102, sm_86).
// Measures the numbers the rest of the project is designed against:
//   1. achieved DRAM streaming-read bandwidth (the decode roofline denominator)
//   2. achieved DRAM copy (read+write) bandwidth
//   3. L2 resident bandwidth
//   4. achieved BF16 tensor-core TFLOPS via mma.sync.m16n8k16
//   5. cp.async (LDGSTS) staged-copy throughput vs plain global->shared
//   6. weight-traffic proxy: time to stream the exact per-token weight footprint
//
// Build: nvcc -O3 -arch=sm_86 -o microbench microbench.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// ---------------------------------------------------------------- read BW
__global__ void k_read(const float4* __restrict__ p, size_t n4, float* sink) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0,0,0,0);
  for (; i < n4; i += stride) {
    float4 v = p[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  float s = acc.x + acc.y + acc.z + acc.w;
  if (s == 1.2345e30f) sink[0] = s;   // never true; defeats DCE
}

__global__ void k_read_rep(const float4* __restrict__ p, size_t n4, float* sink, int rep) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0,0,0,0);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n4; i += stride) {
      float4 v = p[i];
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
  float s = acc.x + acc.y + acc.z + acc.w;
  if (s == 1.2345e30f) sink[0] = s;
}

__global__ void k_copy(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) dst[i] = src[i];
}

// ---------------------------------------------------------------- BF16 MMA
// m16n8k16 bf16 -> f32.  Each warp issues MMAs back to back on register-resident
// fragments; this measures issue-limited tensor-core throughput, not memory.
__global__ void k_mma_bf16(float* sink, int iters) {
  uint32_t a0=0x3f803f80u, a1=0x3f803f80u, a2=0x3f803f80u, a3=0x3f803f80u;
  uint32_t b0=0x3f803f80u, b1=0x3f803f80u;
  float c0=0.f, c1=0.f, c2=0.f, c3=0.f;
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b1),"r"(b0));
  }
  if (c0 == 1.2345e30f) sink[0] = c0+c1+c2+c3;
}

// ---------------------------------------------------------------- cp.async
// Stage a tile global->shared with LDGSTS (16B/thread) and commit/wait.
template<int BYTES>
__global__ void k_cpasync(const float4* __restrict__ src, size_t n4, float* sink, int iters) {
  extern __shared__ __align__(16) char smem[];
  const int tid = threadIdx.x;
  size_t base = (blockIdx.x * (size_t)blockDim.x);
  float acc = 0.f;
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    size_t off = (base + it * (size_t)gridDim.x * blockDim.x) % (n4 - blockDim.x);
    uint32_t sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + tid * 16));
    const float4* g = src + off + tid;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(sptr), "l"(g), "n"(BYTES));
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    acc += reinterpret_cast<float4*>(smem)[tid].x;
  }
  if (acc == 1.2345e30f) sink[0] = acc;
}

__global__ void k_ldgsts_ref(const float4* __restrict__ src, size_t n4, float* sink, int iters) {
  extern __shared__ __align__(16) char smem[];
  const int tid = threadIdx.x;
  size_t base = (blockIdx.x * (size_t)blockDim.x);
  float acc = 0.f;
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    size_t off = (base + it * (size_t)gridDim.x * blockDim.x) % (n4 - blockDim.x);
    reinterpret_cast<float4*>(smem)[tid] = src[off + tid];   // LDG + STS
    __syncthreads();
    acc += reinterpret_cast<float4*>(smem)[tid].x;
  }
  if (acc == 1.2345e30f) sink[0] = acc;
}

// ---------------------------------------------------------------- timing
struct Timer {
  cudaEvent_t a, b;
  Timer(){ CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  ~Timer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
  void start(){ CK(cudaEventRecord(a)); }
  float stop(){ CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
                float ms; CK(cudaEventElapsedTime(&ms,a,b)); return ms; }
};

static float median(std::vector<float>& v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static float p95(std::vector<float>& v){ std::sort(v.begin(),v.end()); return v[(size_t)(v.size()*0.95f)]; }

int main() {
  int dev = 0; CK(cudaSetDevice(dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  size_t freeB, totB; CK(cudaMemGetInfo(&freeB, &totB));

  printf("=== DEVICE ===\n");
  printf("name                     : %s\n", p.name);
  printf("compute capability       : sm_%d%d\n", p.major, p.minor);
  printf("SMs                      : %d\n", p.multiProcessorCount);
  printf("clock (max)              : %.0f MHz\n", p.clockRate/1000.0);
  printf("memory clock             : %.0f MHz\n", p.memoryClockRate/1000.0);
  printf("memory bus width         : %d bit\n", p.memoryBusWidth);
  printf("theoretical peak BW      : %.1f GB/s\n",
         2.0*p.memoryClockRate*1e3*(p.memoryBusWidth/8)/1e9);
  printf("L2 cache                 : %.2f MB\n", p.l2CacheSize/1048576.0);
  printf("shared mem / block (opt-in): %zu KB\n", (size_t)p.sharedMemPerBlockOptin/1024);
  printf("shared mem / SM          : %zu KB\n", (size_t)p.sharedMemPerMultiprocessor/1024);
  printf("regs / SM                : %d\n", p.regsPerMultiprocessor);
  printf("total VRAM               : %.2f GiB\n", totB/1073741824.0);
  printf("free  VRAM               : %.2f GiB  (%.2f GiB already in use)\n",
         freeB/1073741824.0, (totB-freeB)/1073741824.0);

  const int WARMUP = 3, ITERS = 10;
  Timer t;
  float* sink; CK(cudaMalloc(&sink, 256));

  // ---- 1/2. DRAM bandwidth ------------------------------------------------
  printf("\n=== DRAM BANDWIDTH ===\n");
  size_t bytes = 2ull<<30;                       // 2 GiB, >> 6MB L2
  float4 *A, *B;
  CK(cudaMalloc(&A, bytes)); CK(cudaMalloc(&B, bytes));
  CK(cudaMemset(A, 1, bytes));
  size_t n4 = bytes/sizeof(float4);
  int blocks = p.multiProcessorCount * 16, threads = 256;

  {
    std::vector<float> ts;
    for (int i=0;i<WARMUP+ITERS;i++){ t.start(); k_read<<<blocks,threads>>>(A,n4,sink);
      float ms=t.stop(); if(i>=WARMUP) ts.push_back(ms); }
    float m=median(ts), q=p95(ts);
    printf("streaming read  (2 GiB) : %.2f ms median -> %.1f GB/s   [p95 %.2f ms]\n",
           m, bytes/(m*1e-3)/1e9, q);
  }
  {
    std::vector<float> ts;
    for (int i=0;i<WARMUP+ITERS;i++){ t.start(); k_copy<<<blocks,threads>>>(A,B,n4);
      float ms=t.stop(); if(i>=WARMUP) ts.push_back(ms); }
    float m=median(ts);
    printf("copy read+write (2 GiB) : %.2f ms median -> %.1f GB/s effective\n",
           m, 2.0*bytes/(m*1e-3)/1e9);
  }
  {   // L2-resident: 4 MB buffer fits in 6 MB L2. Repeat inside the kernel so the
      // measurement is not dominated by the ~5us launch overhead.
    size_t lb = 4ull<<20; size_t ln4 = lb/sizeof(float4);
    const int REP = 200;
    std::vector<float> ts;
    for (int i=0;i<WARMUP+ITERS;i++){ t.start(); k_read_rep<<<blocks,threads>>>(A,ln4,sink,REP);
      float ms=t.stop(); if(i>=WARMUP) ts.push_back(ms); }
    float m=median(ts);
    printf("L2-resident read (4 MB) : %.4f ms median -> %.1f GB/s  [%d reps in-kernel]\n",
           m, (double)lb*REP/(m*1e-3)/1e9, REP);
  }

  // ---- 3. weight-traffic proxy -------------------------------------------
  printf("\n=== DECODE ROOFLINE PROXY ===\n");
  {
    // Per-token weight footprint measured from the real checkpoint:
    //   INT4 body + scales + zeros + small bf16 = 11.820 GiB
    //   INT4 lm_head g128                       =  0.615 GiB
    const double W_GiB = 11.820 + 0.615;
    std::vector<float> ts;
    for (int i=0;i<WARMUP+ITERS;i++){ t.start(); k_read<<<blocks,threads>>>(A,n4,sink);
      float ms=t.stop(); if(i>=WARMUP) ts.push_back(ms); }
    float m=median(ts);
    double bw = bytes/(m*1e-3);                       // B/s
    double per_tok_ms = W_GiB*1073741824.0/bw*1e3;
    printf("weight footprint/token  : %.3f GiB\n", W_GiB);
    printf("at measured read BW     : %.2f ms/token -> %.1f tok/s AR CEILING\n",
           per_tok_ms, 1000.0/per_tok_ms);
  }
  CK(cudaFree(A)); CK(cudaFree(B));

  // ---- 4. BF16 tensor cores ----------------------------------------------
  printf("\n=== BF16 TENSOR CORES ===\n");
  {
    int nblk = p.multiProcessorCount*8, nthr = 256, iters = 20000;
    std::vector<float> ts;
    for (int i=0;i<WARMUP+ITERS;i++){ t.start(); k_mma_bf16<<<nblk,nthr>>>(sink,iters);
      float ms=t.stop(); if(i>=WARMUP) ts.push_back(ms); }
    float m=median(ts);
    // 2 MMAs per iter; m16n8k16 = 16*8*16*2 flops per warp-MMA
    double warps = (double)nblk*nthr/32.0;
    double flops = warps*iters*2.0*(16.0*8.0*16.0*2.0);
    int sclk=0, mclk=0;
    cudaDeviceGetAttribute(&sclk, cudaDevAttrClockRate, dev);
    printf("mma.sync.m16n8k16.bf16  : %.2f ms -> %.1f TFLOPS\n", m, flops/(m*1e-3)/1e12);
    printf("  datasheet GA102 dense  : %.1f TFLOPS (82 SM x 256 FMA/clk x 2 x %.3f GHz)\n",
           82*256*2*(sclk/1e6)/1e3, sclk/1e6);
  }

  // ---- 5. cp.async --------------------------------------------------------
  printf("\n=== cp.async (LDGSTS) vs LDG+STS ===\n");
  {
    size_t sb = 512ull<<20; float4* S; CK(cudaMalloc(&S, sb)); CK(cudaMemset(S,1,sb));
    size_t sn4 = sb/sizeof(float4);
    int nblk = p.multiProcessorCount*8, nthr = 256, iters = 2000;
    size_t shm = nthr*16;
    std::vector<float> ta, tb;
    for (int i=0;i<WARMUP+ITERS;i++){ t.start(); k_cpasync<16><<<nblk,nthr,shm>>>(S,sn4,sink,iters);
      float ms=t.stop(); if(i>=WARMUP) ta.push_back(ms); }
    for (int i=0;i<WARMUP+ITERS;i++){ t.start(); k_ldgsts_ref<<<nblk,nthr,shm>>>(S,sn4,sink,iters);
      float ms=t.stop(); if(i>=WARMUP) tb.push_back(ms); }
    float ma=median(ta), mb=median(tb);
    double moved = (double)nblk*nthr*16.0*iters;
    printf("cp.async.cg 16B/thread  : %.2f ms -> %.1f GB/s\n", ma, moved/(ma*1e-3)/1e9);
    printf("LDG+STS      16B/thread : %.2f ms -> %.1f GB/s\n", mb, moved/(mb*1e-3)/1e9);
    printf("cp.async speedup        : %.2fx\n", mb/ma);
    CK(cudaFree(S));
  }

  // ---- 6. shared memory ceiling check ------------------------------------
  printf("\n=== SHARED MEMORY CEILING (head_dim 256 attention) ===\n");
  {
    printf("opt-in max smem/block   : %zu KB\n", (size_t)p.sharedMemPerBlockOptin/1024);
    auto tile = [](int br,int bc){ return (br*256*2 + bc*256*2*2)/1024; };
    printf("Br=64,Bc=64 Q+K+V bf16  : %d KB  %s\n", tile(64,64),
           tile(64,64) <= (int)(p.sharedMemPerBlockOptin/1024) ? "fits" : "DOES NOT FIT");
    printf("Br=64,Bc=32 Q+K+V bf16  : %d KB  %s\n", tile(64,32),
           tile(64,32) <= (int)(p.sharedMemPerBlockOptin/1024) ? "fits" : "DOES NOT FIT");
    printf("Br=32,Bc=32 Q+K+V bf16  : %d KB  %s\n", tile(32,32),
           tile(32,32) <= (int)(p.sharedMemPerBlockOptin/1024) ? "fits" : "DOES NOT FIT");
    printf("Br=64,Bc=64 K,V in fp8  : %d KB\n", (64*256*2 + 64*256*1*2)/1024);
  }

  CK(cudaFree(sink));
  printf("\nOK\n");
  return 0;
}
