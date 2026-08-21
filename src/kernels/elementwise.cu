#include "elementwise.cuh"
#include <cfloat>
#include <cuda_runtime.h>

namespace qwen {
namespace {

__global__ void k_rmsnorm(__nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ w, int dim, float eps) {
  const int r = blockIdx.x;
  const size_t off = size_t(r) * dim;
  float a = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = __bfloat162float(x[off + i]);
    a += v * v;
  }
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) a += __shfl_down_sync(0xffffffffu, a, s);
  __shared__ float part[32];
  const int nw = blockDim.x >> 5;
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = a;
  __syncthreads();
  __shared__ float inv;
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t += part[i];
    inv = rsqrtf(t / dim + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    o[off + i] = __float2bfloat16(__bfloat162float(x[off + i]) * inv *
                                  (1.0f + __bfloat162float(w[i])));
}

__global__ void k_add(__nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = __float2bfloat16(__bfloat162float(o[i]) + __bfloat162float(x[i]));
}

__global__ void k_swiglu(__nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ f,
                         int inter) {
  const int r = blockIdx.x;
  const size_t src = size_t(r) * 2 * inter, dst = size_t(r) * inter;
  for (int i = threadIdx.x; i < inter; i += blockDim.x) {
    const float g = __bfloat162float(f[src + i]);
    const float u = __bfloat162float(f[src + inter + i]);
    o[dst + i] = __float2bfloat16((g / (1.f + expf(-g))) * u);
  }
}

__global__ void k_embed(__nv_bfloat16* __restrict__ o, const int8_t* __restrict__ t,
                        const float* __restrict__ sc, const int32_t* __restrict__ ids,
                        int dim) {
  const int r = blockIdx.x;
  const int id = ids[r];
  const float s = sc[id];
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    o[size_t(r) * dim + i] = __float2bfloat16(float(t[size_t(id) * dim + i]) * s);
}

__global__ void k_embed_bf16(__nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ t,
                             const int32_t* __restrict__ ids, int dim) {
  const int r = blockIdx.x;
  const size_t base = size_t(ids[r]) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) o[size_t(r) * dim + i] = t[base + i];
}

__global__ void k_quant_embed(int8_t* __restrict__ q, float* __restrict__ sc,
                              const __nv_bfloat16* __restrict__ src, int dim) {
  const int r = blockIdx.x;
  const size_t off = size_t(r) * dim;
  float mx = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    mx = fmaxf(mx, fabsf(__bfloat162float(src[off + i])));
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, s));
  __shared__ float part[32];
  const int nw = blockDim.x >> 5;
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = mx;
  __syncthreads();
  __shared__ float s_scale, s_inv;
  if (threadIdx.x == 0) {
    float m = 0.f;
    for (int i = 0; i < nw; ++i) m = fmaxf(m, part[i]);
    s_scale = m / 127.0f;
    s_inv = (m > 0.f) ? 127.0f / m : 0.f;
    sc[r] = s_scale;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = __bfloat162float(src[off + i]) * s_inv;
    q[off + i] = int8_t(max(-127, min(127, __float2int_rn(v))));
  }
}

__global__ void k_argmax_partial(int32_t* __restrict__ idx, float* __restrict__ val,
                                 const __nv_bfloat16* __restrict__ x, int n) {
  const int chunk = blockIdx.x;
  const int stride = gridDim.x * blockDim.x;
  float best = -FLT_MAX; int bi = 0;
  for (int i = chunk * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float v = __bfloat162float(x[i]);
    // strict > keeps the LOWEST index on ties, matching torch.argmax
    if (v > best) { best = v; bi = i; }
  }
  __shared__ float sv[256]; __shared__ int si[256];
  sv[threadIdx.x] = best; si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sv[threadIdx.x + s] > sv[threadIdx.x] ||
          (sv[threadIdx.x + s] == sv[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x])) {
        sv[threadIdx.x] = sv[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { val[chunk] = sv[0]; idx[chunk] = si[0]; }
}

__global__ void k_argmax_final(int32_t* __restrict__ out, const int32_t* __restrict__ idx,
                               const float* __restrict__ val, int n) {
  float best = -FLT_MAX; int bi = 0;
  for (int i = 0; i < n; ++i)
    if (val[i] > best || (val[i] == best && idx[i] < bi)) { best = val[i]; bi = idx[i]; }
  out[0] = bi;
}

}  // namespace

void rmsnorm(__nv_bfloat16* o, const __nv_bfloat16* x, const __nv_bfloat16* w,
             int rows, int dim, float eps, cudaStream_t st) {
  k_rmsnorm<<<rows, 256, 0, st>>>(o, x, w, dim, eps);
}
void residual_add(__nv_bfloat16* o, const __nv_bfloat16* x, int n, cudaStream_t st) {
  k_add<<<(n + 255) / 256, 256, 0, st>>>(o, x, n);
}
void swiglu(__nv_bfloat16* o, const __nv_bfloat16* f, int rows, int inter, cudaStream_t st) {
  k_swiglu<<<rows, 256, 0, st>>>(o, f, inter);
}
void embed_int8(__nv_bfloat16* o, const int8_t* t, const float* sc, const int32_t* ids,
                int n, int dim, cudaStream_t st) {
  k_embed<<<n, 256, 0, st>>>(o, t, sc, ids, dim);
}
void embed_bf16(__nv_bfloat16* o, const __nv_bfloat16* t, const int32_t* ids, int n,
                int dim, cudaStream_t st) {
  k_embed_bf16<<<n, 256, 0, st>>>(o, t, ids, dim);
}
void quantize_embed_int8(int8_t* q, float* sc, const __nv_bfloat16* src, int rows,
                         int dim, cudaStream_t st) {
  k_quant_embed<<<rows, 256, 0, st>>>(q, sc, src, dim);
}
void argmax(int32_t* out, const __nv_bfloat16* x, int n, int32_t* scratch, cudaStream_t st) {
  const int chunks = 256;
  int32_t* idx = scratch;
  float* val = reinterpret_cast<float*>(scratch + chunks);
  k_argmax_partial<<<chunks, 256, 0, st>>>(idx, val, x, n);
  k_argmax_final<<<1, 1, 0, st>>>(out, idx, val, chunks);
}

}  // namespace qwen
