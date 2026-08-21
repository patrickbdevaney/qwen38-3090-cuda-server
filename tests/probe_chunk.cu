// Probe: does chunked prefill carry the GDN recurrent and conv state across
// chunk boundaries? If prefilling [0,N) in one call and in K calls disagree,
// then every long prompt is already subtly wrong and prefix caching cannot
// possibly be exact.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "../src/model/model.h"

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const int N = argc > 2 ? atoi(argv[2]) : 1024;

  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = 8192; o.max_batch = 2048; o.lm_head_bits = 8; o.verbose = false;
  qwen::model_load(m, md, o);

  std::vector<int32_t> ids(N);
  uint32_t s = 12345;
  for (int i = 0; i < N; ++i) { s = s*1664525u+1013904223u; ids[i] = int32_t(s % 200000) + 8; }

  // Compare the RECURRENT STATE, not just the logits: if the state matches bit
  // for bit across chunkings but the logits do not, the carry is correct and the
  // difference is arithmetic in the attention path.
  const size_t SE = size_t(m.shape.gdn_state_elems());
  const size_t CE = size_t(m.shape.gdn_conv_state_elems());
  std::vector<float> st_ref(SE), cv_ref(CE), st_v(SE), cv_v(CE);

  auto run = [&](int chunk) {
    cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
    cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
    int p = 0;
    while (p < N) { const int n = std::min(chunk, N - p); qwen::model_prefill(m, ids.data()+p, n, p); p += n; }
    cudaDeviceSynchronize();
    std::vector<uint16_t> lg(m.shape.vocab_size);
    cudaMemcpy(lg.data(), m.logits, lg.size()*2, cudaMemcpyDeviceToHost);
    return lg;
  };
  auto grab = [&](std::vector<float>& st, std::vector<float>& cv) {
    cudaMemcpy(st.data(), m.gdn_state, SE*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(cv.data(), m.gdn_conv, CE*4, cudaMemcpyDeviceToHost);
  };

  auto one = run(N);
  grab(st_ref, cv_ref);
  printf("N=%d, one chunk vs:\n", N);
  for (int c : {512, 256, 128, 64}) {
    auto v = run(c);
    grab(st_v, cv_v);
    size_t sd = 0, cd = 0; double sw = 0, cw = 0;
    for (size_t i = 0; i < SE; ++i) if (st_v[i] != st_ref[i]) { ++sd; sw = std::max(sw, (double)std::fabs(st_v[i]-st_ref[i])); }
    for (size_t i = 0; i < CE; ++i) if (cv_v[i] != cv_ref[i]) { ++cd; cw = std::max(cw, (double)std::fabs(cv_v[i]-cv_ref[i])); }
    printf("    gdn_state %8zu/%zu differ (max %.3e)   gdn_conv %6zu/%zu differ (max %.3e)\n",
           sd, SE, sw, cd, CE, cw);
    size_t diff = 0; double worst = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      if (v[i] != one[i]) ++diff;
      worst = std::max(worst, (double)std::fabs(b2f(v[i]) - b2f(one[i])));
    }
    printf("  chunk %4d : %6zu / %d bits differ, max|d| %.4e\n", c, diff, m.shape.vocab_size, worst);
  }
  return 0;
}
