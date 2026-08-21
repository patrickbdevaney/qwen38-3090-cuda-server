// Layerwise diff against the Phase 1 reference dump: which layer first diverges.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "../third_party/json.hpp"
#include "../src/model/model.h"
using json = nlohmann::json;
static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc, char** argv) {
  const std::string md = argv[1];
  const std::string fx = argv[2];
  const std::string which = argc > 3 ? argv[3] : "p0_factual";
  json man; { std::ifstream f(fx + "/manifest.json"); f >> man; }
  std::vector<int32_t> ids;
  for (auto& v : man["prompts"][which]["ids"]) ids.push_back(v.get<int32_t>());
  const int T = int(ids.size());

  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = 4096; o.max_batch = 256; o.lm_head_bits = 16; o.verbose = false;
  qwen::model_load(m, md, o);
  const int NL = m.shape.num_hidden_layers, H = m.shape.hidden_size;
  cudaMalloc(&m.dbg_hidden, size_t(NL + 1) * m.max_batch * H * 2);
  qwen::model_prefill(m, ids.data(), T, 0);
  cudaDeviceSynchronize();

  std::vector<uint16_t> got(size_t(NL + 1) * m.max_batch * H);
  cudaMemcpy(got.data(), m.dbg_hidden, got.size() * 2, cudaMemcpyDeviceToHost);

  printf("%-8s %12s %12s %12s  %s\n", "layer", "max|diff|", "rel", "ref |max|", "kind");
  for (int L = 0; L <= NL; ++L) {
    char p[512]; snprintf(p, sizeof p, "%s/%s/hidden_%02d.bf16", fx.c_str(), which.c_str(), L);
    std::vector<uint16_t> ref(size_t(T) * H);
    std::ifstream f(p, std::ios::binary);
    if (!f) { printf("missing %s\n", p); break; }
    f.read(reinterpret_cast<char*>(ref.data()), ref.size() * 2);
    double mx = 0, rmx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const size_t gi = size_t(L) * m.max_batch * H + (i / H) * H + (i % H);
      mx = std::max(mx, std::fabs(double(b2f(got[gi])) - b2f(ref[i])));
      rmx = std::max(rmx, std::fabs(double(b2f(ref[i]))));
    }
    // NOTE: HF's last hidden_states entry is AFTER the final norm, so index
    // num_layers is not the raw output of the last decoder layer and must not
    // be compared against it.
    const char* kind = (L == 0) ? "embed" : (L == NL) ? "final-norm (skipped)"
        : (m.shape.layer_types[L-1] == qwen::LayerKind::FullAttention ? "attn" : "gdn");
    if (L == NL) { printf("%-8d %12s %12s %12.4e  %s\n", L, "-", "-", rmx, kind); continue; }
    printf("%-8d %12.4e %12.2e %12.4e  %s%s\n", L, mx, rmx > 0 ? mx / rmx : 0, rmx, kind,
           (rmx > 0 && mx / rmx > 0.05) ? "   <-- DIVERGED" : "");
    if (rmx > 0 && mx / rmx > 0.5) break;
  }
  return 0;
}
