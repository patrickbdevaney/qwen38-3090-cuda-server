// GATE: the CUDA vision tower against the official transformers
// Qwen3_5VisionModel (tools/dump_vision_ref.py).
//
// Synthetic pixel_values on purpose: it separates "does the ViT compute the
// right thing" from "does our image decode and resize match PIL", which are
// different questions with different tolerances. Stage taps so a mismatch names
// the stage rather than just the output.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include "../third_party/json.hpp"
#include "../src/vision/vit.h"

using json = nlohmann::json;
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }

static std::vector<uint16_t> rd(const std::string& p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { printf("missing fixture %s\n", p.c_str()); exit(2); }
  std::vector<uint16_t> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * 2);
  if (size_t(f.gcount()) != n * 2) { printf("short read %s\n", p.c_str()); exit(2); }
  return v;
}

static bool cmp(const char* name, const std::vector<uint16_t>& got,
                const std::vector<uint16_t>& want, double tol) {
  double se = 0, sw = 0, maxa = 0; long nf = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const float a = b2f(got[i]), b = b2f(want[i]);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++nf; continue; }
    const double d = std::fabs(double(a) - double(b));
    maxa = std::max(maxa, d); se += d * d; sw += double(b) * double(b);
  }
  const double rel = sw > 0 ? std::sqrt(se / sw) : 0;
  const bool ok = nf == 0 && rel <= tol;
  printf("  %-14s max|d| %10.3e  rel %10.3e  nonfinite %4ld   %s\n",
         name, maxa, rel, nf, ok ? "OK" : "FAIL");
  return ok;
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const std::string fx = argc > 2 ? argv[2] : "tests/fixtures/vision";

  json man; { std::ifstream f(fx + "/manifest.json");
              if (!f) { printf("no manifest in %s\n", fx.c_str()); return 2; } f >> man; }
  const int gt = man["grid"][0], gh = man["grid"][1], gw = man["grid"][2];
  const int N = man["n_patch"], PD = man["patch_dim"];
  const int HID = man["hidden_size"], OUT = man["out_hidden_size"], NOUT = man["n_out"];

  qwen::VisionTower v;
  qwen::VisionLoadOptions o;
  o.max_patches = std::max(N, 256);
  o.debug = true;
  qwen::vision_load(v, dir, o);

  bool ok = true;
  auto eq = [&](const char* n, int a, int b) {
    if (a != b) { printf("  shape %s: %d != %d\n", n, a, b); ok = false; }
  };
  eq("hidden", v.sh.hidden, HID);
  eq("depth", v.sh.depth, man["depth"].get<int>());
  eq("heads", v.sh.num_heads, man["num_heads"].get<int>());
  eq("head_dim", v.sh.head_dim, man["head_dim"].get<int>());
  eq("merge", v.sh.spatial_merge, man["spatial_merge_size"].get<int>());
  eq("grid_per_side", v.sh.grid_per_side, man["num_grid_per_side"].get<int>());
  eq("patch_dim", v.sh.patch_dim(), PD);
  if (!ok) return 1;

  auto pix = rd(fx + "/pixel_values.bf16", size_t(N) * PD);
  __nv_bfloat16* d_pix = nullptr;
  CK(cudaMalloc(&d_pix, pix.size() * 2));
  CK(cudaMemcpy(d_pix, pix.data(), pix.size() * 2, cudaMemcpyHostToDevice));

  const __nv_bfloat16* out = qwen::vision_forward(v, d_pix, gt, gh, gw);
  CK(cudaDeviceSynchronize());

  printf("\ngate_vision (grid %dx%dx%d, %d patches -> %d tokens)\n", gt, gh, gw, N, NOUT);

  // bf16 through 27 blocks; the reference itself ran on CPU in bf16.
  const double TOL = 6e-2;
  // patch_embed + the bilinearly resampled pos_embed. Gated, because this is
  // where an align_corners or patch-ordering mistake shows up on its own rather
  // than smeared through 27 blocks.
  { auto want = rd(fx + "/stage_post_pos_embed.bf16", size_t(N) * HID);
    std::vector<uint16_t> got(want.size());
    CK(cudaMemcpy(got.data(), v.dbg_post_pos, got.size() * 2, cudaMemcpyDeviceToHost));
    ok &= cmp("patch+pos_embed", got, want, 1e-2); }

  { auto want = rd(fx + "/stage_merger.bf16", size_t(NOUT) * OUT);
    std::vector<uint16_t> got(want.size());
    CK(cudaMemcpy(got.data(), out, got.size() * 2, cudaMemcpyDeviceToHost));
    ok &= cmp("image tokens", got, want, TOL); }

  printf("  RESULT: %s\n", ok ? "PASS" : "FAIL");
  qwen::vision_free(v);
  return ok ? 0 : 1;
}
