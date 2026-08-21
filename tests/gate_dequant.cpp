// GATE (laguna ladder G1): W4A16 dequantization must be BIT-EXACT against
// compressed-tensors' own unpack_from_int32 + dequantize.
//
// This runs before any GEMV exists, because a fast kernel over a wrong unpack
// is not an optimization, it is a bug that produces fluent nonsense.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include "../third_party/json.hpp"
#include "../src/loader/safetensors.h"
#include "../src/loader/w4a16_unpack.h"

using json = nlohmann::json;

static float bf16_to_f32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

int main(int argc, char** argv) {
  const std::string model_dir = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const std::string fx = argc > 2 ? argv[2] : "tests/fixtures/dequant";

  std::ifstream mf(fx + "/manifest.json");
  if (!mf) { fprintf(stderr, "cannot open %s/manifest.json\n", fx.c_str()); return 2; }
  json man; mf >> man;

  qwen::SafeTensors st;
  try { st.open_dir(model_dir); }
  catch (const std::exception& e) { fprintf(stderr, "loader: %s\n", e.what()); return 2; }
  printf("mapped %zu tensors, %.2f GiB\n", st.size(), st.mapped_bytes() / 1073741824.0);

  const int64_t gs = man["group_size"].get<int64_t>();
  size_t nfail = 0, nchecked = 0;
  double worst = 0.0;
  size_t q_mismatch = 0;

  for (const auto& t : man["tensors"]) {
    const std::string base = t["name"].get<std::string>();
    const int64_t out_f = t["out_features"], in_f = t["in_features"];
    const int64_t rows = t["rows"], cols = t["cols"];
    const int64_t G = in_f / gs;

    const auto& packed = st.get(base + ".weight_packed");
    const auto& scale  = st.get(base + ".weight_scale");
    const auto& zpv    = st.get(base + ".weight_zero_point");

    // Shape sanity: these are the assumptions the unpack encodes.
    if (packed.shape != std::vector<int64_t>{out_f, in_f / 8} ||
        scale.shape  != std::vector<int64_t>{out_f, G} ||
        zpv.shape    != std::vector<int64_t>{out_f / 8, G}) {
      printf("  SHAPE MISMATCH %s\n", base.c_str()); ++nfail; continue;
    }

    const auto* pw = reinterpret_cast<const uint32_t*>(packed.data);
    const auto* zw = reinterpret_cast<const uint32_t*>(zpv.data);
    const auto* sc = reinterpret_cast<const uint16_t*>(scale.data);   // BF16

    std::vector<float> want(rows * cols);
    std::vector<int8_t> wantq(rows * cols);
    std::ifstream df(fx + "/" + t["deq_file"].get<std::string>(), std::ios::binary);
    std::ifstream qf(fx + "/" + t["q_file"].get<std::string>(), std::ios::binary);
    df.read(reinterpret_cast<char*>(want.data()), rows * cols * 4);
    qf.read(reinterpret_cast<char*>(wantq.data()), rows * cols);

    double tmax = 0;
    for (int64_t o = 0; o < rows; ++o) {
      for (int64_t i = 0; i < cols; ++i) {
        const int32_t q  = qwen::w4_q(pw, in_f, o, i);
        const int64_t g  = i / gs;
        const int32_t zp = qwen::w4_zp(zw, G, o, g);
        const float   s  = bf16_to_f32(sc[o * G + g]);
        const float   got = static_cast<float>(q - zp) * s;
        const float   ref = want[o * cols + i];
        if (q != wantq[o * cols + i]) ++q_mismatch;
        const double d = std::fabs(static_cast<double>(got) - ref);
        if (d > tmax) tmax = d;
        ++nchecked;
      }
    }
    if (tmax > worst) worst = tmax;
    // The reference multiplies in fp32 from a bf16 scale, exactly as we do, so
    // this should be bit-identical, not merely close.
    const bool ok = (tmax == 0.0);
    printf("  %-52s max|diff| %.3e  %s\n", base.c_str(), tmax, ok ? "exact" : "FAIL");
    if (!ok) ++nfail;
  }

  printf("\ngate_dequant\n");
  printf("  values checked   : %zu\n", nchecked);
  printf("  int4 mismatches  : %zu\n", q_mismatch);
  printf("  worst |diff|     : %.3e\n", worst);
  printf("  tensors failing  : %zu\n", nfail);
  const bool pass = !nfail && !q_mismatch;
  printf("  RESULT           : %s\n", pass ? "PASS (bit-exact)" : "FAIL");
  return pass ? 0 : 1;
}
