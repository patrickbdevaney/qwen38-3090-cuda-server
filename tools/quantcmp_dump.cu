// Dump this server's next-token distribution at EVERY position of each
// comparison prompt, so it can be scored against the BF16 reference.
//
// Teacher forcing, not generation: the prompt is pushed through the model in
// chunks and the logits for every position are kept. That turns an 8-prompt set
// into ~15K independent next-token distributions, which is enough to say
// something about a quantisation rather than about one lucky continuation.
//
// Output per prompt is fp16 [T, vocab], byte-identical in layout to what
// tools/quantcmp_bf16.py writes, plus an ids.json so the comparison can assert
// that our tokenizer produced exactly the same sequence HF did. If the ids ever
// disagree the logits are not comparable and the comparison must fail loudly
// rather than quietly score the wrong positions.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include "../src/model/model.h"
#include "../src/cache/prefix.h"
#include "../src/tokenizer/bpe.h"
#include "../third_party/json.hpp"

using ojson = nlohmann::ordered_json;

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} } while(0)

// bf16 -> fp16 on the host. The reference is fp16 because 15K x 248K x 4 bytes
// of fp32 is 15 GiB of disk for no extra information: bf16 logits carry 8
// mantissa bits and fp16 carries 10, so the conversion is lossless here.
static uint16_t bf16_to_fp16(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f; memcpy(&f, &u, 4);
  __half h = __float2half(f);
  uint16_t o; memcpy(&o, &h, 2);
  return o;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("usage: %s <model_dir> <prompts.json> <out_dir> "
           "[lm_head_bits=8] [kv=0] [chunk=1024]\n", argv[0]);
    return 1;
  }
  const std::string md = argv[1], pj = argv[2], out = argv[3];
  const int lm_bits = argc > 4 ? atoi(argv[4]) : 8;
  const int kvmode  = argc > 5 ? atoi(argv[5]) : 0;
  const int chunk   = argc > 6 ? atoi(argv[6]) : 1024;

  ojson prompts;
  { std::ifstream f(pj); if (!f) { printf("cannot open %s\n", pj.c_str()); return 1; }
    f >> prompts; }

  qwen::Tokenizer tok;
  tok.load(md + "/tokenizer.json");

  // Size the context for the longest prompt, and no larger: this runs next to
  // nothing else, but a 262K arena would push the logits buffer out of VRAM.
  int longest = 0;
  std::vector<std::pair<std::string, std::vector<int32_t>>> work;
  for (auto it = prompts.begin(); it != prompts.end(); ++it) {
    std::vector<int32_t> ids = tok.encode(it.value().get<std::string>());
    longest = std::max(longest, int(ids.size()));
    work.emplace_back(it.key(), std::move(ids));
  }
  printf("%zu prompts, longest %d tokens\n", work.size(), longest);

  qwen::Model m;
  qwen::LoadOptions o;
  o.max_ctx = longest + 64;
  o.max_batch = chunk;
  o.lm_head_bits = lm_bits;
  o.embed_host = true;
  o.verbose = true;
  if (kvmode == 1) { o.kv_k = qwen::KvFmt::FP8;  o.kv_v = qwen::KvFmt::INT4; }
  else if (kvmode == 2) { o.kv_k = qwen::KvFmt::INT4; o.kv_v = qwen::KvFmt::INT4; }
  qwen::model_load(m, md, o);

  const int V = m.shape.vocab_size;
  __nv_bfloat16* d_logits = nullptr;
  CK(cudaMalloc(&d_logits, size_t(chunk) * V * 2));
  std::vector<uint16_t> host(size_t(chunk) * V), fp16(size_t(chunk) * V);

  ojson manifest;
  manifest["model"] = md;
  manifest["lm_head_bits"] = lm_bits;
  manifest["kv"] = kvmode;
  manifest["dtype"] = "float16";

  for (auto& [name, ids] : work) {
    qwen::prefix_cold(m);          // fresh KV and GDN state per prompt
    const int T = int(ids.size());
    const std::string path = out + "/" + name + ".f16";
    std::ofstream of(path, std::ios::binary);
    if (!of) { printf("cannot write %s\n", path.c_str()); return 1; }

    std::vector<int32_t> top1(T);
    for (int s = 0; s < T; s += chunk) {
      const int n = std::min(chunk, T - s);
      qwen::model_forward_all_logits(m, ids.data() + s, n, s, d_logits);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(host.data(), d_logits, size_t(n) * V * 2, cudaMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i) {
        const uint16_t* row = host.data() + size_t(i) * V;
        uint16_t* dst = fp16.data() + size_t(i) * V;
        int best = 0; float bv = -1e30f;
        for (int v = 0; v < V; ++v) {
          dst[v] = bf16_to_fp16(row[v]);
          uint32_t u = uint32_t(row[v]) << 16; float f; memcpy(&f, &u, 4);
          if (f > bv) { bv = f; best = v; }
        }
        top1[s + i] = best;
      }
      of.write(reinterpret_cast<const char*>(fp16.data()), size_t(n) * V * 2);
    }
    of.close();
    manifest["prompts"][name] = {
      {"tokens", T}, {"vocab", V}, {"ids", ids}, {"top1", top1},
      {"file", name + ".f16"}};
    printf("%-16s %6d tok -> %s\n", name.c_str(), T, path.c_str());
    fflush(stdout);
  }

  { std::ofstream f(out + "/manifest.json"); f << manifest.dump(); }
  printf("done\n");
  return 0;
}
