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

// bf16 -> fp16 on the DEVICE. The reference is fp16 because 15K x 248K x 4
// bytes of fp32 is 15 GiB of disk for no extra information: bf16 logits carry 8
// mantissa bits and fp16 10, so the narrowing is lossless here.
//
// This has to be a kernel. Doing it on the host is 3.9e9 conversions for the
// full prompt set, which is over an hour single-threaded -- longer than
// generating the BF16 reference it is being compared against.
__global__ void bf16_to_fp16_k(__half* dst, const __nv_bfloat16* src, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(__bfloat162float(src[i]));
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("usage: %s <model_dir> <prompts.json> <out_dir> "
           "[lm_head_bits=8] [kv=0] [chunk=1024] [weights.gguf]\n", argv[0]);
    return 1;
  }
  const std::string md = argv[1], pj = argv[2], out = argv[3];
  const int lm_bits = argc > 4 ? atoi(argv[4]) : 8;
  // argv[7]: optional GGUF weights; md still supplies config + tokenizer.
  const std::string gguf = argc > 7 ? argv[7] : "";
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
  o.gguf = gguf;
  o.embed_host = true;
  o.verbose = true;
  if (kvmode == 1) { o.kv_k = qwen::KvFmt::FP8;  o.kv_v = qwen::KvFmt::INT4; }
  else if (kvmode == 2) { o.kv_k = qwen::KvFmt::INT4; o.kv_v = qwen::KvFmt::INT4; }
  qwen::model_load(m, md, o);

  const int V = m.shape.vocab_size;
  __nv_bfloat16* d_logits = nullptr;
  __half* d_fp16 = nullptr;
  CK(cudaMalloc(&d_logits, size_t(chunk) * V * 2));
  CK(cudaMalloc(&d_fp16, size_t(chunk) * V * 2));
  std::vector<uint16_t> fp16(size_t(chunk) * V);

  ojson manifest;
  manifest["model"] = md;
  manifest["lm_head_bits"] = lm_bits;
  manifest["kv"] = kvmode;
  if (!gguf.empty()) manifest["gguf"] = gguf;
  manifest["dtype"] = "float16";

  for (auto& [name, ids] : work) {
    qwen::prefix_cold(m);          // fresh KV and GDN state per prompt
    const int T = int(ids.size());
    const std::string path = out + "/" + name + ".f16";
    std::ofstream of(path, std::ios::binary);
    if (!of) { printf("cannot write %s\n", path.c_str()); return 1; }

    for (int s = 0; s < T; s += chunk) {
      const int n = std::min(chunk, T - s);
      qwen::model_forward_all_logits(m, ids.data() + s, n, s, d_logits);
      const size_t elems = size_t(n) * size_t(V);
      bf16_to_fp16_k<<<int((elems + 255) / 256), 256>>>(d_fp16, d_logits, elems);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(fp16.data(), d_fp16, elems * 2, cudaMemcpyDeviceToHost));
      of.write(reinterpret_cast<const char*>(fp16.data()), elems * 2);
    }
    of.close();
    // No top-1 here: the scorer takes the argmax off the logits themselves, so
    // recording it would only create a second thing that can disagree.
    manifest["prompts"][name] = {
      {"tokens", T}, {"vocab", V}, {"ids", ids}, {"file", name + ".f16"}};
    printf("%-16s %6d tok -> %s\n", name.c_str(), T, path.c_str());
    fflush(stdout);
  }

  { std::ofstream f(out + "/manifest.json"); f << manifest.dump(); }
  printf("done\n");
  return 0;
}
