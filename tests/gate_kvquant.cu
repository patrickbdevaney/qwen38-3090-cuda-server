// GATE: quantised KV cache against the FP8 baseline.
//
// KV quantisation threatens ONE thing, and it is not perplexity. It degrades
// long-range retrieval: the keys decide which past position attention lands on,
// and noise in them makes the model miss a fact it can still fluently talk
// around. Averaged token statistics understate that badly, because the damage
// is concentrated on the handful of tokens where the model has to look
// something up.
//
// So this gate measures three things, weakest to strongest:
//
//   1. top-1 agreement and KL against the FP8 run  -- the usual numbers,
//   2. how far the greedy token stream stays identical,
//   3. NEEDLE RETRIEVAL: a fact buried early in a long context, asked about at
//      the end. That is the property a coding agent actually depends on, and it
//      is pass/fail rather than a distance.
//
// Each format is loaded, run, and freed in turn, because the cache layout is
// fixed at load and two of them will not fit at once.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../src/model/model.h"
#include "../src/tokenizer/bpe.h"
#include "../src/tokenizer/chat_template.h"
#include "../src/kernels/elementwise.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }

struct Run {
  std::vector<int32_t> out;
  std::vector<uint16_t> logits;
  std::string text;
  double prefill_s = 0;
};

static Run run_one(const std::string& md, qwen::KvFmt kf, qwen::KvFmt vf,
                   const std::vector<int32_t>& ids, int gen, int max_ctx,
                   const qwen::Tokenizer& tok, size_t* peak_mib) {
  qwen::Model m;
  qwen::LoadOptions o;
  o.max_ctx = max_ctx; o.max_batch = 2048; o.lm_head_bits = 8;
  o.embed_host = true; o.verbose = false;
  o.kv_k = kf; o.kv_v = vf;
  qwen::model_load(m, md, o);

  int p = 0;
  while (p < int(ids.size())) {
    const int n = std::min(m.max_batch, int(ids.size()) - p);
    qwen::model_prefill(m, ids.data() + p, n, p);
    p += n;
  }
  cudaDeviceSynchronize();

  Run r;
  r.logits.resize(m.shape.vocab_size);
  cudaMemcpy(r.logits.data(), m.logits, r.logits.size() * 2, cudaMemcpyDeviceToHost);

  int32_t* d_id = m.argmax_scratch + 512;
  for (int i = 0; i < gen; ++i) {
    qwen::argmax(d_id, m.logits, m.shape.vocab_size, m.argmax_scratch);
    int32_t t = 0;
    cudaMemcpy(&t, d_id, 4, cudaMemcpyDeviceToHost);
    r.out.push_back(t);
    if (m.shape.is_stop_token(t)) break;
    if (i + 1 < gen) qwen::model_decode(m, t, int(ids.size()) + i);
  }
  r.text = tok.decode(r.out, true);

  size_t fb = 0, tb = 0;
  cudaMemGetInfo(&fb, &tb);
  *peak_mib = (tb - fb) >> 20;
  return r;
}

static void compare(const char* name, const Run& base, const Run& got, const char* needle) {
  size_t same = 0;
  while (same < base.out.size() && same < got.out.size() && base.out[same] == got.out[same]) ++same;

  // top-1 and KL on the post-prefill distribution
  size_t ia = 0, ib = 0;
  float va = -1e30f, vb = -1e30f, ma = -1e30f, mb = -1e30f;
  for (size_t i = 0; i < base.logits.size(); ++i) {
    const float x = b2f(base.logits[i]), y = b2f(got.logits[i]);
    if (x > va) { va = x; ia = i; }
    if (y > vb) { vb = y; ib = i; }
    ma = std::max(ma, x); mb = std::max(mb, y);
  }
  double sa = 0, sb = 0;
  for (size_t i = 0; i < base.logits.size(); ++i) {
    sa += std::exp(b2f(base.logits[i]) - ma);
    sb += std::exp(b2f(got.logits[i]) - mb);
  }
  double kl = 0;
  for (size_t i = 0; i < base.logits.size(); ++i) {
    const double p = std::exp(b2f(base.logits[i]) - ma) / sa;
    const double q = std::exp(b2f(got.logits[i]) - mb) / sb;
    if (p > 1e-12) kl += p * std::log(p / std::max(q, 1e-30));
  }
  const bool found = got.text.find(needle) != std::string::npos;
  printf("  %-14s top-1 %s   KL %.3e   identical %zu/%zu   needle %s\n",
         name, ia == ib ? "agree" : "DIFFER", kl, same, base.out.size(),
         found ? "FOUND" : "*** MISSED ***");
}

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const int target_ctx = argc > 2 ? atoi(argv[2]) : 8192;
  const int gen = argc > 3 ? atoi(argv[3]) : 24;

  qwen::Tokenizer tok;
  tok.load(md + "/tokenizer.json");

  // A fact buried near the START of a long context, asked about at the END --
  // the longest possible distance, which is where quantised keys fail first.
  //
  // This MUST go through the chat template. A raw completion prompt makes an
  // instruct model continue the filler pattern instead of answering, and then
  // the baseline misses the needle for reasons that have nothing to do with the
  // KV cache.
  const char* NEEDLE = "84731";
  auto build = [&](int lines) {
    std::string doc = "Project notes.\n\nThe access code for the staging server is ";
    doc += NEEDLE;
    doc += ".\n\n";
    for (int i = 0; i < lines; ++i)
      doc += "Module " + std::to_string(i) +
             " handles subsystem " + std::to_string(i) +
             " and depends on module " + std::to_string(i ? i - 1 : 0) + ".\n";
    doc += "\nWhat is the access code for the staging server? Reply with only the number.";
    qwen::ojson msgs = qwen::ojson::array();
    msgs.push_back({{"role", "user"}, {"content", doc}});
    qwen::ChatOptions copt;
    copt.add_generation_prompt = true;
    copt.enable_thinking = false;    // answer directly; this is a retrieval test
    return tok.encode(qwen::render_chat(msgs, qwen::ojson(), copt));
  };
  // Size the FILLER to hit the target. Truncating the token list instead would
  // cut off the question and the generation prompt, and the model would just
  // continue the filler -- which is a broken test, not a KV result.
  int lo = 0, hi = 1;
  while (int(build(hi).size()) < target_ctx && hi < (1 << 20)) hi *= 2;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (int(build(mid).size()) <= target_ctx) lo = mid; else hi = mid;
  }
  std::vector<int32_t> ids = build(lo);
  printf("gate_kvquant: %zu prompt tokens, needle \"%s\" at the start\n",
         ids.size(), NEEDLE);

  const int max_ctx = int(ids.size()) + gen + 64;
  size_t p_fp8 = 0, p_v4 = 0, p_i4 = 0;

  Run fp8 = run_one(md, qwen::KvFmt::FP8, qwen::KvFmt::FP8, ids, gen, max_ctx, tok, &p_fp8);
  printf("\n  %-14s baseline                                identical  -    needle %s\n",
         "fp8/fp8", fp8.text.find(NEEDLE) != std::string::npos ? "FOUND" : "*** MISSED ***");
  if (fp8.text.find(NEEDLE) == std::string::npos) {
    printf("  the FP8 baseline itself does not retrieve the needle; the test is "
           "inconclusive rather than a KV-quantisation failure.\n");
    printf("  baseline said: %s\n", fp8.text.substr(0, 80).c_str());
    return 2;
  }

  Run v4 = run_one(md, qwen::KvFmt::FP8, qwen::KvFmt::INT4, ids, gen, max_ctx, tok, &p_v4);
  compare("K fp8 / V int4", fp8, v4, NEEDLE);
  Run i4 = run_one(md, qwen::KvFmt::INT4, qwen::KvFmt::INT4, ids, gen, max_ctx, tok, &p_i4);
  compare("int4 / int4", fp8, i4, NEEDLE);

  printf("\n  peak VRAM: fp8 %zu MiB, v4 %zu MiB, int4 %zu MiB (at %d ctx)\n",
         p_fp8, p_v4, p_i4, max_ctx);

  // The gate is retrieval, not distance: a format that still finds the needle
  // has not broken the thing KV quantisation actually threatens.
  const bool ok = v4.text.find(NEEDLE) != std::string::npos &&
                  i4.text.find(NEEDLE) != std::string::npos;
  printf("  RESULT: %s\n", ok ? "PASS (both formats retrieve)" :
                                "FAIL (a quantised format lost the needle)");
  return ok ? 0 : 1;
}
