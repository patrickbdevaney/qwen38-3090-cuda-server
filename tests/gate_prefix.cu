// GATE (directive G8): prefix cache reuse.
//
// Three claims, tested separately, because they do NOT have the same strength
// and collapsing them into one number would hide which is which.
//
//  A. SNAPSHOT FIDELITY -- exact. Storing the recurrent state and restoring it
//     must be a bit-for-bit round trip and must not perturb generation at all.
//     The state is snapshotted in fp32, its native precision, so there is no
//     licence to differ. A failure here is a real bug.
//
//  B. CHUNK-ALIGNED REUSE -- exact. If the snapshot sits on a prefill chunk
//     boundary, the chunks that remain have identical shapes to the ones a cold
//     run would use, so the arithmetic is identical. This is what vLLM calls
//     mamba-cache-mode "align", and it is exact for the same reason.
//
//  C. END-OF-TURN REUSE -- NOT exact, and this gate measures how far off it is
//     instead of pretending otherwise. A snapshot taken after generation was
//     produced by D single-token decode steps; a cold run reaches the same
//     position with a large chunked prefill. Those are different kernels with
//     different summation orders, so they cannot agree bit for bit. Every
//     hybrid-model prefix cache has this property. It is also the only mode
//     that reaches the 20x bar, because it is the only one that can reuse the
//     generated tokens -- so it is reported with a top-1 agreement and a KL,
//     exactly as gate_forward reports the INT4-vs-BF16 gap.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include "../src/model/model.h"
#include "../src/cache/prefix.h"
#include "../src/kernels/elementwise.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }
static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

static qwen::Model* g_m = nullptr;

static void prefill_range(const std::vector<int32_t>& ids, int from, int to) {
  int p = from;
  while (p < to) {
    const int n = std::min(g_m->max_batch, to - p);
    qwen::model_prefill(*g_m, ids.data() + p, n, p);
    p += n;
  }
}

static std::vector<uint16_t> grab_logits() {
  std::vector<uint16_t> v(g_m->shape.vocab_size);
  cudaMemcpy(v.data(), g_m->logits, v.size() * 2, cudaMemcpyDeviceToHost);
  return v;
}

// `processed` counts how many generated tokens were fed back. The last one never
// is: it is produced and returned but its KV and recurrent state do not exist,
// which is exactly the state a server is in when it stops.
static std::vector<int32_t> gen_greedy(int pos, int n, int* processed) {
  std::vector<int32_t> out;
  *processed = 0;
  int32_t* d_id = g_m->argmax_scratch + 512;
  for (int i = 0; i < n; ++i) {
    qwen::argmax(d_id, g_m->logits, g_m->shape.vocab_size, g_m->argmax_scratch);
    int32_t tok = 0;
    cudaMemcpy(&tok, d_id, 4, cudaMemcpyDeviceToHost);
    out.push_back(tok);
    if (g_m->shape.is_stop_token(tok)) break;
    if (i + 1 < n) { qwen::model_decode(*g_m, tok, pos + i); ++(*processed); }
  }
  return out;
}

struct Agree { size_t top1 = 0, n = 0; double kl = 0; size_t bits = 0; };

static Agree compare_logits(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  Agree r; r.n = 1;
  size_t bits = 0;
  for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++bits;
  r.bits = bits;
  // top-1 agreement and KL(cold || warm)
  size_t ia = 0, ib = 0;
  float va = -1e30f, vb = -1e30f, ma = -1e30f, mb = -1e30f;
  for (size_t i = 0; i < a.size(); ++i) {
    const float x = b2f(a[i]), y = b2f(b[i]);
    if (x > va) { va = x; ia = i; }
    if (y > vb) { vb = y; ib = i; }
    ma = std::max(ma, x); mb = std::max(mb, y);
  }
  r.top1 = (ia == ib) ? 1 : 0;
  double sa = 0, sb = 0;
  for (size_t i = 0; i < a.size(); ++i) { sa += std::exp(b2f(a[i]) - ma); sb += std::exp(b2f(b[i]) - mb); }
  double kl = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double p = std::exp(b2f(a[i]) - ma) / sa;
    const double q = std::exp(b2f(b[i]) - mb) / sb;
    if (p > 1e-12) kl += p * std::log(p / std::max(q, 1e-30));
  }
  r.kl = kl;
  return r;
}

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const int P1 = argc > 2 ? atoi(argv[2]) : 3000;
  const int NEW = argc > 3 ? atoi(argv[3]) : 24;
  const int GEN = argc > 4 ? atoi(argv[4]) : 24;
  const int CHUNK = argc > 5 ? atoi(argv[5]) : 512;

  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = 8192; o.max_batch = CHUNK; o.lm_head_bits = 8; o.verbose = false;
  qwen::model_load(m, md, o);
  g_m = &m;

  qwen::PrefixCache pc;
  qwen::prefix_alloc(pc, m, 4);
  printf("prefix cache: 4 slots x %.1f MiB pinned host = %.2f GiB host, 0 device\n"
         "prefill chunk: %d tokens\n",
         double(pc.state_elems + pc.conv_elems) * 4 / (1 << 20),
         double(pc.host_bytes) / (1 << 30), CHUNK);

  auto mk = [&](int n, uint32_t seed) {
    std::vector<int32_t> v(n);
    uint32_t s = seed;
    for (int i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      v[i] = int32_t(s % uint32_t(m.shape.vocab_size - 16)) + 8;
    }
    return v;
  };

  bool ok = true;
  const std::vector<int32_t> t1 = mk(P1, 12345);

  // ---- A. snapshot fidelity -------------------------------------------
  qwen::prefix_cold(m);
  prefill_range(t1, 0, P1);
  CK(cudaDeviceSynchronize());
  const std::vector<uint16_t> base_logits = grab_logits();
  qwen::prefix_set_kv(pc, t1);
  qwen::prefix_store(pc, m, t1, P1);

  int d0 = 0;
  const std::vector<int32_t> ref_cont = gen_greedy(P1, GEN, &d0);

  // restore the snapshot we just took and redo the same continuation
  { int slot = -1;
    const int r = qwen::prefix_lookup(pc, t1, &slot);
    (void)r;
    // look up on t1 itself would refuse (it must leave a token to run), so
    // restore the slot directly: this is the fidelity test, not the reuse test.
    qwen::prefix_restore(pc, m, 0);
  }
  int d1 = 0;
  const std::vector<int32_t> got_cont = gen_greedy(P1, GEN, &d1);
  size_t sameA = 0;
  while (sameA < ref_cont.size() && sameA < got_cont.size() && ref_cont[sameA] == got_cont[sameA]) ++sameA;
  const bool okA = (sameA == ref_cont.size());
  ok &= okA;
  printf("\nA  snapshot fidelity      : %zu / %zu tokens identical after store+restore   %s\n",
         sameA, ref_cont.size(), okA ? "OK" : "FAIL");

  // ---- B. chunk-aligned reuse -----------------------------------------
  // Snapshot on a prefill chunk boundary, then resume there. The remaining
  // chunks have identical shapes to a cold run's, so this must be exact.
  const int N2 = P1 + NEW;
  const std::vector<int32_t> t2 = [&]{
    std::vector<int32_t> v = t1;
    for (int32_t x : mk(NEW, 777)) v.push_back(x);
    return v;
  }();
  const int ALIGN = (P1 / CHUNK) * CHUNK;

  qwen::prefix_cold(m);
  prefill_range(t2, 0, ALIGN);
  CK(cudaDeviceSynchronize());
  qwen::prefix_store(pc, m, t2, ALIGN);
  qwen::prefix_set_kv(pc, std::vector<int32_t>(t2.begin(), t2.begin() + ALIGN));

  // cold reference for the whole of t2
  qwen::prefix_cold(m);
  double a = now_s();
  prefill_range(t2, 0, N2);
  CK(cudaDeviceSynchronize());
  const double coldB_s = now_s() - a;
  const std::vector<uint16_t> coldB = grab_logits();

  { int slot = -1;
    const int reuse = qwen::prefix_lookup(pc, t2, &slot);
    a = now_s();
    if (reuse > 0) qwen::prefix_restore(pc, m, slot); else qwen::prefix_cold(m);
    prefill_range(t2, reuse, N2);
    CK(cudaDeviceSynchronize());
    const double warmB_s = now_s() - a;
    const std::vector<uint16_t> warmB = grab_logits();
    const Agree g = compare_logits(coldB, warmB);
    const bool okB = (reuse == ALIGN) && (g.bits == 0);
    ok &= okB;
    printf("B  chunk-aligned reuse    : reused %d/%d, %zu logit bits differ, %.2fx faster   %s\n",
           reuse, N2, g.bits, coldB_s / warmB_s, okB ? "OK" : "FAIL");
  }

  // ---- C. end-of-turn reuse, the case the cache exists for -------------
  qwen::prefix_reset(pc);
  qwen::prefix_cold(m);
  a = now_s();
  prefill_range(t1, 0, P1);
  CK(cudaDeviceSynchronize());
  const double cold1_s = now_s() - a;
  int done1 = 0;
  const std::vector<int32_t> reply = gen_greedy(P1, GEN, &done1);
  std::vector<int32_t> full = t1;
  for (int32_t x : reply) full.push_back(x);
  const int after1 = P1 + done1;
  qwen::prefix_set_kv(pc, std::vector<int32_t>(full.begin(), full.begin() + after1));
  qwen::prefix_store(pc, m, full, after1);

  std::vector<int32_t> t3 = full;
  for (int32_t x : mk(NEW, 999)) t3.push_back(x);
  const int N3 = int(t3.size());

  int slot = -1;
  const int reuse = qwen::prefix_lookup(pc, t3, &slot);
  a = now_s();
  if (reuse > 0) qwen::prefix_restore(pc, m, slot); else qwen::prefix_cold(m);
  prefill_range(t3, reuse, N3);
  CK(cudaDeviceSynchronize());
  const double warm_s = now_s() - a;
  const std::vector<uint16_t> warmC = grab_logits();
  int dd = 0;
  const std::vector<int32_t> warm_out = gen_greedy(N3, GEN, &dd);

  qwen::prefix_cold(m);
  a = now_s();
  prefill_range(t3, 0, N3);
  CK(cudaDeviceSynchronize());
  const double coldC_s = now_s() - a;
  const std::vector<uint16_t> coldC = grab_logits();
  const std::vector<int32_t> cold_out = gen_greedy(N3, GEN, &dd);

  const Agree g = compare_logits(coldC, warmC);
  size_t sameC = 0;
  while (sameC < cold_out.size() && sameC < warm_out.size() && cold_out[sameC] == warm_out[sameC]) ++sameC;
  const double speedup = coldC_s / warm_s;

  printf("\nC  end-of-turn reuse (the case the cache exists for)\n");
  printf("     turn 1 prompt        : %d tokens, cold prefill %.3f s (%.0f tok/s)\n",
         P1, cold1_s, P1 / cold1_s);
  printf("     turn 2 prompt        : %d tokens, reused %d (%.1f%%)\n",
         N3, reuse, 100.0 * reuse / N3);
  printf("     cold prefill         : %.3f s\n", coldC_s);
  printf("     warm prefill         : %.3f s  (state restore %.1f ms)\n", warm_s, pc.restore_ms);
  printf("     SPEEDUP              : %.1fx   [G8 bar 20x]\n", speedup);
  printf("     top-1 agrees         : %s\n", g.top1 ? "yes" : "NO");
  printf("     KL(cold||warm)       : %.3e\n", g.kl);
  printf("     logit bits differing : %zu / %d\n", g.bits, m.shape.vocab_size);
  printf("     tokens identical     : %zu / %zu\n", sameC, cold_out.size());

  const bool okC = speedup >= 20.0 && g.top1 == 1 && g.kl < 5e-3;
  ok &= okC;
  printf("     %s\n", okC ? "OK (fast, and within the INT4 numerics envelope)"
                          : "FAIL");

  printf("\ngate_prefix RESULT: %s\n", ok ? "PASS" : "FAIL");
  qwen::prefix_free(pc);
  return ok ? 0 : 1;
}
