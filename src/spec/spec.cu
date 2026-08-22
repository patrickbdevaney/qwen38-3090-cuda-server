#include "spec.h"
#include "../kernels/gdn.cuh"
#include "../kernels/elementwise.cuh"
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>

namespace qwen {
namespace {
#define CKS(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } } while(0)

void* salloc(SpecState& s, size_t n) {
  void* p = nullptr; CKS(cudaMalloc(&p, n)); s.owned.push_back(p); return p;
}

// New conv state after committing `n` block positions: the last K-1 inputs of
// [old_state | block_input[0..n)]. A slice, not a recompute.
__global__ void k_conv_state_slice(float* __restrict__ state,
                                   const float* __restrict__ old_state,
                                   const __nv_bfloat16* __restrict__ pre,
                                   int conv_dim, int K, int n, int block) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(conv_dim) * (K - 1)) return;
  const int c = int(i / (K - 1)), j = int(i % (K - 1));
  const int t = n - (K - 1) + j;                 // position relative to block start
  if (t >= 0) state[i] = __bfloat162float(pre[size_t(t) * conv_dim + c]);
  else        state[i] = old_state[size_t(c) * (K - 1) + (K - 1 + t)];
}
}  // namespace

void spec_alloc(SpecState& s, const Model& m, int max_block) {
  CKS(cudaMalloc(&s.argmax_ids, size_t(max_block + 1) * 4));
  s.owned.push_back(s.argmax_ids);
  s.block = max_block;
  const auto& S = m.shape;
  const int NG = S.num_gdn_layers, CD = int(m.gdn.conv_dim()), NV = S.linear_num_value_heads;
  s.gdn_state_bak = static_cast<float*>(salloc(s, size_t(S.gdn_state_elems()) * 4));
  s.gdn_conv_bak  = static_cast<float*>(salloc(s, size_t(S.gdn_conv_state_elems()) * 4));
  s.pre_conv  = static_cast<__nv_bfloat16*>(salloc(s, size_t(NG) * max_block * CD * 2));
  s.post_conv = static_cast<__nv_bfloat16*>(salloc(s, size_t(NG) * max_block * CD * 2));
  s.g    = static_cast<float*>(salloc(s, size_t(NG) * max_block * NV * 4));
  s.beta = static_cast<float*>(salloc(s, size_t(NG) * max_block * NV * 4));
}

void spec_free(SpecState& s) {
  for (void* p : s.owned) cudaFree(p);
  s.owned.clear();
  s = SpecState{};
}

void spec_begin_block(SpecState& s, Model& m) {
  const auto& S = m.shape;
  CKS(cudaMemcpyAsync(s.gdn_state_bak, m.gdn_state, size_t(S.gdn_state_elems()) * 4,
                      cudaMemcpyDeviceToDevice, m.stream));
  CKS(cudaMemcpyAsync(s.gdn_conv_bak, m.gdn_conv, size_t(S.gdn_conv_state_elems()) * 4,
                      cudaMemcpyDeviceToDevice, m.stream));
  s.capturing = true;
  m.spec = &s;
}

void spec_rollback(SpecState& s, Model& m, int n) {
  const auto& S = m.shape;
  const int NG = S.num_gdn_layers, CD = int(m.gdn.conv_dim()), NV = S.linear_num_value_heads;
  // restore the seed
  CKS(cudaMemcpyAsync(m.gdn_state, s.gdn_state_bak, size_t(S.gdn_state_elems()) * 4,
                      cudaMemcpyDeviceToDevice, m.stream));
  if (n <= 0) {
    CKS(cudaMemcpyAsync(m.gdn_conv, s.gdn_conv_bak, size_t(S.gdn_conv_state_elems()) * 4,
                        cudaMemcpyDeviceToDevice, m.stream));
    return;
  }
  const size_t st_per = size_t(NV) * m.gdn.head_k * m.gdn.head_v;
  for (int gi = 0; gi < NG; ++gi) {
    // replay the recurrence over the accepted prefix only; the projections and
    // the conv are NOT redone, which is what makes this cheap
    gdn_scan(m.gdn_core,
             m.gdn_state + size_t(gi) * st_per,
             s.post_conv + size_t(gi) * s.block * CD,
             s.g + size_t(gi) * s.block * NV,
             s.beta + size_t(gi) * s.block * NV,
             m.gdn, n, m.stream);
    const size_t cs = size_t(CD) * (m.gdn.conv_k - 1);
    k_conv_state_slice<<<(cs + 255) / 256, 256, 0, m.stream>>>(
        m.gdn_conv + size_t(gi) * cs, s.gdn_conv_bak + size_t(gi) * cs,
        s.pre_conv + size_t(gi) * s.block * CD, CD, m.gdn.conv_k, n, s.block);
  }
}

int accept_greedy(const std::vector<int32_t>& drafted,
                  const std::vector<int32_t>& target_argmax,
                  int32_t& correction) {
  size_t a = 0;
  while (a < drafted.size() && a < target_argmax.size() && target_argmax[a] == drafted[a]) ++a;
  correction = (a < target_argmax.size()) ? target_argmax[a] : target_argmax.back();
  return int(a);
}

std::vector<int32_t> SuffixDrafter::propose(int k, int min_match, int max_match) const {
  const int n = int(seq_.size());
  if (n < min_match + 1) return {};
  for (int L = std::min(max_match, n - 1); L >= min_match; --L) {
    // most recent earlier occurrence wins. The search must stop BEFORE the
    // current suffix itself, or every query trivially matches itself and the
    // drafter silently never fires -- a bug laguna's suffix.h documents.
    for (int i = n - L - 1; i >= 0; --i) {
      bool ok = true;
      for (int j = 0; j < L; ++j)
        if (seq_[i + j] != seq_[n - L + j]) { ok = false; break; }
      if (!ok) continue;
      std::vector<int32_t> out;
      for (int j = 0; j < k && i + L + j < n; ++j) out.push_back(seq_[i + L + j]);
      if (!out.empty()) return out;
    }
  }
  return {};
}

void model_verify_block(Model& m, const int32_t* ids, int T, int position,
                        __nv_bfloat16* logits_out) {
  model_forward_all_logits(m, ids, T, position, logits_out);
}

namespace {
int32_t argmax_host(const std::vector<uint16_t>& v, size_t off, size_t n) {
  auto f = [](uint16_t h) { uint32_t u = uint32_t(h) << 16; float x; memcpy(&x, &u, 4); return x; };
  float best = -1e30f; int32_t bi = 0;
  for (size_t i = 0; i < n; ++i) {
    const float x = f(v[off + i]);
    if (x > best) { best = x; bi = int32_t(i); }   // strict >: lowest index wins ties
  }
  return bi;
}
}  // namespace

std::vector<int32_t> spec_generate(Model& m, SpecState& s,
                                   const std::vector<int32_t>& prompt,
                                   int max_new, int max_k, SpecStats& stats) {
  const int V = m.shape.vocab_size;
  std::vector<int32_t> out;
  SuffixDrafter drafter;
  for (int32_t t : prompt) drafter.push(t);

  __nv_bfloat16* lg = nullptr;
  CKS(cudaMalloc(&lg, size_t(max_k + 1) * V * 2));
  std::vector<uint16_t> host(size_t(max_k + 1) * V);

  // A verification block of T rows costs more than one decode step, so
  // speculation only pays when the committed tokens per round exceed that cost.
  // bench_block measures the ratio directly on this card and this checkpoint:
  // T=2 -> 1.92x, T=4 -> 2.04x, T=8 -> 2.11x, T=16 -> 2.23x, i.e. very nearly
  // 1.876 + 0.0221*T. A drafter that lands 2.0 tokens per round at k=7 is a NET
  // LOSS, which is exactly what the suffix drafter does on non-repetitive text,
  // so track the realised gain and fall back to plain decode when it stops
  // paying. Cooldown doubles on repeated failure and resets on success, so a
  // prompt that turns repetitive halfway through still gets speculation back.
  const double cost_a = getenv("QWEN_SPEC_COST_A") ? atof(getenv("QWEN_SPEC_COST_A")) : 1.876;
  const double cost_b = getenv("QWEN_SPEC_COST_B") ? atof(getenv("QWEN_SPEC_COST_B")) : 0.0221;
  double ewma = 0.0; bool have_ewma = false;
  int cooldown = 0, penalty = 8;

  int pos = int(prompt.size());
  // chunked prefill
  {
    int p = 0;
    while (p < int(prompt.size())) {
      const int n = std::min(m.max_batch, int(prompt.size()) - p);
      model_prefill(m, prompt.data() + p, n, p);
      p += n;
    }
  }
  stats.per_position.assign(max_k + 1, 0);

  while (int(out.size()) < max_new) {
    // the free token: already computed by the previous step
    std::vector<uint16_t> l1(V);
    CKS(cudaMemcpy(l1.data(), m.logits, size_t(V) * 2, cudaMemcpyDeviceToHost));
    const int32_t t0 = argmax_host(l1, 0, V);
    out.push_back(t0);
    drafter.push(t0);
    if (m.shape.is_stop_token(t0) || int(out.size()) >= max_new) break;

    std::vector<int32_t> drafted = drafter.propose(max_k);
    if (cooldown > 0) { --cooldown; ++stats.fallback_tokens; drafted.clear(); }
    if (drafted.empty()) {
      model_decode(m, t0, pos);
      ++pos;
      continue;
    }
    const int k = int(drafted.size());
    std::vector<int32_t> block;
    block.push_back(t0);
    for (int32_t d : drafted) block.push_back(d);

    spec_begin_block(s, m);
    model_verify_block(m, block.data(), k + 1, pos, lg);
    m.spec = nullptr; s.capturing = false;
    CKS(cudaMemcpy(host.data(), lg, size_t(k + 1) * V * 2, cudaMemcpyDeviceToHost));

    std::vector<int32_t> targ(k + 1);
    for (int i = 0; i <= k; ++i) targ[i] = argmax_host(host, size_t(i) * V, V);

    int32_t correction = 0;
    const int a = accept_greedy(drafted, targ, correction);

    ++stats.rounds;
    stats.drafted += k;
    stats.accepted += a;
    stats.committed += a + 1;                 // t0 plus the accepted drafts
    if (a < int(stats.per_position.size())) stats.per_position[a]++;

    const double gain = double(a + 1);
    ewma = have_ewma ? 0.8 * ewma + 0.2 * gain : gain;
    have_ewma = true;
    if (ewma < cost_a + cost_b * double(k + 1)) {
      cooldown = penalty;
      penalty = std::min(penalty * 2, 256);
    } else {
      penalty = 8;
    }

    for (int i = 0; i < a; ++i) { out.push_back(drafted[i]); drafter.push(drafted[i]); }

    // Roll the recurrent state back to the a+1 positions actually committed.
    // KV needs nothing: the next write simply starts at pos + a + 1 and
    // overwrites the rejected tail.
    if (a + 1 < k + 1) spec_rollback(s, m, a + 1);
    pos += a + 1;

    // the correction's distribution becomes the next round's free token
    CKS(cudaMemcpy(m.logits, lg + size_t(a) * V, size_t(V) * 2, cudaMemcpyDeviceToDevice));
    if (m.shape.is_stop_token(correction)) { out.push_back(correction); break; }
  }
  cudaFree(lg);
  return out;
}

// ---------------------------------------------------------------- DFlash2
void spec_push_taps(DraftModel& d, const __nv_bfloat16* taps, int n, int pos0,
                    int window_floor) {
  const int stride = d.sh.n_taps * d.sh.hidden;
  int start = 0;
  if (pos0 < window_floor) start = std::min(n, window_floor - pos0);
  for (int i = start; i < n; ) {
    const int chunk = std::min(d.max_rows, n - i);
    draft_push(d, taps + size_t(i) * stride, chunk, pos0 + i);
    i += chunk;
  }
}

// ---------------------------------------------------------------- profiling
//
// A speculative round measured 2.6x the cost of an autoregressive step, where
// the bytes it reads say it should cost about 1.2x. That ratio IS the break-even
// acceptance: below it speculation loses, which is why prose (2.6-2.7 accepted
// per round) came out slower than plain decode on the GGUF path. Making the
// round cheaper raises every workload at once and is worth more than routing
// around the bad case, so the round is instrumented rather than guessed at.
//
// Off unless QWEN_SPEC_PROFILE=1, because the stage timers need a device sync
// each and that would itself distort the thing being measured.
struct SpecProfile {
  double anchor = 0, draft = 0, verify = 0, pick = 0, commit = 0;
  uint64_t rounds = 0;
};
static SpecProfile g_prof;
static int g_prof_on = -1;

static bool prof_on() {
  if (g_prof_on < 0) { const char* e = getenv("QWEN_SPEC_PROFILE"); g_prof_on = e && atoi(e); }
  return g_prof_on != 0;
}
static double prof_now() {
  if (!prof_on()) return 0;
  cudaDeviceSynchronize();
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define PROF(field, t0) do { if (prof_on()) g_prof.field += prof_now() - (t0); } while (0)

void spec_profile_dump(const char* label) {
  if (!prof_on() || !g_prof.rounds) return;
  const double n = double(g_prof.rounds);
  const double tot = g_prof.anchor + g_prof.draft + g_prof.verify + g_prof.pick + g_prof.commit;
  printf("\nspec round breakdown (%s, %llu rounds, %.2f ms/round)\n", label,
         (unsigned long long)g_prof.rounds, tot / n);
  auto row = [&](const char* nm, double v) {
    printf("  %-28s %8.3f ms  %5.1f%%\n", nm, v / n, 100.0 * v / tot);
  };
  row("anchor logits D2H + argmax", g_prof.anchor);
  row("draft (drafter + head + pick)", g_prof.draft);
  row("verify forward (T=block)", g_prof.verify);
  row("block logits D2H + argmax", g_prof.pick);
  row("taps + rollback + commit", g_prof.commit);
  g_prof = SpecProfile{};
}

int spec_round(Model& m, SpecState& s, DraftModel& d, int& pos,
               __nv_bfloat16* lg, __nv_bfloat16* dlg,
               std::vector<int32_t>& nids,
               std::deque<int32_t>& out) {
  const int V = m.shape.vocab_size;
  const int BS = d.sh.block_size, NP = BS - 1;

  const double p_t0 = prof_now();
  // The anchor's argmax runs on the device: this used to allocate a 497 KiB
  // host vector, copy the whole vocabulary into it and scan it on the CPU, once
  // per round, sitting between two GPU forwards.
  int32_t t0 = 0;
  argmax_rows(s.argmax_ids, m.logits, 1, V, m.stream);
  CKS(cudaMemcpyAsync(&t0, s.argmax_ids, 4, cudaMemcpyDeviceToHost, m.stream));
  CKS(cudaStreamSynchronize(m.stream));
  out.push_back(t0);
  PROF(anchor, p_t0);
  if (m.shape.is_stop_token(t0)) return 1;
  if (prof_on()) ++g_prof.rounds;

  // ---- draft ----
  const double p_dr = prof_now();
  nids[0] = t0;
  CKS(cudaMemcpy(d.ids_buf, nids.data(), size_t(BS) * 4, cudaMemcpyHostToDevice));
  // The drafter has no embedding table: the noise comes from the TARGET's.
  // With --embed-host (and always under --gguf) that table is in host memory
  // and m.embed_bf/m.embed_q are null, so the gather has to run on the host.
  if (m.embed_on_host)
    embed_rows_host_into(m, nids.data(), BS, d.noise, 0);
  else if (m.embed_quantized)
    embed_int8(d.noise, m.embed_q, m.embed_scale, d.ids_buf, BS, m.shape.hidden_size, 0);
  else
    embed_bf16(d.noise, m.embed_bf, d.ids_buf, BS, m.shape.hidden_size, 0);
  const __nv_bfloat16* dh = draft_block(d, pos);
  model_apply_head(m, dlg, dh + size_t(1) * m.shape.hidden_size, NP);
  draft_select(d, dh + size_t(1) * m.shape.hidden_size, dlg, NP, t0);

  std::vector<int32_t> drafted(NP);
  CKS(cudaMemcpy(drafted.data(), d.path, size_t(NP) * 4, cudaMemcpyDeviceToHost));
  PROF(draft, p_dr);

  // ---- verify ----
  std::vector<int32_t> block;
  block.push_back(t0);
  for (int32_t x : drafted) block.push_back(x);

  const double p_vf = prof_now();
  spec_begin_block(s, m);
  model_verify_block(m, block.data(), BS, pos, lg);
  m.spec = nullptr; s.capturing = false;
  PROF(verify, p_vf);

  const double p_pk = prof_now();
  // Likewise for the verification block: block_size argmaxes over the vocabulary,
  // on the device, returning block_size integers rather than a 4 MiB tile.
  std::vector<int32_t> targ(BS);
  argmax_rows(s.argmax_ids, lg, BS, V, m.stream);
  CKS(cudaMemcpyAsync(targ.data(), s.argmax_ids, size_t(BS) * 4,
                      cudaMemcpyDeviceToHost, m.stream));
  CKS(cudaStreamSynchronize(m.stream));
  PROF(pick, p_pk);
  const double p_cm = prof_now();

  int32_t correction = 0;
  const int a = accept_greedy(drafted, targ, correction);
  for (int i = 0; i < a; ++i) out.push_back(drafted[i]);

  // The verification forward wrote taps for all BS rows; the committed prefix
  // is rows 0..a and becomes the drafter's next context.
  spec_push_taps(d, m.taps, a + 1, pos, 0);

  if (a + 1 < BS) spec_rollback(s, m, a + 1);
  pos += a + 1;
  CKS(cudaMemcpy(m.logits, lg + size_t(a) * V, size_t(V) * 2, cudaMemcpyDeviceToDevice));
  PROF(commit, p_cm);
  return a + 1;
}

std::vector<int32_t> spec_generate_dflash(Model& m, SpecState& s, DraftModel& d,
                                          const std::vector<int32_t>& prompt,
                                          int max_new, SpecStats& stats) {
  const int V = m.shape.vocab_size;
  const int BS = d.sh.block_size;      // 8: the anchor plus 7 mask slots
  const int NP = BS - 1;
  std::vector<int32_t> out;

  model_enable_taps(m, d.sh.target_layer_ids);
  draft_reset(d);

  __nv_bfloat16 *lg = nullptr, *dlg = nullptr;
  CKS(cudaMalloc(&lg, size_t(BS) * V * 2));
  CKS(cudaMalloc(&dlg, size_t(NP) * V * 2));

  const int P = int(prompt.size());
  // Everything before this is outside the drafter's sliding window by the time
  // the first block runs, so it never contributes.
  const int window_floor = d.sh.sliding_window > 0
                               ? std::max(0, P - d.sh.sliding_window + 1) : 0;
  {
    int p = 0;
    while (p < P) {
      const int n = std::min(m.max_batch, P - p);
      model_prefill(m, prompt.data() + p, n, p);
      spec_push_taps(d, m.taps, n, p, window_floor);
      p += n;
    }
  }

  int pos = P;
  stats.per_position.assign(BS, 0);

  // noise ids: the anchor then BS-1 mask tokens. Only slot 0 ever changes.
  std::vector<int32_t> nids(BS, d.sh.mask_token_id);

  while (int(out.size()) < max_new) {
    const double p_t0 = prof_now();
    int32_t t0 = 0;
    argmax_rows(s.argmax_ids, m.logits, 1, V, m.stream);
    CKS(cudaMemcpyAsync(&t0, s.argmax_ids, 4, cudaMemcpyDeviceToHost, m.stream));
    CKS(cudaStreamSynchronize(m.stream));
    out.push_back(t0);
    PROF(anchor, p_t0);
    if (m.shape.is_stop_token(t0) || int(out.size()) >= max_new) break;
    if (prof_on()) ++g_prof.rounds;
    const double p_dr = prof_now();

    // ---- draft ----
    nids[0] = t0;
    CKS(cudaMemcpy(d.ids_buf, nids.data(), size_t(BS) * 4, cudaMemcpyHostToDevice));
    // The drafter has no embedding table: the noise comes from the TARGET's.
    // With --embed-host (and always under --gguf) that table is in host memory
    // and m.embed_bf/m.embed_q are null, so the gather has to run on the host.
    if (m.embed_on_host)
      embed_rows_host_into(m, nids.data(), BS, d.noise, 0);
    else if (m.embed_quantized)
      embed_int8(d.noise, m.embed_q, m.embed_scale, d.ids_buf, BS, m.shape.hidden_size, 0);
    else
      embed_bf16(d.noise, m.embed_bf, d.ids_buf, BS, m.shape.hidden_size, 0);
    const __nv_bfloat16* dh = draft_block(d, pos);
    model_apply_head(m, dlg, dh + size_t(1) * m.shape.hidden_size, NP);
    draft_select(d, dh + size_t(1) * m.shape.hidden_size, dlg, NP, t0);

    std::vector<int32_t> drafted(NP);
    CKS(cudaMemcpy(drafted.data(), d.path, size_t(NP) * 4, cudaMemcpyDeviceToHost));

    // ---- verify ----
    std::vector<int32_t> block;
    block.push_back(t0);
    for (int32_t x : drafted) block.push_back(x);

    PROF(draft, p_dr);
    const double p_vf = prof_now();
    spec_begin_block(s, m);
    model_verify_block(m, block.data(), BS, pos, lg);
    m.spec = nullptr; s.capturing = false;
    PROF(verify, p_vf);

    const double p_pk = prof_now();
    std::vector<int32_t> targ(BS);
    argmax_rows(s.argmax_ids, lg, BS, V, m.stream);
    CKS(cudaMemcpyAsync(targ.data(), s.argmax_ids, size_t(BS) * 4,
                        cudaMemcpyDeviceToHost, m.stream));
    CKS(cudaStreamSynchronize(m.stream));
    PROF(pick, p_pk);
    const double p_cm2 = prof_now();

    int32_t correction = 0;
    const int a = accept_greedy(drafted, targ, correction);

    ++stats.rounds;
    stats.drafted += NP;
    stats.accepted += a;
    stats.committed += a + 1;
    if (a < int(stats.per_position.size())) stats.per_position[a]++;
    for (int i = 0; i < a; ++i) out.push_back(drafted[i]);

    // The verification forward wrote taps for all BS rows; the committed prefix
    // is rows 0..a and becomes the drafter's next context.
    spec_push_taps(d, m.taps, a + 1, pos, 0);

    if (a + 1 < BS) spec_rollback(s, m, a + 1);
    pos += a + 1;

    CKS(cudaMemcpy(m.logits, lg + size_t(a) * V, size_t(V) * 2, cudaMemcpyDeviceToDevice));
    if (m.shape.is_stop_token(correction)) { out.push_back(correction); break; }
  }
  cudaFree(lg);
  cudaFree(dlg);
  model_disable_taps(m);
  return out;
}

}  // namespace qwen
