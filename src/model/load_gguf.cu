// Load a GGUF checkpoint into the same Model the AWQ loader fills.
//
// The two formats disagree about more than block layout, and the disagreements
// are what this file is for:
//
//  * FUSING. AWQ ships q|k|v (and GDN's qkv|z, and gate|up) pre-concatenated
//    into one packed tensor. GGUF ships them separately and lets each carry its
//    own quant type -- blk.3 of UD-Q3_K_XL has attn_q as IQ4_NL, attn_k as Q4_K
//    and attn_v as Q5_K -- so they can neither be concatenated at load nor read
//    by one kernel. Linear holds them as parts and the kernels' ldy stride
//    writes each into its own column range. Some layers ship a pre-fused
//    attn_qkv instead, so both spellings are accepted.
//
//  * NAMES. GGUF uses llama.cpp's vocabulary: blk.N.attn_norm, ssm_out,
//    ffn_gate. The GDN layers borrow the attention names for their input
//    projections (attn_qkv is the GDN qkv, attn_gate is its z).
//
//  * SMALL TENSORS. Norms, the conv1d kernel, A_log and dt_bias arrive as F32
//    (and occasionally quantised, e.g. ssm_alpha/beta as Q8_0). The rest of the
//    model wants bf16, so those are dequantised once at load into bf16 and the
//    quantised original is dropped. They are a few MB in total; only the big
//    projections stay in block format.
//
//  * CONVENTIONS. llama.cpp's converter rewrites three things on the way in,
//    and every one of them is silent -- the tensor loads, runs, and produces
//    fluent nonsense. All three were confirmed element-for-element against the
//    true BF16 checkpoint (tools/cmp_gguf_conventions.py), not guessed:
//
//      norms   every *norm.weight EXCEPT linear_attn.norm gets +1, because HF's
//              RMSNorm for this model is (1 + w) * x and ggml's is w * x. Our
//              k_rmsnorm follows HF, so the +1 has to come back off. ssm_norm
//              is the exception at both ends and is used raw.
//      ssm_a   holds -exp(A_log), not A_log. gdn_gates wants A_log, so this is
//              log(-x). Exact: bf16(log(-x)) recovers the original bf16 A_log.
//      v heads reordered from grouped to tiled -- see GdnDims::v_tiled. Handled
//              by the scan's head pairing rather than by permuting here.
//
//  * NEXTN. GGUF files for this model carry an extra blk.<n_layer> holding the
//    MTP/next-token head (nextn.*). This server drafts with DFlash2, so those
//    tensors are skipped.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "model.h"
#include "../gguf/gguf.h"
#include "../gguf/dequant.cuh"
#include "../gguf/gemv.cuh"

namespace qwen {

#define CKL(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  fprintf(stderr,"gguf load %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); abort(); } } while(0)

// Upload a tensor in its GGUF block format, untouched. cudaMalloc gives 256-byte
// alignment, which satisfies the 16-byte contract the fused GEMV relies on.
GgufWeight gguf_upload_weight(Model& m, const GgufTensor& t) {
  GgufWeight w;
  void* d = nullptr;
  CKL(cudaMalloc(&d, t.nbytes));
  CKL(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
  m.owned.push_back(d);
  w.data = d;
  w.type = t.type;
  w.in_f = int(t.row_len());
  w.out_f = int(t.rows());
  w.bytes = t.nbytes;
  return w;
}

namespace {

// GGUF norms carry llama.cpp's +1; k_rmsnorm applies it itself.
__global__ void k_sub1(__nv_bfloat16* w, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) w[i] = __float2bfloat16(__bfloat162float(w[i]) - 1.0f);
}

// ssm_a holds -exp(A_log); gdn_gates wants A_log.
__global__ void k_log_neg(__nv_bfloat16* w, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) w[i] = __float2bfloat16(logf(-__bfloat162float(w[i])));
}

// Dequantise a small tensor to bf16 and keep only the bf16. Used for norms,
// biases, conv kernels and the two tiny per-head projections.
const __nv_bfloat16* upload_bf16(Model& m, const GgufTensor& t) {
  const int64_t n = int64_t(t.numel());
  __nv_bfloat16* dst = nullptr;
  CKL(cudaMalloc(&dst, size_t(n) * 2));
  m.owned.push_back(dst);
  if (t.type == GgmlType::BF16) {
    CKL(cudaMemcpy(dst, t.data, size_t(n) * 2, cudaMemcpyHostToDevice));
    return dst;
  }
  void* src = nullptr;
  CKL(cudaMalloc(&src, t.nbytes));
  CKL(cudaMemcpy(src, t.data, t.nbytes, cudaMemcpyHostToDevice));
  if (!gguf_dequant_supported(t.type)) {
    fprintf(stderr, "gguf load: no dequantiser for %s (tensor %s)\n",
            ggml_type_name(t.type), t.name.c_str());
    abort();
  }
  gguf_dequant_bf16(dst, src, t.type, n);
  CKL(cudaDeviceSynchronize());
  CKL(cudaFree(src));
  return dst;
}

// A norm as ggml stores it: strip the +1 the converter added.
const __nv_bfloat16* upload_norm(Model& m, const GgufTensor& t) {
  const __nv_bfloat16* w = upload_bf16(m, t);
  const int n = int(t.numel());
  k_sub1<<<(n + 255) / 256, 256>>>(const_cast<__nv_bfloat16*>(w), n);
  CKL(cudaDeviceSynchronize());
  return w;
}

// ssm_a -> A_log.
const __nv_bfloat16* upload_a_log(Model& m, const GgufTensor& t) {
  const __nv_bfloat16* w = upload_bf16(m, t);
  const int n = int(t.numel());
  k_log_neg<<<(n + 255) / 256, 256>>>(const_cast<__nv_bfloat16*>(w), n);
  CKL(cudaDeviceSynchronize());
  return w;
}

// Fill a Linear from the first spelling that exists. `names` is tried in order;
// a name may itself be a '+'-joined list of parts to run as separate pieces.
bool fill_linear(Model& m, const GgufFile& f, Linear& L,
                 const std::vector<std::string>& names, const char* what) {
  for (const std::string& spec : names) {
    std::vector<std::string> parts;
    size_t s = 0;
    while (s <= spec.size()) {
      const size_t e = spec.find('+', s);
      parts.push_back(spec.substr(s, e == std::string::npos ? e : e - s));
      if (e == std::string::npos) break;
      s = e + 1;
    }
    bool all = true;
    for (const auto& n : parts) if (!f.has(n)) { all = false; break; }
    if (!all) continue;
    if (parts.size() > 3) { fprintf(stderr, "gguf load: %s has %zu parts\n", what, parts.size()); abort(); }

    L.gguf = true;
    L.n_part = int(parts.size());
    L.out_f = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
      L.part[i] = gguf_upload_weight(m, f.get(parts[i]));
      if (!gguf_gemv_supported(L.part[i].type)) {
        fprintf(stderr, "gguf load: fused GEMV does not implement %s (tensor %s)\n",
                ggml_type_name(L.part[i].type), parts[i].c_str());
        abort();
      }
      if (i == 0) L.in_f = L.part[i].in_f;
      else if (L.part[i].in_f != L.in_f) {
        fprintf(stderr, "gguf load: %s part %zu in_f %d != %d\n", what, i,
                L.part[i].in_f, L.in_f);
        abort();
      }
      L.out_f += L.part[i].out_f;
    }
    return true;
  }
  return false;
}

void need(bool ok, const char* what, int layer) {
  if (!ok) { fprintf(stderr, "gguf load: missing %s for layer %d\n", what, layer); abort(); }
}

}  // namespace

// Populate the per-layer weights of an already-shaped Model from a GGUF file.
// The caller has read config.json for the shape; this only fills weights, so a
// GGUF and an AWQ checkpoint of the same model produce the same Model layout.
size_t model_load_gguf_weights(Model& m, GgufFile& f, bool verbose) {
  const ModelShape& S = m.shape;
  size_t body_bytes = 0;
  m.gdn.v_tiled = true;   // see GdnDims::v_tiled

  m.layers.resize(S.num_hidden_layers);
  for (int i = 0; i < S.num_hidden_layers; ++i) {
    LayerWeights& L = m.layers[i];
    const std::string b = "blk." + std::to_string(i) + ".";
    L.is_attn = S.layer_types[i] == LayerKind::FullAttention;
    L.attn_slot = S.attn_layer_index[i];

    L.input_ln = upload_norm(m, f.get(b + "attn_norm.weight"));
    L.post_ln  = upload_norm(m, f.get(b + "post_attention_norm.weight"));

    need(fill_linear(m, f, L.mlp_gate_up,
                     {b + "ffn_gate.weight+" + b + "ffn_up.weight"}, "ffn gate/up"), "ffn_gate/up", i);
    need(fill_linear(m, f, L.mlp_down, {b + "ffn_down.weight"}, "ffn_down"), "ffn_down", i);

    if (L.is_attn) {
      need(fill_linear(m, f, L.attn_qkv,
                       {b + "attn_qkv.weight",
                        b + "attn_q.weight+" + b + "attn_k.weight+" + b + "attn_v.weight"},
                       "attn qkv"), "attn qkv", i);
      need(fill_linear(m, f, L.attn_o,
                       {b + "attn_output.weight"}, "attn_output"), "attn_output", i);
      L.q_norm = upload_norm(m, f.get(b + "attn_q_norm.weight"));
      L.k_norm = upload_norm(m, f.get(b + "attn_k_norm.weight"));
    } else {
      // GDN borrows the attention names: attn_qkv is the qkv projection and
      // attn_gate is z. AWQ fuses them; here they stay two parts.
      need(fill_linear(m, f, L.gdn_in_qkvz,
                       {b + "attn_qkv.weight+" + b + "attn_gate.weight"},
                       "gdn qkv/z"), "gdn qkv+z", i);
      need(fill_linear(m, f, L.gdn_out, {b + "ssm_out.weight"}, "ssm_out"), "ssm_out", i);
      L.gdn_a       = upload_bf16(m, f.get(b + "ssm_alpha.weight"));
      L.gdn_b       = upload_bf16(m, f.get(b + "ssm_beta.weight"));
      L.gdn_conv_w  = upload_bf16(m, f.get(b + "ssm_conv1d.weight"));
      L.gdn_A_log   = upload_a_log(m, f.get(b + "ssm_a"));
      L.gdn_dt_bias = upload_bf16(m, f.get(b + "ssm_dt.bias"));
      L.gdn_norm    = upload_bf16(m, f.get(b + "ssm_norm.weight"));  // raw at both ends
    }
    body_bytes += L.mlp_gate_up.total_bytes() + L.mlp_down.total_bytes();
    body_bytes += L.is_attn ? (L.attn_qkv.total_bytes() + L.attn_o.total_bytes())
                            : (L.gdn_in_qkvz.total_bytes() + L.gdn_out.total_bytes());
    if (verbose && (i + 1) % 16 == 0) {
      printf("  loaded %d/%d layers\n", i + 1, S.num_hidden_layers);
      fflush(stdout);
    }
  }
  m.final_norm = upload_norm(m, f.get("output_norm.weight"));

  // Prefill scratch: one dequantised copy of the largest single part. Sized from
  // what was actually loaded rather than from a constant, so a differently
  // quantised file of the same model needs no code change.
  size_t max_elems = 0;
  for (const LayerWeights& L : m.layers) {
    const Linear* all[4] = {&L.mlp_gate_up, &L.mlp_down,
                            L.is_attn ? &L.attn_qkv : &L.gdn_in_qkvz,
                            L.is_attn ? &L.attn_o   : &L.gdn_out};
    for (const Linear* q : all)
      for (int i = 0; i < q->n_part; ++i)
        max_elems = std::max(max_elems, size_t(q->part[i].out_f) * size_t(q->part[i].in_f));
  }
  CKL(cudaMalloc(&m.gguf_deq, max_elems * 2));
  m.owned.push_back(m.gguf_deq);
  m.gguf_deq_elems = max_elems;
  if (verbose)
    printf("  gguf prefill scratch: %.3f GiB (largest part %zu elements)\n",
           double(max_elems * 2) / (1 << 30), max_elems);
  return body_bytes;
}

}  // namespace qwen
