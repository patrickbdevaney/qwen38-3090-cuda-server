#include "model_shape.h"

#include <cstdio>
#include <fstream>
#include "../../third_party/json.hpp"

using json = nlohmann::json;

namespace qwen {
namespace {

// Every accessor below throws on a missing key rather than defaulting.
const json& req(const json& j, const char* k, const char* where) {
  auto it = j.find(k);
  if (it == j.end())
    throw ConfigError(std::string("config: missing required key '") + k + "' in " + where);
  return *it;
}

template <typename T>
T req_as(const json& j, const char* k, const char* where) {
  const json& v = req(j, k, where);
  try { return v.get<T>(); }
  catch (const std::exception&) {
    throw ConfigError(std::string("config: key '") + k + "' in " + where +
                      " has unexpected type: " + v.dump().substr(0, 80));
  }
}

}  // namespace

ModelShape ModelShape::from_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw ConfigError("config: cannot open " + path);
  json root;
  try { f >> root; }
  catch (const std::exception& e) { throw ConfigError(std::string("config: bad JSON: ") + e.what()); }

  // The VL wrapper nests the language model under text_config.
  const json& tc = root.contains("text_config") ? root["text_config"] : root;

  ModelShape s;
  s.hidden_size             = req_as<int32_t>(tc, "hidden_size", "text_config");
  s.vocab_size              = req_as<int32_t>(tc, "vocab_size", "text_config");
  s.num_hidden_layers       = req_as<int32_t>(tc, "num_hidden_layers", "text_config");
  s.intermediate_size       = req_as<int32_t>(tc, "intermediate_size", "text_config");
  s.rms_norm_eps            = req_as<float>(tc, "rms_norm_eps", "text_config");
  s.max_position_embeddings = req_as<int32_t>(tc, "max_position_embeddings", "text_config");
  s.full_attention_interval = req_as<int32_t>(tc, "full_attention_interval", "text_config");

  // tie_word_embeddings can live at either level; the outer one wins if present
  // because that is what transformers honours.
  if (root.contains("tie_word_embeddings"))    s.tie_word_embeddings = root["tie_word_embeddings"].get<bool>();
  else                                         s.tie_word_embeddings = req_as<bool>(tc, "tie_word_embeddings", "text_config");

  // --- layer map -------------------------------------------------------
  const json& lt = req(tc, "layer_types", "text_config");
  if (!lt.is_array() || static_cast<int32_t>(lt.size()) != s.num_hidden_layers)
    throw ConfigError("config: layer_types length != num_hidden_layers");
  s.layer_types.reserve(lt.size());
  s.attn_layer_index.assign(lt.size(), -1);
  for (size_t i = 0; i < lt.size(); ++i) {
    const std::string t = lt[i].get<std::string>();
    if (t == "linear_attention") { s.layer_types.push_back(LayerKind::GatedDeltaNet); ++s.num_gdn_layers; }
    else if (t == "full_attention") {
      s.attn_layer_index[i] = s.num_attn_layers++;
      s.layer_types.push_back(LayerKind::FullAttention);
    } else {
      throw ConfigError("config: unknown layer_type '" + t + "'");
    }
  }

  // --- full attention ---------------------------------------------------
  s.num_attention_heads   = req_as<int32_t>(tc, "num_attention_heads", "text_config");
  s.num_key_value_heads   = req_as<int32_t>(tc, "num_key_value_heads", "text_config");
  s.head_dim              = req_as<int32_t>(tc, "head_dim", "text_config");
  s.partial_rotary_factor = req_as<float>(tc, "partial_rotary_factor", "text_config");
  s.attn_output_gate      = req_as<bool>(tc, "attn_output_gate", "text_config");
  s.output_gate_type      = req_as<std::string>(tc, "output_gate_type", "text_config");
  s.rotary_dims = static_cast<int32_t>(s.head_dim * s.partial_rotary_factor);

  // --- mrope ------------------------------------------------------------
  const json& rp = req(tc, "rope_parameters", "text_config");
  s.rope_theta       = req_as<double>(rp, "rope_theta", "rope_parameters");
  s.rope_type        = req_as<std::string>(rp, "rope_type", "rope_parameters");
  s.mrope_interleaved = rp.contains("mrope_interleaved") && rp["mrope_interleaved"].get<bool>();
  const json& ms = req(rp, "mrope_section", "rope_parameters");
  if (!ms.is_array() || ms.size() != 3)
    throw ConfigError("config: mrope_section must have 3 entries");
  int32_t msum = 0;
  for (size_t i = 0; i < 3; ++i) { s.mrope_section[i] = ms[i].get<int32_t>(); msum += s.mrope_section[i]; }
  // The sections tile the rotary HALF-dimension (one entry per cos/sin pair).
  // If this ever fails, the mrope kernel would silently rotate the wrong dims,
  // which is directive failure mode #1: fluent output that decays at long
  // context. Fail here instead.
  if (msum != s.rotary_dims / 2)
    throw ConfigError("config: mrope_section sums to " + std::to_string(msum) +
                      " but rotary_dims/2 is " + std::to_string(s.rotary_dims / 2));

  // --- gated delta net --------------------------------------------------
  s.linear_num_value_heads = req_as<int32_t>(tc, "linear_num_value_heads", "text_config");
  s.linear_num_key_heads   = req_as<int32_t>(tc, "linear_num_key_heads", "text_config");
  s.linear_key_head_dim    = req_as<int32_t>(tc, "linear_key_head_dim", "text_config");
  s.linear_value_head_dim  = req_as<int32_t>(tc, "linear_value_head_dim", "text_config");
  s.linear_conv_kernel_dim = req_as<int32_t>(tc, "linear_conv_kernel_dim", "text_config");
  s.mamba_ssm_dtype        = req_as<std::string>(tc, "mamba_ssm_dtype", "text_config");

  // --- extras -----------------------------------------------------------
  if (tc.contains("mtp_num_hidden_layers")) s.mtp_num_hidden_layers = tc["mtp_num_hidden_layers"].get<int32_t>();
  if (tc.contains("eos_token_id") && tc["eos_token_id"].is_number())
    s.eos_token_id = tc["eos_token_id"].get<int32_t>();
  s.has_vision = root.contains("vision_config");
  // generation_config.json is the authority on stopping.
  {
    const size_t slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);
    std::ifstream gf(dir + "/generation_config.json");
    if (gf) {
      json g; gf >> g;
      if (g.contains("eos_token_id")) {
        if (g["eos_token_id"].is_array())
          for (auto& v : g["eos_token_id"]) s.stop_token_ids.push_back(v.get<int32_t>());
        else if (g["eos_token_id"].is_number())
          s.stop_token_ids.push_back(g["eos_token_id"].get<int32_t>());
      }
    }
  }
  if (s.stop_token_ids.empty() && s.eos_token_id >= 0)
    s.stop_token_ids.push_back(s.eos_token_id);
  if (root.contains("image_token_id")) s.image_token_id = root["image_token_id"].get<int32_t>();
  if (root.contains("video_token_id")) s.video_token_id = root["video_token_id"].get<int32_t>();

  // --- quantization -----------------------------------------------------
  const json* q = nullptr;
  if (root.contains("quantization_config"))    q = &root["quantization_config"];
  else if (tc.contains("quantization_config")) q = &tc["quantization_config"];
  if (q) {
    const json& g0 = req(req(*q, "config_groups", "quantization_config"), "group_0", "config_groups");
    const json& w  = req(g0, "weights", "config_groups.group_0");
    s.quant_bits       = req_as<int32_t>(w, "num_bits", "weights");
    s.quant_group_size = req_as<int32_t>(w, "group_size", "weights");
    s.quant_symmetric  = w.contains("symmetric") && w["symmetric"].is_boolean() && w["symmetric"].get<bool>();
    s.quant_format     = req_as<std::string>(g0, "format", "config_groups.group_0");
    if (s.quant_bits != 4)
      throw ConfigError("config: only 4-bit weights are supported, got " + std::to_string(s.quant_bits));
    if (s.quant_format != "pack-quantized")
      throw ConfigError("config: unsupported quantization format '" + s.quant_format + "'");
    if (s.hidden_size % s.quant_group_size != 0)
      throw ConfigError("config: hidden_size not divisible by quant group_size");
  }

  // --- invariants the kernels rely on -----------------------------------
  if (s.head_dim * s.partial_rotary_factor != static_cast<float>(s.rotary_dims))
    throw ConfigError("config: head_dim * partial_rotary_factor is not an integer");
  if (s.num_attention_heads % s.num_key_value_heads != 0)
    throw ConfigError("config: num_attention_heads not divisible by num_key_value_heads");
  if (s.num_gdn_layers + s.num_attn_layers != s.num_hidden_layers)
    throw ConfigError("config: layer kinds do not sum to num_hidden_layers");
  if (s.linear_conv_kernel_dim < 2)
    throw ConfigError("config: linear_conv_kernel_dim must be >= 2");
  return s;
}

std::string ModelShape::describe() const {
  char b[4096];
  int n = snprintf(b, sizeof b,
    "ModelShape\n"
    "  hidden %d  vocab %d  layers %d (%d GDN + %d attn, interval %d)  ffn %d\n"
    "  attn : %d Q / %d KV heads, head_dim %d, rotary %d of %d (%d pass through)\n"
    "         gate=%s(%s)  qk-norm implied by weights\n"
    "  mrope: sections {%d,%d,%d} sum %d == rotary/2, interleaved=%s, theta %.0f, type %s\n"
    "  gdn  : %d V-heads / %d QK-heads, dk %d dv %d, conv width %d, ssm dtype %s\n"
    "         qkv proj out %lld, z proj out %lld\n"
    "  quant: %d-bit group %d %s, format %s\n"
    "  tie_word_embeddings %s   mtp layers %d   vision %s (deferred)\n"
    "  derived:\n"
    "    KV/token      %lld elems = %lld KiB bf16, %lld KiB fp8\n"
    "    GDN state     %lld elems = %.1f MiB fp32 (constant in context)\n"
    "    GDN conv state %lld elems = %.1f MiB fp32\n"
    "    ctx 131072 fp8 KV = %.2f GiB\n",
    hidden_size, vocab_size, num_hidden_layers, num_gdn_layers, num_attn_layers,
    full_attention_interval, intermediate_size,
    num_attention_heads, num_key_value_heads, head_dim, rotary_dims, head_dim,
    head_dim - rotary_dims,
    attn_output_gate ? "yes" : "no", output_gate_type.c_str(),
    mrope_section[0], mrope_section[1], mrope_section[2],
    mrope_section[0] + mrope_section[1] + mrope_section[2],
    mrope_interleaved ? "true" : "false", rope_theta, rope_type.c_str(),
    linear_num_value_heads, linear_num_key_heads, linear_key_head_dim,
    linear_value_head_dim, linear_conv_kernel_dim, mamba_ssm_dtype.c_str(),
    static_cast<long long>(gdn_qkv_dim()), static_cast<long long>(gdn_z_dim()),
    quant_bits, quant_group_size, quant_symmetric ? "symmetric" : "asymmetric",
    quant_format.c_str(),
    tie_word_embeddings ? "true" : "false", mtp_num_hidden_layers,
    has_vision ? "present" : "absent",
    static_cast<long long>(kv_elems_per_token()),
    static_cast<long long>(kv_bytes_per_token(2) / 1024),
    static_cast<long long>(kv_bytes_per_token(1) / 1024),
    static_cast<long long>(gdn_state_elems()), gdn_state_elems() * 4.0 / (1 << 20),
    static_cast<long long>(gdn_conv_state_elems()), gdn_conv_state_elems() * 4.0 / (1 << 20),
    kv_bytes_per_token(1) * 131072.0 / (1ull << 30));
  return std::string(b, n > 0 ? n : 0);
}

}  // namespace qwen
