// config.json -> ModelShape. Every shape the runtime uses is derived HERE and
// nowhere else; no kernel, loader or server file may hardcode a dimension.
//
// Policy, inherited from laguna-s1-cuda-server: there are NO DEFAULTS. A
// missing or unexpected key is a hard error at startup. A server that quietly
// substitutes a plausible constant for a shape it could not read will produce
// fluent garbage, which is the worst failure mode available.
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen {

struct ConfigError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class LayerKind : uint8_t { GatedDeltaNet, FullAttention };

struct ModelShape {
  // --- core -----------------------------------------------------------
  int32_t hidden_size            = 0;   // 5120
  int32_t vocab_size             = 0;   // 248320
  int32_t num_hidden_layers      = 0;   // 64
  int32_t intermediate_size      = 0;   // 17408
  float   rms_norm_eps           = 0.f;
  bool    tie_word_embeddings    = false;
  int32_t max_position_embeddings = 0;  // 262144

  // --- layer map ------------------------------------------------------
  std::vector<LayerKind> layer_types;   // 64 entries, 16 x (L,L,L,A)
  int32_t num_gdn_layers  = 0;          // 48
  int32_t num_attn_layers = 0;          // 16
  int32_t full_attention_interval = 0;  // 4
  std::vector<int32_t> attn_layer_index;  // model layer idx -> KV slot, -1 if GDN

  // --- full attention -------------------------------------------------
  int32_t num_attention_heads = 0;      // 24
  int32_t num_key_value_heads = 0;      // 4
  int32_t head_dim            = 0;      // 256
  float   partial_rotary_factor = 0.f;  // 0.25
  int32_t rotary_dims         = 0;      // 64 -> the other 192 pass through
  bool    attn_output_gate    = false;
  std::string output_gate_type;         // "swish"

  // --- mrope ----------------------------------------------------------
  double rope_theta = 0.0;              // 1e7
  std::array<int32_t, 3> mrope_section{0, 0, 0};   // {11,11,10}, sums to 32
  bool   mrope_interleaved = false;
  std::string rope_type;                // "default"; YaRN is configured, not read

  // --- gated delta net ------------------------------------------------
  int32_t linear_num_value_heads = 0;   // 48
  int32_t linear_num_key_heads   = 0;   // 16
  int32_t linear_key_head_dim    = 0;   // 128
  int32_t linear_value_head_dim  = 0;   // 128
  int32_t linear_conv_kernel_dim = 0;   // 4 -> conv state is 3 timesteps
  std::string mamba_ssm_dtype;          // "float32"

  // --- extras ---------------------------------------------------------
  int32_t mtp_num_hidden_layers = 0;    // 1, built-in MTP head
  bool    has_vision = false;           // deferred; loader skips model.visual.*
  int32_t eos_token_id = -1;
  int32_t image_token_id = -1;
  int32_t video_token_id = -1;

  // --- quantization ---------------------------------------------------
  int32_t quant_bits       = 0;         // 4
  int32_t quant_group_size = 0;         // 128 (philbert440) or 32 (cyankiwi)
  bool    quant_symmetric  = false;     // false -> explicit zero points
  std::string quant_format;             // "pack-quantized"

  // --- derived --------------------------------------------------------
  int64_t gdn_qkv_dim()  const {        // conv1d width, = 10240
    return int64_t(linear_num_key_heads) * linear_key_head_dim * 2 +
           int64_t(linear_num_value_heads) * linear_value_head_dim;
  }
  int64_t gdn_z_dim()    const {        // = 6144
    return int64_t(linear_num_value_heads) * linear_value_head_dim;
  }
  int64_t gdn_state_elems() const {     // 48*48*128*128 = 37,748,736
    return int64_t(num_gdn_layers) * linear_num_value_heads *
           linear_key_head_dim * linear_value_head_dim;
  }
  int64_t gdn_conv_state_elems() const {
    return int64_t(num_gdn_layers) * gdn_qkv_dim() * (linear_conv_kernel_dim - 1);
  }
  int64_t kv_elems_per_token() const {  // 16*4*256*2 = 32,768
    return int64_t(num_attn_layers) * num_key_value_heads * head_dim * 2;
  }
  int64_t kv_bytes_per_token(int elem_bytes) const {
    return kv_elems_per_token() * elem_bytes;
  }
  int64_t q_proj_out() const { return int64_t(num_attention_heads) * head_dim; }
  int64_t kv_proj_out() const { return int64_t(num_key_value_heads) * head_dim; }

  static ModelShape from_file(const std::string& config_json_path);
  std::string describe() const;
};

}  // namespace qwen
