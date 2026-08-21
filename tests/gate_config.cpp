// GATE: the config parser must derive every shape the directive's Gate 0 table
// asserts, from the checkpoint itself, with no hardcoded fallbacks.
#include <cstdio>
#include <cstring>
#include "../src/config/model_shape.h"

static int fails = 0;
#define CHECK(expr, fmt, ...) do { if (!(expr)) { \
  printf("  FAIL %-34s " fmt "\n", #expr, ##__VA_ARGS__); ++fails; } } while (0)

int main(int argc, char** argv) {
  const char* p = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ/config.json";
  qwen::ModelShape s;
  try { s = qwen::ModelShape::from_file(p); }
  catch (const std::exception& e) { printf("  THREW: %s\n", e.what()); return 2; }

  printf("%s", s.describe().c_str());

  CHECK(s.hidden_size == 5120, "got %d", s.hidden_size);
  CHECK(s.vocab_size == 248320, "got %d", s.vocab_size);
  CHECK(s.num_hidden_layers == 64, "got %d", s.num_hidden_layers);
  CHECK(s.num_gdn_layers == 48, "got %d", s.num_gdn_layers);
  CHECK(s.num_attn_layers == 16, "got %d", s.num_attn_layers);
  CHECK(s.intermediate_size == 17408, "got %d", s.intermediate_size);
  CHECK(s.num_attention_heads == 24, "got %d", s.num_attention_heads);
  CHECK(s.num_key_value_heads == 4, "got %d", s.num_key_value_heads);
  CHECK(s.head_dim == 256, "got %d", s.head_dim);
  CHECK(s.rotary_dims == 64, "got %d", s.rotary_dims);
  CHECK(s.mrope_section[0] == 11 && s.mrope_section[1] == 11 && s.mrope_section[2] == 10,
        "got {%d,%d,%d}", s.mrope_section[0], s.mrope_section[1], s.mrope_section[2]);
  CHECK(s.mrope_interleaved, "expected true");
  CHECK(s.rope_theta == 10000000.0, "got %f", s.rope_theta);
  CHECK(s.linear_num_value_heads == 48, "got %d", s.linear_num_value_heads);
  CHECK(s.linear_num_key_heads == 16, "got %d", s.linear_num_key_heads);
  CHECK(s.linear_key_head_dim == 128, "got %d", s.linear_key_head_dim);
  CHECK(s.linear_value_head_dim == 128, "got %d", s.linear_value_head_dim);
  CHECK(s.linear_conv_kernel_dim == 4, "got %d", s.linear_conv_kernel_dim);
  CHECK(!s.tie_word_embeddings, "expected false: lm_head is separate");
  CHECK(s.attn_output_gate, "expected true");
  CHECK(s.mtp_num_hidden_layers == 1, "got %d", s.mtp_num_hidden_layers);
  CHECK(s.has_vision, "expected a vision_config to be present and skipped");
  CHECK(s.kv_elems_per_token() == 32768, "got %lld", (long long)s.kv_elems_per_token());
  CHECK(s.gdn_state_elems() == 37748736, "got %lld", (long long)s.gdn_state_elems());
  CHECK(s.gdn_qkv_dim() == 10240, "got %lld", (long long)s.gdn_qkv_dim());
  CHECK(s.gdn_z_dim() == 6144, "got %lld", (long long)s.gdn_z_dim());
  CHECK(s.quant_bits == 4, "got %d", s.quant_bits);
  CHECK(s.quant_group_size == 128 || s.quant_group_size == 32, "got %d", s.quant_group_size);
  CHECK(!s.quant_symmetric, "expected asymmetric (explicit zero points)");

  // the 16 x (L,L,L,A) pattern
  for (int i = 0; i < s.num_hidden_layers; ++i) {
    const bool want_attn = (i % 4 == 3);
    const bool is_attn = s.layer_types[i] == qwen::LayerKind::FullAttention;
    if (is_attn != want_attn) { printf("  FAIL layer %d kind\n", i); ++fails; break; }
  }

  printf("gate_config: %d failures -> %s\n", fails, fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
