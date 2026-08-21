// Qwen3.5 vision tower: a 27-block SigLIP-style ViT that turns image patches
// into language-model tokens.
//
// It is far simpler than the language model -- everything is a plain GEMM,
// LayerNorm or GELU -- but four details are not guessable and each one silently
// produces a plausible wrong answer:
//
//  1. `deepstack_visual_indexes` is EMPTY in this checkpoint. Qwen-VL normally
//     injects visual features at several language-model layers; here it does
//     not, so the tower's output is spliced in at the embedding layer and
//     nowhere else. That removes the hardest part of the architecture.
//  2. Patches arrive in SPATIAL-MERGE-BLOCK order (block_row, block_col,
//     in_row, in_col), not raster order. That is what makes the 2x2 merge a
//     pure reshape at the end, and it is also what the position-embedding
//     resample has to match.
//  3. The learned pos_embed is a fixed 48x48 grid BILINEARLY RESAMPLED to each
//     image's patch grid. It is not a lookup.
//  4. The block MLP uses gelu_pytorch_tanh; the MERGER uses nn.GELU(), the
//     exact erf one. Two different functions in the same tower.
//
// On top of the learned pos_embed there is a 2D rotary embedding over (row,
// col), applied inside every block.
#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <vector>

namespace qwen {

struct VisionShape {
  int depth = 0, hidden = 0, intermediate = 0, num_heads = 0, head_dim = 0;
  int in_channels = 0, patch_size = 0, temporal_patch = 0, spatial_merge = 0;
  int num_pos_embed = 0, grid_per_side = 0, out_hidden = 0;
  float eps = 1e-6f, rope_theta = 10000.0f;

  int patch_dim() const { return in_channels * temporal_patch * patch_size * patch_size; }
  int merged_dim() const { return hidden * spatial_merge * spatial_merge; }
  // 2D rope: head_dim/2 frequencies, split between the two axes.
  int rope_dim() const { return head_dim / 2; }
};

struct VisionBlock {
  __nv_bfloat16 *norm1_w = nullptr, *norm1_b = nullptr;
  __nv_bfloat16 *norm2_w = nullptr, *norm2_b = nullptr;
  __nv_bfloat16 *qkv_w = nullptr, *qkv_b = nullptr;     // [3*hidden, hidden]
  __nv_bfloat16 *proj_w = nullptr, *proj_b = nullptr;   // [hidden, hidden]
  __nv_bfloat16 *fc1_w = nullptr, *fc1_b = nullptr;     // [inter, hidden]
  __nv_bfloat16 *fc2_w = nullptr, *fc2_b = nullptr;     // [hidden, inter]
};

struct VisionTower {
  VisionShape sh;
  std::vector<VisionBlock> blocks;

  __nv_bfloat16 *patch_w = nullptr, *patch_b = nullptr;   // [hidden, patch_dim]
  __nv_bfloat16 *pos_embed = nullptr;                     // [num_pos_embed, hidden]
  __nv_bfloat16 *merger_norm_w = nullptr, *merger_norm_b = nullptr;
  __nv_bfloat16 *merger_fc1_w = nullptr, *merger_fc1_b = nullptr;
  __nv_bfloat16 *merger_fc2_w = nullptr, *merger_fc2_b = nullptr;

  // scratch, sized for max_patches
  int max_patches = 0, qtile = 512;
  __nv_bfloat16 *x = nullptr, *xn = nullptr, *qkv = nullptr, *attn = nullptr;
  __nv_bfloat16 *mlp = nullptr, *probs = nullptr, *merged = nullptr, *out = nullptr;
  float *scores = nullptr, *cos_tab = nullptr, *sin_tab = nullptr;
  int32_t *pos_ids = nullptr;

  // Debug tap for gate_vision: x right after patch_embed + the resampled
  // pos_embed. v.x itself is the running residual stream and by the end of the
  // forward holds block 27's output, not this.
  bool debug = false;
  __nv_bfloat16* dbg_post_pos = nullptr;

  cublasHandle_t cublas = nullptr;
  cudaStream_t stream = 0;
  std::vector<void*> owned;
  size_t bytes = 0;
};

struct VisionLoadOptions {
  int  max_patches = 4096;   // 1024 image tokens after the 2x2 merge
  bool debug = false;
  bool verbose = true;
};

void vision_load(VisionTower& v, const std::string& dir, const VisionLoadOptions& opt);
void vision_free(VisionTower& v);

// One image. `pixel_values` is [t*h*w, patch_dim] on device, already in
// spatial-merge-block order, normalised to [-1, 1]. Writes
// [t*h*w/(merge*merge), out_hidden] image tokens and returns the buffer.
const __nv_bfloat16* vision_forward(VisionTower& v, const __nv_bfloat16* pixel_values,
                                    int grid_t, int grid_h, int grid_w);

// Number of language-model tokens one image of this grid becomes.
inline int vision_num_tokens(const VisionShape& s, int t, int h, int w) {
  return t * h * w / (s.spatial_merge * s.spatial_merge);
}

}  // namespace qwen
