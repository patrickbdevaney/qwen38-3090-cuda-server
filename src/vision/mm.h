// Turning a prompt with images into something the language model can prefill.
//
// Two things have to happen between the chat template and model_prefill_mm:
//
//  1. The template emits ONE <|image_pad|> per image. It has to become one per
//     image TOKEN, which is grid_h*grid_w/(merge^2) of them. transformers does
//     this in the processor; we do it after tokenising.
//
//  2. mrope positions stop being sequential. A text token advances all three
//     axes by one; an image spans a (t, h, w) box, and afterwards the position
//     advances by max(h, w) -- NOT by the number of image tokens. Getting that
//     wrong puts every token after the first image at the wrong position, which
//     looks like the model simply ignoring the image.
#pragma once
#include <cstdint>
#include <vector>

namespace qwen {

// One image's span in the final token sequence, in MERGED grid units.
struct ImageSpan {
  int start = 0;      // index of the first image token
  int n_tokens = 0;   // t * h * w
  int t = 1, h = 0, w = 0;
};

// Replace each occurrence of `pad_id` with `counts[i]` copies of it, in order.
// Returns the expanded ids and fills `spans`. Throws std::runtime_error if the
// number of placeholders does not match `counts.size()`.
std::vector<int32_t> expand_image_pads(const std::vector<int32_t>& ids, int32_t pad_id,
                                       const std::vector<ImageSpan>& images,
                                       std::vector<ImageSpan>& spans);

// mrope positions for a sequence containing those spans.
//
//   text token : t = h = w = cur, cur += 1
//   image      : t = cur + (j / (H*W)), h = cur + ((j / W) % H), w = cur + (j % W)
//                then cur += max(H, W)
void mrope_positions(int n_tokens, const std::vector<ImageSpan>& spans,
                     std::vector<int32_t>& pt, std::vector<int32_t>& ph,
                     std::vector<int32_t>& pw);

}  // namespace qwen
