#include "mm.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace qwen {

std::vector<int32_t> expand_image_pads(const std::vector<int32_t>& ids, int32_t pad_id,
                                       const std::vector<ImageSpan>& images,
                                       std::vector<ImageSpan>& spans) {
  size_t n_pad = 0;
  for (int32_t t : ids) if (t == pad_id) ++n_pad;
  if (n_pad != images.size())
    throw std::runtime_error("image placeholder count " + std::to_string(n_pad) +
                             " does not match " + std::to_string(images.size()) +
                             " images");
  std::vector<int32_t> out;
  out.reserve(ids.size());
  spans.clear();
  size_t k = 0;
  for (int32_t t : ids) {
    if (t != pad_id) { out.push_back(t); continue; }
    ImageSpan s = images[k++];
    s.start = int(out.size());
    s.n_tokens = s.t * s.h * s.w;
    for (int i = 0; i < s.n_tokens; ++i) out.push_back(pad_id);
    spans.push_back(s);
  }
  return out;
}

void mrope_positions(int n_tokens, const std::vector<ImageSpan>& spans,
                     std::vector<int32_t>& pt, std::vector<int32_t>& ph,
                     std::vector<int32_t>& pw) {
  pt.assign(n_tokens, 0); ph.assign(n_tokens, 0); pw.assign(n_tokens, 0);
  int cur = 0, i = 0;
  size_t si = 0;
  while (i < n_tokens) {
    if (si < spans.size() && spans[si].start == i) {
      const ImageSpan& s = spans[si];
      const int HW = s.h * s.w;
      for (int j = 0; j < s.n_tokens && i + j < n_tokens; ++j) {
        pt[i + j] = cur + (j / HW);
        ph[i + j] = cur + ((j / s.w) % s.h);
        pw[i + j] = cur + (j % s.w);
      }
      // The position advances by the LARGER spatial extent, not by the token
      // count: an image occupies a box in mrope space, not a run.
      cur += std::max(s.h, s.w);
      i += s.n_tokens;
      ++si;
    } else {
      pt[i] = ph[i] = pw[i] = cur;
      ++cur; ++i;
    }
  }
}

}  // namespace qwen
