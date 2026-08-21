// Image decode and preprocessing for the vision tower.
//
// Reproduces Qwen2VLImageProcessor: smart_resize to a multiple of
// patch_size * spatial_merge_size within [min_pixels, max_pixels], normalise
// with mean 0.5 / std 0.5 (i.e. to [-1, 1]), then patchify.
//
// THE PATCH ORDER IS NOT RASTER. The reference does
//     reshape(b, c, gh/m, m, ps, gw/m, m, ps).permute(0, 2, 5, 3, 6, 1, 4, 7)
// so patches come out in (block_row, block_col, in_row, in_col) order, and each
// patch's vector is laid out [channel][temporal][py][px]. Getting either wrong
// produces an image the tower will happily encode into nonsense.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace qwen {

struct ImageGrid { int t = 1, h = 0, w = 0; };   // in PATCHES, before the merge

struct PreprocessedImage {
  std::vector<float> pixel_values;   // [t*h*w, patch_dim]
  ImageGrid grid;
  int patch_dim = 0;
  int src_w = 0, src_h = 0;          // as decoded, before resize
  int res_w = 0, res_h = 0;          // after smart_resize
};

struct ImageOptions {
  int patch_size = 16;
  int temporal_patch = 2;
  int spatial_merge = 2;
  int64_t min_pixels = 65536;        // preprocessor_config.json shortest_edge
  int64_t max_pixels = 16777216;     // longest_edge
  float mean = 0.5f, std_ = 0.5f;
};

// Decode PNG/JPEG/BMP/GIF from memory. Throws std::runtime_error on failure.
// `rgb` comes back tightly packed, 3 bytes per pixel.
void decode_image(const uint8_t* bytes, size_t n, std::vector<uint8_t>& rgb,
                  int& w, int& h);

// The reference's smart_resize.
void smart_resize(int height, int width, int factor, int64_t min_pixels,
                  int64_t max_pixels, int& out_h, int& out_w);

// Decode + resize + normalise + patchify.
PreprocessedImage preprocess_image(const uint8_t* bytes, size_t n,
                                   const ImageOptions& opt);

// Decode a base64 payload, tolerating a `data:image/...;base64,` prefix and
// whitespace. Returns false if the input is not valid base64.
bool base64_decode(const std::string& in, std::vector<uint8_t>& out);

}  // namespace qwen
