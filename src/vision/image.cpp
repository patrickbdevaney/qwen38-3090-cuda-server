#include "image.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "../../third_party/stb_image.h"

namespace qwen {
namespace {

// Keys bicubic kernel with a = -0.75, the one PIL and torchvision use.
inline double keys(double x) {
  const double a = -0.75;
  x = std::fabs(x);
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
  return 0.0;
}

// One separable resize pass with antialiasing: when downscaling, the kernel is
// stretched by 1/scale so each output pixel averages every input pixel it
// covers. Without that, downscaling an image aliases badly and the tower sees
// something quite different from what the reference saw.
void resize_axis(const std::vector<float>& src, int src_w, int src_h,
                 std::vector<float>& dst, int dst_len, bool horizontal, int chan) {
  const int in_len = horizontal ? src_w : src_h;
  const int other = horizontal ? src_h : src_w;
  const double scale = double(in_len) / double(dst_len);
  const double filter_scale = std::max(1.0, scale);
  const double support = 2.0 * filter_scale;

  const int out_w = horizontal ? dst_len : src_w;
  const int out_h = horizontal ? src_h : dst_len;
  dst.assign(size_t(out_w) * out_h * chan, 0.f);

  std::vector<double> wts;
  for (int o = 0; o < dst_len; ++o) {
    const double center = (o + 0.5) * scale;
    int lo = int(std::floor(center - support + 0.5));
    int hi = int(std::ceil(center + support - 0.5));
    lo = std::max(lo, 0);
    hi = std::min(hi, in_len);
    if (hi <= lo) { lo = std::min(std::max(int(center), 0), in_len - 1); hi = lo + 1; }
    wts.assign(size_t(hi - lo), 0.0);
    double sum = 0.0;
    for (int i = lo; i < hi; ++i) {
      const double w = keys((i + 0.5 - center) / filter_scale);
      wts[i - lo] = w;
      sum += w;
    }
    if (sum == 0.0) { wts.assign(wts.size(), 1.0 / double(wts.size())); sum = 1.0; }
    for (double& w : wts) w /= sum;

    for (int p = 0; p < other; ++p) {
      for (int c = 0; c < chan; ++c) {
        double acc = 0.0;
        for (int i = lo; i < hi; ++i) {
          const size_t si = horizontal ? (size_t(p) * src_w + i) * chan + c
                                       : (size_t(i) * src_w + p) * chan + c;
          acc += wts[i - lo] * src[si];
        }
        const size_t di = horizontal ? (size_t(p) * out_w + o) * chan + c
                                     : (size_t(o) * out_w + p) * chan + c;
        dst[di] = float(acc);
      }
    }
  }
}

}  // namespace

void decode_image(const uint8_t* bytes, size_t n, std::vector<uint8_t>& rgb,
                  int& w, int& h) {
  int comp = 0;
  stbi_uc* p = stbi_load_from_memory(bytes, int(n), &w, &h, &comp, 3);
  if (!p) throw std::runtime_error(std::string("image decode failed: ") + stbi_failure_reason());
  rgb.assign(p, p + size_t(w) * h * 3);
  stbi_image_free(p);
}

void smart_resize(int height, int width, int factor, int64_t min_pixels,
                  int64_t max_pixels, int& out_h, int& out_w) {
  const double mx = double(std::max(height, width)), mn = double(std::min(height, width));
  if (mn <= 0) throw std::runtime_error("image has a zero dimension");
  if (mx / mn > 200.0) throw std::runtime_error("absolute aspect ratio must be smaller than 200");
  auto round_f = [&](double v) { return int(std::llround(v / factor)) * factor; };
  int hb = round_f(height), wb = round_f(width);
  if (int64_t(hb) * wb > max_pixels) {
    const double beta = std::sqrt(double(height) * double(width) / double(max_pixels));
    hb = std::max(factor, int(std::floor(height / beta / factor)) * factor);
    wb = std::max(factor, int(std::floor(width / beta / factor)) * factor);
  } else if (int64_t(hb) * wb < min_pixels) {
    const double beta = std::sqrt(double(min_pixels) / (double(height) * double(width)));
    hb = int(std::ceil(height * beta / factor)) * factor;
    wb = int(std::ceil(width * beta / factor)) * factor;
  }
  out_h = std::max(hb, factor);
  out_w = std::max(wb, factor);
}

PreprocessedImage preprocess_image(const uint8_t* bytes, size_t n,
                                   const ImageOptions& opt) {
  PreprocessedImage out;
  std::vector<uint8_t> rgb;
  int w = 0, h = 0;
  decode_image(bytes, n, rgb, w, h);
  out.src_w = w; out.src_h = h;

  const int factor = opt.patch_size * opt.spatial_merge;
  int rh = 0, rw = 0;
  smart_resize(h, w, factor, opt.min_pixels, opt.max_pixels, rh, rw);
  out.res_w = rw; out.res_h = rh;

  // uint8 -> float in [0,1], then a separable bicubic resize.
  std::vector<float> f(size_t(w) * h * 3);
  for (size_t i = 0; i < f.size(); ++i) f[i] = float(rgb[i]) / 255.0f;

  std::vector<float> tmp, resized;
  resize_axis(f, w, h, tmp, rw, /*horizontal=*/true, 3);
  resize_axis(tmp, rw, h, resized, rh, /*horizontal=*/false, 3);

  const int ps = opt.patch_size, m = opt.spatial_merge, Tp = opt.temporal_patch;
  const int gh = rh / ps, gw = rw / ps;
  out.grid = {1, gh, gw};
  out.patch_dim = 3 * Tp * ps * ps;
  const int n_patch = gh * gw;
  out.pixel_values.assign(size_t(n_patch) * out.patch_dim, 0.f);

  const int blocks_w = gw / m;
  for (int p = 0; p < n_patch; ++p) {
    // merge-block order -> (row, col)
    const int in_col = p % m;
    const int in_row = (p / m) % m;
    const int block_col = (p / (m * m)) % blocks_w;
    const int block_row = p / (m * m * blocks_w);
    const int row = block_row * m + in_row;
    const int col = block_col * m + in_col;
    float* dst = out.pixel_values.data() + size_t(p) * out.patch_dim;
    for (int c = 0; c < 3; ++c) {
      for (int py = 0; py < ps; ++py) {
        for (int px = 0; px < ps; ++px) {
          const size_t si = (size_t(row * ps + py) * rw + (col * ps + px)) * 3 + c;
          const float v = (resized[si] - opt.mean) / opt.std_;
          // [channel][temporal][py][px]; a still image is repeated across the
          // temporal axis, which is what the reference's expand() does.
          for (int t = 0; t < Tp; ++t)
            dst[((c * Tp + t) * ps + py) * ps + px] = v;
        }
      }
    }
  }
  return out;
}

bool base64_decode(const std::string& in, std::vector<uint8_t>& out) {
  size_t start = 0;
  const size_t comma = in.find(",");
  if (in.compare(0, 5, "data:") == 0 && comma != std::string::npos) start = comma + 1;

  static int8_t T[256];
  static bool init = false;
  if (!init) {
    memset(T, -1, sizeof T);
    const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) T[uint8_t(A[i])] = int8_t(i);
    T[uint8_t('-')] = 62; T[uint8_t('_')] = 63;   // url-safe
    init = true;
  }
  out.clear();
  out.reserve((in.size() - start) * 3 / 4 + 3);
  uint32_t acc = 0;
  int bits = 0;
  for (size_t i = start; i < in.size(); ++i) {
    const unsigned char ch = uint8_t(in[i]);
    if (ch == '=' ) break;
    if (ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t') continue;
    const int8_t v = T[ch];
    if (v < 0) return false;
    acc = (acc << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(uint8_t((acc >> bits) & 0xFF));
    }
  }
  return !out.empty();
}

}  // namespace qwen
