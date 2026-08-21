// GGUF container parsing.
//
// Just the file format: header, metadata key/values, the tensor directory, and
// an mmap of the data section. Nothing here knows what a Qwen layer is, and
// nothing here dequantises.
//
// Layout (v2/v3):
//   magic "GGUF" | u32 version | u64 tensor_count | u64 kv_count
//   kv_count  x { string key, u32 type, value }
//   tensor_count x { string name, u32 n_dims, u64 dims[n_dims], u32 ggml_type, u64 offset }
//   padding to general.alignment (default 32)
//   data
//
// Tensor dims are in ggml order: ne[0] is the FASTEST axis. A weight that HF
// stores as [out, in] appears here as ne = {in, out}, so the memory is still
// row-major [out][in] and no transpose is involved.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace qwen {

// Subset of ggml_type we care about; the numeric values are the file format's.
enum class GgmlType : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
  Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
  IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20,
  IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
  I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
};

const char* ggml_type_name(GgmlType t);
// Elements per block and bytes per block. Non-quantised types are 1 element per
// "block" of sizeof(elem).
int  ggml_block_elems(GgmlType t);
int  ggml_block_bytes(GgmlType t);
bool ggml_type_known(GgmlType t);

struct GgufTensor {
  std::string name;
  GgmlType type = GgmlType::F32;
  std::vector<uint64_t> ne;    // ggml order, ne[0] fastest
  uint64_t offset = 0;         // from the start of the data section
  const uint8_t* data = nullptr;
  uint64_t nbytes = 0;

  uint64_t numel() const {
    uint64_t n = 1;
    for (uint64_t d : ne) n *= d;
    return n;
  }
  // The HF-style [out, in] view: rows = ne[1..], row length = ne[0].
  uint64_t row_len() const { return ne.empty() ? 0 : ne[0]; }
  uint64_t rows() const { return ne.empty() ? 0 : numel() / ne[0]; }
};

struct GgufError : std::runtime_error { using std::runtime_error::runtime_error; };

class GgufFile {
 public:
  ~GgufFile();
  GgufFile() = default;
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;

  void open(const std::string& path);
  void close();

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const GgufTensor& get(const std::string& name) const;
  const GgufTensor* find(const std::string& name) const;
  const std::unordered_map<std::string, GgufTensor>& all() const { return tensors_; }

  // Metadata. Integer-ish values are widened to int64, floats to double.
  bool        meta_has(const std::string& k) const { return meta_.count(k) != 0; }
  int64_t     meta_int(const std::string& k) const;
  int64_t     meta_int(const std::string& k, int64_t dflt) const;
  double      meta_f64(const std::string& k, double dflt) const;
  std::string meta_str(const std::string& k, const std::string& dflt = "") const;
  const std::vector<int64_t>& meta_int_array(const std::string& k) const;

  uint32_t version() const { return version_; }
  size_t   mapped_bytes() const { return len_; }
  const std::vector<std::string>& meta_keys() const { return meta_order_; }

 private:
  struct Value {
    uint32_t type = 0;
    int64_t i = 0;
    double f = 0;
    std::string s;
    std::vector<int64_t> ia;
    std::vector<std::string> sa;
  };
  void* addr_ = nullptr;
  size_t len_ = 0;
  uint32_t version_ = 0;
  std::unordered_map<std::string, GgufTensor> tensors_;
  std::unordered_map<std::string, Value> meta_;
  std::vector<std::string> meta_order_;
};

}  // namespace qwen
