// mmap'd safetensors reader.
//
// The file is mapped, never read into a host buffer: the directive's loader
// requirement is a direct-to-device staged upload with no host double copy, and
// the prior Jetson work hit exactly that bug. On discrete VRAM the failure mode
// is host RAM pressure and a slow cold start rather than a crash, but the fix is
// the same shape.
//
// Header format: [u64 little-endian header length][JSON header][tensor data]
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen {

struct LoaderError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class Dtype : uint8_t { BF16, F16, F32, F64, I8, U8, I16, I32, I64, BOOL, F8_E4M3, F8_E5M2 };

size_t dtype_size(Dtype d);
const char* dtype_name(Dtype d);

struct TensorView {
  Dtype dtype = Dtype::BF16;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;   // points into the mapping; valid while open
  size_t nbytes = 0;
  int shard = -1;

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

// One or more shards presented as a single tensor namespace.
class SafeTensors {
 public:
  ~SafeTensors();
  SafeTensors() = default;
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;

  // Opens every *.safetensors in `dir`, honouring model.safetensors.index.json
  // if present but not requiring it.
  void open_dir(const std::string& dir);
  void open_files(const std::vector<std::string>& paths);
  void close();

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorView& get(const std::string& name) const;
  const TensorView* find(const std::string& name) const;

  const std::unordered_map<std::string, TensorView>& all() const { return tensors_; }
  size_t size() const { return tensors_.size(); }
  size_t mapped_bytes() const;

  // Total bytes of tensors whose name starts with `prefix`. Used for budget
  // reporting and to size the vision-tower skip.
  size_t bytes_with_prefix(const std::string& prefix) const;

 private:
  struct Mapping { void* addr = nullptr; size_t len = 0; std::string path; };
  std::vector<Mapping> maps_;
  std::unordered_map<std::string, TensorView> tensors_;

  void open_one(const std::string& path);
};

}  // namespace qwen
