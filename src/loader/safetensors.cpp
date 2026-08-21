#include "safetensors.h"

#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../../third_party/json.hpp"

using json = nlohmann::json;

namespace qwen {

size_t dtype_size(Dtype d) {
  switch (d) {
    case Dtype::BF16: case Dtype::F16: case Dtype::I16: return 2;
    case Dtype::F32:  case Dtype::I32:                  return 4;
    case Dtype::F64:  case Dtype::I64:                  return 8;
    case Dtype::I8:   case Dtype::U8: case Dtype::BOOL:
    case Dtype::F8_E4M3: case Dtype::F8_E5M2:           return 1;
  }
  return 0;
}

const char* dtype_name(Dtype d) {
  switch (d) {
    case Dtype::BF16: return "BF16";  case Dtype::F16: return "F16";
    case Dtype::F32:  return "F32";   case Dtype::F64: return "F64";
    case Dtype::I8:   return "I8";    case Dtype::U8:  return "U8";
    case Dtype::I16:  return "I16";   case Dtype::I32: return "I32";
    case Dtype::I64:  return "I64";   case Dtype::BOOL: return "BOOL";
    case Dtype::F8_E4M3: return "F8_E4M3"; case Dtype::F8_E5M2: return "F8_E5M2";
  }
  return "?";
}

static Dtype parse_dtype(const std::string& s) {
  if (s == "BF16") return Dtype::BF16;
  if (s == "F16")  return Dtype::F16;
  if (s == "F32")  return Dtype::F32;
  if (s == "F64")  return Dtype::F64;
  if (s == "I8")   return Dtype::I8;
  if (s == "U8")   return Dtype::U8;
  if (s == "I16")  return Dtype::I16;
  if (s == "I32")  return Dtype::I32;
  if (s == "I64")  return Dtype::I64;
  if (s == "BOOL") return Dtype::BOOL;
  if (s == "F8_E4M3") return Dtype::F8_E4M3;
  if (s == "F8_E5M2") return Dtype::F8_E5M2;
  throw LoaderError("safetensors: unsupported dtype " + s);
}

SafeTensors::~SafeTensors() { close(); }

void SafeTensors::close() {
  for (auto& m : maps_) if (m.addr) ::munmap(m.addr, m.len);
  maps_.clear();
  tensors_.clear();
}

size_t SafeTensors::mapped_bytes() const {
  size_t n = 0;
  for (const auto& m : maps_) n += m.len;
  return n;
}

void SafeTensors::open_one(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw LoaderError("safetensors: cannot open " + path);
  struct stat st{};
  if (::fstat(fd, &st) != 0) { ::close(fd); throw LoaderError("safetensors: fstat " + path); }
  const size_t len = static_cast<size_t>(st.st_size);
  if (len < 8) { ::close(fd); throw LoaderError("safetensors: too small " + path); }

  void* addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (addr == MAP_FAILED) throw LoaderError("safetensors: mmap failed " + path);
  // Sequential during load, and we do not want the page cache to keep 20 GB
  // resident afterwards; the arena upload drops it.
  ::madvise(addr, len, MADV_SEQUENTIAL);

  const int shard = static_cast<int>(maps_.size());
  maps_.push_back({addr, len, path});

  const auto* base = static_cast<const uint8_t*>(addr);
  uint64_t hdr_len = 0;
  std::memcpy(&hdr_len, base, 8);
  if (hdr_len == 0 || 8 + hdr_len > len)
    throw LoaderError("safetensors: bad header length in " + path);

  json hdr;
  try {
    hdr = json::parse(reinterpret_cast<const char*>(base + 8),
                      reinterpret_cast<const char*>(base + 8 + hdr_len));
  } catch (const std::exception& e) {
    throw LoaderError("safetensors: bad header JSON in " + path + ": " + e.what());
  }

  const uint8_t* data = base + 8 + hdr_len;
  const size_t data_len = len - 8 - hdr_len;

  for (auto it = hdr.begin(); it != hdr.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const json& t = it.value();
    TensorView v;
    v.dtype = parse_dtype(t.at("dtype").get<std::string>());
    v.shape = t.at("shape").get<std::vector<int64_t>>();
    const auto off = t.at("data_offsets").get<std::vector<uint64_t>>();
    if (off.size() != 2 || off[1] < off[0] || off[1] > data_len)
      throw LoaderError("safetensors: bad data_offsets for " + it.key());
    v.nbytes = static_cast<size_t>(off[1] - off[0]);
    v.data   = data + off[0];
    v.shard  = shard;
    if (v.nbytes != static_cast<size_t>(v.numel()) * dtype_size(v.dtype))
      throw LoaderError("safetensors: size mismatch for " + it.key());
    // A duplicate name across shards means the index is inconsistent; that is
    // a corrupted download, not something to paper over.
    if (!tensors_.emplace(it.key(), v).second)
      throw LoaderError("safetensors: duplicate tensor " + it.key());
  }
}

void SafeTensors::open_files(const std::vector<std::string>& paths) {
  for (const auto& p : paths) open_one(p);
}

void SafeTensors::open_dir(const std::string& dir) {
  DIR* d = ::opendir(dir.c_str());
  if (!d) throw LoaderError("safetensors: cannot list " + dir);
  std::vector<std::string> files;
  while (dirent* e = ::readdir(d)) {
    const std::string n = e->d_name;
    if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0)
      files.push_back(dir + "/" + n);
  }
  ::closedir(d);
  if (files.empty()) throw LoaderError("safetensors: no .safetensors in " + dir);
  std::sort(files.begin(), files.end());
  open_files(files);
}

const TensorView* SafeTensors::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& SafeTensors::get(const std::string& name) const {
  const TensorView* v = find(name);
  if (!v) throw LoaderError("safetensors: missing tensor " + name);
  return *v;
}

size_t SafeTensors::bytes_with_prefix(const std::string& prefix) const {
  size_t n = 0;
  for (const auto& [k, v] : tensors_)
    if (k.compare(0, prefix.size(), prefix) == 0) n += v.nbytes;
  return n;
}

}  // namespace qwen
