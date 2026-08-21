#include "gguf.h"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace qwen {
namespace {

struct TypeInfo { const char* name; int elems; int bytes; };

// Block geometry, straight from ggml. `elems` is how many weights one block
// holds and `bytes` is how big the block is on disk; the ratio is the real
// bits-per-weight, e.g. IQ4_XS is 136 bytes per 256 weights = 4.25 bits.
const TypeInfo kTypes[] = {
  /* 0 F32     */ {"F32", 1, 4},
  /* 1 F16     */ {"F16", 1, 2},
  /* 2 Q4_0    */ {"Q4_0", 32, 18},
  /* 3 Q4_1    */ {"Q4_1", 32, 20},
  /* 4         */ {nullptr, 0, 0},
  /* 5         */ {nullptr, 0, 0},
  /* 6 Q5_0    */ {"Q5_0", 32, 22},
  /* 7 Q5_1    */ {"Q5_1", 32, 24},
  /* 8 Q8_0    */ {"Q8_0", 32, 34},
  /* 9 Q8_1    */ {"Q8_1", 32, 36},
  /*10 Q2_K    */ {"Q2_K", 256, 84},
  /*11 Q3_K    */ {"Q3_K", 256, 110},
  /*12 Q4_K    */ {"Q4_K", 256, 144},
  /*13 Q5_K    */ {"Q5_K", 256, 176},
  /*14 Q6_K    */ {"Q6_K", 256, 210},
  /*15 Q8_K    */ {"Q8_K", 256, 292},
  /*16 IQ2_XXS */ {"IQ2_XXS", 256, 66},
  /*17 IQ2_XS  */ {"IQ2_XS", 256, 74},
  /*18 IQ3_XXS */ {"IQ3_XXS", 256, 98},
  /*19 IQ1_S   */ {"IQ1_S", 256, 50},
  /*20 IQ4_NL  */ {"IQ4_NL", 32, 18},
  /*21 IQ3_S   */ {"IQ3_S", 256, 110},
  /*22 IQ2_S   */ {"IQ2_S", 256, 82},
  /*23 IQ4_XS  */ {"IQ4_XS", 256, 136},
  /*24 I8      */ {"I8", 1, 1},
  /*25 I16     */ {"I16", 1, 2},
  /*26 I32     */ {"I32", 1, 4},
  /*27 I64     */ {"I64", 1, 8},
  /*28 F64     */ {"F64", 1, 8},
  /*29 IQ1_M   */ {"IQ1_M", 256, 56},
  /*30 BF16    */ {"BF16", 1, 2},
};
constexpr int kNTypes = int(sizeof(kTypes) / sizeof(kTypes[0]));

const TypeInfo* info(GgmlType t) {
  const int i = int(t);
  if (i < 0 || i >= kNTypes || !kTypes[i].name) return nullptr;
  return &kTypes[i];
}

// GGUF metadata value type tags.
enum : uint32_t {
  KV_U8 = 0, KV_I8 = 1, KV_U16 = 2, KV_I16 = 3, KV_U32 = 4, KV_I32 = 5,
  KV_F32 = 6, KV_BOOL = 7, KV_STR = 8, KV_ARR = 9, KV_U64 = 10, KV_I64 = 11,
  KV_F64 = 12,
};

struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  void need(size_t n) const {
    if (size_t(end - p) < n) throw GgufError("gguf: truncated file");
  }
  template <class T> T pod() {
    need(sizeof(T));
    T v;
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string str() {
    const uint64_t n = pod<uint64_t>();
    need(n);
    std::string s(reinterpret_cast<const char*>(p), size_t(n));
    p += n;
    return s;
  }
};

int64_t read_scalar_int(Reader& r, uint32_t t) {
  switch (t) {
    case KV_U8:  return int64_t(r.pod<uint8_t>());
    case KV_I8:  return int64_t(r.pod<int8_t>());
    case KV_U16: return int64_t(r.pod<uint16_t>());
    case KV_I16: return int64_t(r.pod<int16_t>());
    case KV_U32: return int64_t(r.pod<uint32_t>());
    case KV_I32: return int64_t(r.pod<int32_t>());
    case KV_BOOL:return int64_t(r.pod<uint8_t>());
    case KV_U64: return int64_t(r.pod<uint64_t>());
    case KV_I64: return r.pod<int64_t>();
    default: throw GgufError("gguf: not an integer kv type");
  }
}

}  // namespace

const char* ggml_type_name(GgmlType t) {
  const TypeInfo* i = info(t);
  return i ? i->name : "UNKNOWN";
}
int ggml_block_elems(GgmlType t) {
  const TypeInfo* i = info(t);
  if (!i) throw GgufError("gguf: unknown ggml type " + std::to_string(int(t)));
  return i->elems;
}
int ggml_block_bytes(GgmlType t) {
  const TypeInfo* i = info(t);
  if (!i) throw GgufError("gguf: unknown ggml type " + std::to_string(int(t)));
  return i->bytes;
}
bool ggml_type_known(GgmlType t) { return info(t) != nullptr; }

GgufFile::~GgufFile() { close(); }

void GgufFile::close() {
  if (addr_) { munmap(addr_, len_); addr_ = nullptr; len_ = 0; }
  tensors_.clear();
  meta_.clear();
  meta_order_.clear();
}

void GgufFile::open(const std::string& path) {
  close();
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw GgufError("gguf: cannot open " + path);
  struct stat st{};
  if (fstat(fd, &st) != 0) { ::close(fd); throw GgufError("gguf: fstat failed"); }
  len_ = size_t(st.st_size);
  addr_ = mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (addr_ == MAP_FAILED) { addr_ = nullptr; throw GgufError("gguf: mmap failed"); }

  Reader r{static_cast<const uint8_t*>(addr_), static_cast<const uint8_t*>(addr_) + len_};
  const uint32_t magic = r.pod<uint32_t>();
  if (magic != 0x46554747u) throw GgufError("gguf: bad magic");
  version_ = r.pod<uint32_t>();
  if (version_ < 2 || version_ > 3)
    throw GgufError("gguf: unsupported version " + std::to_string(version_));
  const uint64_t n_tensors = r.pod<uint64_t>();
  const uint64_t n_kv = r.pod<uint64_t>();

  for (uint64_t i = 0; i < n_kv; ++i) {
    const std::string key = r.str();
    const uint32_t t = r.pod<uint32_t>();
    Value v;
    v.type = t;
    if (t == KV_STR) {
      v.s = r.str();
    } else if (t == KV_F32) {
      v.f = double(r.pod<float>());
    } else if (t == KV_F64) {
      v.f = r.pod<double>();
    } else if (t == KV_ARR) {
      const uint32_t et = r.pod<uint32_t>();
      const uint64_t n = r.pod<uint64_t>();
      for (uint64_t j = 0; j < n; ++j) {
        if (et == KV_STR) v.sa.push_back(r.str());
        else if (et == KV_F32) v.f = double(r.pod<float>());
        else if (et == KV_F64) v.f = r.pod<double>();
        else v.ia.push_back(read_scalar_int(r, et));
      }
    } else {
      v.i = read_scalar_int(r, t);
    }
    if (!meta_.count(key)) meta_order_.push_back(key);
    meta_[key] = std::move(v);
  }

  std::vector<GgufTensor> list;
  list.reserve(size_t(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensor t;
    t.name = r.str();
    const uint32_t nd = r.pod<uint32_t>();
    if (nd == 0 || nd > 4) throw GgufError("gguf: tensor " + t.name + " has " +
                                           std::to_string(nd) + " dims");
    for (uint32_t d = 0; d < nd; ++d) t.ne.push_back(r.pod<uint64_t>());
    t.type = GgmlType(r.pod<uint32_t>());
    t.offset = r.pod<uint64_t>();
    if (!ggml_type_known(t.type))
      throw GgufError("gguf: tensor " + t.name + " has unknown type " +
                      std::to_string(uint32_t(t.type)));
    const int be = ggml_block_elems(t.type), bb = ggml_block_bytes(t.type);
    if (t.numel() % uint64_t(be) != 0)
      throw GgufError("gguf: tensor " + t.name + " numel not a multiple of the block");
    t.nbytes = t.numel() / uint64_t(be) * uint64_t(bb);
    list.push_back(std::move(t));
  }

  const int64_t align = meta_int("general.alignment", 32);
  const size_t hdr = size_t(r.p - static_cast<const uint8_t*>(addr_));
  const size_t data_start = (hdr + size_t(align) - 1) / size_t(align) * size_t(align);
  for (auto& t : list) {
    const size_t off = data_start + size_t(t.offset);
    if (off + t.nbytes > len_)
      throw GgufError("gguf: tensor " + t.name + " runs past the end of the file");
    t.data = static_cast<const uint8_t*>(addr_) + off;
    tensors_.emplace(t.name, std::move(t));
  }
}

const GgufTensor& GgufFile::get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw GgufError("gguf: no tensor named " + name);
  return it->second;
}
const GgufTensor* GgufFile::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

int64_t GgufFile::meta_int(const std::string& k) const {
  auto it = meta_.find(k);
  if (it == meta_.end()) throw GgufError("gguf: missing metadata key " + k);
  return it->second.i;
}
int64_t GgufFile::meta_int(const std::string& k, int64_t dflt) const {
  auto it = meta_.find(k);
  return it == meta_.end() ? dflt : it->second.i;
}
double GgufFile::meta_f64(const std::string& k, double dflt) const {
  auto it = meta_.find(k);
  return it == meta_.end() ? dflt : it->second.f;
}
std::string GgufFile::meta_str(const std::string& k, const std::string& dflt) const {
  auto it = meta_.find(k);
  return it == meta_.end() ? dflt : it->second.s;
}
const std::vector<int64_t>& GgufFile::meta_int_array(const std::string& k) const {
  auto it = meta_.find(k);
  if (it == meta_.end()) throw GgufError("gguf: missing metadata array " + k);
  return it->second.ia;
}

}  // namespace qwen
