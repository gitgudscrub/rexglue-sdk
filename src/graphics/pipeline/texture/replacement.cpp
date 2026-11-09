/**
 * Texture replacement packs - see replacement.h.
 */

#include <rex/graphics/pipeline/texture/replacement.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <system_error>
#include <thread>

#include <rex/cvar.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/filesystem.h>
#include <rex/hash.h>
#include <rex/logging.h>

REXCVAR_DEFINE_STRING(texture_pack_root, "textures", "GPU",
                      "Texture pack folder: dump/ receives dumped textures, replace/ holds "
                      "replacement DDS files named by texture hash");
REXCVAR_DEFINE_BOOL(texture_dump, false, "GPU",
                    "Write each unique game texture once to <texture_pack_root>/dump as DDS");
REXCVAR_DEFINE_BOOL(texture_replacement, true, "GPU",
                    "Show replacement textures from <texture_pack_root>/replace (false = the "
                    "original textures). Can be changed while running.");

namespace rex::graphics::texture_replacement {

namespace {

// DXGI_FORMAT values, so this file doesn't need the D3D headers.
enum : uint32_t {
  kDxgiR8G8B8A8Unorm = 28,
  kDxgiR8G8B8A8UnormSrgb = 29,
  kDxgiR8G8Unorm = 49,
  kDxgiR8Unorm = 61,
  kDxgiBC1Unorm = 71,
  kDxgiBC1UnormSrgb = 72,
  kDxgiBC2Unorm = 74,
  kDxgiBC2UnormSrgb = 75,
  kDxgiBC3Unorm = 77,
  kDxgiBC3UnormSrgb = 78,
  kDxgiBC4Unorm = 80,
  kDxgiBC5Unorm = 83,
  kDxgiB8G8R8A8Unorm = 87,
  kDxgiB8G8R8A8UnormSrgb = 91,
  kDxgiBC7Unorm = 98,
  kDxgiBC7UnormSrgb = 99,
};

constexpr uint32_t FourCC(char a, char b, char c, char d) {
  return uint32_t(uint8_t(a)) | (uint32_t(uint8_t(b)) << 8) | (uint32_t(uint8_t(c)) << 16) |
         (uint32_t(uint8_t(d)) << 24);
}

#pragma pack(push, 1)
struct DdsPixelFormat {
  uint32_t size;
  uint32_t flags;
  uint32_t four_cc;
  uint32_t rgb_bit_count;
  uint32_t r_mask, g_mask, b_mask, a_mask;
};
struct DdsHeader {
  uint32_t size;
  uint32_t flags;
  uint32_t height;
  uint32_t width;
  uint32_t pitch_or_linear_size;
  uint32_t depth;
  uint32_t mip_map_count;
  uint32_t reserved1[11];
  DdsPixelFormat pixel_format;
  uint32_t caps, caps2, caps3, caps4;
  uint32_t reserved2;
};
struct DdsHeaderDx10 {
  uint32_t dxgi_format;
  uint32_t resource_dimension;
  uint32_t misc_flag;
  uint32_t array_size;
  uint32_t misc_flags2;
};
#pragma pack(pop)
static_assert(sizeof(DdsHeader) == 124);
static_assert(sizeof(DdsHeaderDx10) == 20);

constexpr uint32_t kDdsMagic = FourCC('D', 'D', 'S', ' ');
constexpr uint32_t kDdsdCaps = 0x1, kDdsdHeight = 0x2, kDdsdWidth = 0x4, kDdsdPitch = 0x8,
                   kDdsdPixelFormat = 0x1000, kDdsdMipMapCount = 0x20000,
                   kDdsdLinearSize = 0x80000;
constexpr uint32_t kDdpfAlphaPixels = 0x1, kDdpfFourCC = 0x4, kDdpfRgb = 0x40,
                   kDdpfLuminance = 0x20000;
constexpr uint32_t kDdsCapsTexture = 0x1000;
constexpr uint32_t kDdsCaps2Cubemap = 0x200, kDdsCaps2Volume = 0x200000;
constexpr uint32_t kD3D10ResourceDimensionTexture2D = 3;

// How a supported guest format is dumped: which DXGI format the untiled data
// is (matching the host texture), and whether it can be written with a legacy
// DDS header (better tool support) instead of DX10.
struct DumpFormat {
  uint32_t dxgi_format;
  uint32_t legacy_four_cc;  // 0 = RGBA masks (32bpp) or DX10.
  bool legacy_rgba;
  const char* name;
};

bool GetDumpFormat(xenos::TextureFormat format, DumpFormat& out) {
  switch (format) {
    case xenos::TextureFormat::k_DXT1:
      out = {kDxgiBC1Unorm, FourCC('D', 'X', 'T', '1'), false, "DXT1"};
      return true;
    case xenos::TextureFormat::k_DXT2_3:
      out = {kDxgiBC2Unorm, FourCC('D', 'X', 'T', '3'), false, "DXT3"};
      return true;
    case xenos::TextureFormat::k_DXT4_5:
      out = {kDxgiBC3Unorm, FourCC('D', 'X', 'T', '5'), false, "DXT5"};
      return true;
    case xenos::TextureFormat::k_DXN:
      out = {kDxgiBC5Unorm, 0, false, "DXN"};
      return true;
    case xenos::TextureFormat::k_DXT5A:
      out = {kDxgiBC4Unorm, 0, false, "DXT5A"};
      return true;
    case xenos::TextureFormat::k_8_8_8_8:
      out = {kDxgiR8G8B8A8Unorm, 0, true, "RGBA8"};
      return true;
    case xenos::TextureFormat::k_8:
      out = {kDxgiR8Unorm, 0, false, "R8"};
      return true;
    default:
      return false;
  }
}

// Format properties of DXGI formats replacements may use.
bool GetDxgiFormatLayout(uint32_t dxgi_format, uint32_t& bytes_per_block_out,
                         bool& block_compressed_out) {
  switch (dxgi_format) {
    case kDxgiBC1Unorm:
    case kDxgiBC4Unorm:
      bytes_per_block_out = 8;
      block_compressed_out = true;
      return true;
    case kDxgiBC2Unorm:
    case kDxgiBC3Unorm:
    case kDxgiBC5Unorm:
    case kDxgiBC7Unorm:
      bytes_per_block_out = 16;
      block_compressed_out = true;
      return true;
    case kDxgiR8G8B8A8Unorm:
    case kDxgiB8G8R8A8Unorm:
      bytes_per_block_out = 4;
      block_compressed_out = false;
      return true;
    case kDxgiR8G8Unorm:
      bytes_per_block_out = 2;
      block_compressed_out = false;
      return true;
    case kDxgiR8Unorm:
      bytes_per_block_out = 1;
      block_compressed_out = false;
      return true;
    default:
      return false;
  }
}

uint32_t SrgbToUnorm(uint32_t dxgi_format) {
  switch (dxgi_format) {
    case kDxgiR8G8B8A8UnormSrgb:
      return kDxgiR8G8B8A8Unorm;
    case kDxgiBC1UnormSrgb:
      return kDxgiBC1Unorm;
    case kDxgiBC2UnormSrgb:
      return kDxgiBC2Unorm;
    case kDxgiBC3UnormSrgb:
      return kDxgiBC3Unorm;
    case kDxgiB8G8R8A8UnormSrgb:
      return kDxgiB8G8R8A8Unorm;
    case kDxgiBC7UnormSrgb:
      return kDxgiBC7Unorm;
    default:
      return dxgi_format;
  }
}

// Guest endian swap of one block (the conversion helpers live in rexcore and
// are not exported to this plugin).
void CopySwap(xenos::Endian endian, uint8_t* out, const uint8_t* in, uint32_t length) {
  switch (endian) {
    case xenos::Endian::k8in16:
      for (uint32_t i = 0; i + 1 < length; i += 2) {
        out[i] = in[i + 1];
        out[i + 1] = in[i];
      }
      break;
    case xenos::Endian::k8in32:
      for (uint32_t i = 0; i + 3 < length; i += 4) {
        out[i] = in[i + 3];
        out[i + 1] = in[i + 2];
        out[i + 2] = in[i + 1];
        out[i + 3] = in[i];
      }
      break;
    case xenos::Endian::k16in32:
      for (uint32_t i = 0; i + 3 < length; i += 4) {
        out[i] = in[i + 2];
        out[i + 1] = in[i + 3];
        out[i + 2] = in[i];
        out[i + 3] = in[i + 1];
      }
      break;
    default:
      std::memcpy(out, in, length);
      break;
  }
}

std::string HashPrefix(uint64_t hash) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llX", static_cast<unsigned long long>(hash));
  return buffer;
}

// Parses the 16 leading hex characters of a file name.
bool ParseHashPrefix(const std::string& name, uint64_t& hash_out) {
  if (name.size() < 16) {
    return false;
  }
  uint64_t hash = 0;
  for (size_t i = 0; i < 16; ++i) {
    char c = name[i];
    uint32_t digit;
    if (c >= '0' && c <= '9') {
      digit = uint32_t(c - '0');
    } else if (c >= 'A' && c <= 'F') {
      digit = uint32_t(c - 'A' + 10);
    } else if (c >= 'a' && c <= 'f') {
      digit = uint32_t(c - 'a' + 10);
    } else {
      return false;
    }
    hash = (hash << 4) | digit;
  }
  hash_out = hash;
  return true;
}

bool WriteDds(const std::filesystem::path& path, const DumpFormat& format, uint32_t width,
              uint32_t height, const std::vector<uint8_t>& data) {
  DdsHeader header = {};
  header.size = sizeof(DdsHeader);
  header.flags = kDdsdCaps | kDdsdHeight | kDdsdWidth | kDdsdPixelFormat | kDdsdMipMapCount;
  header.width = width;
  header.height = height;
  header.mip_map_count = 1;
  header.caps = kDdsCapsTexture;
  header.pixel_format.size = sizeof(DdsPixelFormat);
  bool dx10 = false;
  uint32_t bytes_per_block;
  bool block_compressed;
  GetDxgiFormatLayout(format.dxgi_format, bytes_per_block, block_compressed);
  if (block_compressed) {
    header.flags |= kDdsdLinearSize;
    header.pitch_or_linear_size = uint32_t(data.size());
  } else {
    header.flags |= kDdsdPitch;
    header.pitch_or_linear_size = width * bytes_per_block;
  }
  if (format.legacy_four_cc) {
    header.pixel_format.flags = kDdpfFourCC;
    header.pixel_format.four_cc = format.legacy_four_cc;
  } else if (format.legacy_rgba) {
    header.pixel_format.flags = kDdpfRgb | kDdpfAlphaPixels;
    header.pixel_format.rgb_bit_count = 32;
    header.pixel_format.r_mask = 0x000000FF;
    header.pixel_format.g_mask = 0x0000FF00;
    header.pixel_format.b_mask = 0x00FF0000;
    header.pixel_format.a_mask = 0xFF000000;
  } else {
    header.pixel_format.flags = kDdpfFourCC;
    header.pixel_format.four_cc = FourCC('D', 'X', '1', '0');
    dx10 = true;
  }

  // Write to a temporary name first so a crash never leaves a truncated file
  // that would be skipped as "already dumped" next session.
  std::filesystem::path temp_path = path;
  temp_path += ".tmp";
  {
    std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
    if (!file) {
      return false;
    }
    file.write(reinterpret_cast<const char*>(&kDdsMagic), sizeof(kDdsMagic));
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (dx10) {
      DdsHeaderDx10 header_dx10 = {};
      header_dx10.dxgi_format = format.dxgi_format;
      header_dx10.resource_dimension = kD3D10ResourceDimensionTexture2D;
      header_dx10.array_size = 1;
      file.write(reinterpret_cast<const char*>(&header_dx10), sizeof(header_dx10));
    }
    file.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    if (!file) {
      return false;
    }
  }
  std::error_code ec;
  std::filesystem::rename(temp_path, path, ec);
  return !ec;
}

}  // namespace

bool IsFormatSupported(xenos::TextureFormat format) {
  DumpFormat dump_format;
  return GetDumpFormat(format, dump_format);
}

uint32_t DdsImage::MipRows(uint32_t mip) const {
  uint32_t h = MipHeight(mip);
  return block_compressed ? (h + 3) / 4 : h;
}

uint32_t DdsImage::MipRowBytes(uint32_t mip) const {
  uint32_t w = MipWidth(mip);
  return (block_compressed ? (w + 3) / 4 : w) * bytes_per_block;
}

bool LoadDds(const std::filesystem::path& path, DdsImage& image_out) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }
  std::streamoff file_size = file.tellg();
  file.seekg(0);
  uint32_t magic = 0;
  DdsHeader header = {};
  if (file_size < std::streamoff(sizeof(magic) + sizeof(header))) {
    return false;
  }
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!file || magic != kDdsMagic || header.size != sizeof(DdsHeader)) {
    return false;
  }
  if (header.caps2 & (kDdsCaps2Cubemap | kDdsCaps2Volume)) {
    return false;
  }
  std::streamoff data_offset = sizeof(magic) + sizeof(header);

  uint32_t dxgi_format = 0;
  const DdsPixelFormat& pf = header.pixel_format;
  if (pf.flags & kDdpfFourCC) {
    switch (pf.four_cc) {
      case FourCC('D', 'X', 'T', '1'):
        dxgi_format = kDxgiBC1Unorm;
        break;
      case FourCC('D', 'X', 'T', '2'):
      case FourCC('D', 'X', 'T', '3'):
        dxgi_format = kDxgiBC2Unorm;
        break;
      case FourCC('D', 'X', 'T', '4'):
      case FourCC('D', 'X', 'T', '5'):
        dxgi_format = kDxgiBC3Unorm;
        break;
      case FourCC('A', 'T', 'I', '1'):
      case FourCC('B', 'C', '4', 'U'):
        dxgi_format = kDxgiBC4Unorm;
        break;
      case FourCC('A', 'T', 'I', '2'):
      case FourCC('B', 'C', '5', 'U'):
        dxgi_format = kDxgiBC5Unorm;
        break;
      case FourCC('D', 'X', '1', '0'): {
        DdsHeaderDx10 header_dx10 = {};
        file.read(reinterpret_cast<char*>(&header_dx10), sizeof(header_dx10));
        if (!file || header_dx10.resource_dimension != kD3D10ResourceDimensionTexture2D ||
            header_dx10.array_size > 1 || (header_dx10.misc_flag & 0x4 /* cube */)) {
          return false;
        }
        dxgi_format = header_dx10.dxgi_format;
        data_offset += sizeof(header_dx10);
      } break;
      default:
        return false;
    }
  } else if ((pf.flags & kDdpfRgb) && pf.rgb_bit_count == 32) {
    if (pf.r_mask == 0x000000FF && pf.g_mask == 0x0000FF00 && pf.b_mask == 0x00FF0000) {
      dxgi_format = kDxgiR8G8B8A8Unorm;
    } else if (pf.r_mask == 0x00FF0000 && pf.g_mask == 0x0000FF00 && pf.b_mask == 0x000000FF) {
      dxgi_format = kDxgiB8G8R8A8Unorm;
    }
  } else if ((pf.flags & (kDdpfLuminance | kDdpfRgb)) && pf.rgb_bit_count == 8) {
    dxgi_format = kDxgiR8Unorm;
  }
  dxgi_format = SrgbToUnorm(dxgi_format);

  DdsImage image;
  if (!GetDxgiFormatLayout(dxgi_format, image.bytes_per_block, image.block_compressed)) {
    return false;
  }
  image.dxgi_format = dxgi_format;
  image.width = header.width;
  image.height = header.height;
  if (!image.width || !image.height || image.width > 16384 || image.height > 16384) {
    return false;
  }
  uint32_t max_mips = 1;
  while ((std::max(image.width, image.height) >> max_mips) != 0) {
    ++max_mips;
  }
  image.mip_count = (header.flags & kDdsdMipMapCount) && header.mip_map_count
                        ? std::min(header.mip_map_count, max_mips)
                        : 1;
  // Block-compressed resources need 4-aligned top-level sizes on D3D12 hosts
  // without unaligned block texture support; require it to be safe.
  if (image.block_compressed && ((image.width & 3) || (image.height & 3))) {
    return false;
  }
  uint64_t total = 0;
  image.mip_offsets.resize(image.mip_count);
  for (uint32_t mip = 0; mip < image.mip_count; ++mip) {
    image.mip_offsets[mip] = total;
    total += uint64_t(image.MipRows(mip)) * image.MipRowBytes(mip);
  }
  if (uint64_t(file_size - data_offset) < total) {
    return false;
  }
  image.data.resize(size_t(total));
  file.seekg(data_offset);
  file.read(reinterpret_cast<char*>(image.data.data()), std::streamsize(total));
  if (!file) {
    return false;
  }
  image_out = std::move(image);
  return true;
}

struct TextureReplacer::DumpJob {
  std::filesystem::path path;
  DumpFormat format;
  uint32_t width;
  uint32_t height;
  std::vector<uint8_t> data;
};

class TextureReplacer::DumpWorker {
 public:
  DumpWorker() : thread_([this] { Run(); }) {}
  ~DumpWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }

  // Waits while the queue is full: a texture whose dump is dropped might
  // never be loaded again, and dumping is a collection mode where a short
  // stall is fine.
  bool Push(DumpJob&& job) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      space_cv_.wait(lock, [this] { return jobs_.size() < 512; });
      jobs_.push_back(std::move(job));
    }
    cv_.notify_one();
    return true;
  }

 private:
  void Run() {
    while (true) {
      DumpJob job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
        if (jobs_.empty()) {
          return;  // Stopping with nothing left to write.
        }
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      space_cv_.notify_one();
      if (!WriteDds(job.path, job.format, job.width, job.height, job.data)) {
        REXGPU_WARN("Texture dump: failed to write {}", rex::path_to_utf8(job.path));
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable space_cv_;
  std::deque<DumpJob> jobs_;
  bool stop_ = false;
  std::thread thread_;
};

class TextureReplacer::LoadWorker {
 public:
  LoadWorker() : thread_([this] { Run(); }) {}
  ~LoadWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }

  void Request(uint64_t hash, const std::filesystem::path& path) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!in_flight_.insert(hash).second) {
        return;  // Already queued or being read.
      }
      queue_.emplace_back(hash, path);
    }
    cv_.notify_one();
  }

  void Collect(std::vector<LoadedImage>& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (LoadedImage& loaded : done_) {
      in_flight_.erase(loaded.hash);
      out.push_back(std::move(loaded));
    }
    done_.clear();
  }

 private:
  void Run() {
    while (true) {
      std::pair<uint64_t, std::filesystem::path> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_) {
          return;
        }
        job = std::move(queue_.front());
        queue_.pop_front();
      }
      LoadedImage loaded;
      loaded.hash = job.first;
      loaded.path = job.second;
      auto image = std::make_unique<DdsImage>();
      if (LoadDds(job.second, *image)) {
        loaded.image = std::move(image);
      }
      std::lock_guard<std::mutex> lock(mutex_);
      done_.push_back(std::move(loaded));
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::pair<uint64_t, std::filesystem::path>> queue_;
  std::vector<LoadedImage> done_;
  std::unordered_set<uint64_t> in_flight_;
  bool stop_ = false;
  std::thread thread_;
};

TextureReplacer::TextureReplacer() = default;
TextureReplacer::~TextureReplacer() { Shutdown(); }

void TextureReplacer::Initialize() {
  Shutdown();
  const std::string& root_utf8 = REXCVAR_GET(texture_pack_root);
  std::filesystem::path root = std::filesystem::u8path(root_utf8);
  if (root.empty()) {
    return;
  }
  std::error_code ec;

  std::filesystem::path replace_dir = root / "replace";
  if (std::filesystem::is_directory(replace_dir, ec)) {
    for (auto it = std::filesystem::recursive_directory_iterator(
             replace_dir, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
      if (!it->is_regular_file(ec)) {
        continue;
      }
      const std::filesystem::path& path = it->path();
      std::string extension = rex::path_to_utf8(path.extension());
      if (extension != ".dds" && extension != ".DDS") {
        continue;
      }
      uint64_t hash;
      if (ParseHashPrefix(rex::path_to_utf8(path.filename()), hash)) {
        replacements_[hash] = path;
      }
    }
  }
  REXGPU_INFO("Texture pack: {} replacement textures in {}", replacements_.size(),
              rex::path_to_utf8(replace_dir));

  dump_enabled_ = REXCVAR_GET(texture_dump);
  if (dump_enabled_) {
    dump_dir_ = root / "dump";
    std::filesystem::create_directories(dump_dir_, ec);
    if (!std::filesystem::is_directory(dump_dir_, ec)) {
      REXGPU_ERROR("Texture dump: can't create {}", rex::path_to_utf8(dump_dir_));
      dump_enabled_ = false;
    } else {
      // Skip textures dumped in earlier sessions.
      for (auto it = std::filesystem::directory_iterator(dump_dir_, ec);
           !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        uint64_t hash;
        std::string name = rex::path_to_utf8(it->path().filename());
        if (name.size() > 4 && name.substr(name.size() - 4) == ".dds" &&
            ParseHashPrefix(name, hash)) {
          dumped_.insert(hash);
        }
      }
      dump_worker_ = std::make_unique<DumpWorker>();
      REXGPU_INFO("Texture dump: on, {} already in {}", dumped_.size(),
                  rex::path_to_utf8(dump_dir_));
    }
  }
}

void TextureReplacer::Shutdown() {
  load_worker_.reset();
  dump_worker_.reset();
  dump_enabled_ = false;
  replacements_.clear();
  dumped_.clear();
  last_untiled_.clear();
  last_untiled_.shrink_to_fit();
}

bool TextureReplacer::Hash(const GuestTexture& texture, uint64_t& hash_out) {
  DumpFormat dump_format;
  if (!GetDumpFormat(texture.format, dump_format) || !texture.data) {
    return false;
  }
  const FormatInfo* format_info = FormatInfo::Get(texture.format);
  uint32_t block_width = format_info->block_width;
  uint32_t block_height = format_info->block_height;
  uint32_t bytes_per_block = format_info->bytes_per_block();
  uint32_t bytes_per_block_log2 = 0;
  while ((UINT32_C(1) << bytes_per_block_log2) < bytes_per_block) {
    ++bytes_per_block_log2;
  }
  uint32_t width_blocks = (texture.width + block_width - 1) / block_width;
  uint32_t height_blocks = (texture.height + block_height - 1) / block_height;
  if (!width_blocks || !height_blocks || !texture.row_pitch_bytes) {
    return false;
  }
  uint32_t pitch_blocks = texture.row_pitch_bytes / bytes_per_block;

  last_untiled_.resize(size_t(width_blocks) * height_blocks * bytes_per_block);
  uint8_t* out = last_untiled_.data();
  for (uint32_t y = 0; y < height_blocks; ++y) {
    for (uint32_t x = 0; x < width_blocks; ++x) {
      uint64_t offset;
      if (texture.tiled) {
        offset = uint32_t(texture_util::GetTiledOffset2D(int32_t(x), int32_t(y), pitch_blocks,
                                                         bytes_per_block_log2));
      } else {
        offset = uint64_t(y) * texture.row_pitch_bytes + uint64_t(x) * bytes_per_block;
      }
      if (offset + bytes_per_block > texture.data_size) {
        return false;
      }
      CopySwap(texture.endianness, out, texture.data + offset, bytes_per_block);
      out += bytes_per_block;
    }
  }

  uint64_t seed = uint64_t(texture.format) | (uint64_t(texture.width) << 8) |
                  (uint64_t(texture.height) << 24);
  hash_out = XXH3_64bits_withSeed(last_untiled_.data(), last_untiled_.size(), seed);
  last_texture_ = texture;
  last_texture_.data = nullptr;
  return true;
}

void TextureReplacer::DumpLastHashed(uint64_t hash) {
  if (!dump_enabled_ || !dump_worker_ || last_untiled_.empty()) {
    return;
  }
  if (!dumped_.insert(hash).second) {
    return;
  }
  DumpJob job;
  GetDumpFormat(last_texture_.format, job.format);
  job.width = last_texture_.width;
  job.height = last_texture_.height;
  job.path = dump_dir_ / (HashPrefix(hash) + "_" + std::to_string(job.width) + "x" +
                          std::to_string(job.height) + "_" + job.format.name + ".dds");
  job.data = last_untiled_;
  if (!dump_worker_->Push(std::move(job))) {
    // Queue full - forget it so a later load can try again.
    dumped_.erase(hash);
  }
}

std::filesystem::path TextureReplacer::FindReplacement(uint64_t hash) const {
  auto it = replacements_.find(hash);
  return it != replacements_.end() ? it->second : std::filesystem::path();
}

bool TextureReplacer::RequestLoad(uint64_t hash) {
  auto it = replacements_.find(hash);
  if (it == replacements_.end()) {
    return false;
  }
  if (!load_worker_) {
    load_worker_ = std::make_unique<LoadWorker>();
  }
  load_worker_->Request(hash, it->second);
  return true;
}

void TextureReplacer::CollectLoaded(std::vector<LoadedImage>& out) {
  if (load_worker_) {
    load_worker_->Collect(out);
  }
}

}  // namespace rex::graphics::texture_replacement
