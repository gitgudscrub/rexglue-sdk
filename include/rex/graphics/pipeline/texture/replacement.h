/**
 * Texture replacement packs: dumping guest textures to DDS files and loading
 * replacement DDS files (for example AI-upscaled versions) in their place.
 *
 * A texture is identified by a content hash of its base level: the guest data
 * untiled and endian-swapped into the same layout the host texture has, seeded
 * with the format and size. The hash does not depend on the guest address, so
 * a texture keeps its name across sessions and machines with the same game
 * files.
 *
 * Pack layout (under the texture_pack_root cvar):
 *   dump/<hash>_<width>x<height>_<format>.dds   written when texture_dump is on
 *   replace/<any subfolders>/<hash>....dds      loaded in place of the original
 *   replace/<any subfolders>/<hash>....edds     the same, encrypted with a key
 *                                               derived from the original
 *                                               texture (see LoadKeyedDds in
 *                                               replacement.cpp), so a pack
 *                                               only opens next to the game
 * Only the first 16 characters of a replacement file name (the hash) matter,
 * so tools can keep the rest of the dump name or add their own suffixes.
 *
 * A replacement must keep the channel layout of its dump (RGBA stays RGBA,
 * a two-channel DXN dump stays a two-channel BC5 or R8G8), because the guest
 * swizzle is applied on top of it. It must keep the aspect ratio; its size
 * and mip count are free.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/xenos.h>

REXCVAR_DECLARE(std::string, texture_pack_root);
REXCVAR_DECLARE(bool, texture_dump);
REXCVAR_DECLARE(bool, texture_replacement);

namespace rex::graphics::texture_replacement {

// What the hasher needs to know about a guest texture's base level.
struct GuestTexture {
  xenos::TextureFormat format;
  xenos::Endian endianness;
  uint32_t width;
  uint32_t height;
  bool tiled;
  // Row pitch of the base level in bytes (guest layout).
  uint32_t row_pitch_bytes;
  const uint8_t* data;
  uint32_t data_size;
};

// Whether textures of this format can be hashed, dumped and replaced.
bool IsFormatSupported(xenos::TextureFormat format);

// A DDS file loaded for upload. `data` holds the mips back to back, each one
// tightly packed (rows of blocks with no padding).
struct DdsImage {
  uint32_t dxgi_format = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mip_count = 0;
  // Bytes per 4x4 block for block-compressed formats, else bytes per texel.
  uint32_t bytes_per_block = 0;
  bool block_compressed = false;
  std::vector<uint64_t> mip_offsets;
  std::vector<uint8_t> data;

  uint32_t MipWidth(uint32_t mip) const { return std::max(width >> mip, uint32_t(1)); }
  uint32_t MipHeight(uint32_t mip) const { return std::max(height >> mip, uint32_t(1)); }
  // Number of rows of blocks (or texels) and bytes per row of a mip.
  uint32_t MipRows(uint32_t mip) const;
  uint32_t MipRowBytes(uint32_t mip) const;
};

// Reads a DDS file (legacy or DX10 header, 2D, no arrays or cubes). sRGB
// formats are read as their UNORM versions since guest textures are sampled
// without sRGB conversion.
bool LoadDds(const std::filesystem::path& path, DdsImage& image_out);

class TextureReplacer {
 public:
  TextureReplacer();
  ~TextureReplacer();

  // Reads the texture_pack_root, texture_dump cvars and indexes replace/.
  void Initialize();
  void Shutdown();

  // Whether hashing loaded textures is useful at all (dumping on, or a pack
  // with at least one replacement).
  bool IsActive() const {
    return dump_enabled_ || !replacements_.empty() || !highlights_.empty();
  }
  bool dump_enabled() const { return dump_enabled_; }

  // Untiles the base level into an internal buffer and returns its hash.
  // Returns false if the texture can't be hashed (format, size, bounds).
  bool Hash(const GuestTexture& texture, uint64_t& hash_out);
  // Writes the texture last passed to Hash to dump/ once per hash (on a
  // worker thread). No-op if dumping is off.
  void DumpLastHashed(uint64_t hash);

  // Replacement file for a hash, or an empty path.
  std::filesystem::path FindReplacement(uint64_t hash) const;

  // Starts reading the replacement for a hash on a background thread.
  // Returns false if the pack has no replacement for it.
  bool RequestLoad(uint64_t hash);
  struct LoadedImage {
    uint64_t hash;
    std::filesystem::path path;
    // Null if the file couldn't be read.
    std::unique_ptr<DdsImage> image;
    // The file is a keyed (.edds) replacement that did not open with this
    // texture's key: made from different game data.
    bool bad_key = false;
  };
  // Moves out the loads finished since the last call.
  void CollectLoaded(std::vector<LoadedImage>& out);
  // Whether the hash is in the texture_highlight list (see the cvar).
  bool IsHighlighted(uint64_t hash) const;

 private:
  struct DumpJob;
  class DumpWorker;
  class LoadWorker;

  bool dump_enabled_ = false;
  std::filesystem::path dump_dir_;
  std::unordered_map<uint64_t, std::filesystem::path> replacements_;
  std::unordered_set<uint64_t> dumped_;
  // texture_highlight: upper-case hex hash prefixes.
  std::vector<std::string> highlights_;

  // Last Hash() input, untiled.
  GuestTexture last_texture_{};
  std::vector<uint8_t> last_untiled_;

  std::unique_ptr<DumpWorker> dump_worker_;
  std::unique_ptr<LoadWorker> load_worker_;
};

}  // namespace rex::graphics::texture_replacement
