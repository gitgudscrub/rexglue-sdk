#include "sparrow_capture.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/logging.h>

REXCVAR_DEFINE_INT32(sparrow_capture, 0, "GPU/D3D12",
                     "Project Sparrow: set to 1 to save every draw of the next frame (shaders, "
                     "registers, constants, vertex and index data, texture hashes) to "
                     "<sparrow_capture_dir>/<frame>/ for tools/sparrow; goes back to 0 when done")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_STRING(sparrow_capture_dir, "sparrow_capture", "GPU/D3D12",
                      "Project Sparrow: folder that scene captures are written to")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics::d3d12::sparrow_capture {

namespace {

// Register ranges saved with every draw: the context registers (render
// targets, viewport, depth, blend, ...) and the shader constants (float, fetch,
// bool and loop).
constexpr uint32_t kContextBase = 0x2000;
constexpr uint32_t kContextCount = 0x400;
constexpr uint32_t kConstantBase = 0x4000;
constexpr uint32_t kConstantCount = 0x928;
// Larger buffers are skipped (recorded with no data).
constexpr uint32_t kMaxBufferBytes = 32u << 20;

enum class State { kIdle, kCapturing };

struct Capture {
  State state = State::kIdle;
  uint64_t frame = 0;
  uint32_t draws = 0;
  std::vector<uint8_t> blob;
  std::string json;
  // (guest address, size) -> blob offsets of the copies saved so far, so an
  // unchanged buffer drawn many times is stored once.
  std::unordered_map<uint64_t, std::vector<uint64_t>> buffers;
};
Capture capture;

uint64_t AddBlob(const void* data, size_t size) {
  uint64_t offset = capture.blob.size();
  // Keep everything 16-byte aligned for easy reading.
  capture.blob.resize(offset + ((size + 15) & ~size_t(15)));
  std::memcpy(capture.blob.data() + offset, data, size);
  return offset;
}

uint64_t AddBuffer(const uint8_t* data, uint32_t guest_address, uint32_t size) {
  uint64_t key = (uint64_t(guest_address) << 32) | size;
  auto& copies = capture.buffers[key];
  for (uint64_t offset : copies) {
    if (std::memcmp(capture.blob.data() + offset, data, size) == 0) {
      return offset;
    }
  }
  uint64_t offset = AddBlob(data, size);
  copies.push_back(offset);
  return offset;
}

void Finish() {
  std::filesystem::path dir =
      std::filesystem::path(REXCVAR_GET(sparrow_capture_dir)) / fmt::format("{}", capture.frame);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  bool ok = !ec;
  if (ok) {
    std::ofstream blob(dir / "blob.bin", std::ios::binary);
    blob.write(reinterpret_cast<const char*>(capture.blob.data()),
               std::streamsize(capture.blob.size()));
    ok = bool(blob);
  }
  if (ok) {
    std::ofstream json(dir / "capture.json", std::ios::binary);
    json << fmt::format(
        "{{\"version\":1,\"frame\":{},\"ctx_base\":{},\"ctx_count\":{},\"const_base\":{},"
        "\"const_count\":{},\"draws\":[\n",
        capture.frame, kContextBase, kContextCount, kConstantBase, kConstantCount);
    json << capture.json << "\n]}\n";
    ok = bool(json);
  }
  if (ok) {
    REXGPU_INFO("[sparrow-capture] frame {}: {} draws, {} MB, written to {}", capture.frame,
                capture.draws, capture.blob.size() >> 20, dir.string());
  } else {
    REXGPU_ERROR("[sparrow-capture] could not write {}", dir.string());
  }
  capture = {};
}

}  // namespace

bool IsCapturing() { return capture.state == State::kCapturing; }

void OnDraw(const RegisterFile& regs, memory::Memory& memory, const Shader& vertex_shader,
            const Shader* pixel_shader, xenos::PrimitiveType primitive_type,
            uint32_t index_count, const DrawIndexInfo& index_info, uint32_t used_texture_mask,
            const uint64_t* texture_hashes, uint32_t draw_resolution_scale_x,
            uint32_t draw_resolution_scale_y) {
  if (capture.state != State::kCapturing) {
    return;
  }
  std::string& j = capture.json;
  if (capture.draws) {
    j += ",\n";
  }
  uint64_t ctx = AddBlob(&regs.values[kContextBase], kContextCount * sizeof(uint32_t));
  uint64_t consts = AddBlob(&regs.values[kConstantBase], kConstantCount * sizeof(uint32_t));
  j += fmt::format(
      "{{\"i\":{},\"vs\":\"{:016X}\",\"ps\":\"{:016X}\",\"prim\":{},\"count\":{},"
      "\"res_scale\":[{},{}],\"ctx\":{},\"consts\":{},\"index\":",
      capture.draws, vertex_shader.ucode_data_hash(),
      pixel_shader ? pixel_shader->ucode_data_hash() : uint64_t(0), uint32_t(primitive_type),
      index_count, draw_resolution_scale_x, draw_resolution_scale_y, ctx, consts);
  if (index_info.indexed && index_info.length && index_info.length <= kMaxBufferBytes) {
    const uint8_t* data = memory.TranslatePhysical<const uint8_t*>(index_info.guest_base);
    uint64_t offset = AddBuffer(data, index_info.guest_base, index_info.length);
    j += fmt::format("{{\"fmt\":{},\"endian\":{},\"base\":{},\"len\":{},\"blob\":{}}}",
                     uint32_t(index_info.format), uint32_t(index_info.endianness),
                     index_info.guest_base, index_info.length, offset);
  } else {
    j += "null";
  }
  j += ",\"vb\":[";
  bool first_binding = true;
  for (const Shader::VertexBinding& binding : vertex_shader.vertex_bindings()) {
    xenos::xe_gpu_vertex_fetch_t fetch = regs.GetVertexFetch(binding.fetch_constant);
    uint32_t address = fetch.address << 2;
    uint32_t size = fetch.size << 2;
    std::string blob = "null";
    if (fetch.type == xenos::FetchConstantType::kVertex && size && size <= kMaxBufferBytes) {
      const uint8_t* data = memory.TranslatePhysical<const uint8_t*>(address);
      blob = fmt::format("{}", AddBuffer(data, address, size));
    }
    j += fmt::format(
        "{}{{\"fc\":{},\"stride\":{},\"addr\":{},\"size\":{},\"endian\":{},\"blob\":{},\"attrs\":[",
        first_binding ? "" : ",", binding.fetch_constant, binding.stride_words, address, size,
        uint32_t(fetch.endian), blob);
    first_binding = false;
    bool first_attr = true;
    for (const Shader::VertexBinding::Attribute& attr : binding.attributes) {
      const auto& a = attr.fetch_instr.attributes;
      j += fmt::format(
          "{}{{\"fmt\":{},\"off\":{},\"signed\":{},\"int\":{},\"exp\":{},\"rf\":{}}}",
          first_attr ? "" : ",", uint32_t(a.data_format), a.offset, a.is_signed ? 1 : 0,
          a.is_integer ? 1 : 0, a.exp_adjust, uint32_t(a.signed_rf_mode));
      first_attr = false;
    }
    j += "]}";
  }
  // Raw guest memory of textures the tools can't get from the texture dump:
  // everything the vertex shader reads (terrain heights, lookup tables) as
  // "vtex", and pixel shader textures with no content hash (render-to-texture
  // results such as the baked terrain colour; run with readback_resolve=full
  // so they are in guest memory) as "ptex". Base level of 2D textures only, as
  // stored (tiled or linear).
  auto add_texture = [&](uint32_t fetch_constant, bool& first) {
    xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(fetch_constant);
    const FormatInfo* format = FormatInfo::Get(uint32_t(fetch.format));
    if (fetch.type != xenos::FetchConstantType::kTexture || !format || !format->block_width) {
      return;
    }
    uint32_t height = fetch.size_2d.height + 1;
    uint32_t block_rows = (height + format->block_height - 1) / format->block_height;
    if (fetch.tiled) {
      block_rows = (block_rows + 31) & ~uint32_t(31);
    }
    uint32_t pitch_blocks = (fetch.pitch << 5) / format->block_width;
    uint32_t address = fetch.base_address << 12;
    uint32_t size = pitch_blocks * block_rows * format->bytes_per_block();
    if (!size || size > kMaxBufferBytes) {
      return;
    }
    const uint8_t* data = memory.TranslatePhysical<const uint8_t*>(address);
    j += fmt::format("{}{{\"fc\":{},\"addr\":{},\"size\":{},\"blob\":{}}}", first ? "" : ",",
                     fetch_constant, address, size, AddBuffer(data, address, size));
    first = false;
  };
  j += "],\"vtex\":[";
  bool first_vtex = true;
  for (const Shader::TextureBinding& binding : vertex_shader.texture_bindings()) {
    add_texture(binding.fetch_constant, first_vtex);
  }
  j += "],\"ptex\":[";
  bool first_ptex = true;
  for (uint32_t i = 0; i < 32; ++i) {
    if ((used_texture_mask & (uint32_t(1) << i)) && !texture_hashes[i]) {
      bool in_vs = false;
      for (const Shader::TextureBinding& binding : vertex_shader.texture_bindings()) {
        in_vs |= binding.fetch_constant == i;
      }
      if (!in_vs) {
        add_texture(i, first_ptex);
      }
    }
  }
  j += "],\"tex\":[";
  bool first_texture = true;
  for (uint32_t i = 0; i < 32; ++i) {
    if (used_texture_mask & (uint32_t(1) << i)) {
      j += fmt::format("{}[{},\"{:016X}\"]", first_texture ? "" : ",", i, texture_hashes[i]);
      first_texture = false;
    }
  }
  j += "]}";
  ++capture.draws;
}

void OnSwap(uint64_t frame) {
  if (capture.state == State::kCapturing) {
    Finish();
    REXCVAR_SET(sparrow_capture, 0);
    return;
  }
  if (REXCVAR_GET(sparrow_capture) > 0) {
    capture = {};
    capture.state = State::kCapturing;
    capture.frame = frame;
    capture.blob.reserve(64u << 20);
    REXGPU_INFO("[sparrow-capture] capturing frame {}", frame);
  }
}

}  // namespace rex::graphics::d3d12::sparrow_capture
