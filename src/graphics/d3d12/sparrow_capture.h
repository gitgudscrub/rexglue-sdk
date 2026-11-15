// Project Sparrow scene capture: saves every draw of one frame (shaders, the
// context registers, shader and fetch constants, vertex layout, bound texture
// hashes, and the vertex and index data read from guest memory) so that
// tools/sparrow can rebuild the scene outside the emulated renderer.
//
// Set the sparrow_capture cvar to 1 (it can change while running); the capture
// starts at the next swap, covers one whole frame, and is written to
// <sparrow_capture_dir>/<frame>/ as capture.json + blob.bin. The cvar goes back
// to 0 when the files are written.

#pragma once

#include <cstdint>

#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/xenos.h>
#include <rex/memory.h>

namespace rex::graphics::d3d12::sparrow_capture {

// True while a frame is being captured, so the caller can skip gathering what
// OnDraw needs otherwise.
bool IsCapturing();

struct DrawIndexInfo {
  bool indexed = false;
  xenos::IndexFormat format = xenos::IndexFormat::kInt16;
  xenos::Endian endianness = xenos::Endian::kNone;
  uint32_t guest_base = 0;
  uint32_t length = 0;
};

// Records one draw. texture_hashes holds the content hash bound to each texture
// fetch constant in used_texture_mask (others are ignored).
void OnDraw(const RegisterFile& regs, memory::Memory& memory, const Shader& vertex_shader,
            const Shader* pixel_shader, xenos::PrimitiveType primitive_type,
            uint32_t index_count, const DrawIndexInfo& index_info, uint32_t used_texture_mask,
            const uint64_t* texture_hashes, uint32_t draw_resolution_scale_x,
            uint32_t draw_resolution_scale_y);

// Called on every swap: starts a capture that was asked for, or finishes and
// writes the one in progress.
void OnSwap(uint64_t frame);

}  // namespace rex::graphics::d3d12::sparrow_capture
