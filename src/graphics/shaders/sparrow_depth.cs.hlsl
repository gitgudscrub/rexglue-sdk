// Project Sparrow: copies the scene depth at the end of the opaque scene
// into an R32_FLOAT texture (sample 0 when multisampled) that the material
// shaders read for the lamp shadows (SystemBindlessView::kSparrowDepth,
// shaders/sparrow_common.hlsli SparrowLampShadow). Runs in
// D3D12CommandProcessor::SparrowCopySceneDepth on the post-grade root
// signature: the depth is t0 (kSource), the copy u0 (kDestination). The
// game draws the screen in EDRAM tiles, each from the target's top: rows
// src_y.. of the target go to rows dst_y.. of the copy.

cbuffer SparrowDepthConstants : register(b0) {
  uint2 depth_size;  // width, rows to copy
  uint src_y;
  uint dst_y;
  float depth_scale;  // 2 for 20e4 depth: the host keeps [0, 2) as [0, 1)
  uint3 pad0;
  uint4 pad1[11];  // 52 root constants in all, as the post passes
};

#if MSAA
Texture2DMS<float> scene_depth : register(t0);
float LoadDepth(int2 p) { return scene_depth.Load(p, 0); }
#else
Texture2D<float> scene_depth : register(t0);
float LoadDepth(int2 p) { return scene_depth[p]; }
#endif
RWTexture2D<float> depth_out : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= depth_size)) {
    return;
  }
  depth_out[uint2(id.x, id.y + dst_y)] = LoadDepth(int2(id.x, id.y + src_y)) * depth_scale;
}
