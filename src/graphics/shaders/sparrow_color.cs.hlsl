// Project Sparrow: copies the scene colour at the end of each EDRAM tile
// (with the depth, sparrow_depth.cs.hlsl) into an R16G16B16A16_FLOAT
// texture (samples averaged when multisampled) that the material shaders
// read for the reflections (SystemBindlessView::kSparrowColor,
// shaders/sparrow_common.hlsli SparrowReflection). The values are the
// game's target as written (its colour exponent bias included). Runs in
// D3D12CommandProcessor::SparrowCopySceneColor on the post-grade root
// signature: the colour is t0 (kSource), the copy u0 (kDestination).

cbuffer SparrowColorConstants : register(b0) {
  uint2 color_size;  // width, rows to copy
  uint src_y;
  uint dst_y;
  uint4 pad1[12];  // 52 root constants in all, as the post passes
};

#if MSAA
Texture2DMS<float4> scene_color : register(t0);
float3 LoadColor(int2 p) {
  uint w, h, n;
  scene_color.GetDimensions(w, h, n);
  float3 c = 0.0;
  for (uint i = 0; i < n; ++i) c += scene_color.Load(p, i).rgb;
  return c / float(max(n, 1u));
}
#else
Texture2D<float4> scene_color : register(t0);
float3 LoadColor(int2 p) { return scene_color[p].rgb; }
#endif
RWTexture2D<float4> color_out : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= color_size)) {
    return;
  }
  float3 c = LoadColor(int2(id.x, id.y + src_y));
  // NaN / Inf would spread through the reflections.
  c = (c == c && abs(c) < 65000.0) ? max(c, 0.0) : 0.0;
  color_out[uint2(id.x, id.y + dst_y)] = float4(c, 1.0);
}
