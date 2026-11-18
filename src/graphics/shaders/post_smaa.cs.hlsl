// Project Sparrow anti-aliasing: SMAA 1x (Jimenez et al., smaa/SMAA.hlsl,
// MIT licence in smaa/LICENSE.txt) in place of FXAA (post_smaa), on the
// colour grade's output. Three compute passes on the post-grade root
// signature, one entry point each (fxc /E):
//   edges   t0 colour            -> u0 edges   (R8G8)
//   weights t0 edges, t1 area, t2 search -> u0 weights (R8G8B8A8)
//   blend   t0 colour, t1 weights -> u0 output
// SMAA's vertex-shader offsets are worked out per pixel here; its "no edge"
// discards write zero (compute shaders cannot discard, and that also clears
// the edges texture where there is nothing).

cbuffer SmaaConstants : register(b0) {
  uint2 size;
  uint2 pad0;
  uint4 pad1[12];  // 52 root constants in all, as the post passes
};

SamplerState linear_clamp : register(s0);
SamplerState point_clamp : register(s1);

#define SMAA_RT_METRICS float4(1.0 / float(size.x), 1.0 / float(size.y), float(size.x), float(size.y))
#define SMAA_PRESET_HIGH 1
#define SMAA_CUSTOM_SL 1
#define SMAATexture2D(tex) Texture2D tex
#define SMAATexturePass2D(tex) tex
#define SMAASampleLevelZero(tex, coord) tex.SampleLevel(linear_clamp, coord, 0)
#define SMAASampleLevelZeroPoint(tex, coord) tex.SampleLevel(point_clamp, coord, 0)
#define SMAASampleLevelZeroOffset(tex, coord, offset) tex.SampleLevel(linear_clamp, coord, 0, offset)
#define SMAASample(tex, coord) tex.SampleLevel(linear_clamp, coord, 0)
#define SMAASamplePoint(tex, coord) tex.SampleLevel(point_clamp, coord, 0)
#define SMAASampleOffset(tex, coord, offset) tex.SampleLevel(linear_clamp, coord, 0, offset)
#define SMAA_FLATTEN [flatten]
#define SMAA_BRANCH [branch]
#define SMAATexture2DMS2(tex) Texture2DMS<float4, 2> tex
#define SMAALoad(tex, pos, sample) tex.Load(pos, sample)
#define SMAAGather(tex, coord) tex.Gather(linear_clamp, coord, 0)
#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 1
#define discard return float2(0.0, 0.0)
#include "smaa/SMAA.hlsl"
#undef discard

Texture2D tex0 : register(t0);
Texture2D tex1 : register(t1);
Texture2D tex2 : register(t2);
RWTexture2D<float4> dest : register(u0);

float2 TexCoord(uint2 id) { return (float2(id) + 0.5) / float2(size); }

[numthreads(8, 8, 1)]
void edges(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= size)) return;
  float2 uv = TexCoord(id.xy);
  float4 offset[3];
  SMAAEdgeDetectionVS(uv, offset);
  dest[id.xy] = float4(SMAAColorEdgeDetectionPS(uv, offset, tex0), 0.0, 0.0);
}

[numthreads(8, 8, 1)]
void weights(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= size)) return;
  float2 uv = TexCoord(id.xy);
  float2 pixcoord;
  float4 offset[3];
  SMAABlendingWeightCalculationVS(uv, pixcoord, offset);
  dest[id.xy] = SMAABlendingWeightCalculationPS(uv, pixcoord, offset, tex0, tex1, tex2, 0.0);
}

[numthreads(8, 8, 1)]
void blend(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= size)) return;
  float2 uv = TexCoord(id.xy);
  float4 offset;
  SMAANeighborhoodBlendingVS(uv, offset);
  dest[id.xy] = float4(SMAANeighborhoodBlendingPS(uv, offset, tex0, tex1).rgb, 1.0);
}
