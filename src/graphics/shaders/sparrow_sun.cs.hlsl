// Project Sparrow: hands the sun shadow map (post_sun_map) to the material
// shaders. Runs in D3D12CommandProcessor::CaptureSceneDepth, after the
// game's depth pre-pass and before its colour pass, on the post-grade root
// signature: the map is t2 (kMean), the Sparrow texture u0 (kDestination).
//
// The Sparrow texture is R32_FLOAT, size x (size + size / 8 + 1):
//   rows 0 .. size-1  the map's depth (sample 0 when it is multisampled),
//                     reversed: larger is nearer the sun, 0 = nothing;
//   rows size ..      the ground ambient grid, size / 8 rows
//                     (sparrow_ambient.cs.hlsl writes them);
//   last row          frame data, one float per texel (SparrowFrame in
//                     shaders/sparrow_common.hlsli reads it):
//                     0-2 map centre (world), 3 half width (negative: up
//                     axis flipped), 4 half depth along the sun, 5 size,
//                     6 valid (1), 7 bias in texels, 8-10 world direction
//                     toward the sun, 11 frame counter (wraps at 2^24),
//                     12 debug mode (sparrow_debug), 13 ambient grid
//                     written this frame (1),
//                     16-31 camera world -> clip rows (last frame's),
//                     32-35 alpha, beta (clip z = alpha * w + beta),
//                     depth z scale, z offset, 36-39 depth texture
//                     width, height, y down (1), camera valid (1).
// Materials bind it through the bindless table at a fixed index
// (SystemBindlessView::kSparrowSun).

cbuffer SparrowSunConstants : register(b0) {
  uint size;
  uint frame;
  uint debug;
  uint grid;  // 1 when sparrow_ambient.cs.hlsl writes the grid this frame
  float4 centre_radius;  // xyz centre, w signed half width
  float4 depth_size;     // half depth, size, valid, bias texels
  float4 sun_dir;        // xyz toward the sun
  float4 cam[4];         // world -> clip rows
  float4 cam2;           // alpha, beta, z scale, z offset
  float4 cam3;           // depth width, height, y down, valid
  uint4 pad1[3];         // 52 root constants in all, as the post passes
};

#if MSAA
Texture2DMS<float> sun_map : register(t2);
float LoadMap(int2 p) { return sun_map.Load(p, 0); }
#else
Texture2D<float> sun_map : register(t2);
float LoadMap(int2 p) { return sun_map[p]; }
#endif
RWTexture2D<float> sparrow_out : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= size || id.y > size) {
    return;
  }
  if (id.y < size) {
    sparrow_out[id.xy] = LoadMap(int2(id.xy));
    return;
  }
  // The frame row.
  float v = 0.0;
  switch (id.x) {
    case 0: v = centre_radius.x; break;
    case 1: v = centre_radius.y; break;
    case 2: v = centre_radius.z; break;
    case 3: v = centre_radius.w; break;
    case 4: v = depth_size.x; break;
    case 5: v = depth_size.y; break;
    case 6: v = depth_size.z; break;
    case 7: v = depth_size.w; break;
    case 8: v = sun_dir.x; break;
    case 9: v = sun_dir.y; break;
    case 10: v = sun_dir.z; break;
    case 11: v = float(frame & 0xFFFFFFu); break;
    case 12: v = float(debug); break;
    case 13: v = float(grid); break;
    default:
      if (id.x >= 16u && id.x < 32u) {
        v = cam[(id.x - 16u) / 4u][(id.x - 16u) % 4u];
      } else if (id.x >= 32u && id.x < 36u) {
        v = cam2[id.x - 32u];
      } else if (id.x >= 36u && id.x < 40u) {
        v = cam3[id.x - 36u];
      }
      break;
  }
  sparrow_out[int2(id.x, size + size / 8u)] = v;
}
