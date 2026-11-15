// Project Sparrow: sky visibility and bounce light for upward-facing ground,
// once a frame on a coarse grid, so the ground materials read one value
// instead of searching the map per pixel. Runs right after
// sparrow_sun.cs.hlsl in D3D12CommandProcessor::CaptureSceneDepth, with the
// same bindings and constants: the sun map is t2, the Sparrow texture u0.
//
// The grid is G = size / 8 cells a side, one per 8 x 8 map texels, in the
// rows size .. size + G - 1 of the Sparrow texture. A cell stands for the
// line along the sun through its centre: every world point the material
// shaders map to that cell lies on it, whatever its height. The ground
// itself is usually not in the map (the cobbles, terrain layers and splats
// are blended colour-pass layers, not in the depth pre-pass the map
// replays), but the occluders around are, and the samples around the cell
// (map-plane offsets from its centre) are the same for every point on the
// line. So the sky visibility is worked out at four heights spanning the
// occluders found, and the material interpolates at its own height:
//   columns 0G .. 4G-1   sky visibility (0 closed in .. 1 open) at heights
//                        lerp(lo, hi, i / 3), i = 0..3;
//   columns 4G, 5G       lo and hi (world z);
//   columns 6G .. 7G-1   bounce light per unit sun colour at the second
//                        height (times the bounce albedo in the material).
// The search is the one SparrowSkyVisibility / SparrowBounce do per pixel in
// shaders/sparrow_common.hlsli (Fable-2-Recomp); keep the two in step.

cbuffer SparrowSunConstants : register(b0) {
  uint size;
  uint frame;
  uint debug;
  uint grid;
  float4 centre_radius;  // xyz centre, w signed half width
  float4 depth_size;     // half depth, size, valid, bias texels
  float4 sun_dir;        // xyz toward the sun
  uint4 pad1[9];
};

#if MSAA
Texture2DMS<float> sun_map : register(t2);
float LoadMap(int2 p) { return sun_map.Load(p, 0); }
#else
Texture2D<float> sun_map : register(t2);
float LoadMap(int2 p) { return sun_map[p]; }
#endif
RWTexture2D<float> sparrow_out : register(u0);

static float3 g_centre, g_l, g_right, g_up;
static float g_radius, g_depth_half;

// The sunlit point at map-plane position m (world units along right / up
// from the centre), bilinear over the texels that have something drawn.
bool SunlitPoint(float2 m, out float3 hit, out float along) {
  hit = 0.0;
  along = 0.0;
  float2 n = m / g_radius;
  if (max(abs(n.x), abs(n.y)) >= 1.0) return false;
  float fsize = float(size);
  float2 uv = float2(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5) * fsize - 0.5;
  int2 b = int2(floor(uv));
  float2 fr = uv - float2(b);
  int2 hi = int2(int(size) - 1, int(size) - 1);
  float d00 = LoadMap(clamp(b, 0, hi)), d10 = LoadMap(clamp(b + int2(1, 0), 0, hi));
  float d01 = LoadMap(clamp(b + int2(0, 1), 0, hi)), d11 = LoadMap(clamp(b + int2(1, 1), 0, hi));
  float4 w4 = float4((1.0 - fr.x) * (1.0 - fr.y), fr.x * (1.0 - fr.y), (1.0 - fr.x) * fr.y, fr.x * fr.y);
  w4 *= float4(d00 > 0.0, d10 > 0.0, d01 > 0.0, d11 > 0.0);
  float wsum = dot(w4, 1.0);
  if (wsum <= 1e-4) return false;
  float d = dot(w4, float4(d00, d10, d01, d11)) / wsum;
  along = (d - 0.5) * 2.0 * g_depth_half;
  hit = g_centre + g_right * m.x + g_up * m.y + g_l * along;
  return true;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint g = size / 8u;
  if (id.x >= g || id.y >= g) {
    return;
  }
  int row = int(size + id.y);
  g_centre = centre_radius.xyz;
  g_radius = abs(centre_radius.w);
  g_depth_half = depth_size.x;
  g_l = sun_dir.xyz;
  float3 ref = abs(g_l.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
  g_right = normalize(cross(ref, g_l));
  g_up = cross(g_l, g_right);
  if (centre_radius.w < 0.0) g_up = -g_up;

  // The cell centre on the map plane (texel rows run down the up axis).
  float2 t = (float2(id.xy) + 0.5) * 8.0 / float(size);
  float2 m = float2(t.x * 2.0 - 1.0, 1.0 - t.y * 2.0) * g_radius;
  float3 base = g_centre + g_right * m.x + g_up * m.y;  // on the line, map plane

  const float radii[3] = {0.6, 1.8, 5.0};
  // The nearest ring counts less: with eight directions a thin pole or a
  // person would block a whole 45-degree slice.
  const float ring_weight[3] = {0.5, 1.0, 1.0};
  float3 hits[24];
  bool hit_ok[24];
  float zmin = 1e9, zmax = -1e9;
  float3 hit;
  float hit_along;
  bool valid = g_radius > 0.0 && g_depth_half > 0.0 && g_l.z > 0.05;
  if (valid && SunlitPoint(m, hit, hit_along)) {
    zmin = hit.z;
    zmax = hit.z;
  }
  [unroll] for (int k = 0; k < 8; ++k) {
    float a = (float(k) + 0.5) * (6.2831853 / 8.0);
    float2 dir = float2(cos(a), sin(a));
    [unroll] for (int j = 0; j < 3; ++j) {
      bool ok = valid && SunlitPoint(m + dir * radii[j], hit, hit_along);
      hits[k * 3 + j] = hit;
      hit_ok[k * 3 + j] = ok;
      if (ok) {
        zmin = min(zmin, hit.z);
        zmax = max(zmax, hit.z);
      }
    }
  }
  float lo = 0.0, hi = 1.0;
  float vis[4] = {1.0, 1.0, 1.0, 1.0};
  float bounce = 0.0;
  if (zmax >= zmin) {
    lo = zmin - 2.0;
    hi = max(zmax, lo + 0.5);
    [unroll] for (int i = 0; i < 4; ++i) {
      float h = lerp(lo, hi, float(i) / 3.0);
      // The point on the line at height h, searched from half a metre up.
      float3 origin = base + g_l * ((h - base.z) / g_l.z) + float3(0.0, 0.0, 0.5);
      float open = 0.0;
      [unroll] for (int k = 0; k < 8; ++k) {
        float sin_h = 0.0;
        [unroll] for (int j = 0; j < 3; ++j) {
          if (hit_ok[k * 3 + j]) {
            float3 w = hits[k * 3 + j] - origin;
            float len = length(w);
            if (len > 0.05) sin_h = max(sin_h, ring_weight[j] * w.z / len);
          }
        }
        open += 1.0 - sin_h * sin_h;
      }
      vis[i] = open / 8.0;
    }
    // Bounce at the second height: 16 points on a golden-angle spiral
    // within 6 m, one bounce, Lambert, an upward-facing receiver.
    const float kRadius = 6.0;
    float3 pos = base + g_l * ((lerp(lo, hi, 1.0 / 3.0) - base.z) / g_l.z);
    float sum = 0.0;
    [loop] for (int s = 0; s < 16; ++s) {
      float r = kRadius * sqrt((float(s) + 0.5) / 16.0);
      float b = float(s) * 2.3999632;
      if (!SunlitPoint(m + float2(cos(b), sin(b)) * r, hit, hit_along)) continue;
      float3 w = hit - pos;
      float d2 = max(dot(w, w), 0.25);
      w *= rsqrt(d2);
      sum += saturate(w.z) * saturate(dot(-w, g_l) + 0.3) / d2;
    }
    bounce = sum * (3.14159265 * kRadius * kRadius / 16.0) / 3.14159265;
  }
  [unroll] for (int i = 0; i < 4; ++i) {
    sparrow_out[int2(uint(i) * g + id.x, row)] = vis[i];
  }
  sparrow_out[int2(4u * g + id.x, row)] = lo;
  sparrow_out[int2(5u * g + id.x, row)] = hi;
  sparrow_out[int2(6u * g + id.x, row)] = bounce;
}
