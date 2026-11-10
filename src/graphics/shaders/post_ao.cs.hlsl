// Modern-graphics ambient occlusion. Runs when the game resolves its scene
// depth (D3D12RenderTargetCache::Resolve -> D3D12CommandProcessor::
// CaptureSceneDepth), reading the host depth render target directly, and
// leaves a half-resolution occlusion texture that the colour grade at swap
// (post_grade.cs.hlsl) multiplies into the frame. Three passes, one entry
// point each (fxc /E):
//
//   ao      host depth -> half-res (occlusion, view depth)      (t0 -> u0)
//   blur_h  depth-aware blur, horizontal                         (t1 -> u0)
//   blur_v  same, vertical                                       (t1 -> u0)
//
// The game's projection isn't known, so positions are rebuilt in units of
// the near plane (view z = 1 / (1 - depth), or 1 / depth for reversed Z) with
// an assumed field of view. Everything is a ratio of those units, so the
// near plane cancels out; the radius is tuned in the same units.

cbuffer PostAoConstants : register(b0) {
  uint2 depth_size;     // used region of the host depth texture
  uint2 ao_size;        // half of depth_size
  float radius;         // sampling radius, near-plane units
  float bias;           // self-occlusion bias, fraction of view depth
  float intensity;      // occlusion gain
  float tan_half_fov;   // assumed vertical tan(fov / 2)
  uint reversed_z;      // 1 if depth 0 is far
  float max_radius_px;  // cap on the projected radius (pixels of depth_size)
  float fade_distance;  // view depth where AO is gone (the game's fog has
                        // taken over by then); near-plane units
  uint pad1, pad2, pad3, pad4, pad5;
};

// The game's scene depth is multisampled; the MSAA variant (fxc /D MSAA=1)
// reads sample 0.
#if MSAA
Texture2DMS<float> depth_in : register(t0);
float LoadDepth(int2 p) { return depth_in.Load(p, 0); }
#else
Texture2D<float> depth_in : register(t0);
float LoadDepth(int2 p) { return depth_in[p]; }
#endif
Texture2D<float2> ao_in : register(t1);
RWTexture2D<float2> ao_out : register(u0);

bool IsSky(float d) { return reversed_z != 0 ? d <= 0.0 : d >= 1.0; }

float ViewZ(float d) {
  return reversed_z != 0 ? 1.0 / max(d, 1e-7) : 1.0 / max(1.0 - d, 1e-7);
}

float FocalPx() { return 0.5 * float(depth_size.y) / tan_half_fov; }

// View-space position of a full-res depth texel.
float3 ViewPos(int2 p) {
  p = clamp(p, int2(0, 0), int2(depth_size) - 1);
  float z = ViewZ(LoadDepth(p));
  float2 c = (float2(p) + 0.5) - 0.5 * float2(depth_size);
  return float3(c * (z / FocalPx()), z);
}

// --- ao ---------------------------------------------------------------------
static const uint kSamples = 12;

[numthreads(8, 8, 1)]
void ao(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= ao_size)) return;
  int2 p = int2(id.xy) * 2;
  float d = LoadDepth(clamp(p, int2(0, 0), int2(depth_size) - 1));
  if (IsSky(d)) {
    ao_out[id.xy] = float2(1.0, 60000.0);  // fits in a half float
    return;
  }
  float3 pos = ViewPos(p);
  // Normal from the smaller of the one-sided differences (keeps edges clean).
  float3 l = pos - ViewPos(p - int2(1, 0)), r = ViewPos(p + int2(1, 0)) - pos;
  float3 u = pos - ViewPos(p - int2(0, 1)), dn = ViewPos(p + int2(0, 1)) - pos;
  float3 dx = abs(l.z) < abs(r.z) ? l : r;
  float3 dy = abs(u.z) < abs(dn.z) ? u : dn;
  float3 n = normalize(cross(dy, dx));
  if (n.z > 0.0) n = -n;  // face the camera (view z points into the screen)

  float radius_px = min(radius * FocalPx() / pos.z, max_radius_px);
  if (radius_px < 1.0) {
    ao_out[id.xy] = float2(1.0, pos.z);
    return;
  }
  // Per-pixel rotation (interleaved gradient noise), removed by the blur.
  float noise = frac(52.9829189 * frac(dot(float2(id.xy), float2(0.06711056, 0.00583715))));
  float angle0 = noise * 6.2831853;
  float r2 = radius * radius;
  float sum = 0.0;
  for (uint i = 0; i < kSamples; ++i) {
    float t = (float(i) + 0.5) / float(kSamples);
    float a = angle0 + float(i) * 2.3999632;  // golden angle spiral
    float2 offset = float2(cos(a), sin(a)) * (radius_px * sqrt(t));
    float3 v = ViewPos(p + int2(round(offset))) - pos;
    float vv = dot(v, v);
    float falloff = saturate(1.0 - vv / r2);
    sum += max(0.0, dot(v, n) - bias * pos.z) / (vv + 0.01 * r2) * falloff;
  }
  float occlusion = saturate(1.0 - intensity * radius * sum / float(kSamples));
  occlusion = lerp(occlusion, 1.0, smoothstep(0.25 * fade_distance, fade_distance, pos.z));
  ao_out[id.xy] = float2(occlusion, pos.z);
}

// --- blur: 9 taps, weighted by view-depth similarity ------------------------
float2 BlurAlong(int2 p, int2 dir) {
  float2 center = ao_in[p];
  float sum = 0.0, wsum = 0.0;
  for (int i = -4; i <= 4; ++i) {
    float2 s = ao_in[clamp(p + dir * i, int2(0, 0), int2(ao_size) - 1)];
    float w = exp(-float(i * i) / 8.0) *
              exp(-abs(s.y - center.y) / (0.03 * center.y));
    sum += s.x * w;
    wsum += w;
  }
  return float2(sum / max(wsum, 1e-5), center.y);
}

[numthreads(8, 8, 1)]
void blur_h(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= ao_size)) return;
  ao_out[id.xy] = BlurAlong(int2(id.xy), int2(1, 0));
}

[numthreads(8, 8, 1)]
void blur_v(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= ao_size)) return;
  ao_out[id.xy] = BlurAlong(int2(id.xy), int2(0, 1));
}
