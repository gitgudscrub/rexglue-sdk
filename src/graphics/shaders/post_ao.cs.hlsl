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
  // Sun shadows (see SunShadow below).
  uint sun_enabled;
  float sun_strength;   // darkening of a shadowed pixel
  float sun_length;     // march length toward the sun, world units
  float sun_thickness;  // assumed depth of surfaces, world units
  uint sun_y_down;     // 1: clip +y is down the screen
  row_major float4x4 inv_view_proj;  // the game's clip -> world
  float4 sun_clip;      // clip-space change per world unit toward the sun
  float4 sun_world;     // xyz: world direction toward the sun; w: world-space
                        // AO radius in world units (0 = near-plane AO below)
  // Sun shadow map (see SunMapShadow): centre xyz, half width w (negative:
  // up axis flipped); half depth along the sun, size, bias texels, mode
  // (0 off, 1 on, 2 debug coverage).
  float4 sun_map;
  float4 sun_map2;
};

// The game's scene depth is multisampled; the MSAA variant (fxc /D MSAA=1)
// reads sample 0.
#if MSAA
Texture2DMS<float> depth_in : register(t0);
Texture2DMS<float> sun_map_tex : register(t2);
float LoadSunMap(int2 p) { return sun_map_tex.Load(p, 0); }
float LoadDepth(int2 p) { return depth_in.Load(p, 0); }
#else
Texture2D<float> depth_in : register(t0);
Texture2D<float> sun_map_tex : register(t2);
float LoadSunMap(int2 p) { return sun_map_tex[p]; }
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

// --- sun shadows ------------------------------------------------------------
// The game's own sun shadow map covers only nearby objects and the ground has
// only its baked light map, so rocks, roots, trunks and walls cast no sharp
// shadow on it. With the game's camera matrix (captured from the ground draws)
// the depth buffer gives exact world positions: march from the pixel toward
// the sun and look for something in front of the ray.
float2 NdcOf(float2 p) {
  float y = p.y / float(depth_size.y) * 2.0 - 1.0;
  return float2(p.x / float(depth_size.x) * 2.0 - 1.0, sun_y_down != 0 ? y : -y);
}
float4 WorldH(float2 ndc, float d) { return mul(inv_view_proj, float4(ndc, d, 1.0)); }
float3 WorldPos(int2 p) {
  p = clamp(p, int2(0, 0), int2(depth_size) - 1);
  float4 h = WorldH(NdcOf(float2(p) + 0.5), LoadDepth(p));
  return h.xyz / h.w;
}

float SunShadow(int2 p, float d, float noise) {
  float2 ndc0 = NdcOf(float2(p) + 0.5);
  float4 h = WorldH(ndc0, d);
  float3 pos = h.xyz / h.w;
  float w0 = 1.0 / h.w;  // the pixel's clip w (its view depth, world units)
  // Surface normal from the neighbours, turned to face the camera.
  float3 dx = WorldPos(p + int2(1, 0)) - pos, dx2 = pos - WorldPos(p - int2(1, 0));
  float3 dy = WorldPos(p + int2(0, 1)) - pos, dy2 = pos - WorldPos(p - int2(0, 1));
  if (dot(dx2, dx2) < dot(dx, dx)) dx = dx2;
  if (dot(dy2, dy2) < dot(dy, dy)) dy = dy2;
  float3 n = normalize(cross(dy, dx));
  float4 hn = WorldH(ndc0, reversed_z != 0 ? 1.0 : 0.0);  // same pixel on the near plane
  if (dot(n, hn.xyz / hn.w - pos) < 0.0) n = -n;
  float ndl = dot(n, sun_world.xyz);
  if (sun_enabled == 2) return ndl * 0.5 + 0.5;               // debug: facing the sun
  if (sun_enabled == 3) return frac(pos.x / 4.0) * 0.5 + frac(pos.y / 4.0) * 0.5;  // debug: world grid
  if (sun_enabled == 4) return frac(pos.z / 4.0);
  if (ndl < 0.05) return 1.0;  // facing away: the game's lighting has it dark already
  float4 c0 = float4(ndc0, d, 1.0) * w0;
  const uint kSteps = 16;
  float shadow = 0.0;
  for (uint i = 0; i < kSteps; ++i) {
    float f = (float(i) + noise) / float(kSteps);
    float t = sun_length * (0.02 + f * f);  // denser close to the surface
    float4 c = c0 + sun_clip * t;
    if (c.w <= 0.0) break;
    float2 nq = c.xy / c.w;
    if (any(abs(nq) > 1.0)) break;
    float qy = sun_y_down != 0 ? nq.y : -nq.y;
    int2 q = int2((nq.x * 0.5 + 0.5) * float(depth_size.x), (qy * 0.5 + 0.5) * float(depth_size.y));
    float ds = LoadDepth(clamp(q, int2(0, 0), int2(depth_size) - 1));
    if (IsSky(ds)) continue;
    float ws = 1.0 / dot(inv_view_proj[3], float4(nq, ds, 1.0));
    float gap = c.w - ws;  // > 0: the depth buffer has something in front of the ray
    if (gap > 0.02 + 0.004 * c.w && gap < sun_thickness) {
      shadow = 1.0 - f * 0.5;  // far occluders give a lighter, less certain shadow
      break;
    }
  }
  // Screen-space rays get unreliable far away; the game's fog takes over.
  shadow *= 1.0 - smoothstep(sun_length * 12.0, sun_length * 25.0, w0);
  return 1.0 - sun_strength * shadow * saturate(ndl * 4.0);
}

// --- sun shadow map ---------------------------------------------------------
// D3D12CommandProcessor::ReplayDrawIntoSunMap draws the game's depth pre-pass
// a second time from the sun into a square orthographic map (reversed Z: the
// stored depth is larger toward the sun). Returns -1 outside the map, where
// the screen-space march takes over.
float SunMapShadow(int2 p, float d) {
  float2 ndc0 = NdcOf(float2(p) + 0.5);
  float4 h = WorldH(ndc0, d);
  float3 pos = h.xyz / h.w;
  float3 dx = WorldPos(p + int2(1, 0)) - pos, dx2 = pos - WorldPos(p - int2(1, 0));
  float3 dy = WorldPos(p + int2(0, 1)) - pos, dy2 = pos - WorldPos(p - int2(0, 1));
  if (dot(dx2, dx2) < dot(dx, dx)) dx = dx2;
  if (dot(dy2, dy2) < dot(dy, dy)) dy = dy2;
  float3 n = normalize(cross(dy, dx));
  float4 hn = WorldH(ndc0, reversed_z != 0 ? 1.0 : 0.0);
  if (dot(n, hn.xyz / hn.w - pos) < 0.0) n = -n;
  float3 l = sun_world.xyz;
  float ndl = dot(n, l);
  float3 ref = abs(l.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
  float3 right = normalize(cross(ref, l));
  float3 up = cross(l, right);
  float radius = abs(sun_map.w);
  if (sun_map.w < 0.0) up = -up;
  float depth_half = sun_map2.x;
  float size = sun_map2.y;
  float bias_texels = sun_map2.z;
  // Mode 1 on, 2 coverage, 3 the map itself.
  float mode = sun_map2.w;
  bool debug = mode > 1.5;
  if (mode > 2.5) {  // debug: the map itself stretched over the screen
    float m = LoadSunMap(int2(float2(p) / float2(depth_size) * size));
    return m <= 0.0 ? 0.0 : frac(m * 50.0) * 0.8 + 0.2;
  }
  float texel = 2.0 * radius / size;
  // Normal offset: step off the surface so it does not shadow itself.
  float3 v = pos + n * (texel * bias_texels) - sun_map.xyz;
  float x = dot(right, v) / radius, y = dot(up, v) / radius;
  float z = dot(l, v) / (2.0 * depth_half) + 0.5;
  float edge = max(abs(x), abs(y));
  if (edge >= 1.0 || z >= 1.0 || z <= 0.0) {
    return debug ? 0.5 : -1.0;  // debug: grey outside the map
  }
  if (ndl < 0.05 && !debug) return 1.0;
  float2 uv = float2(x * 0.5 + 0.5, 0.5 - y * 0.5) * size - 0.5;
  float z_bias = texel * bias_texels / (2.0 * depth_half);
  int2 base = int2(floor(uv));
  float2 f = uv - float2(base);
  // 4x4 taps, bilinear-weighted 3x3 PCF.
  float lit[4][4];
  [unroll] for (int j = 0; j < 4; ++j) {
    [unroll] for (int i = 0; i < 4; ++i) {
      int2 q = clamp(base + int2(i - 1, j - 1), int2(0, 0), int2(size, size) - 1);
      lit[j][i] = LoadSunMap(q) > z + z_bias ? 0.0 : 1.0;
    }
  }
  float sum = 0.0;
  [unroll] for (int j2 = 0; j2 < 3; ++j2) {
    [unroll] for (int i2 = 0; i2 < 3; ++i2) {
      float top = lerp(lit[j2][i2], lit[j2][i2 + 1], f.x);
      float bottom = lerp(lit[j2 + 1][i2], lit[j2 + 1][i2 + 1], f.x);
      sum += lerp(top, bottom, f.y);
    }
  }
  float shadow = 1.0 - sum / 9.0;
  if (debug) return 1.0 - shadow * 0.8;  // bright = lit, dark = shadowed
  float result = 1.0 - sun_strength * shadow * saturate(ndl * 4.0);
  // Blend into the screen-space march near the map's edge.
  float blend = smoothstep(0.85, 1.0, edge);
  if (blend > 0.0) {
    result = lerp(result, sun_enabled != 0 ? SunShadow(p, d, 0.5) : 1.0, blend);
  }
  return result;
}

// --- world-space ao -----------------------------------------------------------
// With the game's camera the radius is in world units, so a rock gets the same
// contact shading near and far (the near-plane AO can only reach a few pixels
// and puts dark halos round foreground edges). Samples on a screen spiral
// sized to the radius at the pixel's depth; each one occludes by how far it
// rises above the surface, falling off to zero at the radius, so things more
// than a radius away (a wall far behind a rock) add nothing.
static const uint kWorldSamples = 16;

float WorldAo(int2 p, float d, float noise, float radius_w) {
  float2 ndc0 = NdcOf(float2(p) + 0.5);
  float4 h = WorldH(ndc0, d);
  float3 pos = h.xyz / h.w;
  float w0 = 1.0 / h.w;
  // World size of one pixel at this depth (across the view, not the surface).
  float4 h1 = WorldH(NdcOf(float2(p) + float2(1.5, 0.5)), d);
  float px_w = length(h1.xyz / h1.w - pos);
  float radius_px = min(radius_w / max(px_w, 1e-6), max_radius_px);
  if (radius_px < 1.5) return 1.0;
  float3 dx = WorldPos(p + int2(1, 0)) - pos, dx2 = pos - WorldPos(p - int2(1, 0));
  float3 dy = WorldPos(p + int2(0, 1)) - pos, dy2 = pos - WorldPos(p - int2(0, 1));
  if (dot(dx2, dx2) < dot(dx, dx)) dx = dx2;
  if (dot(dy2, dy2) < dot(dy, dy)) dy = dy2;
  float3 n = normalize(cross(dy, dx));
  float4 hn = WorldH(ndc0, reversed_z != 0 ? 1.0 : 0.0);
  if (dot(n, hn.xyz / hn.w - pos) < 0.0) n = -n;
  float r2 = radius_w * radius_w;
  float b = bias * radius_w;
  float sum = 0.0;
  float angle0 = noise * 6.2831853;
  for (uint i = 0; i < kWorldSamples; ++i) {
    float t = (float(i) + 0.5) / float(kWorldSamples);
    float a = angle0 + float(i) * 2.3999632;
    float2 offset = float2(cos(a), sin(a)) * (radius_px * sqrt(t));
    int2 q = clamp(p + int2(round(offset)), int2(0, 0), int2(depth_size) - 1);
    float dq = LoadDepth(q);
    if (IsSky(dq)) continue;
    float3 v = WorldPos(q) - pos;
    float vv = dot(v, v);
    float falloff = saturate(1.0 - vv / r2);
    // cos of the elevation above the surface plane, less a small bias
    sum += saturate((dot(v, n) - b) * rsqrt(vv + 1e-6)) * falloff * falloff;
  }
  float occ = saturate(1.0 - intensity * 2.0 * sum / float(kWorldSamples));
  // Far away the radius is a few pixels and the game's fog takes over.
  return lerp(occ, 1.0, smoothstep(fade_distance * 0.5, fade_distance, w0));
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
  float noise = frac(52.9829189 * frac(dot(float2(id.xy), float2(0.06711056, 0.00583715))));
  float sun = sun_map2.w > 0.0 ? SunMapShadow(p, d) : -1.0;
  if (sun < 0.0) sun = sun_enabled != 0 ? SunShadow(p, d, noise) : 1.0;
  if (sun_world.w > 0.0) {
    ao_out[id.xy] = float2(WorldAo(p, d, noise, sun_world.w) * sun, ViewPos(p).z);
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
    ao_out[id.xy] = float2(sun, pos.z);
    return;
  }
  // Per-pixel rotation (interleaved gradient noise), removed by the blur.
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
  ao_out[id.xy] = float2(occlusion * sun, pos.z);
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
