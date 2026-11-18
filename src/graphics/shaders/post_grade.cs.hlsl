// Modern-graphics colour grade, applied to the guest output after the gamma
// ramp (D3D12 swap, see D3D12CommandProcessor::IssueSwap). Five passes, one
// entry point each (compiled separately with fxc /E):
//
//   down    full-res colour -> quarter-res luma           (u0 = luma)
//   blur_h  quarter-res luma -> blurred, horizontal         (t1 -> u0)
//   blur_v  same, vertical                                  (t1 -> u0)
//   mean    average luma of the frame, smoothed over time   (t1 -> u1)
//   grade   colour + blurred luma + mean -> output          (t0, t1, t2 -> u0)
//
// Optional bloom (bloom_strength > 0), run before the grade:
//
//   bloom_down    full-res colour -> quarter-res highlights   (t0 -> u0)
//   bloom_blur_h  quarter-res highlights, horizontal           (t1 -> u0)
//   bloom_blur_v  same, vertical                               (t1 -> u0)
//
// Optional light shafts (shafts_sun.z > 0), in the bloom textures instead:
//
//   shafts_mask    sky near the sun, quarter res          (t0, t3 -> u0)
//   shafts_coarse  radial blur toward the sun, long steps  (t1 -> u0)
//   shafts_fine    the same with short steps (fills gaps)   (t1 -> u0)
//
// Optional distance dehaze (haze_params.x > 0), after the mean:
//
//   haze           colour of the far, fogged scene, smoothed  (t0, t3 -> u1)
//
// The grade writes perceptual luma to alpha so the SDK's FXAA pass can run
// on its output (post_aa).
//
// The grade adds depth without changing the game's look: a slight dehaze,
// local contrast ("clarity") from the blurred luma, a gentle S-curve,
// vibrance for muted colours, a light 3x3 sharpen, and an exposure fix so
// the frame keeps its original average brightness. The maths matches the
// prototype used to tune it (scratch Python, not in the repo).

cbuffer PostGradeConstants : register(b0) {
  uint2 size;         // full-res size
  uint2 small_size;   // quarter-res size
  float dehaze;       // black floor pulled down (0..0.1)
  float clarity;      // local contrast amount
  float contrast;     // S-curve amount (0..1)
  float vibrance;     // muted-colour saturation boost
  float sharpen;      // 3x3 unsharp amount; negative: CAS of strength -sharpen
  float keep_exposure;  // 0..1, how much the average brightness is restored
  float mean_blend;   // per-frame blend of the brightness average (0..1)
  uint pass_param;    // blur: unused; mean: first frame flag
  float ao_strength;  // grade: ambient occlusion amount (0..1)
  uint ao_mode;       // grade: 0 no AO, 1 apply, 2 show AO, 3 show AO depth
  float bloom_threshold;  // highlight luma where bloom starts
  float bloom_strength;   // 0 = no bloom
  // Light shafts: sun position in frame UV (may lie off screen), strength
  // (0 = off), radius of the sky around the sun that feeds them (in frame
  // heights); ray length (fraction of the way to the sun), per-sample decay,
  // sky brightness threshold, glow kept on dark pixels.
  float4 shafts_sun;
  float4 shafts_params;
  // Distance dehaze: amount of the fog removed at full distance (0 = off),
  // view depth where it starts and where it is full (near-plane units, as in
  // the AO texture), debug view (1 = show the amount).
  float4 haze_params;
};

Texture2D<float4> source : register(t0);
Texture2D<float> luma_in : register(t1);
Texture2D<float> mean_in : register(t2);
// Half-res (occlusion, view depth) from post_ao.cs.hlsl, stretched over the
// frame (the game scales its 3D view to the full output).
Texture2D<float2> ao_tex : register(t3);
// Bloom: blur input (same slot as luma_in) and the result for the grade.
Texture2D<float4> bloom_in : register(t1);
Texture2D<float4> bloom_tex : register(t4);
RWTexture2D<float4> dest : register(u0);
RWTexture2D<float> mean_out : register(u1);
// Single-channel destination for the luma passes (same slot as dest; each
// entry point uses only one of them).
RWTexture2D<float> dest_luma : register(u0);
// Distance dehaze: the fog colour (1x1), written by `haze` (same slot as
// mean_out) and read by the grade.
RWTexture2D<float4> haze_out : register(u1);
Texture2D<float4> haze_tex : register(t5);
SamplerState linear_clamp : register(s0);

float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

float3 Dehaze(float3 c) { return saturate((c - dehaze) / (1.0 - dehaze)); }

// --- down: each thread averages a 4x4 block of the source ------------------
[numthreads(8, 8, 1)]
void down(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  float sum = 0.0;
  uint2 base = id.xy * 4;
  for (uint y = 0; y < 4; ++y) {
    for (uint x = 0; x < 4; ++x) {
      uint2 p = min(base + uint2(x, y), size - 1);
      sum += Luma(Dehaze(source[p].rgb));
    }
  }
  dest_luma[id.xy] = sum / 16.0;
}

// --- blur: 25-tap gaussian (sigma ~6 quarter-res texels = ~24 full-res) ----
static const int kRadius = 12;
float Gauss(int i) { return exp(-float(i * i) / (2.0 * 6.0 * 6.0)); }

float BlurAlong(int2 p, int2 dir) {
  float sum = 0.0, wsum = 0.0;
  for (int i = -kRadius; i <= kRadius; ++i) {
    int2 q = clamp(p + dir * i, int2(0, 0), int2(small_size) - 1);
    float w = Gauss(i);
    sum += luma_in[q] * w;
    wsum += w;
  }
  return sum / wsum;
}

[numthreads(8, 8, 1)]
void blur_h(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  dest_luma[id.xy] = BlurAlong(int2(id.xy), int2(1, 0));
}

[numthreads(8, 8, 1)]
void blur_v(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  dest_luma[id.xy] = BlurAlong(int2(id.xy), int2(0, 1));
}

// --- mean: one group sums the quarter-res luma ------------------------------
// Writes the mean dehazed luma, smoothed over frames so the exposure fix
// in the grade doesn't flicker.
groupshared float g_sum[256];

[numthreads(16, 16, 1)]
void mean(uint3 gtid : SV_GroupThreadID) {
  uint index = gtid.y * 16 + gtid.x;
  float sum = 0.0;
  uint count = 0;
  for (uint y = gtid.y; y < small_size.y; y += 16) {
    for (uint x = gtid.x; x < small_size.x; x += 16) {
      sum += luma_in[uint2(x, y)];
      ++count;
    }
  }
  g_sum[index] = sum;
  GroupMemoryBarrierWithGroupSync();
  for (uint s = 128; s > 0; s >>= 1) {
    if (index < s) g_sum[index] += g_sum[index + s];
    GroupMemoryBarrierWithGroupSync();
  }
  if (index == 0) {
    float m = g_sum[0] / float(small_size.x * small_size.y);
    float old = mean_out[uint2(0, 0)];
    mean_out[uint2(0, 0)] = pass_param != 0 ? m : lerp(old, m, mean_blend);
  }
}

// --- bloom -------------------------------------------------------------------
[numthreads(8, 8, 1)]
void bloom_down(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  float3 sum = 0.0;
  uint2 base = id.xy * 4;
  for (uint y = 0; y < 4; ++y) {
    for (uint x = 0; x < 4; ++x) {
      sum += source[min(base + uint2(x, y), size - 1)].rgb;
    }
  }
  float3 c = sum / 16.0;
  // Soft knee above the threshold, so only real highlights glow.
  float k = saturate((Luma(c) - bloom_threshold) / max(1.0 - bloom_threshold, 1e-3));
  dest[id.xy] = float4(c * (k * k), 1.0);
}

float3 BloomBlurAlong(int2 p, int2 dir) {
  float3 sum = 0.0;
  float wsum = 0.0;
  for (int i = -kRadius; i <= kRadius; ++i) {
    int2 q = clamp(p + dir * i, int2(0, 0), int2(small_size) - 1);
    float w = Gauss(i);
    sum += bloom_in[q].rgb * w;
    wsum += w;
  }
  return sum / wsum;
}

[numthreads(8, 8, 1)]
void bloom_blur_h(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  dest[id.xy] = float4(BloomBlurAlong(int2(id.xy), int2(1, 0)), 1.0);
}

[numthreads(8, 8, 1)]
void bloom_blur_v(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  dest[id.xy] = float4(BloomBlurAlong(int2(id.xy), int2(0, 1)), 1.0);
}

// --- light shafts ------------------------------------------------------------
// Bright sky seen around the sun, smeared along lines toward the sun: where
// leaves, branches or buildings cut the sky the smear leaves dark gaps, which
// read as beams of light through the trees.
static const uint kShaftSamples = 24;

[numthreads(8, 8, 1)]
void shafts_mask(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  float3 sum = 0.0;
  float sky = 0.0;
  uint2 base = id.xy * 4;
  for (uint y = 0; y < 4; ++y) {
    for (uint x = 0; x < 4; ++x) {
      uint2 q = min(base + uint2(x, y), size - 1);
      float2 quv = (float2(q) + 0.5) / float2(size);
      float is_sky = ao_tex.SampleLevel(linear_clamp, quv, 0.0).y > 50000.0 ? 1.0 : 0.0;
      sum += source[q].rgb * is_sky;
      sky += is_sky;
    }
  }
  float3 c = sum / 16.0;
  float2 uv = (float2(id.xy) + 0.5) / float2(small_size);
  float aspect = float(size.x) / float(size.y);
  float2 d = (uv - shafts_sun.xy) * float2(aspect, 1.0);
  float near_sun = saturate(1.0 - length(d) / max(shafts_sun.w, 1e-3));
  float k = saturate((Luma(c) - shafts_params.z) / max(1.0 - shafts_params.z, 1e-3));
  dest[id.xy] = float4(c * k * near_sun * near_sun, 1.0);
}

float3 ShaftBlur(uint2 id, float step_scale) {
  float2 uv = (float2(id) + 0.5) / float2(small_size);
  float2 delta = (shafts_sun.xy - uv) * shafts_params.x * step_scale / float(kShaftSamples);
  float3 sum = 0.0;
  float w = 1.0, wsum = 0.0;
  for (uint i = 0; i < kShaftSamples; ++i) {
    sum += bloom_in.SampleLevel(linear_clamp, uv + delta * float(i), 0.0).rgb * w;
    wsum += w;
    w *= shafts_params.y;
  }
  return sum / wsum;
}

[numthreads(8, 8, 1)]
void shafts_coarse(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  dest[id.xy] = float4(ShaftBlur(id.xy, 1.0), 1.0);
}

[numthreads(8, 8, 1)]
void shafts_fine(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= small_size)) return;
  dest[id.xy] = float4(ShaftBlur(id.xy, 1.0 / float(kShaftSamples)), 1.0);
}

// --- distance dehaze ---------------------------------------------------------
// The game's fog blends distant pixels toward one colour. Average the frame
// weighted toward far (non-sky) depth to find that colour, smoothed over
// frames; the grade then takes part of it back out by distance.
float HazeAmount(float z) {
  if (z > 50000.0) return 0.0;  // sky
  return haze_params.x * smoothstep(haze_params.y, haze_params.z, z);
}

groupshared float4 g_haze[256];

[numthreads(16, 16, 1)]
void haze(uint3 gtid : SV_GroupThreadID) {
  uint index = gtid.y * 16 + gtid.x;
  float4 sum = 0.0;
  // 64 x 36 samples over the frame.
  for (uint y = gtid.y; y < 36; y += 16) {
    for (uint x = gtid.x; x < 64; x += 16) {
      float2 uv = (float2(x, y) + 0.5) / float2(64.0, 36.0);
      float z = ao_tex.SampleLevel(linear_clamp, uv, 0.0).y;
      // The farthest band (past the full depth) is mostly fog colour.
      float w = z > 50000.0 ? 0.0 : smoothstep(haze_params.z, haze_params.z * 4.0, z);
      sum += float4(source[uint2(uv * float2(size))].rgb * w, w);
    }
  }
  g_haze[index] = sum;
  GroupMemoryBarrierWithGroupSync();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (index < stride) g_haze[index] += g_haze[index + stride];
    GroupMemoryBarrierWithGroupSync();
  }
  if (index == 0) {
    float4 s = g_haze[0];
    float4 old = haze_out[uint2(0, 0)];
    if (s.w < 1.0) {
      // Hardly anything far on screen: keep the last colour.
      haze_out[uint2(0, 0)] = pass_param != 0 ? float4(0.7, 0.75, 0.85, 1.0) : old;
      return;
    }
    float3 c = s.rgb / s.w;
    haze_out[uint2(0, 0)] = pass_param != 0 ? float4(c, 1.0) : lerp(old, float4(c, 1.0), 0.05);
  }
}

// --- grade -----------------------------------------------------------------
float3 Grade(float3 c, float blurred) {
  c = Dehaze(c);
  // Local contrast on luma, weighted away from black and white (no halos).
  float y = Luma(c);
  float wgt = 4.0 * y * (1.0 - y);
  float ny = saturate(y + (y - blurred) * clarity * wgt);
  c *= (ny + 1e-4) / (y + 1e-4);
  // Gentle S-curve.
  float3 s = c * c * (3.0 - 2.0 * c);
  c = c + (s - c) * contrast;
  // Vibrance.
  y = Luma(c);
  float sat = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
  c = y + (c - y) * (1.0 + vibrance * (1.0 - sat));
  return c;
}

// Brightness of the graded frame relative to the dehazed input, estimated
// from the blurred luma at mid-grey of the frame: grade a grey of the mean
// luma and compare. Cheap, and stable because the mean is smoothed.
float ExposureScale(float mean_luma) {
  float3 g = Grade(mean_luma.xxx * (1.0 - dehaze) + dehaze, mean_luma);
  float after = max(Luma(saturate(g)), 1e-4);
  float scale = pow(max(mean_luma, 1e-4) / after, 0.8);
  return lerp(1.0, scale, keep_exposure);
}

[numthreads(16, 8, 1)]
void grade(uint3 id : SV_DispatchThreadID) {
  if (any(id.xy >= size)) return;
  int2 p = int2(id.xy);
  int2 hi = int2(size) - 1;
  float3 c = source[p].rgb;
  // 3x3 unsharp on the input (before grading keeps it one pass).
  if (sharpen < 0.0) {
    // Contrast-adaptive sharpening (after AMD FidelityFX CAS): the cross's
    // weight follows how much headroom the 3x3 neighbourhood leaves, so
    // soft detail sharpens and edges that are already crisp do not ring.
    float3 a = source[clamp(p + int2(-1, -1), int2(0, 0), hi)].rgb;
    float3 b = source[clamp(p + int2(0, -1), int2(0, 0), hi)].rgb;
    float3 cc = source[clamp(p + int2(1, -1), int2(0, 0), hi)].rgb;
    float3 d = source[clamp(p + int2(-1, 0), int2(0, 0), hi)].rgb;
    float3 f = source[clamp(p + int2(1, 0), int2(0, 0), hi)].rgb;
    float3 g = source[clamp(p + int2(-1, 1), int2(0, 0), hi)].rgb;
    float3 h = source[clamp(p + int2(0, 1), int2(0, 0), hi)].rgb;
    float3 i = source[clamp(p + int2(1, 1), int2(0, 0), hi)].rgb;
    float3 mn = min(min(min(d, c), min(f, b)), h);
    float3 mx = max(max(max(d, c), max(f, b)), h);
    mn += min(mn, min(min(a, cc), min(g, i)));
    mx += max(mx, max(max(a, cc), max(g, i)));
    float3 amp = sqrt(saturate(min(mn, 2.0 - mx) / max(mx, 1e-5)));
    float3 w = amp * (-1.0 / lerp(8.0, 5.0, saturate(-sharpen)));
    c = saturate(((b + d + f + h) * w + c) / (1.0 + 4.0 * w));
  } else if (sharpen > 0.0) {
    float3 sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        sum += source[clamp(p + int2(x, y), int2(0, 0), hi)].rgb;
      }
    }
    c = c + (c - sum / 9.0) * sharpen;
  }
  float2 uv = (float2(id.xy) + 0.5) / float2(size);
  float blurred = luma_in.SampleLevel(linear_clamp, uv, 0.0);
  float mean_luma = mean_in[uint2(0, 0)];
  if (ao_mode != 0) {
    float2 ao = ao_tex.SampleLevel(linear_clamp, uv, 0.0);
    if (ao_mode == 2) {
      dest[id.xy] = float4(ao.xxx, 1.0);
      return;
    }
    if (ao_mode == 3) {
      // Bands every doubling of view depth; sky white.
      float band = ao.y > 50000.0 ? 1.0 : frac(log2(max(ao.y, 1.0)) * 0.5);
      dest[id.xy] = float4(band.xxx, 1.0);
      return;
    }
    if (haze_params.x > 0.0) {
      float f = min(HazeAmount(ao.y), 0.85);
      if (haze_params.w > 0.5) {
        dest[id.xy] = float4(f.xxx, 1.0);
        return;
      }
      // Take out a half-desaturated fog colour, then cap the colour gain so
      // the far scene regains contrast without turning garish.
      float3 a = haze_tex[uint2(0, 0)].rgb;
      a = lerp(Luma(a).xxx, a, 0.5);
      float3 r = saturate((c - a * f) / (1.0 - f));
      float3 chroma_in = c - Luma(c);
      float3 chroma_out = r - Luma(r);
      float limit = length(chroma_in) * (1.0 + 0.5 * f) + 0.01;
      float len = length(chroma_out);
      if (len > limit) r = Luma(r) + chroma_out * (limit / len);
      c = saturate(r);
    }
    // The frame is gamma-encoded: occlusion is linear light, so ^(1/2.2).
    c *= pow(saturate(lerp(1.0, ao.x, ao_strength)), 1.0 / 2.2);
  }
  if (bloom_strength > 0.0) {
    // Screen blend: brightens without clipping.
    float3 b = saturate(bloom_tex.SampleLevel(linear_clamp, uv, 0.0).rgb * bloom_strength);
    c = 1.0 - (1.0 - saturate(c)) * (1.0 - b);
  }
  if (shafts_sun.z > 0.0) {
    float3 b = saturate(bloom_tex.SampleLevel(linear_clamp, uv, 0.0).rgb * shafts_sun.z);
    // Keep shadowed surfaces dark: the glow fades to shafts_params.w of its
    // strength on dark pixels, so the beams don't veil the whole frame.
    b *= lerp(shafts_params.w, 1.0, smoothstep(0.05, 0.6, Luma(saturate(c))));
    c = 1.0 - (1.0 - saturate(c)) * (1.0 - b);
  }
  float3 g = saturate(Grade(saturate(c), blurred) * ExposureScale(mean_luma));
  dest[id.xy] = float4(g, dot(g, float3(0.299, 0.587, 0.114)));
}
