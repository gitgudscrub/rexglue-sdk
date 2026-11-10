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
  float sharpen;      // 3x3 unsharp amount
  float keep_exposure;  // 0..1, how much the average brightness is restored
  float mean_blend;   // per-frame blend of the brightness average (0..1)
  uint pass_param;    // blur: unused; mean: first frame flag
};

Texture2D<float4> source : register(t0);
Texture2D<float> luma_in : register(t1);
Texture2D<float> mean_in : register(t2);
RWTexture2D<float4> dest : register(u0);
RWTexture2D<float> mean_out : register(u1);
// Single-channel destination for the luma passes (same slot as dest; each
// entry point uses only one of them).
RWTexture2D<float> dest_luma : register(u0);
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
  if (sharpen > 0.0) {
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
  float3 g = Grade(saturate(c), blurred) * ExposureScale(mean_luma);
  dest[id.xy] = float4(saturate(g), 1.0);
}
