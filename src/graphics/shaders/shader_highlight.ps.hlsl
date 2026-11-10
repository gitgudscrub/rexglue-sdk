// Replaces game pixel shaders listed in the shader_highlight cvar, so the
// surfaces they draw show up in a solid colour (finding which shader draws
// water, sky, skin...). The n-th listed shader gets colour n (fxc /D
// COLOR=n): magenta, red, green, blue, yellow, cyan, white, orange. Reads
// nothing, so it fits any vertex shader and root signature.
static const float3 kColors[8] = {
  float3(1, 0, 1), float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1),
  float3(1, 1, 0), float3(0, 1, 1), float3(1, 1, 1), float3(1, 0.5, 0)};
float4 main() : SV_Target0 { return float4(kColors[COLOR], 1.0); }
