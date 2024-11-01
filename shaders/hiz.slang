// A rewrite of SPD to support HiZ correctly and moar wave ops for good measure.
// Translated to slang/hlsl 6.5
import shader_struct;

[vk::binding(0, 0)]
globallycoherent RWTexture2D uImages[13];

[vk::binding(1, 0)]
Sampler2D uTexture;

[vk::binding(1, 1)]
RWStructuredBuffer<uint> atomic_counter;

[vk::push_constant]
cbuffer hiz
{
    HiZRegisters registers;
};

int2 mip_resolution(int mip)
{
    return max(registers.resolution >> mip, int2(1));
}

#define REDUCE_OPERATOR max
float reduce(float4 v)
{
    float2 v0 = REDUCE_OPERATOR(v.xy, v.zw);
    return REDUCE_OPERATOR(v0.x, v0.y);
}

static const int SHUFFLE_X0 = 1 << 0;
static const int SHUFFLE_Y0 = 1 << 1;
static const int SHUFFLE_Y1 = 1 << 2;
static const int SHUFFLE_X1 = 1 << 3;

uint2 unswizzle16x16(uint index)
{
    uint x0 = (index >> 0) & 1;//bitfieldExtract(index, 0, 1);
    uint y01 = (index >> 1) & 3;
    uint x12 = (index >> 3) & 3;
    uint y23 = (index >> 5) & 3;
    uint x3 = (index >> 7) & 1;

    uint x_final = x0 | x12 << 1 | x3 << 3;
    uint y_final = y01 | y23 << 2;

    return uint2(x_final, y_final);
}

float4 transform_z(float4 zs)
{
    float2 z0 = mul(registers.z_transform, float2(zs.x, 1.0));
    float2 z1 = mul(registers.z_transform, float2(zs.y, 1.0));
    float2 z2 = mul(registers.z_transform, float2(zs.z, 1.0));
    float2 z3 = mul(registers.z_transform, float2(zs.w, 1.0));
    return float4(z0.x, z1.x, z2.x, z3.x) / float4(z0.y, z1.y, z2.y, z3.y);
}

void write_image(int2 coord, int mip, float v)
{
    // Rely on image robustness to clean up the OOB writes here.
    uImages[mip][coord] = float4(v);
}

void write_image4(int2 coord, int mip, float4 v)
{
    uImages[mip][coord + int2(0, 0)] = v.xxxx;
    uImages[mip][coord + int2(1, 0)] = v.yyyy;
    uImages[mip][coord + int2(0, 1)] = v.zzzz;
    uImages[mip][coord + int2(1, 1)] = v.wwww;
}

groupshared float shared_buffer[256 / 16];
//groupshared bool shared_is_last_workgroup;

float4x4 fetch_4x4_texture(int2 base_coord)
{
    float2 fcoord = float2(base_coord) * registers.inv_resolution;
    float4 q00 = uTexture.Gather(fcoord, int2(1, 1)).wzxy;
    float4 q10 = uTexture.Gather(fcoord, int2(3, 1)).wzxy;
    float4 q01 = uTexture.Gather(fcoord, int2(1, 3)).wzxy;
    float4 q11 = uTexture.Gather(fcoord, int2(3, 3)).wzxy;
    return float4x4(q00, q10, q01, q11);
}

float4 fetch_2x2_image_mip6(int2 base_coord)
{
    int2 max_coord = mip_resolution(6) - 1;
    float d0 = uImages[6].Load(min(base_coord + int2(0, 0), max_coord)).x;
    float d1 = uImages[6].Load(min(base_coord + int2(1, 0), max_coord)).x;
    float d2 = uImages[6].Load(min(base_coord + int2(0, 1), max_coord)).x;
    float d3 = uImages[6].Load(min(base_coord + int2(1, 1), max_coord)).x;
    return float4(d0, d1, d2, d3);
}

float4x4 fetch_4x4_image_mip6(int2 base_coord)
{
    float4 q0 = fetch_2x2_image_mip6(base_coord + int2(0, 0));
    float4 q1 = fetch_2x2_image_mip6(base_coord + int2(2, 0));
    float4 q2 = fetch_2x2_image_mip6(base_coord + int2(0, 2));
    float4 q3 = fetch_2x2_image_mip6(base_coord + int2(2, 2));
    return float4x4(q0, q1, q2, q3);
}

float4x4 write_mip0_transformed(float4x4 M, int2 base_coord)
{
    float4 q00 = transform_z(M[0]);
    float4 q10 = transform_z(M[1]);
    float4 q01 = transform_z(M[2]);
    float4 q11 = transform_z(M[3]);

    // Write out transformed LOD 0
    write_image4(base_coord + int2(0, 0), 0, q00);
    write_image4(base_coord + int2(2, 0), 0, q10);
    write_image4(base_coord + int2(0, 2), 0, q01);
    write_image4(base_coord + int2(2, 2), 0, q11);

    return float4x4(q00, q10, q01, q11);
}

float reduce_mip_registers(float4x4 M, int2 base_coord, int mip, bool full_res_pass)
{
    float4 q00 = M[0];
    float4 q10 = M[1];
    float4 q01 = M[2];
    float4 q11 = M[3];

    int2 mip_res = mip_resolution(mip);

    float d00 = reduce(q00);
    float d10 = reduce(q10);
    float d01 = reduce(q01);
    float d11 = reduce(q11);

    if (!full_res_pass)
    {
        if (base_coord.x + 1 == mip_res.x) // LOD math chops off data. Need to fold border values into the reduction.
        {
            d00 = REDUCE_OPERATOR(d00, d10);
            d01 = REDUCE_OPERATOR(d01, d11);
        }

        if (base_coord.y + 1 == mip_res.y)
        {
            d01 = REDUCE_OPERATOR(d01, d00);
            d11 = REDUCE_OPERATOR(d11, d10);
        }
    }

    q00 = float4(d00, d10, d01, d11);
    write_image4(base_coord, mip, q00);

    return reduce(q00);
}

float reduce_mips_simd16(int2 base_coord, uint local_index, int mip, float d, bool full_res_pass)
{
    int2 mip_res = mip_resolution(mip);
    float d_horiz, d_vert, d_diag;
    bool swap_horiz, swap_vert;

    // It is possible that our thread is barely in range, but horiz/vert neighbor is not.
#define CUTOFF_REDUCE() { \
    swap_horiz = base_coord.x + 1 == mip_res.x; \
    swap_vert = base_coord.y + 1 == mip_res.y; \
    if (swap_horiz) \
        d = REDUCE_OPERATOR(d, d_horiz); \
    if (swap_vert) \
        d = REDUCE_OPERATOR(d, d_vert); \
    if (swap_vert && swap_horiz) \
        d = REDUCE_OPERATOR(d, d_diag); }

    d_horiz = QuadReadAcrossX(d);
    d_vert = QuadReadAcrossY(d);
    d_diag = QuadReadAcrossDiagonal (d);
    if (!full_res_pass)
        CUTOFF_REDUCE();
    write_image(base_coord, mip, d);

    if (registers.mips > mip + 1)
    {
        uint lane_index = WaveGetLaneIndex();
        base_coord >>= 1;
        mip_res = mip_resolution(mip + 1);
        d = reduce(float4(d, d_horiz, d_vert, d_diag));

        // This requires only SIMD16, which everyone can do.
        d_horiz = WaveShuffle(d, lane_index ^ SHUFFLE_X1);
        d_vert = WaveShuffle(d, lane_index ^ SHUFFLE_Y1);
        d_diag = WaveShuffle(d, lane_index ^ (SHUFFLE_X1 | SHUFFLE_Y1));
        if (!full_res_pass)
            CUTOFF_REDUCE();
        if ((local_index & 3) == 0)
            write_image(base_coord, mip + 1, d);
    }

    return reduce(float4(d, d_horiz, d_vert, d_diag));
}

// Each workgroup reduces 64x64 on its own.
// Allows reducing up to a 4096x4096 texture, like SPD.

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(in uint3 groupID: SV_GroupID, 
                 in uint3 groupThreadID: SV_GroupThreadID)
{
    //each group handles 64-wide region, each thread handles 4-wide region
    uint local_index = groupThreadID.x;//gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uint2 local_coord = unswizzle16x16(local_index);

    // LOD 0 feedback
    int2 base_coord = int2(local_coord) * 4 + int2(groupID.xy * 64u);
    float4x4 M = fetch_4x4_texture(base_coord);
    M = write_mip0_transformed(M, base_coord);

    // Write LOD 1, Compute LOD 2
    if (registers.mips <= 1)
        return;
    float d = reduce_mip_registers(M, base_coord >> 1, 1, true);
    if (registers.mips <= 2)
        return;

    // Write LOD 2, Compute LOD 3-4
    d = reduce_mips_simd16(base_coord >> 2, local_index, 2, d, true);
    if (registers.mips <= 4)
        return;

    // Write LOD 4 to shared
    if ((local_index & 15) == 0)
        shared_buffer[local_index >> 4] = d;
    GroupMemoryBarrierWithGroupSync();

    // Write LOD 4, Compute LOD 5-6.
    if (local_index < 16)
        d = reduce_mips_simd16(int2(groupID.xy * 4u + local_coord), local_index, 4, shared_buffer[local_index], true);

    // Write LOD 6.
    if (registers.mips <= 6)
        return;
    if (local_index == 0)
        write_image(int2(groupID.xy), 6, d);
    if (registers.mips <= 7)
        return;

    // Persistent waves
    AllMemoryBarrierWithGroupSync();
  /*  if (local_index == 0) {
        uint counter_val;
        InterlockedAdd(atomic_counter[0], 1u, counter_val);
        shared_is_last_workgroup = counter_val + 1u == registers.target_counter;
    } */
    GroupMemoryBarrierWithGroupSync();
    //if (!shared_is_last_workgroup)
    //    return;
    if(groupID.x + groupID.y != 0) {
        return;
    }


    // Reset counter for next iteration.
    //if (local_index == 0)
    //    atomic_counter[0] = 0u;

    // Write LOD 7, Compute LOD 8
    base_coord = int2(local_coord) * 4;
    d = reduce_mip_registers(fetch_4x4_image_mip6(base_coord), base_coord >> 1, 7, false);
    if (registers.mips <= 8)
        return;

    // Write LOD 8-9, Compute LOD 10
    d = reduce_mips_simd16(int2(local_coord), local_index, 8, d, false);
    if (registers.mips <= 10)
        return;
    if ((local_index & 15) == 0)
        shared_buffer[local_index >> 4] = d;
    GroupMemoryBarrierWithGroupSync();

    if (local_index < 16)
        d = reduce_mips_simd16(int2(local_coord), local_index, 10, shared_buffer[local_index], false);
    if (registers.mips <= 12)
        return;
    if (local_index == 0)
        write_image(int2(0), 12, d);
}