module shader_struct;
// slang parallel of engine/shader_struct.rs

static public const int MAX_MESHLET_VERTS = 64;
static public const int MAX_MESHLETS_TRIANGLES = 124; // needs to be divisible by 4, closest to 126 for nvidia
static public const uint MAX_PARTICLE_COUNT = 100000;

// standard meshlets
public struct VertexStageOutput
{
    public CoarseVertex coarseVertex : CoarseVertex;
    public float4 sv_position : SV_Position;
};

public struct CoarseVertex {
    public float2 uv;
    public uint iid;
    public uint meshletID;
};

// particle renders
public struct VertexStageOutputParticle
{
    public ParticleVarying coarseVertex : ParticleVarying;
    public float4 sv_position : SV_Position;
};

public struct ParticleVarying {
    public uint particleID;
};

// engine structures
public struct Scene 
{
    public float4x4 viewProjMatrix;
    public float4x4 viewMatrix;
    public float4 rotationQuaternion;
    public uint frame_index;
}

// meshlet definition structures
public struct VertexEntry
{
    public float4 v1; // position = v1.xyz, normal = (v1.w,v2.xy), uv = v2.zw
    public float4 v2;
    public float3 getPosition() {
        return v1.xyz;
    }
    public float3 getNormal() {
        return float3(v1.w, v2.xy);
    }
    public float2 getUV() {
        return v2.zw;
    }
};

public struct MeshletEntry 
{
    public float4 pos_radius;
    public VertexEntry verts[MAX_MESHLET_VERTS];
    public uint8_t indices[MAX_MESHLETS_TRIANGLES * 3];
    public uint vertex_count;
    public uint triangle_count;
    public uint triangle_offset_in_primitive;
    public uint primitive_id;
}

public struct ObjectEntry
{
    public float4x4 modelViewProj;
    public float4 position;
    //public float sphere_size; // bounding sphere
    public uint ibo_offset;
    public uint index_count;
    public uint vbo_offset;
    public uint albedo_handle;
    public uint metallic_roughness_handle;
    public uint occlusion_handle;
    public uint normal_handle;
    public uint emissive_handle;
};

// Provided by VK_VERSION_1_0
public struct VkDrawIndexedIndirectCommand {
    public uint32_t indexCount;
    public uint32_t instanceCount;
    public uint32_t firstIndex;
    public int32_t vertexOffset;
    public uint32_t firstInstance;
};

// Provided by VK_VERSION_1_0
public struct MeshletIndirectCommand {
    public uint groupSizeX;
    public uint groupSizeY;
    public uint groupSizeZ;
    public uint totalParticles;
};

public struct DrawBuffer {
    public uint32_t drawCount;
    public VkDrawIndexedIndirectCommand draws[10000];
};

public struct HiZRegisters
{
    public float2x2 z_transform;
    public int2 resolution;
    public float2 inv_resolution;
    public int mips;
    public uint target_counter;
};

public struct Particle
{
    public float4 position;
    public float4 velocity;
    public float4 color;
    public float age;
    public float radius;
};

public uint HashUintSeeded(uint index, uint seed)
{
    uint x = index;
    x ^= seed * 0x9e3779b9; // mix seed (golden ratio)
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

public float HashToFloat(uint hash)
{
    // Use upper 24 bits for good float precision
    return (hash >> 8) * (1.0 / 16777216.0); // 2^24
}

public float HashToSignedFloat(uint hash)
{
    // Use upper 24 bits for good float precision
    return (HashToFloat(hash) - 0.5) * 2;
}