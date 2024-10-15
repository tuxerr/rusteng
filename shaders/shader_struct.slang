module shader_struct;
// slang parallel of engine/shader_struct.rs

static const int MAX_MESHLET_VERTS = 64;
static const int MAX_MESHLETS_TRIANGLES = 124; // needs to be divisible by 4, closest to 126 for nvidia

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
    public uint triangle_count;
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

public struct DrawBuffer {
    public uint32_t drawCount;
    public VkDrawIndexedIndirectCommand draws[10000];
};