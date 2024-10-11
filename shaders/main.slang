#define MAX_HANDLE 200000

struct VertexStageOutput
{
    CoarseVertex       coarseVertex : CoarseVertex;
    float4          sv_position     : SV_Position;
};

struct CoarseVertex {
    float2 uv;
    uint iid;
};

struct VertexEntry
{
    float4 v1; //position = v1.xyz, normal = (v1.w,v2.xy), uv = v2.zw
    float4 v2;
    float3 getPosition() {
        return v1.xyz;
    }
    float3 getNormal() {
        return float3(v1.w,v2.xy);
    }
    float2 getUV() {
        return v2.zw;
    }
};

struct ObjectEntry
{
    float4x4 modelViewProj;
    uint vbo_offset;
    uint albedo_handle;
    uint metallic_roughness_handle;
    uint occlusion_handle;
    uint normal_handle;
    uint emissive_handle;
}

[vk::binding(0, 0)] 
Sampler2D bindlessTextures[]; 

[vk::push_constant]
cbuffer Constants {
    ConstBufferPointer<VertexEntry> vbo;
    ConstBufferPointer<ObjectEntry> objects;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    uint vid : SV_VertexID, 
    uint iid : SV_InstanceID)
{
    VertexStageOutput output;
    ObjectEntry obj = objects[iid];
    VertexEntry av = vbo[vid + obj.vbo_offset];
    output.sv_position = mul(objects[iid].modelViewProj, float4(av.getPosition(), 1.0));
    output.coarseVertex.uv = av.getUV();
    output.coarseVertex.iid = iid;

    return output;
}

[shader("fragment")]
float4 fragmentMain(CoarseVertex coarseVertex: CoarseVertex) : SV_Target
{

    ObjectEntry obj = objects[coarseVertex.iid];
    float4 color = float4(1.0, 0.0, 1.0, 1.0);
    if (obj.albedo_handle < MAX_HANDLE) {
        color = bindlessTextures[obj.albedo_handle].Sample(coarseVertex.uv);
    }

    if (obj.emissive_handle < MAX_HANDLE) {
        color += 2 * bindlessTextures[obj.emissive_handle].Sample(coarseVertex.uv);
    }

    if (obj.occlusion_handle < MAX_HANDLE) {
        color *= bindlessTextures[obj.occlusion_handle].Sample(coarseVertex.uv);
    }

    return color;
}