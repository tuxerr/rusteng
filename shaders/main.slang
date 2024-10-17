#define MAX_HANDLE 200000

import shader_struct;

struct VertexStageOutput
{
    CoarseVertex       coarseVertex : CoarseVertex;
    float4          sv_position     : SV_Position;
};

struct CoarseVertex {
    float2 uv;
    uint iid;
    uint meshletID;
};

[vk::binding(0, 0)] 
Sampler2D bindlessTextures[];

[vk::push_constant]
cbuffer Constants {
    uint64_t objectCount;
    ConstBufferPointer<VertexEntry> vbo;
    ConstBufferPointer<ObjectEntry> objects;
    ConstBufferPointer<MeshletEntry> meshlets;
    DrawBuffer *drawbuf;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    uint vid : SV_VertexID, 
    uint iid : SV_InstanceID)
{
    VertexStageOutput output;
    //    ObjectEntry obj = objects[iid];
    //    VertexEntry av = vbo[vid + obj.vbo_offset];
    MeshletEntry meshlet = meshlets[iid];
    ObjectEntry obj = objects[meshlet.primitive_id];
    VertexEntry av = meshlet.verts[vid];
    output.sv_position = mul(obj.modelViewProj, float4(av.getPosition(), 1.0));
    output.coarseVertex.uv = av.getUV();
    output.coarseVertex.iid = meshlet.primitive_id;
    output.coarseVertex.meshletID = iid;

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

    color.x = (coarseVertex.meshletID % 12) / 12.0;
    color.y = ((coarseVertex.meshletID / 12) % 12) / 12.0;
    color.z = ((coarseVertex.meshletID / 144) % 12) / 12.0;
    return color;
}