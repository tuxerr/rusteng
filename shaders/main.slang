#define MAX_HANDLE 200000

import shader_struct;

[vk::binding(0, 0)] 
Sampler2D bindlessTextures[];

[vk::push_constant]
cbuffer Constants {
    uint64_t objectCount;
    ConstBufferPointer<VertexEntry> vbo;
    ConstBufferPointer<ObjectEntry> objects;
    MeshletEntry* meshlets;
    DrawBuffer *drawbuf;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)] // 1 thread per vertex, 2 triangle per thread
void meshMain(in uint3 groupID: SV_GroupID,
              in uint3 groupThreadID: SV_GroupThreadID,
              out vertices VertexStageOutput verts[64],
              out indices uint3 triangles[124])
{
    uint meshletid = groupID.x;
    if (meshletid >= objectCount) {
        return;
    }

    const MeshletEntry *meshlet = meshlets + meshletid;

    ObjectEntry obj = objects[meshlet->primitive_id];
    SetMeshOutputCounts(meshlet->vertex_count, meshlet->triangle_count);

    uint vertexID = groupThreadID.x;
    if (vertexID < meshlet->vertex_count) {
        VertexStageOutput vso;
        VertexEntry av = meshlet->verts[vertexID];
        vso.sv_position = mul(obj.modelViewProj, float4(av.getPosition(), 1.0));
        vso.coarseVertex.uv = av.getUV();
        vso.coarseVertex.iid = meshlet->primitive_id;
        vso.coarseVertex.meshletID = meshletid;
        verts[vertexID] = vso;
    }

    uint triangle_id = groupThreadID.x;
    if (triangle_id < meshlet->triangle_count) {
        triangles[triangle_id] = uint3(meshlet->indices[triangle_id * 3], meshlet->indices[triangle_id * 3 + 1], meshlet.indices[triangle_id * 3 + 2]);
    }

    triangle_id += 64;
    if (triangle_id < meshlet->triangle_count) {
        triangles[triangle_id] = uint3(meshlet->indices[triangle_id * 3], meshlet->indices[triangle_id * 3 + 1], meshlet.indices[triangle_id * 3 + 2]);
    }
}

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
    //color.z = ((coarseVertex.meshletID / 144) % 12) / 12.0;
    return color;
}