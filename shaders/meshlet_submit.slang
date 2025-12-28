import shader_struct;
import shader_struct;

[vk::push_constant]
cbuffer Constants {
    uint64_t objectCount;
    ConstBufferPointer<VertexEntry> vbo;
    ConstBufferPointer<ObjectEntry> objects;
    ConstBufferPointer<MeshletEntry> meshlets;
    DrawBuffer *drawbuf;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint meshletid = dispatchThreadID.x;
    if (meshletid >= objectCount) {
        return;
    }

    MeshletEntry meshlet = meshlets[meshletid];
    ObjectEntry obj = objects[meshlet.primitive_id];

    // if we decide to draw the object (frustum culling, occlusion culling)
    uint new_object_index;
    InterlockedAdd(drawbuf->drawCount, 1, new_object_index);
    drawbuf->draws[meshletid].firstIndex = obj.ibo_offset + meshlet.triangle_offset_in_primitive;
    drawbuf->draws[meshletid].firstInstance = meshletid;
    drawbuf->draws[meshletid].indexCount = meshlet.triangle_count * 3;
    drawbuf->draws[meshletid].instanceCount = 1;
    drawbuf->draws[meshletid].vertexOffset = 0; // bindless manual vertex fetching
}