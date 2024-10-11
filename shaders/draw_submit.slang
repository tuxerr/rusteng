import shader_struct;

[vk::push_constant]
cbuffer Constants {
    uint64_t objectCount;
    ConstBufferPointer<VertexEntry> vbo;
    ConstBufferPointer<ObjectEntry> objects;
    DrawBuffer *drawbuf;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint objid = dispatchThreadID.x;
    if (objid >= objectCount) {
        return;
    }

    ObjectEntry obj = objects[objid];

    // if we decide to draw the object (frustum culling, occlusion culling)
    uint new_object_index;
    InterlockedAdd(drawbuf->drawCount, 1, new_object_index);
    drawbuf->draws[new_object_index].firstIndex = obj.ibo_offset;
    drawbuf->draws[new_object_index].firstInstance = new_object_index;
    drawbuf->draws[new_object_index].indexCount = obj.index_count;
    drawbuf->draws[new_object_index].instanceCount = 1;
    drawbuf->draws[new_object_index].vertexOffset = 0; // bindless manual vertex fetching
}