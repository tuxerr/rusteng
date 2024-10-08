struct VertexStageOutput
{
    float4          sv_position     : SV_Position;
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
}

[vk::push_constant]
cbuffer Constants {
    ConstBufferPointer<VertexEntry> vbo;
    ConstBufferPointer<ObjectEntry> objects;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    uint vid : SV_VertexID)
{
    VertexStageOutput output;
    VertexEntry av = vbo[vid];
    output.sv_position = mul(objects[0].modelViewProj,float4(av.getPosition(),1.0));

    return output;
}

[shader("fragment")]
float4 fragmentMain() : SV_Target
{
    return float4(1.0, 0.0, 1.0, 1.0);
}