struct VertexStageOutput
{
    float4          sv_position     : SV_Position;
};

struct AssembledVertex
{
    float4 v1; //position = v1.xyz, normal = (v1.w,v2.xy), uv = v2.zw
    float4 v2;
};

struct SceneData
{

}

//this is shadowing rust's object.rs
//struct VertexEntry {
//    pub pos: cgmath::Vector3<f32>,
//    pub norm: cgmath::Vector3<f32>,
//    pub uv: cgmath::Vector2<f32>,
//}

[vk::push_constant]
cbuffer Constants {
    ConstBufferPointer<AssembledVertex> vbo;
    ConstBufferPointer<SceneData> vbo2;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    uint vid : SV_VertexID)
{
    VertexStageOutput output;
    AssembledVertex av = vbo[vid];
    output.sv_position = float4(av.v1.xyz,1.0);

    return output;
}

[shader("fragment")]
float4 fragmentMain()
{
    return float4(1.0, 0.0, 1.0, 1.0);
}