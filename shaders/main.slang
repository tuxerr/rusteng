struct VertexStageOutput
{
    float4          sv_position     : SV_Position;
};

struct AssembledVertex
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : UV;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;
    output.sv_position = float4(0.2,0.2,0.2, 1.0);

    return output;
}

[shader("fragment")]
float4 fragmentMain()
{
    return float4(1.0, 0.0, 1.0, 1.0);
}