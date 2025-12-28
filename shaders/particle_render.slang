#define MAX_HANDLE 200000

import shader_struct;

[vk::binding(0, 0)] 
Sampler2D bindlessTextures[];

[vk::push_constant]
cbuffer Constants {
    Particle *particles;
    Scene *scene;
    MeshletIndirectCommand *commands;
    uint commandOffset;
    uint particleOffset;
};

float3 RotateByQuaternion(float3 v, float4 q)
{
    // q.xyz = vector part, q.w = scalar part
    float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float4 CoordinateToWorldSpacePosition(float2 c, Particle p)
{
    float3 worldSpaceCoordinate = RotateByQuaternion(float3(c.x,c.y,0),scene->rotationQuaternion);
    float3 worldSpacePos = p.position.xyz + worldSpaceCoordinate * p.radius;
    return mul(scene->viewProjMatrix,float4(worldSpacePos,1));
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)] // 1 thread per particle
void meshMain(uint3 threadID : SV_GroupThreadID,
              uint3 dispatchThreadID: SV_DispatchThreadID,
              out vertices VertexStageOutputParticle verts[256],
              out indices uint3 triangles[128])
{
    uint particleID = dispatchThreadID.x;
    uint nParticleCount = commands[commandOffset].totalParticles;

    if (particleID >= nParticleCount) {
        return;
    }

    const Particle particle = particles[particleOffset + particleID];
    
    // only the first active thread will be used
    if ( WaveIsFirstLane() )
    {
        uint nNumParticlesInGroup = min(64, nParticleCount - particleID);
        SetMeshOutputCounts( nNumParticlesInGroup * 4, nNumParticlesInGroup * 2 );
    }

    const float2[4] coords = {
        float2(-1,-1),
        float2(1,-1),
        float2(1,1),
        float2(-1,1) 
    };

    VertexStageOutputParticle vsop;
    vsop.coarseVertex.particleID = particleOffset+particleID;

    uint nInitialVertOffset = threadID.x * 4;
    uint nInitialTriangleOffset = threadID.x * 2;

    for ( uint nVertexIndex = 0; nVertexIndex < 4; nVertexIndex++ )
    {
        vsop.sv_position = CoordinateToWorldSpacePosition(coords[nVertexIndex], particle);
        verts[nInitialVertOffset + nVertexIndex] = vsop;
    }

    triangles[nInitialTriangleOffset + 0] = uint3(nInitialVertOffset + 0, nInitialVertOffset + 1, nInitialVertOffset + 2);
    triangles[nInitialTriangleOffset + 1] = uint3(nInitialVertOffset + 0, nInitialVertOffset + 2, nInitialVertOffset + 3);
}

[shader("fragment")]
float4 fragmentMain(ParticleVarying part: ParticleVarying) : SV_Target
{

    Particle particle = particles[part.particleID];
    return particle.color;
}