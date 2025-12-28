import shader_struct;
import gpu_particles;

[vk::push_constant]
cbuffer Constants {
    Particle *particles;
    MeshletIndirectCommand *meshletCommands;
    uint nParticleOptions;
    float flTimeRatio;
    float flDrag;
};

public void SimulateParticle(inout Particle p)
{
    p.position = p.position + p.velocity * flTimeRatio;
    p.velocity = p.velocity * (1 - flDrag);
    p.velocity.y -= 0.05;
    p.age = p.age - flTimeRatio;
    p.color[1] = p.age / 3.0f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    bool bIsPing = (nParticleOptions & 0x1) == 0;
    uint nCurrentIndex = bIsPing ? 0 : 1;
    uint nPrevIndex = bIsPing ? 1 : 0;
    uint nCurrentOffset = nCurrentIndex * MAX_PARTICLE_COUNT;
    uint nPrevOffset = nPrevIndex * MAX_PARTICLE_COUNT;

    MeshletIndirectCommand prevFrameCommand = meshletCommands[nPrevIndex];

    // this stores any new particles emitted + the previous frame particle count
    uint nParticleCount = prevFrameCommand.totalParticles;
    if (dispatchThreadID.x >= nParticleCount)
        return;

    Particle newParticleToSim = particles[ nPrevOffset + dispatchThreadID.x ];

    SimulateParticle(newParticleToSim);

    if (newParticleToSim.age > 0)
    {
        uint new_particle_index;
        InterlockedAdd(meshletCommands[nCurrentIndex].totalParticles, 1u, new_particle_index);

        particles[new_particle_index + nCurrentOffset] = newParticleToSim;

        if ( ( ( new_particle_index )  % 64 ) == 0 )
        {
            InterlockedAdd(meshletCommands[nCurrentIndex].groupSizeX, 1u);
            meshletCommands[nCurrentIndex].groupSizeY = 1;
            meshletCommands[nCurrentIndex].groupSizeZ = 1;
        }
    }
}