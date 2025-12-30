import shader_struct;
import gpu_particles;

[vk::push_constant]
cbuffer Constants {
    Particle *particles;
    MeshletIndirectCommand *meshletCommands;
    Scene *scene;
    uint nParticleOptions;
    uint nParticlesToEmit;
    float3 vInitialPosition;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    bool bIsPing = (nParticleOptions & 0x1) == 0;
    uint nCurrentIndex = bIsPing ? 0 : 1;
    uint nCurrentOffset = nCurrentIndex * MAX_PARTICLE_COUNT;

    // this stores the previous frame particle count + any new particles emitted
    if ( dispatchThreadID.x >= nParticlesToEmit ||
        meshletCommands[nCurrentIndex].totalParticles >= MAX_PARTICLE_COUNT )
        return;

    uint new_particle_index;
    InterlockedAdd(meshletCommands[nCurrentIndex].totalParticles, 1u, new_particle_index);
    if ( new_particle_index < MAX_PARTICLE_COUNT )
    {
        Particle currParticle;
        currParticle.age = 1 + HashToFloat(HashUintSeeded(new_particle_index, 5 + scene->frame_index));
        currParticle.color = float4(1, 0, 0, HashToFloat(HashUintSeeded(new_particle_index, 6 + scene->frame_index)));
        currParticle.position.xyz = vInitialPosition + float3(0, 0, 0);
        currParticle.position.w = 1;
        currParticle.radius = 0.01 + HashToFloat(HashUintSeeded(new_particle_index, 4 + scene->frame_index)) * 0.02;
        currParticle.velocity.xyz = 2 * float3(
            HashToSignedFloat(HashUintSeeded(new_particle_index + scene->frame_index, 1 + scene->frame_index)),
            HashToSignedFloat(HashUintSeeded(new_particle_index + scene->frame_index, 2 + scene->frame_index)),
            HashToSignedFloat(HashUintSeeded(new_particle_index, 3 + scene->frame_index)));
        currParticle.velocity.w = 0;

        particles[nCurrentOffset + new_particle_index] = currParticle;

        if ( ( ( new_particle_index )  % 64 ) == 0 )
        {
            InterlockedAdd(meshletCommands[nCurrentIndex].groupSizeX, 1u);
            meshletCommands[nCurrentIndex].groupSizeY = 1;
            meshletCommands[nCurrentIndex].groupSizeZ = 1;
        }
    }
    else
    {
        InterlockedAdd(meshletCommands[nCurrentIndex].totalParticles, -1u);
    }
}