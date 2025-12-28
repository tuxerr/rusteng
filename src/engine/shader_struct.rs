use zerocopy_derive::*;

pub const MAX_MESHLET_VERTS: usize = 64;
pub const MAX_MESHLETS_TRIANGLES: usize = 124; // needs to be divisible by 4, closest to 126 for nvidia

#[repr(C)]
#[derive(IntoBytes, Immutable, Clone, Copy)]
pub struct VertexEntry {
    pub pos: [f32; 3],
    pub norm: [f32; 3],
    pub uv: [f32; 2],
}

impl Default for VertexEntry {
    fn default() -> Self {
        VertexEntry {
            pos: [0.0, 0.0, 0.0],
            norm: [0.0, 0.0, 0.0],
            uv: [0.0, 0.0],
        }
    }
}

#[repr(C)]
pub struct MeshletEntry {
    pub pos_radius: [f32; 4],
    pub verts: [VertexEntry; MAX_MESHLET_VERTS],
    pub indices: [u8; MAX_MESHLETS_TRIANGLES * 3],
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub triangle_offset_in_primitive: u32,
    pub primitive_id: u32,
}

#[repr(C)]
pub struct PrimitiveEntry {
    pub model_view_projection: cgmath::Matrix4<f32>,
    pub position: cgmath::Vector4<f32>,
    //pub sphere_size: f32,
    pub ibo_offset: u32,
    pub index_count: u32,
    pub vbo_offset: u32,
    pub albedo_handle: u32,
    pub metallic_roughness_handle: u32,
    pub occlusion_handle: u32,
    pub normal_handle: u32,
    pub emissive_handle: u32,
}

#[repr(C)]
pub struct PushConstants {
    pub object_count: u64,
    pub vbo: u64,
    pub objects: u64,
    pub meshlets: u64,
    pub drawbuf: u64,
}

#[repr(C)]
pub struct HiZRegisters {
    pub z_transform: cgmath::Matrix2<f32>,
    pub resolution: cgmath::Vector2<i32>,
    pub inv_resolution: cgmath::Vector2<f32>,
    pub mips: i32,
    pub target_counter: u32,
}

pub struct Particle {
    pub position: cgmath::Vector4<f32>,
    pub velocity: cgmath::Vector4<f32>,
    pub color: cgmath::Vector4<f32>,
    pub age: f32,
    pub radius: f32,
}

// Provided by VK_VERSION_1_0
pub struct MeshletIndirectCommand {
    pub group_size_x: u32,
    pub group_size_y: u32,
    pub group_size_z: u32,
    pub total_particles: u32,
}

#[repr(C)]
#[derive(IntoBytes, Immutable, Clone, Copy)]
pub struct Scene {
    pub view_proj_matrix: [[f32; 4]; 4],
    pub view_matrix: [[f32; 4]; 4],
    pub rotation_quaternion: [f32; 4],
}

// individual shaders push constant definitions (wip : reflect!)
pub struct PushConstantsParticleEmit {
    pub particles: u64,
    pub meshletCommands: u64,
    pub nParticleOptions: u32,
    pub nParticlesToEmit: u32,
    pub vInitialPosition: [f32; 3],
}

pub struct PushConstantsParticleSimulate {
    pub particles: u64,
    pub meshletCommands: u64,
    pub nParticleOptions: u32,
    pub flTimeRatio: f32,
    pub flDrag: f32,
}

pub struct PushConstantsParticleRender {
    pub particles: u64,
    pub scene: u64,
    pub meshlet_commands: u64,
    pub commandOffset: u32,
    pub particleOffset: u32,
}
