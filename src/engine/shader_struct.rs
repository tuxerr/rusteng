use zerocopy_derive::*;

pub const MAX_MESHLET_VERTS: usize = 64;
pub const MAX_MESHLETS_TRIANGLES: usize = 124; // needs to be divisible by 4, closest to 126 for nvidia

#[repr(C)]
#[derive(IntoBytes, Immutable)]
#[derive(Clone, Copy)]
pub struct VertexEntry {
    pub pos: [f32; 3],
    pub norm: [f32; 3],
    pub uv: [f32; 2],
}

impl Default for VertexEntry {
    fn default() -> Self {
         VertexEntry {
            pos : [0.0, 0.0, 0.0],
            norm : [0.0, 0.0, 0.0],
            uv : [0.0, 0.0]
         }
    }
}

#[repr(C)]
pub struct MeshletEntry {
    pub pos_radius: [f32; 4],
    pub verts: [VertexEntry; MAX_MESHLET_VERTS],
    pub indices: [u8; MAX_MESHLETS_TRIANGLES * 3],
    pub triangle_count: u32,
    pub primitive_id: u32
}

#[repr(C)]
pub struct ObjectEntry {
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
    pub emissive_handle: u32
}

#[repr(C)]
pub struct PushConstants {
    pub object_count : u64,
    pub vbo : u64,
    pub objects : u64,
    pub meshlets : u64,
    pub drawbuf : u64
}

