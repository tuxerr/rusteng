#[repr(C)]
pub struct VertexEntry {
    pub pos: cgmath::Vector3<f32>,
    pub norm: cgmath::Vector3<f32>,
    pub uv: cgmath::Vector2<f32>,
}

#[repr(C)]
pub struct ObjectEntry {
    pub model_view_projection: cgmath::Matrix4<f32>,
    pub position: cgmath::Vector4<f32>,
    pub sphere_size: f32,
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
    pub drawbuf : u64
}

