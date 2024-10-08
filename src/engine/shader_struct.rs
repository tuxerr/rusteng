#[repr(C)]
pub struct VertexEntry {
    pub pos: cgmath::Vector3<f32>,
    pub norm: cgmath::Vector3<f32>,
    pub uv: cgmath::Vector2<f32>,
}

#[repr(C)]
pub struct ObjectEntry {
    pub model_view_projection: cgmath::Matrix4<f32>
}

