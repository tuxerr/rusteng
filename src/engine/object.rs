use super::{vkutil, Engine};
use ash::vk;
use cgmath::{SquareMatrix, Vector3};
use gltf;
use std::fs::File;
use std::path::Path;
use std::u64;
use std::rc::Rc;

#[derive(Default)]
pub struct BufferSlice {
    pub size: u32,
    pub offset: u32,
}

pub struct Object {
    pub name: String,
    pub transform: cgmath::Matrix4<f32>,
    pub gltf_document: gltf::Document,
    pub ibo_slice: BufferSlice,
    pub vbo_slice: BufferSlice,
    pub pipeline: Rc<vkutil::Pipeline>,
}

#[repr(C)]
struct VertexEntry {
    pub pos: cgmath::Vector3<f32>,
    pub norm: cgmath::Vector3<f32>,
    pub uv: cgmath::Vector2<f32>,
}

impl Object {
    pub fn loadObjectInEngine(eng: &mut Engine, name: String, pipeline: Rc<vkutil::Pipeline>) -> Self {
        let mesh_path_str = format!("assets/{}.glb", name);
        let mesh_path = Path::new(&mesh_path_str);

        let (document, buffers, _) =
            gltf::import(mesh_path).expect("Unable to load Fox model");

        let mut vboentry = Vec::new();
        let mut iboentry: Vec<u32> = Vec::new();

        for mesh in document.meshes() {
            println!("Mesh #{}", mesh.index());
            for primitive in mesh.primitives() {
                println!("- Primitive #{}", primitive.index());
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // indices
                if let Some(iter) = reader.read_indices() {
                    for index in iter.into_u32() {
                        iboentry.push(index);
                    }
                }

                // positions
                if let Some(iter_pos) = reader.read_positions() {
                    for vertex_position in iter_pos {
                        vboentry.push(VertexEntry {
                            pos: cgmath::Vector3::from(vertex_position),
                            norm: cgmath::Vector3::new(0.0f32, 0.0f32, 0.0f32),
                            uv: cgmath::Vector2::new(0.0f32, 0.0f32),
                        });
                    }
                }

                // normals
                if let Some(iter) = reader.read_normals() {
                    for (idx, norm) in iter.take(vboentry.len()).enumerate() {
                        vboentry[idx].norm = cgmath::Vector3::from(norm);
                    }
                }

                // uvs
                if let Some(iter) = reader.read_tex_coords(0) {
                    for (idx, uv) in iter.into_f32().take(vboentry.len()).enumerate() {
                        vboentry[idx].uv = cgmath::Vector2::from(uv);
                    }
                }
            }
        }

        if(iboentry.is_empty()) {
            for i in 0..vboentry.len() {
                iboentry.push(i as u32);
            }
        }

        let alloc_info = eng
            .context
            .vma_alloc
            .get_allocation_info(&eng.staging_buf.mem_alloc);

        let vbo_size = vboentry.len() * std::mem::size_of::<VertexEntry>();
        let ibo_size = iboentry.len() * std::mem::size_of::<u32>();
        let alloc_ptr = alloc_info.mapped_data;
        assert!(
            alloc_info.size > (vbo_size + ibo_size) as u64,
            "Staging buffer too small! {} < {}",
            alloc_info.size,
            (vbo_size + ibo_size)
        );
        println!(
            "Copying {} bytes from {:#?} to {:#?}",
            vbo_size,
            vboentry.as_ptr(),
            alloc_ptr as *mut VertexEntry
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                vboentry.as_ptr(),
                alloc_ptr as *mut VertexEntry,
                vboentry.len(),
            );
            let alloc_ptr_for_ibo = (alloc_ptr as *mut VertexEntry).offset(vboentry.len() as isize);
            std::ptr::copy_nonoverlapping(
                iboentry.as_ptr(),
                alloc_ptr_for_ibo as *mut u32,
                iboentry.len(),
            );

            //eng.context.vma_alloc.flush_allocation(&eng.staging_buf.mem_alloc, 0, (vbo_size + ibo_size) as vk::DeviceSize).expect("Failure to flush");
            //eng.context.vma_alloc.unmap_memory(&mut eng.staging_buf.mem_alloc);
        }

        let commandbuffer = eng
            .context
            .allocate_and_begin_commandbuffer(&eng.commandpools[0]);
        let vbo_regions = [ash::vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(vbo_size as u64)];
        let ibo_regions = [ash::vk::BufferCopy::default()
            .src_offset(vbo_size as u64)
            .dst_offset(0)
            .size(ibo_size as u64)];

        unsafe {
            eng.context.device.cmd_copy_buffer(
                commandbuffer,
                eng.staging_buf.vk_buffer,
                eng.global_vbo.vk_buffer,
                &vbo_regions,
            );
            eng.context.device.cmd_copy_buffer(
                commandbuffer,
                eng.staging_buf.vk_buffer,
                eng.global_ibo.vk_buffer,
                &ibo_regions,
            );

            eng.context
                .device
                .end_command_buffer(commandbuffer)
                .expect("Failure to end commandbuffer recording");

            eng.context
                .device
                .reset_fences(&[eng.fences[0]])
                .unwrap();

            let commandbuffers_submit = [commandbuffer];
            let queue_submit_info =
                vk::SubmitInfo::default().command_buffers(&commandbuffers_submit);
            eng.context
                .device
                .queue_submit(
                    eng.context.queues.gfx_queue,
                    &[queue_submit_info],
                    eng.fences[0],
                )
                .expect("Failure to submit commandbuffer into gfx queue");

            eng.context
                .device
                .wait_for_fences(&[eng.fences[0]], true, u64::MAX)
                .expect("Failure to wait for staging fence");
        }

        let ibo_slice = BufferSlice {
            offset: 0, 
            size: ibo_size as u32,
        };

        let vbo_slice = BufferSlice {
            offset: 0, 
            size: vbo_size as u32,
        };

        Object {
            name,
            transform: cgmath::Matrix4::identity(),
            gltf_document: document,
            ibo_slice,
            vbo_slice,
            pipeline: pipeline
        }
    }
}
