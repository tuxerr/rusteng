use crate::engine::vkutil::Texture;

use super::{shader_struct::VertexEntry, vkutil, Engine};
use ash::vk;
use cgmath::SquareMatrix;
use gltf;
use std::collections::HashSet;
use std::path::Path;
use std::rc::Rc;
use std::{u32, u64};
use derivative::Derivative;

fn gltf_format_to_vulkan_format(format: gltf::image::Format, srgb : bool) -> vk::Format {
    match format {
        gltf::image::Format::R8 => if srgb { vk::Format::R8_SRGB } else { vk::Format::R8_UNORM},
        gltf::image::Format::R8G8 => if srgb { vk::Format::R8G8_SRGB } else { vk::Format::R8G8_UNORM},
        gltf::image::Format::R8G8B8 => if srgb { vk::Format::R8G8B8A8_SRGB } else { vk::Format::R8G8B8A8_UNORM},
        gltf::image::Format::R8G8B8A8 => if srgb { vk::Format::R8G8B8A8_SRGB } else { vk::Format::R8G8B8A8_UNORM},
        gltf::image::Format::R16 => vk::Format::R16_UNORM,
        gltf::image::Format::R16G16 => vk::Format::R16G16_UNORM,
        gltf::image::Format::R16G16B16 => vk::Format::R16G16B16_UNORM,
        gltf::image::Format::R16G16B16A16 => vk::Format::R16G16B16A16_UNORM,
        gltf::image::Format::R32G32B32FLOAT => vk::Format::R32G32B32_SFLOAT,
        gltf::image::Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
    }
}

pub struct Object {
    pub name: String,
    pub transform: cgmath::Matrix4<f32>,
    pub gltf_document: gltf::Document,
    pub primitives: Vec<Primitive>,
    pub pipeline: Rc<vkutil::Pipeline>,
    pub textures: Vec<super::vkutil::Texture>,
}

#[derive(Derivative)]
#[derivative(Default)]
pub struct Primitive {
    pub ibo_slice: vkutil::BufferSlice,
    pub vbo_slice: vkutil::BufferSlice,
    #[derivative(Default(value = "u32::MAX"))]
    pub base_color_tex: u32,
    #[derivative(Default(value = "u32::MAX"))]
    pub metallic_roughness_tex: u32,
    #[derivative(Default(value = "u32::MAX"))]
    pub occlusion_tex: u32,
    #[derivative(Default(value = "u32::MAX"))]
    pub normal_tex: u32,
    #[derivative(Default(value = "u32::MAX"))]
    pub emissive_tex: u32
}

impl Object {
    pub fn loadObjectInEngine(
        eng: &mut Engine,
        name: String,
        pipeline: Rc<vkutil::Pipeline>,
    ) -> Self {
        let mesh_path_str = format!("assets/{}", name);
        let mesh_path = Path::new(&mesh_path_str);

        let (document, buffers, images) =
            gltf::import(mesh_path).expect("Unable to load Fox model");

        let alloc_info = eng
            .context
            .vma_alloc
            .get_allocation_info(&eng.staging_buf.mem_alloc);
        let mut alloc_ptr = alloc_info.mapped_data as *mut u8;

        // calculate image (texture.source) indices which have to be srgb
        let mut srgb_indices = HashSet::new();
        document.materials().for_each(|mat| {
            mat.pbr_metallic_roughness().base_color_texture().map(|tex| {
                srgb_indices.insert(tex.texture().source().index());
            });

            mat.emissive_texture().map(|tex| {
                srgb_indices.insert(tex.texture().source().index());
            });
        });

        // textures need to be updated to add their bindless tex handles
        let images: Vec<Texture> = images
            .iter().enumerate()
            .map(|(idx,image)| {
                // create image
                let vk_format = gltf_format_to_vulkan_format(image.format, srgb_indices.contains(&idx));
                let vk_extent = vk::Extent2D::default()
                    .width(image.width)
                    .height(image.height);
                let mut tex = vkutil::Texture::new_from_extent_format_and_flags(
                    vk_extent,
                    vk_format,
                    vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                    vk_mem::AllocationCreateFlags::empty(),
                    &eng.context,
                );

                // copy image into staging buffer
                {
                    let mut res = Vec::new();
                    let copyvec = if image.format == gltf::image::Format::R8G8B8 {
                        for i in 0..image.pixels.len() / 3 {
                            res.push(image.pixels[i * 3]);
                            res.push(image.pixels[i * 3 + 1]);
                            res.push(image.pixels[i * 3 + 2]);
                            res.push(255);
                        }
                        &res
                    } else {
                        &image.pixels
                    };

                    assert!(copyvec.len() < alloc_info.size as usize, "Staging buffer too small for texture upload");

                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            copyvec.as_ptr(),
                            alloc_ptr as *mut u8,
                            copyvec.len(),
                        )
                    }
                }

                // copy staging buffer into texture and wait
                eng.execute_synchronous_on_queue(eng.context.queues.gfx_queue, |commandbuffer| {
                    let transition_barrier = vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .image(tex.vk_image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(vk::REMAINING_MIP_LEVELS)
                                .layer_count(vk::REMAINING_ARRAY_LAYERS),
                        );

                    let image_region = vk::BufferImageCopy::default()
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(
                            vk::Extent3D::default()
                                .width(tex.extent.width)
                                .height(tex.extent.height)
                                .depth(1),
                        );

                    unsafe {
                        eng.context.device.cmd_pipeline_barrier(
                            commandbuffer,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &[transition_barrier],
                        );

                        eng.context.device.cmd_copy_buffer_to_image(
                            commandbuffer,
                            eng.staging_buf.vk_buffer,
                            tex.vk_image,
                            vk::ImageLayout::GENERAL,
                            &[image_region],
                        );
                    }

                    // associate texture with its global bindless handle
                    tex.bindless_handle = Some(eng.bindless_handle);

                    // write into bindless descriptorset with new data
                    let image_info = [vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(tex.vk_imageview)
                        .sampler(tex.vk_sampler)];
                    let descriptor_write = [vk::WriteDescriptorSet::default()
                        .dst_set(eng.bindless_texture_descriptorset)
                        .dst_binding(0)
                        .dst_array_element(tex.bindless_handle.unwrap())
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info)];

                    unsafe {
                        eng.context
                            .device
                            .update_descriptor_sets(&descriptor_write, &[]);
                    }
                });
                eng.bindless_handle += 1;

                tex
            })
            .collect();

        //let texture_handles : Vec<u32> = document.textures().map(|texture| images[texture.source().index()].bindless_handle.unwrap()).collect();

        // read VBO, IBOs and materials
        let mut vboentry = Vec::new();
        let mut iboentry: Vec<u32> = Vec::new();
        let mut total_primitives = Vec::new();

        for mesh in document.meshes() {
            println!("Mesh #{}", mesh.index());

            let mut primitives: Vec<Primitive> = mesh.primitives().map(|primitive| {
                println!("- Primitive #{}", primitive.index());
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let vbo_start_size = vboentry.len() * std::mem::size_of::<VertexEntry>();
                let ibo_start_size = iboentry.len() * std::mem::size_of::<u32>();

                let mat = primitive.material();

                let color_tex = mat.pbr_metallic_roughness().base_color_texture().map_or(u32::MAX, |tex| {
                    images[tex.texture().source().index()].bindless_handle.unwrap()
                });
                let metal_rough_tex = mat.pbr_metallic_roughness().metallic_roughness_texture().map_or(u32::MAX, |tex| {
                    images[tex.texture().source().index()].bindless_handle.unwrap()
                });
                let emissive_tex = mat.emissive_texture().map_or(u32::MAX, |tex| {
                    images[tex.texture().source().index()].bindless_handle.unwrap()
                });
                let normal_tex = mat.normal_texture().map_or(u32::MAX, |tex| {
                    images[tex.texture().source().index()].bindless_handle.unwrap()
                });
                let occlusion_tex = mat.occlusion_texture().map_or(u32::MAX, |tex| {
                    images[tex.texture().source().index()].bindless_handle.unwrap()
                });
                
                // indices
                let mut indices_pushed = false;
                if let Some(iter) = reader.read_indices() {
                    for index in iter.into_u32() {
                        iboentry.push(index);
                        indices_pushed = true;
                    }
                }

                let uvs = reader.read_tex_coords(0).map_or(vec![], |c| c.into_f32().collect());
                let normals = reader.read_normals().map_or(vec![], |c| c.collect());

                // positions
                let vboentry_size = vboentry.len();
                if let Some(iter_pos) = reader.read_positions() {
                    for (idx,vertex_position) in iter_pos.enumerate() {
                        if indices_pushed == false {
                            iboentry.push(vboentry.len() as u32);
                        }

                        let uv = uvs.get(idx).map_or(cgmath::Vector2::new(0.0f32, 0.0f32), |uv| cgmath::Vector2::from(*uv));
                        let norm = normals.get(idx).map_or(cgmath::Vector3::new(0.0f32, 0.0f32, 0.0f32), |norm| cgmath::Vector3::from(*norm));

                        vboentry.push(VertexEntry {
                            pos: cgmath::Vector3::from(vertex_position),
                            norm: norm,
                            uv: uv,
                        });
                    }
                }

                let vbo_end_size = vboentry.len() * std::mem::size_of::<VertexEntry>();
                let ibo_end_size = iboentry.len() * std::mem::size_of::<u32>();

                let vbo_slice = vkutil::BufferSlice {
                    offset: vbo_start_size,
                    size: vbo_end_size - vbo_start_size,
                };

                let ibo_slice = vkutil::BufferSlice {
                    offset: ibo_start_size,
                    size: ibo_end_size - ibo_start_size,
                };
                
                Primitive {
                    base_color_tex : color_tex,
                    metallic_roughness_tex : metal_rough_tex,
                    occlusion_tex : occlusion_tex,
                    normal_tex : normal_tex,
                    emissive_tex : emissive_tex,
                    vbo_slice : vbo_slice,
                    ibo_slice : ibo_slice,
                }
            }).collect();

            total_primitives.append(&mut primitives);
        }
 
        // calculate offset into global bindless buffers
        let vbo_slice = eng.global_vbo.allocate_slice_of_size(vboentry.len() * std::mem::size_of::<VertexEntry>());
        let ibo_slice = eng.global_ibo.allocate_slice_of_size(iboentry.len() * std::mem::size_of::<u32>());

        total_primitives.iter_mut().for_each(|prim| {
            prim.ibo_slice.offset += ibo_slice.offset;
            prim.vbo_slice.offset += vbo_slice.offset;
        });
        
        // copy into staging buffer
        assert!(
            alloc_info.size > (vbo_slice.size + ibo_slice.size) as u64,
            "Staging buffer too small! {} < {}",
            alloc_info.size,
            (vbo_slice.size + ibo_slice.size)
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                vboentry.as_ptr(),
                alloc_ptr as *mut VertexEntry,
                vboentry.len(),
            );
            alloc_ptr = alloc_ptr.offset(vbo_slice.size as isize);

            std::ptr::copy_nonoverlapping(iboentry.as_ptr(), alloc_ptr as *mut u32, iboentry.len());
            alloc_ptr = alloc_ptr.offset(ibo_slice.size as isize);
        }

        // copy from staging to final VBO/IBO ssbos
        eng.execute_synchronous_on_queue(eng.context.queues.gfx_queue, |commandbuffer| {
            let vbo_regions = [ash::vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(vbo_slice.offset as u64)
                .size(vbo_slice.size as u64)];
            let ibo_regions = [ash::vk::BufferCopy::default()
                .src_offset(vbo_slice.size as u64) // offset into the staging buffer
                .dst_offset(ibo_slice.offset as u64)
                .size(ibo_slice.size as u64)];

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
            } 
        });

        Object {
            name,
            transform: cgmath::Matrix4::identity(),
            gltf_document: document,
            primitives: total_primitives,
            pipeline: pipeline,
            textures: images,
        }
    }
}
