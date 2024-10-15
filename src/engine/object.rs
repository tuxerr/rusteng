use crate::engine::shader_struct::MeshletEntry;
use crate::engine::vkutil::Texture;

use super::shader_struct::{self, MAX_MESHLET_VERTS, MAX_MESHLETS_TRIANGLES};
use super::{shader_struct::VertexEntry, vkutil, Engine};
use ash::vk;
use cgmath::SquareMatrix;
use derivative::Derivative;
use gltf;
use zerocopy::IntoBytes;
use std::collections::HashSet;
use std::path::Path;
use std::rc::Rc;
use std::{u32, u64};
use meshopt;

const MESHLETS_CONE_WEIGHT: f32 = 0.5;

fn gltf_format_to_vulkan_format(format: gltf::image::Format, srgb: bool) -> vk::Format {
    match format {
        gltf::image::Format::R8 => {
            if srgb {
                vk::Format::R8_SRGB
            } else {
                vk::Format::R8_UNORM
            }
        }
        gltf::image::Format::R8G8 => {
            if srgb {
                vk::Format::R8G8_SRGB
            } else {
                vk::Format::R8G8_UNORM
            }
        }
        gltf::image::Format::R8G8B8 => {
            if srgb {
                vk::Format::R8G8B8A8_SRGB
            } else {
                vk::Format::R8G8B8A8_UNORM
            }
        }
        gltf::image::Format::R8G8B8A8 => {
            if srgb {
                vk::Format::R8G8B8A8_SRGB
            } else {
                vk::Format::R8G8B8A8_UNORM
            }
        }
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
    pub meshlet_slice: vkutil::BufferSlice,
    #[derivative(Default(value = "u32::MAX"))]
    pub base_color_tex: u32,
    #[derivative(Default(value = "u32::MAX"))]
    pub metallic_roughness_tex: u32,
    #[derivative(Default(value = "u32::MAX"))]
    pub occlusion_tex: u32,
    #[derivative(Default(value = "u32::MAX"))]
    pub normal_tex: u32,
    #[derivative(Default(value = "u32::MAX"))]
    pub emissive_tex: u32,
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
            mat.pbr_metallic_roughness()
                .base_color_texture()
                .map(|tex| {
                    srgb_indices.insert(tex.texture().source().index());
                });

            mat.emissive_texture().map(|tex| {
                srgb_indices.insert(tex.texture().source().index());
            });
        });

        // textures need to be updated to add their bindless tex handles
        let images: Vec<Texture> = images
            .iter()
            .enumerate()
            .map(|(idx, image)| {
                // create image
                let vk_format =
                    gltf_format_to_vulkan_format(image.format, srgb_indices.contains(&idx));
                let vk_extent = vk::Extent2D::default()
                    .width(image.width)
                    .height(image.height);
                let mut tex = vkutil::Texture::new_from_extent_format_and_flags(
                    vk_extent,
                    vk_format,
                    vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
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
                    tex.load_pixel_data(eng, copyvec);
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

                eng.bindless_handle += 1;
                tex
            })
            .collect();

        // read VBO, IBOs and materials
        let mut vboentry = Vec::new();
        let mut iboentry: Vec<u32> = Vec::new();
        let mut meshletentry = Vec::new();
        let mut total_primitives = Vec::new();

        for mesh in document.meshes() {
            println!("Mesh #{}", mesh.index());

            let mut primitives: Vec<Primitive> = mesh
                .primitives()
                .map(|primitive| {
                    println!("- Primitive #{}", primitive.index());
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                    let vbo_start_size = vboentry.len() * std::mem::size_of::<VertexEntry>();
                    let ibo_start_size = iboentry.len() * std::mem::size_of::<u32>();
                    let meshlet_start_size = meshletentry.len() * std::mem::size_of::<MeshletEntry>();

                    let mut primitive_vbo = Vec::new();
                    let mut primitive_ibo = Vec::new();

                    let mat = primitive.material();

                    let color_tex =
                        mat.pbr_metallic_roughness()
                            .base_color_texture()
                            .map_or(u32::MAX, |tex| {
                                images[tex.texture().source().index()]
                                    .bindless_handle
                                    .unwrap()
                            });
                    let metal_rough_tex = mat
                        .pbr_metallic_roughness()
                        .metallic_roughness_texture()
                        .map_or(u32::MAX, |tex| {
                            images[tex.texture().source().index()]
                                .bindless_handle
                                .unwrap()
                        });
                    let emissive_tex = mat.emissive_texture().map_or(u32::MAX, |tex| {
                        images[tex.texture().source().index()]
                            .bindless_handle
                            .unwrap()
                    });
                    let normal_tex = mat.normal_texture().map_or(u32::MAX, |tex| {
                        images[tex.texture().source().index()]
                            .bindless_handle
                            .unwrap()
                    });
                    let occlusion_tex = mat.occlusion_texture().map_or(u32::MAX, |tex| {
                        images[tex.texture().source().index()]
                            .bindless_handle
                            .unwrap()
                    });

                    // indices
                    if let Some(iter) = reader.read_indices() {
                        for index in iter.into_u32() {
                            primitive_ibo.push(index);
                        }
                    }

                    let uvs = reader
                        .read_tex_coords(0)
                        .map_or(vec![], |c| c.into_f32().collect());
                    let normals = reader.read_normals().map_or(vec![], |c| c.collect());
                    let indices_pushed = !primitive_ibo.is_empty();

                    // positions
                    if let Some(iter_pos) = reader.read_positions() {
                        for (idx, vertex_position) in iter_pos.enumerate() {
                            if indices_pushed == false {
                                primitive_ibo.push(vboentry.len() as u32);
                            }

                            let uv = uvs
                                .get(idx)
                                .map_or([0.0f32, 0.0f32], |uv| *uv);
                            let norm = normals
                                .get(idx)
                                .map_or([0.0f32, 0.0f32, 0.0f32], |norm| *norm);

                            primitive_vbo.push(VertexEntry {
                                pos: vertex_position,
                                norm: norm,
                                uv: uv,
                            });
                        }
                    }

                    let (mut new_ibo, mut new_vbo) = Self::optimize_primitive(primitive_ibo, primitive_vbo);
                    let mut generated_meshlets = Self::generate_meshlets(&new_ibo, &new_vbo);

                    vboentry.append(&mut new_vbo);
                    iboentry.append(&mut new_ibo);
                    meshletentry.append(&mut generated_meshlets);

                    let vbo_end_size = vboentry.len() * std::mem::size_of::<VertexEntry>();
                    let ibo_end_size = iboentry.len() * std::mem::size_of::<u32>();
                    let meshlet_end_size = meshletentry.len() * std::mem::size_of::<MeshletEntry>();

                    let vbo_slice = vkutil::BufferSlice {
                        offset: vbo_start_size,
                        size: vbo_end_size - vbo_start_size,
                    };

                    let ibo_slice = vkutil::BufferSlice {
                        offset: ibo_start_size,
                        size: ibo_end_size - ibo_start_size,
                    };

                    let meshlet_slice = vkutil::BufferSlice {
                        offset: meshlet_start_size,
                        size: meshlet_end_size - meshlet_start_size,
                    };

                    Primitive {
                        base_color_tex: color_tex,
                        metallic_roughness_tex: metal_rough_tex,
                        occlusion_tex: occlusion_tex,
                        normal_tex: normal_tex,
                        emissive_tex: emissive_tex,
                        vbo_slice: vbo_slice,
                        ibo_slice: ibo_slice,
                        meshlet_slice: meshlet_slice,
                    }
                })
                .collect();

            total_primitives.append(&mut primitives);
        }

        // calculate offset into global bindless buffers
        let vbo_slice = eng
            .global_vbo
            .allocate_slice_of_size(vboentry.len() * std::mem::size_of::<VertexEntry>());
        let ibo_slice = eng
            .global_ibo
            .allocate_slice_of_size(iboentry.len() * std::mem::size_of::<u32>());
        let meshlet_slice = eng
            .global_meshlets
            .allocate_slice_of_size(meshletentry.len() * std::mem::size_of::<MeshletEntry>());

        total_primitives.iter_mut().for_each(|prim| {
            prim.ibo_slice.offset += ibo_slice.offset;
            prim.vbo_slice.offset += vbo_slice.offset;
            prim.meshlet_slice.offset += meshlet_slice.offset;
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

            std::ptr::copy_nonoverlapping(meshletentry.as_ptr(), alloc_ptr as *mut MeshletEntry, meshletentry.len());
            alloc_ptr = alloc_ptr.offset(meshlet_slice.size as isize);
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
            let meshlet_regions = [ash::vk::BufferCopy::default()
                .src_offset((vbo_slice.size + ibo_slice.size) as u64) // offset into the staging buffer
                .dst_offset(meshlet_slice.offset as u64)
                .size(meshlet_slice.size as u64)];

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
                eng.context.device.cmd_copy_buffer(
                    commandbuffer,
                    eng.staging_buf.vk_buffer,
                    eng.global_meshlets.vk_buffer,
                    &meshlet_regions,
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

    fn optimize_primitive(ibo: Vec<u32>, vbo: Vec<VertexEntry>) -> (Vec<u32>, Vec<VertexEntry>) {
        let vbo_size = vbo.len();
        let ibo_size = ibo.len();

        let (unique_vertices,remap_table) = meshopt::generate_vertex_remap(&vbo, Some(&ibo));
        let new_ibo = meshopt::remap_index_buffer(Some(&ibo), unique_vertices, &remap_table);
        let new_vbo = meshopt::remap_vertex_buffer(&vbo, unique_vertices, &remap_table);

        let mut new_ibo = meshopt::optimize_vertex_cache(&new_ibo, unique_vertices);

        let new_vbo = meshopt::optimize_vertex_fetch(&mut new_ibo, &new_vbo);

        println!("Remapped VBO from {} to {} vertices, and IBO from {} to {} indices", 
            vbo_size, 
            new_vbo.len(), 
            ibo_size,
            new_ibo.len()
        );

        (new_ibo, new_vbo)
    }

    fn generate_meshlets(ibo: &Vec<u32>, vbo: &Vec<VertexEntry>) -> Vec<shader_struct::MeshletEntry> {
        let u8_vbo_slice = vbo.as_slice().as_bytes();
        let vertex_stride = std::mem::size_of::<VertexEntry>();
        let position_offset = std::mem::offset_of!(VertexEntry,pos);
        let vs_adapter = meshopt::VertexDataAdapter::new(u8_vbo_slice, vertex_stride, position_offset).expect("Failure to create meshopt VS adapter");

        let meshlets = meshopt::build_meshlets(ibo.as_slice(), &vs_adapter, shader_struct::MAX_MESHLET_VERTS, shader_struct::MAX_MESHLETS_TRIANGLES, MESHLETS_CONE_WEIGHT);

        let meshlets : Vec<shader_struct::MeshletEntry> = meshlets.iter().map(|m| {
            let mut verts = [VertexEntry::default(); MAX_MESHLET_VERTS];
            let mut indices = [0; MAX_MESHLETS_TRIANGLES * 3];
            m.vertices.iter().enumerate().for_each(|(idx,v_idx)| {
                verts[idx] = vbo[*v_idx as usize];
            });
            m.triangles.iter().enumerate().for_each(|(idx,i_idx)| {
                indices[idx] = *i_idx;
            });

            let bounds = meshopt::compute_meshlet_bounds(m, &vs_adapter);
            let pos_radius = [bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius];

            shader_struct::MeshletEntry {
                pos_radius,
                verts,
                indices,
                triangle_count : (m.triangles.len() / 3) as u32,
                primitive_id: 0,
            }
        }).collect();
        
        println!("Generated {} meshlets", 
            meshlets.len()
        );
        meshlets
    }
}
