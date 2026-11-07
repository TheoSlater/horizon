//! Draws the simulated water (`crate::sim_water`, a Horizon experimental setting): the solver's
//! cells go up as two textures each frame and a fixed grid mesh, finer than the cells, samples
//! them with a cubic B-spline, so the surface is smooth (C2) with no cell edges to see. Waves,
//! detail normals, Fresnel reflection, depth colour and foam are the shader's (`sim_water.wgsl`);
//! splash droplets are camera-facing billboards. Both draw at the end of the world pass, into
//! its targets (resolution, reversed depth, TAA jitter), blended over the world without writing
//! depth. The terrain shader leaves out vanilla water where this water stands (`Surface::hide`).

use glam::Mat4;

/// Mesh vertices along each cell's side.
const SUBDIVISIONS: u32 = 2;

/// How the water looks. Colours are linear 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    /// Water seen through a little depth, and through a lot.
    pub shallow: [f32; 3],
    pub deep: [f32; 3],
    pub foam: [f32; 3],
    /// How quickly the water turns opaque with depth, per block.
    pub opacity: f32,
    /// Reflectance looking straight down (Fresnel's F0).
    pub reflectance: f32,
    /// The sun glint's sharpness.
    pub shininess: f32,
    /// Wave height in open water, blocks.
    pub waves: f32,
    /// Blocks over which the water fades out at the region's edge.
    pub edge_fade: f32,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            // Vanilla's default water tint (#3F76E4) and its deep water fog, softened.
            shallow: [0.25, 0.47, 0.82],
            deep: [0.03, 0.10, 0.24],
            foam: [0.92, 0.95, 0.97],
            opacity: 0.9,
            reflectance: 0.02,
            shininess: 320.0,
            waves: 0.08,
            edge_fade: 3.0,
        }
    }
}

/// One frame of simulated water, camera-relative.
#[derive(Clone, Copy, Debug)]
pub struct Surface<'a> {
    /// Cells along a side.
    pub size: u32,
    /// A cell's side, blocks.
    pub cell: f32,
    /// Cell (0, 0)'s corner; heights are camera-relative already, so Y is 0.
    pub origin: [f32; 3],
    /// The corner's world X and Z (wrapped), so waves stay put as the region moves.
    pub phase: [f32; 2],
    /// Seconds of simulation.
    pub time: f32,
    pub rain: f32,
    /// Per cell, `z * size + x`: surface height, depth (negative: dry), velocity X and Z.
    pub heights: &'a [[f32; 4]],
    /// Per cell: foam, sky light and block light (0 to 255), unused.
    pub aux: &'a [[u8; 4]],
    /// Per droplet: position, radius, opacity, three unused.
    pub droplets: &'a [[f32; 8]],
    /// Where vanilla water is left out: X and Z bounds, then Y bounds and whether to.
    pub hide: [[f32; 4]; 2],
    pub style: Style,
}

/// The light and sky the water reflects, from the frame's own.
#[derive(Clone, Copy, Debug)]
pub struct Lighting {
    pub sky: [f32; 3],
    /// How strongly the sky lights the world (`sky_light_factor`).
    pub day: f32,
    pub fog_color: [f32; 4],
    /// Environmental start and end, render-distance start and end.
    pub fog: [f32; 4],
    pub sun: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    /// xyz the corner, w the cell size.
    origin: [f32; 4],
    /// Cells along a side, time, rain, wave height.
    grid: [f32; 4],
    /// World phase X and Z, edge fade, mesh subdivisions.
    phase: [f32; 4],
    /// Sky colour, w the daylight.
    sky: [f32; 4],
    fog_color: [f32; 4],
    fog: [f32; 4],
    sun: [f32; 4],
    /// Colours; w opacity, reflectance and shininess.
    shallow: [f32; 4],
    deep: [f32; 4],
    foam: [f32; 4],
    /// The camera's right and up, for droplets.
    right: [f32; 4],
    up: [f32; 4],
}

pub struct SimWater {
    surface_pipeline: wgpu::RenderPipeline,
    droplet_pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    textures: Option<(u32, wgpu::Texture, wgpu::Texture)>,
    indices: wgpu::Buffer,
    index_count: u32,
    droplets: wgpu::Buffer,
    droplet_count: u32,
    group: Option<wgpu::BindGroup>,
}

impl SimWater {
    /// Pipelines drawing into `color_format` over a reversed Depth32Float depth.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Simulated water"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sim_water.wgsl").into()),
        });
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Simulated water"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Simulated water"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let droplet_attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        let droplet_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: 32,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &droplet_attributes,
        })];
        let pipeline = |label: &str,
                        entry: [&str; 2],
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        topology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry[0]),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    // Seen from under the water too.
                    cull_mode: None,
                    ..Default::default()
                },
                // Blended over the world, writing no depth. Reversed depth: nearer is greater.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry[1]),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            surface_pipeline: pipeline(
                "Simulated water surface",
                ["vs_surface", "fs_surface"],
                &[],
                wgpu::PrimitiveTopology::TriangleList,
            ),
            droplet_pipeline: pipeline(
                "Simulated water droplets",
                ["vs_droplet", "fs_droplet"],
                &droplet_buffers,
                wgpu::PrimitiveTopology::TriangleStrip,
            ),
            layout,
            uniforms: buffer(
                "Simulated water uniforms",
                std::mem::size_of::<Uniforms>() as u64,
                wgpu::BufferUsages::UNIFORM,
            ),
            textures: None,
            indices: buffer("Simulated water indices", 0, wgpu::BufferUsages::INDEX),
            index_count: 0,
            droplets: buffer("Simulated water droplets", 0, wgpu::BufferUsages::VERTEX),
            droplet_count: 0,
            group: None,
        }
    }

    /// Uploads `surface` for this frame, drawn with `view_proj` (camera-relative world to clip,
    /// jittered as the world is); `inverse_view` gives the droplets' facing.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &Surface<'_>,
        lighting: &Lighting,
        view_proj: Mat4,
        inverse_view: Mat4,
    ) {
        let size = surface.size;
        let cells = (size * size) as usize;
        if size == 0 || surface.heights.len() < cells || surface.aux.len() < cells {
            self.group = None;
            return;
        }
        if self.textures.as_ref().is_none_or(|(s, ..)| *s != size) {
            let texture = |label, format| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            };
            self.textures = Some((
                size,
                texture("Simulated water heights", wgpu::TextureFormat::Rgba32Float),
                texture(
                    "Simulated water foam and light",
                    wgpu::TextureFormat::Rgba8Unorm,
                ),
            ));
            let m = size * SUBDIVISIONS;
            let indices: Vec<u32> = (0..m * m)
                .flat_map(|q| {
                    let (x, z) = (q % m, q / m);
                    let v = z * (m + 1) + x;
                    [v, v + 1, v + m + 1, v + 1, v + m + 2, v + m + 1]
                })
                .collect();
            self.index_count = indices.len() as u32;
            self.indices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Simulated water indices"),
                size: (indices.len() * 4) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&self.indices, 0, bytemuck::cast_slice(&indices));
        }
        let Some((_, heights, aux)) = &self.textures else {
            return;
        };
        let write = |texture: &wgpu::Texture, bytes: &[u8], texel: u32| {
            queue.write_texture(
                texture.as_image_copy(),
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * texel),
                    rows_per_image: Some(size),
                },
                texture.size(),
            );
        };
        write(heights, bytemuck::cast_slice(&surface.heights[..cells]), 16);
        write(aux, bytemuck::cast_slice(&surface.aux[..cells]), 4);

        self.droplet_count = surface.droplets.len() as u32;
        if self.droplet_count > 0 {
            let bytes: &[u8] = bytemuck::cast_slice(surface.droplets);
            if self.droplets.size() < bytes.len() as u64 {
                self.droplets = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Simulated water droplets"),
                    size: (bytes.len() as u64).next_power_of_two(),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&self.droplets, 0, bytes);
        }
        let s = &surface.style;
        let rgb = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            origin: [
                surface.origin[0],
                surface.origin[1],
                surface.origin[2],
                surface.cell,
            ],
            grid: [size as f32, surface.time, surface.rain, s.waves],
            phase: [
                surface.phase[0],
                surface.phase[1],
                s.edge_fade,
                SUBDIVISIONS as f32,
            ],
            sky: rgb(lighting.sky, lighting.day),
            fog_color: lighting.fog_color,
            fog: lighting.fog,
            sun: rgb(lighting.sun, 0.0),
            shallow: rgb(s.shallow, s.opacity),
            deep: rgb(s.deep, s.reflectance),
            foam: rgb(s.foam, s.shininess),
            right: inverse_view.x_axis.to_array(),
            up: inverse_view.y_axis.to_array(),
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (heights, aux) = (view(heights), view(aux));
        self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Simulated water"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&heights),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&aux),
                },
            ],
        }));
    }

    /// Forgets the frame: nothing draws until the next [`Self::prepare`].
    pub fn clear(&mut self) {
        self.group = None;
    }

    /// Draws what [`Self::prepare`] uploaded: the surface, then the droplets over it.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(group) = &self.group else {
            return;
        };
        pass.set_bind_group(0, group, &[]);
        pass.set_pipeline(&self.surface_pipeline);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
        if self.droplet_count > 0 {
            pass.set_pipeline(&self.droplet_pipeline);
            pass.set_vertex_buffer(0, self.droplets.slice(..));
            pass.draw(0..4, 0..self.droplet_count);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_shader_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("sim_water.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}
