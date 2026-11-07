//! Native extension point for Horizon's experimental realistic renderer.
//!
//! World geometry, visibility, camera transforms, entity drawing and texture
//! ownership stay in the main renderer. This module owns only its post-world
//! pipeline and the optional scratch target needed when existing FXAA runs.

use crate::graphics::{RealisticQuality, RendererMode};

struct Scratch {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: [u32; 2],
}

pub(super) struct Pipeline {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
    format: wgpu::TextureFormat,
    scratch: Option<Scratch>,
}

impl Pipeline {
    pub(super) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        _quality: RealisticQuality,
    ) -> Self {
        // Reuse Horizon's established full-screen copy shader. Realistic shaders
        // added in Phase 2 belong beside this module in `gfx/realistic/`.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Horizon realistic copy shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../world_post.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Horizon realistic copy bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Horizon realistic copy"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Horizon realistic copy"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_post"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_copy"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Horizon realistic copy"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            layout,
            sampler,
            pipeline,
            format,
            scratch: None,
        }
    }

    pub(super) fn scratch_view(
        &mut self,
        device: &wgpu::Device,
        size: [u32; 2],
    ) -> wgpu::TextureView {
        if self
            .scratch
            .as_ref()
            .is_none_or(|target| target.size != size)
        {
            self.scratch = None;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Horizon realistic FXAA intermediate"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.scratch = Some(Scratch {
                _texture: texture,
                view,
                size,
            });
        }
        self.scratch
            .as_ref()
            .expect("realistic scratch target was created")
            .view
            .clone()
    }

    pub(super) fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        target: &wgpu::TextureView,
        timing: Option<&mut super::pack_timing::PassTiming>,
    ) {
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Horizon realistic copy"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Horizon realistic copy"),
            timestamp_writes: timing.and_then(|timer| timer.writes("Realistic copy")),
            color_attachments: &attachments,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Shader packs take precedence because their own passes own the final world image.
pub(super) fn enabled(mode: RendererMode, shader_pack_active: bool) -> bool {
    mode == RendererMode::Realistic && !shader_pack_active
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_packs_suspend_the_realistic_pipeline() {
        assert!(!enabled(RendererMode::Vanilla, false));
        assert!(!enabled(RendererMode::Vanilla, true));
        assert!(enabled(RendererMode::Realistic, false));
        assert!(!enabled(RendererMode::Realistic, true));
    }

    #[test]
    fn copy_pipeline_runs_across_resizes() {
        let instance = crate::gfx::test_gpu::instance();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            return;
        };
        let Ok((device, queue)) = pollster::block_on(adapter.request_device(&Default::default()))
        else {
            return;
        };
        let mut pipeline = Pipeline::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            RealisticQuality::Medium,
        );
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        for size in [[64, 36], [91, 53]] {
            let source = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Realistic copy test source"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Realistic copy test target"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let source_view = source.create_view(&Default::default());
            let target_view = target.create_view(&Default::default());
            let scratch = pipeline.scratch_view(&device, size);
            let mut encoder = device.create_command_encoder(&Default::default());
            pipeline.run(&device, &mut encoder, &source_view, &scratch, None);
            pipeline.run(&device, &mut encoder, &scratch, &target_view, None);
            queue.submit([encoder.finish()]);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(pollster::block_on(validation.pop()).is_none());
    }
}
