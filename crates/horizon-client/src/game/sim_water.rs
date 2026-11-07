//! Simulated water (Horizon experimental setting, `Graphics::simulated_water`): steps
//! `horizon_client::sim_water` once a frame around the camera, reading the client's world, and
//! hands the renderer its surface. Presentation only: the game's water stays vanilla.

use super::Game;
use horizon_client::sim_water::{Body, Input, WaterSim, scan_column};
use std::time::Instant;

/// Ticks a second: `PlayerMotion::velocity` is per tick.
const TICKS_PER_SECOND: f64 = 20.0;

pub(super) struct SimWaterState {
    sim: Box<WaterSim>,
    last: Instant,
}

impl SimWaterState {
    pub(super) fn surface(&self) -> horizon_client::gfx::sim_water::Surface<'_> {
        self.sim.surface()
    }
}

impl Game<'_> {
    /// Steps the simulated water for this frame, seen from `camera` with `rain` falling. Off
    /// (and its memory freed) without the setting, on the title screen, or while a shader pack
    /// draws the world.
    pub(super) fn update_sim_water(&mut self, camera: glam::DVec3, rain: f32) {
        let on = self.settings.horizon.simulated_water
            && self.title.is_none()
            && self
                .renderer
                .as_ref()
                .is_some_and(|r| !r.shader_pack_active());
        if !on {
            self.sim_water = None;
            return;
        }
        let player = self.player.as_ref().map(|p| Body {
            position: [p.position.x, p.position.y, p.position.z],
            velocity: [p.velocity.x, p.velocity.y, p.velocity.z]
                .map(|v| (v * TICKS_PER_SECOND) as f32),
        });
        let now = Instant::now();
        let state = self.sim_water.get_or_insert_with(|| SimWaterState {
            sim: Box::default(),
            last: now,
        });
        let input = Input {
            camera: camera.to_array(),
            dt: now.duration_since(state.last).as_secs_f32(),
            rain,
            player,
        };
        state.last = now;
        let (world, facts) = (&self.world, self.res.facts());
        state
            .sim
            .update(&input, &mut |x, z, y| scan_column(world, facts, x, z, y));
    }
}
