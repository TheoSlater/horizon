//! Splash droplets: spray thrown up by impacts, wakes and waterfalls. Presentation only (the
//! solver's water does not lose or gain them); a droplet that falls back into the water reports
//! where it landed, so it can leave a ripple.

/// Droplets alive at once; the oldest give way.
pub const MAX_DROPLETS: usize = 4096;
const DRAG: f32 = 0.6;

/// A droplet in region-local coordinates: X and Z in blocks from the region's corner, Y the
/// world height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Droplet {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Seconds left, and its whole life (for the fade).
    pub life: f32,
    pub lifetime: f32,
    /// Radius, blocks.
    pub size: f32,
}

#[derive(Default)]
pub struct Splashes {
    pub drops: Vec<Droplet>,
}

impl Splashes {
    pub fn spawn(&mut self, drop: Droplet) {
        if self.drops.len() >= MAX_DROPLETS {
            self.drops.swap_remove(0);
        }
        self.drops.push(drop);
    }

    /// Moves every droplet by whole blocks (the region moved the other way).
    pub fn shift(&mut self, dx: f32, dz: f32) {
        for d in &mut self.drops {
            d.position[0] -= dx;
            d.position[2] -= dz;
        }
    }

    /// Advances `dt` seconds. `surface(x, z)` is the water's height there (`None`: dry), `bed`
    /// the ground's; a droplet falling back into water calls `landed` with it.
    pub fn update(
        &mut self,
        dt: f32,
        surface: impl Fn(f32, f32) -> Option<f32>,
        bed: impl Fn(f32, f32) -> f32,
        mut landed: impl FnMut(&Droplet),
    ) {
        let damp = (-DRAG * dt).exp();
        self.drops.retain_mut(|d| {
            d.life -= dt;
            if d.life <= 0.0 {
                return false;
            }
            d.velocity[1] -= super::GRAVITY * dt;
            for v in &mut d.velocity {
                *v *= damp;
            }
            for (p, v) in d.position.iter_mut().zip(d.velocity) {
                *p += v * dt;
            }
            let [x, y, z] = d.position;
            if d.velocity[1] < 0.0 {
                if let Some(s) = surface(x, z).filter(|&s| y <= s) {
                    d.position[1] = s;
                    landed(d);
                    return false;
                }
                if y <= bed(x, z) {
                    return false;
                }
            }
            true
        });
    }
}

/// A small xorshift generator: splashes need variety, not quality.
#[derive(Clone, Debug)]
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    pub fn unit(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drop_at(y: f32, vy: f32) -> Droplet {
        Droplet {
            position: [1.0, y, 1.0],
            velocity: [0.0, vy, 0.0],
            life: 5.0,
            lifetime: 5.0,
            size: 0.05,
        }
    }

    #[test]
    fn droplets_rise_fall_and_land_in_the_water() {
        let mut s = Splashes::default();
        s.spawn(drop_at(0.0, 5.0));
        let mut landed = Vec::new();
        let mut peak = 0.0f32;
        for _ in 0..200 {
            s.update(
                1.0 / 60.0,
                |_, _| Some(0.0),
                |_, _| -3.0,
                |d| landed.push(*d),
            );
            if let Some(d) = s.drops.first() {
                peak = peak.max(d.position[1]);
            }
        }
        assert!(peak > 0.3, "it rose: {peak}");
        assert!(s.drops.is_empty());
        assert_eq!(landed.len(), 1);
        assert_eq!(landed[0].position[1], 0.0);
    }

    #[test]
    fn droplets_over_dry_ground_vanish_without_a_ripple() {
        let mut s = Splashes::default();
        s.spawn(drop_at(1.0, 0.0));
        let mut landed = 0;
        for _ in 0..200 {
            s.update(1.0 / 60.0, |_, _| None, |_, _| 0.0, |_| landed += 1);
        }
        assert!(s.drops.is_empty());
        assert_eq!(landed, 0);
    }

    #[test]
    fn the_oldest_droplet_gives_way() {
        let mut s = Splashes::default();
        for _ in 0..MAX_DROPLETS + 10 {
            s.spawn(drop_at(1.0, 0.0));
        }
        assert_eq!(s.drops.len(), MAX_DROPLETS);
    }
}
