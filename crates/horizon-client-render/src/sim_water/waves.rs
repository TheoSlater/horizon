//! Dispersive surface waves on top of the shallow-water flow, after Tessendorf's iWave
//! ("Interactive Water Surfaces", 2004). The shallow-water equations move every wave at the same
//! speed, so a splash leaves a single ring; real water sorts a disturbance by wavelength, long
//! waves running ahead of short ones (deep-water dispersion, `omega^2 = g k`). This layer adds that:
//! rings spread into trains of ripples, and a body moving through the surface leaves a Kelvin
//! wake. It carries no mass; the flow underneath does.
//!
//! `h'' = -g * (K * h) - damping * h'`, with `K` the operator whose spectrum is `|k|` (the
//! vertical derivative of a potential flow), as a truncated convolution kernel.

use super::solver::GRAVITY;

/// The kernel's reach, in cells either side.
const RADIUS: i32 = 4;
const SPAN: usize = (2 * RADIUS + 1) as usize;
/// The fixed step of the wave equation, seconds (frames between steps interpolate).
pub const STEP: f32 = 1.0 / 30.0;
const MAX_STEPS: usize = 2;
/// Waves lower than this are still water: a calm surface costs no steps.
const CALM: f32 = 1.0e-4;
/// Viscous loss, per second: ripples ring for several seconds, as on still water.
const DAMPING: f32 = 0.25;
/// The highest a wave stands, as a fraction of the water under it (it breaks past that).
const BREAK: f32 = 0.4;

pub struct Waves {
    size: usize,
    pub height: Vec<f32>,
    previous: Vec<f32>,
    next: Vec<f32>,
    kernel: [f32; SPAN * SPAN],
    clock: f32,
}

impl Waves {
    pub fn new(size: usize, cell: f32) -> Self {
        Self {
            size,
            height: vec![0.0; size * size],
            previous: vec![0.0; size * size],
            next: vec![0.0; size * size],
            kernel: kernel(cell),
            clock: 0.0,
        }
    }

    /// Cell `i`'s height now, between the last two fixed steps, so the surface moves smoothly
    /// at any frame rate.
    pub fn at(&self, i: usize) -> f32 {
        let t = self.clock / STEP;
        self.previous[i] + (self.height[i] - self.previous[i]) * t
    }

    /// How fast cell `i`'s surface is rising or falling, blocks a second.
    pub fn rate(&self, i: usize) -> f32 {
        (self.height[i] - self.previous[i]) / STEP
    }

    /// Pushes the surface down by `amount` blocks at cell `i` (a pressure on the water).
    pub fn press(&mut self, i: usize, amount: f32) {
        self.height[i] -= amount;
        self.previous[i] -= amount;
    }

    /// Moves the field by `d` cells (the region moved that way).
    pub fn shift(&mut self, d: [i32; 2]) {
        let n = self.size as i32;
        for field in [&mut self.height, &mut self.previous] {
            let old = std::mem::take(field);
            *field = (0..n * n)
                .map(|i| {
                    let (x, z) = (i % n + d[0], i / n + d[1]);
                    if (0..n).contains(&x) && (0..n).contains(&z) {
                        old[(z * n + x) as usize]
                    } else {
                        0.0
                    }
                })
                .collect();
        }
    }

    /// Advances `dt` seconds in fixed steps; `depth` is the flow's water per cell, and dry
    /// cells hold the surface flat (the shore reflects).
    pub fn advance(&mut self, dt: f32, depth: &[f32]) {
        self.clock = (self.clock + dt).min(STEP * MAX_STEPS as f32);
        while self.clock >= STEP {
            self.clock -= STEP;
            self.step(depth);
        }
    }

    /// The cells where waves stand, `[x0, z0, x1, z1]` inclusive; `None`: calm everywhere.
    fn active(&self) -> Option<[i32; 4]> {
        let n = self.size;
        let mut b = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
        for (i, (h, p)) in self.height.iter().zip(&self.previous).enumerate() {
            if h.abs() > CALM || p.abs() > CALM {
                let (x, z) = ((i % n) as i32, (i / n) as i32);
                b = [b[0].min(x), b[1].min(z), b[2].max(x), b[3].max(z)];
            }
        }
        (b[0] <= b[2]).then_some(b)
    }

    fn step(&mut self, depth: &[f32]) {
        let n = self.size as i32;
        let Some([x0, z0, x1, z1]) = self.active() else {
            return;
        };
        // Waves reach the kernel's radius further each step.
        let (x0, z0) = ((x0 - RADIUS).max(0), (z0 - RADIUS).max(0));
        let (x1, z1) = ((x1 + RADIUS).min(n - 1), (z1 + RADIUS).min(n - 1));
        let a = DAMPING * STEP;
        let g = GRAVITY * STEP * STEP;
        // Outside the active box everything is calm: the next field is too.
        self.next.fill(0.0);
        for z in z0..=z1 {
            let inner_z = z >= RADIUS && z < n - RADIUS;
            for x in x0..=x1 {
                let i = (z * n + x) as usize;
                if depth[i] <= super::DRY {
                    continue;
                }
                let mut pull = 0.0;
                if inner_z && x >= RADIUS && x < n - RADIUS {
                    // Away from the edge: whole kernel rows straight from the field.
                    for (row, kz) in (-RADIUS..=RADIUS).enumerate() {
                        let start = ((z + kz) * n + x - RADIUS) as usize;
                        let field = &self.height[start..start + SPAN];
                        let weights = &self.kernel[row * SPAN..(row + 1) * SPAN];
                        pull += field.iter().zip(weights).map(|(h, k)| h * k).sum::<f32>();
                    }
                } else {
                    for kz in -RADIUS..=RADIUS {
                        let zz = (z + kz).clamp(0, n - 1);
                        let row = ((kz + RADIUS) as usize) * SPAN;
                        for kx in -RADIUS..=RADIUS {
                            let xx = (x + kx).clamp(0, n - 1);
                            pull += self.kernel[row + (kx + RADIUS) as usize]
                                * self.height[(zz * n + xx) as usize];
                        }
                    }
                }
                let h = (self.height[i] * (2.0 - a) - self.previous[i] - g * pull) / (1.0 + a);
                let limit = BREAK * depth[i];
                self.next[i] = h.clamp(-limit, limit);
            }
        }
        std::mem::swap(&mut self.previous, &mut self.height);
        std::mem::swap(&mut self.height, &mut self.next);
    }
}

/// The convolution kernel of `|k|` on cells of side `cell` blocks: the inverse Fourier
/// transform of `|k|` (softened toward the grid's limit, so the shortest waves don't alias),
/// truncated to [`RADIUS`] and balanced so a level surface feels no pull.
fn kernel(cell: f32) -> [f32; SPAN * SPAN] {
    const MODES: i32 = 48;
    let mut out = [0.0f32; SPAN * SPAN];
    let unit = std::f32::consts::TAU / (MODES as f32 * cell);
    let nyquist = std::f32::consts::PI / cell;
    for kz in -RADIUS..=RADIUS {
        for kx in -RADIUS..=RADIUS {
            let mut sum = 0.0f64;
            for m in -MODES / 2..MODES / 2 {
                for p in -MODES / 2..MODES / 2 {
                    let (wx, wz) = (m as f32 * unit, p as f32 * unit);
                    let k = (wx * wx + wz * wz).sqrt();
                    let soft = (-(k / nyquist).powi(2) * 2.0).exp();
                    let phase = (wx * kx as f32 + wz * kz as f32) * cell;
                    sum += f64::from(k * soft * phase.cos());
                }
            }
            out[((kz + RADIUS) as usize) * SPAN + (kx + RADIUS) as usize] =
                (sum / f64::from(MODES * MODES)) as f32;
        }
    }
    let total: f32 = out.iter().sum();
    out[(RADIUS as usize) * SPAN + RADIUS as usize] -= total;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where along +X the crest of a disturbance's ring has reached, in cells.
    fn front(w: &Waves, z: usize, threshold: f32) -> usize {
        (0..w.size)
            .rev()
            .find(|&x| w.height[z * w.size + x].abs() > threshold)
            .unwrap_or(0)
    }

    #[test]
    fn a_level_surface_stays_level() {
        let mut w = Waves::new(32, 0.5);
        let depth = vec![2.0; 32 * 32];
        w.advance(1.0, &depth);
        assert!(w.height.iter().all(|h| *h == 0.0));
    }

    #[test]
    fn a_splash_spreads_with_long_waves_ahead_of_short_ones() {
        // Deep-water group speed for a 4-block wave is about 1.25 blocks a second: after two
        // seconds the leading waves are some blocks out, not still at the centre nor at the edge.
        let size = 96;
        let mut w = Waves::new(size, 0.5);
        let depth = vec![3.0; size * size];
        let c = size / 2;
        w.press(c * size + c, 0.3);
        for _ in 0..60 {
            w.step(&depth);
        }
        let reach = (front(&w, c, 1.0e-3) - c) as f32 * 0.5;
        assert!((2.0..20.0).contains(&reach), "front at {reach} blocks");
        assert!(w.height.iter().all(|h| h.is_finite()));
    }

    #[test]
    fn ripples_die_down_on_their_own() {
        let size = 48;
        let mut w = Waves::new(size, 0.5);
        let depth = vec![2.0; size * size];
        w.press(24 * size + 24, 0.3);
        for _ in 0..30 * 30 {
            w.step(&depth);
        }
        let largest = w.height.iter().fold(0.0f32, |m, h| m.max(h.abs()));
        assert!(largest < 1.0e-3, "still {largest}");
    }
}
