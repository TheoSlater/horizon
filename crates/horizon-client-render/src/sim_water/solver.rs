//! The shallow-water step, after flow-engine's `ShallowWater2D`: a first-order finite-volume
//! scheme with hydrostatic reconstruction at every face (Audusse et al. 2004), the HLL flux and
//! a CFL-limited explicit step. It is well balanced: a lake at rest stays at rest over any bed,
//! steps included. A wall is a bed no water reaches, so blocks and the region's edge need no
//! special case: the wall pushes back with the water's hydrostatic pressure.

/// Gravity on the water, in blocks per second squared (a block is a metre).
pub const GRAVITY: f32 = 9.81;
/// Below this depth a cell is dry: it holds no momentum and moves no water.
pub const DRY: f32 = 1.0e-3;
/// The bed of a cell water never enters (a block, an unloaded column, outside the region).
pub const WALL: f32 = 1.0e6;
/// The fastest the water may move, blocks per second: keeps the explicit step bounded.
const MAX_SPEED: f32 = 10.0;
/// The Courant number of a step.
const CFL: f32 = 0.45;
/// Velocity desingularisation depth (Kurganov and Petrova): near-dry cells move slowly.
const DESINGULARIZE: f32 = 0.01;

/// A face's flux in world axes (mass, x momentum, z momentum) and the hydrostatic corrections
/// to the normal momentum of the cell on its low side and on its high side.
#[derive(Clone, Copy, Debug, Default)]
struct Face {
    flux: [f32; 3],
    low: f32,
    high: f32,
}

/// A square grid of water columns, structure of arrays. Index `z * size + x`.
pub struct Grid {
    pub size: usize,
    /// A cell's side, in blocks.
    pub cell: f32,
    /// The height of the ground under each cell (absolute Y), or [`WALL`].
    pub bed: Vec<f32>,
    pub depth: Vec<f32>,
    /// Discharge (depth times velocity) along X and Z.
    pub qx: Vec<f32>,
    pub qz: Vec<f32>,
    vx: Vec<f32>,
    vz: Vec<f32>,
    /// `(size + 1) * size` faces across X, then `size * (size + 1)` across Z.
    x_faces: Vec<Face>,
    z_faces: Vec<Face>,
}

impl Grid {
    pub fn new(size: usize, cell: f32) -> Self {
        let cells = size * size;
        Self {
            size,
            cell,
            bed: vec![WALL; cells],
            depth: vec![0.0; cells],
            qx: vec![0.0; cells],
            qz: vec![0.0; cells],
            vx: vec![0.0; cells],
            vz: vec![0.0; cells],
            x_faces: vec![Face::default(); (size + 1) * size],
            z_faces: vec![Face::default(); size * (size + 1)],
        }
    }

    pub fn index(&self, x: usize, z: usize) -> usize {
        z * self.size + x
    }

    pub fn wet(&self, i: usize) -> bool {
        self.depth[i] > DRY
    }

    pub fn surface(&self, i: usize) -> f32 {
        self.bed[i] + self.depth[i]
    }

    /// The cell's velocity, desingularised near dry.
    pub fn velocity(&self, i: usize) -> [f32; 2] {
        [self.vx[i], self.vz[i]]
    }

    /// The water held, in cubic blocks.
    pub fn volume(&self) -> f32 {
        self.depth.iter().sum::<f32>() * self.cell * self.cell
    }

    /// Clears a cell's water and motion.
    pub fn clear(&mut self, i: usize) {
        self.depth[i] = 0.0;
        self.qx[i] = 0.0;
        self.qz[i] = 0.0;
        self.vx[i] = 0.0;
        self.vz[i] = 0.0;
    }

    fn update_velocities(&mut self) {
        for i in 0..self.depth.len() {
            let h = self.depth[i];
            if h <= DRY {
                self.qx[i] = 0.0;
                self.qz[i] = 0.0;
                self.vx[i] = 0.0;
                self.vz[i] = 0.0;
                continue;
            }
            let scale = 2.0 * h / (h * h + (h * h).max(DESINGULARIZE * DESINGULARIZE));
            let (mut u, mut v) = (self.qx[i] * scale, self.qz[i] * scale);
            let speed = (u * u + v * v).sqrt();
            if speed > MAX_SPEED {
                let k = MAX_SPEED / speed;
                u *= k;
                v *= k;
                self.qx[i] = u * h;
                self.qz[i] = v * h;
            }
            self.vx[i] = u;
            self.vz[i] = v;
        }
    }

    /// The longest step the scheme stays stable for, seconds.
    pub fn stable_step(&mut self) -> f32 {
        self.update_velocities();
        let mut fastest = 0.0f32;
        for i in 0..self.depth.len() {
            if self.depth[i] > DRY {
                let c = (GRAVITY * self.depth[i]).sqrt();
                fastest = fastest.max(self.vx[i].abs().max(self.vz[i].abs()) + c);
            }
        }
        if fastest <= f32::EPSILON {
            f32::INFINITY
        } else {
            CFL * self.cell / fastest
        }
    }

    /// The state of cell `(x, z)` for a face, or a wall outside the grid.
    fn side(&self, x: isize, z: isize) -> (f32, f32, f32, f32) {
        let n = self.size as isize;
        if x < 0 || z < 0 || x >= n || z >= n {
            return (WALL, 0.0, 0.0, 0.0);
        }
        let i = z as usize * self.size + x as usize;
        (self.bed[i], self.depth[i], self.vx[i], self.vz[i])
    }

    /// Advances `dt` seconds; `dt` must not exceed [`Self::stable_step`], which also refreshes
    /// the velocities this step reads.
    pub fn step(&mut self, dt: f32) {
        let n = self.size;
        for z in 0..n {
            for x in 0..=n {
                let (bl, hl, ul, vl) = self.side(x as isize - 1, z as isize);
                let (br, hr, ur, vr) = self.side(x as isize, z as isize);
                let (f, low, high) = face(bl, hl, ul, vl, br, hr, ur, vr);
                self.x_faces[z * (n + 1) + x] = Face { flux: f, low, high };
            }
        }
        for z in 0..=n {
            for x in 0..n {
                let (bl, hl, ul, vl) = self.side(x as isize, z as isize - 1);
                let (br, hr, ur, vr) = self.side(x as isize, z as isize);
                // Normal frame: Z is the normal, X the transverse direction.
                let (f, low, high) = face(bl, hl, vl, ul, br, hr, vr, ur);
                self.z_faces[z * n + x] = Face {
                    flux: [f[0], f[2], f[1]],
                    low,
                    high,
                };
            }
        }
        let k = dt / self.cell;
        for z in 0..n {
            for x in 0..n {
                let i = z * n + x;
                let west = self.x_faces[z * (n + 1) + x];
                let east = self.x_faces[z * (n + 1) + x + 1];
                let north = self.z_faces[z * n + x];
                let south = self.z_faces[(z + 1) * n + x];
                let h = self.depth[i]
                    - k * (east.flux[0] - west.flux[0] + south.flux[0] - north.flux[0]);
                if h <= DRY {
                    self.depth[i] = h.max(0.0);
                    self.qx[i] = 0.0;
                    self.qz[i] = 0.0;
                    continue;
                }
                self.depth[i] = h;
                self.qx[i] -= k
                    * ((east.flux[1] + east.low) - (west.flux[1] + west.high) + south.flux[1]
                        - north.flux[1]);
                self.qz[i] -= k
                    * (east.flux[2] - west.flux[2] + (south.flux[2] + south.low)
                        - (north.flux[2] + north.high));
            }
        }
    }

    /// Manning bed friction over `dt`, implicit so thin sheets stop instead of overshooting.
    pub fn friction(&mut self, dt: f32, manning: f32) {
        for i in 0..self.depth.len() {
            let h = self.depth[i];
            if h <= DRY {
                continue;
            }
            let speed = (self.qx[i] * self.qx[i] + self.qz[i] * self.qz[i]).sqrt() / h;
            let k = 1.0 / (1.0 + dt * GRAVITY * manning * manning * speed / h.powf(4.0 / 3.0));
            self.qx[i] *= k;
            self.qz[i] *= k;
        }
    }

    /// Advances `duration` seconds in stable substeps, at most `max_steps` of them; returns how
    /// many it took.
    pub fn advance(&mut self, duration: f32, max_steps: usize, manning: f32) -> usize {
        let mut left = duration;
        let mut steps = 0;
        while left > 1.0e-5 && steps < max_steps {
            let dt = self.stable_step().min(left);
            self.step(dt);
            self.friction(dt, manning);
            left -= dt;
            steps += 1;
        }
        self.update_velocities();
        steps
    }
}

/// The flux across a face between a low cell `l` and a high cell `r` in the face's normal frame
/// (u normal, v transverse), and the hydrostatic corrections for each cell's normal momentum.
#[allow(clippy::too_many_arguments)]
fn face(
    bed_l: f32,
    h_l: f32,
    u_l: f32,
    v_l: f32,
    bed_r: f32,
    h_r: f32,
    u_r: f32,
    v_r: f32,
) -> ([f32; 3], f32, f32) {
    let interface = bed_l.max(bed_r);
    let hl = (bed_l + h_l - interface).max(0.0);
    let hr = (bed_r + h_r - interface).max(0.0);
    let low = 0.5 * GRAVITY * (h_l * h_l - hl * hl);
    let high = 0.5 * GRAVITY * (h_r * h_r - hr * hr);
    (hll(hl, u_l, v_l, hr, u_r, v_r), low, high)
}

/// flow-engine's `hll_flux_2d` in the normal frame: mass, normal and transverse momentum.
fn hll(hl: f32, ul: f32, vl: f32, hr: f32, ur: f32, vr: f32) -> [f32; 3] {
    if hl <= DRY && hr <= DRY {
        return [0.0; 3];
    }
    let (ul, vl) = if hl > DRY { (ul, vl) } else { (0.0, 0.0) };
    let (ur, vr) = if hr > DRY { (ur, vr) } else { (0.0, 0.0) };
    let (cl, cr) = ((GRAVITY * hl).sqrt(), (GRAVITY * hr).sqrt());
    let sl = (ul - cl).min(ur - cr);
    let sr = (ul + cl).max(ur + cr);
    let fl = physical(hl, ul, vl);
    let fr = physical(hr, ur, vr);
    if sl >= 0.0 {
        return fl;
    }
    if sr <= 0.0 {
        return fr;
    }
    let d = sr - sl;
    if d <= 1.0e-6 {
        return [0, 1, 2].map(|k| 0.5 * (fl[k] + fr[k]));
    }
    let left = [hl, hl * ul, hl * vl];
    let right = [hr, hr * ur, hr * vr];
    [0, 1, 2].map(|k| (sr * fl[k] - sl * fr[k] + sl * sr * (right[k] - left[k])) / d)
}

fn physical(h: f32, u: f32, v: f32) -> [f32; 3] {
    if h <= DRY {
        return [0.0; 3];
    }
    let q = h * u;
    [q, q * u + 0.5 * GRAVITY * h * h, q * v]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basin(size: usize, bed: impl Fn(usize, usize) -> f32) -> Grid {
        let mut g = Grid::new(size, 0.5);
        for z in 0..size {
            for x in 0..size {
                let i = g.index(x, z);
                g.bed[i] = bed(x, z);
            }
        }
        g
    }

    #[test]
    fn a_lake_at_rest_over_steps_stays_at_rest() {
        // Terraced bed, walls inside: the surface at 70 must not move.
        let mut g = basin(16, |x, z| match (x + z) % 5 {
            0 => 66.0,
            1 => 67.5,
            2 => 64.0,
            3 if x == 7 => WALL,
            _ => 68.0,
        });
        for i in 0..g.depth.len() {
            if g.bed[i] < WALL {
                g.depth[i] = 70.0 - g.bed[i];
            }
        }
        g.advance(5.0, 10_000, 0.025);
        for i in 0..g.depth.len() {
            if g.bed[i] < WALL {
                assert!((g.surface(i) - 70.0).abs() < 1.0e-3, "cell {i}");
                assert!(g.velocity(i)[0].abs() < 1.0e-3 && g.velocity(i)[1].abs() < 1.0e-3);
            }
        }
    }

    #[test]
    fn a_closed_basin_keeps_its_water() {
        let mut g = basin(24, |x, _| 60.0 + x as f32 * 0.1);
        for z in 0..24 {
            for x in 0..6 {
                let i = g.index(x, z);
                g.depth[i] = 2.0;
            }
        }
        let before = g.volume();
        g.advance(8.0, 100_000, 0.0);
        assert!((g.volume() - before).abs() / before < 1.0e-3);
    }

    #[test]
    fn water_runs_downhill_and_settles_level() {
        // A ramp falling toward +X, water released at the top: it ends at the bottom, level.
        let mut g = basin(32, |x, _| 70.0 - x as f32 * 0.25);
        for z in 0..32 {
            for x in 0..4 {
                let i = g.index(x, z);
                g.depth[i] = 0.5;
            }
        }
        g.advance(40.0, 1_000_000, 0.03);
        let (top, bottom) = (g.index(1, 16), g.index(30, 16));
        assert!(g.depth[top] < 0.01, "the top drained: {}", g.depth[top]);
        assert!(
            g.depth[bottom] > 0.2,
            "the bottom filled: {}",
            g.depth[bottom]
        );
        let wet: Vec<f32> = (0..32)
            .map(|x| g.index(x, 16))
            .filter(|&i| g.depth[i] > 0.05)
            .map(|i| g.surface(i))
            .collect();
        let (lo, hi) = wet
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &s| (a.min(s), b.max(s)));
        assert!(hi - lo < 0.05, "the pool settles level: {lo}..{hi}");
    }

    #[test]
    fn a_wall_holds_water_back() {
        let mut g = basin(8, |x, _| if x == 4 { WALL } else { 64.0 });
        for z in 0..8 {
            for x in 0..4 {
                let i = g.index(x, z);
                g.depth[i] = 1.0;
            }
        }
        g.advance(5.0, 100_000, 0.0);
        for z in 0..8 {
            for x in 5..8 {
                assert_eq!(g.depth[g.index(x, z)], 0.0);
            }
        }
    }
}
