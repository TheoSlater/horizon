//! Simulated water (Horizon, experimental, off by default: Video Settings, Experimental
//! renderer, Simulated Water). A shallow-water solver over a square region around the camera,
//! seeded from the world's water and terrain: water runs downhill under gravity, pools level,
//! falls off ledges, carries ripples, wakes and waves, and churns foam and spray where it is
//! fast, converging or plunging. Vanilla source blocks act as springs holding their level, so
//! lakes and seas keep theirs and a placed source feeds a stream that finds its own way down.
//!
//! Presentation only: gameplay water (swimming, flow pushing entities, the blocks a server
//! sends) stays vanilla. Inside the region the renderer draws this water instead of vanilla's
//! (`gfx::sim_water`, the terrain shader's `sim_water_hides`).
//!
//! The solver follows flow-engine's `ShallowWater2D` (`solver`); a column's water is at most
//! [`MAX_DEPTH`] deep for the solver, which keeps deep oceans' waves at a stable step size.

mod solver;
mod splash;

use crate::blocks::{BlockFacts, Fluid};
use crate::gfx::sim_water::{Style, Surface};
use crate::world::World;
pub use solver::{DRY, GRAVITY, Grid, WALL};
pub use splash::{Droplet, MAX_DROPLETS, Rng, Splashes};

/// Solver cells along a block.
pub const CELLS_PER_BLOCK: usize = 2;
/// The region's side, in blocks.
pub const REGION_BLOCKS: usize = 64;
/// The region's side, in cells.
pub const SIZE: usize = REGION_BLOCKS * CELLS_PER_BLOCK;
const CELL: f32 = 1.0 / CELLS_PER_BLOCK as f32;
/// The region moves in whole steps of this many blocks, keeping the solver's state.
const RECENTER_STEP: i32 = 8;
/// How far above and below the camera a column is scanned for its water and ground.
const SCAN_UP: i32 = 12;
const SCAN_DOWN: i32 = 32;
/// How far the scan climbs out of water the camera is under, looking for its surface.
const SCAN_CLIMB: i32 = 32;
/// The deepest water the solver holds in a column; deeper water rests on a virtual bed.
pub const MAX_DEPTH: f32 = 3.0;
/// Block rows rescanned a frame, so placed and broken blocks reach the solver.
const ROWS_PER_FRAME: usize = 2;
/// The longest frame simulated; a longer one (a hitch) slows the water instead.
const MAX_FRAME: f32 = 0.05;
const MAX_SUBSTEPS: usize = 6;
/// Manning's roughness of the bed.
const MANNING: f32 = 0.025;
/// How quickly a vanilla source restores its level, per second.
const SPRING_RATE: f32 = 1.5;
/// Sheets thinner than this, fed by no spring, soak away at [`SOAK_RATE`] blocks a second.
const THIN: f32 = 0.02;
const SOAK_RATE: f32 = 0.01;
/// Cells along the edge whose motion is damped, so waves leave instead of reflecting.
const SPONGE: usize = 6;
/// Blocks along the edge where vanilla water still draws, under the simulated water's fade.
const HIDE_INSET: f32 = 2.0;

/// What a column of the world holds for the solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Column {
    /// The top of the ground water rests on; `None`: not loaded (a wall).
    pub bed: Option<f32>,
    /// The vanilla water's surface over it.
    pub water: Option<f32>,
    /// That water is a source block: a spring holding its level.
    pub source: bool,
    /// Sky and block light at the surface, 0 to 15.
    pub sky: u8,
    pub block: u8,
}

impl Column {
    pub const UNLOADED: Self = Self {
        bed: None,
        water: None,
        source: false,
        sky: 15,
        block: 0,
    };
}

/// Reads the column at `(x, z)` near height `y`: its top water (falling water skipped: the
/// solver makes its own falls) and the ground under it. Leaves don't hold water up.
pub fn scan_column(world: &World, facts: &BlockFacts, x: i32, z: i32, y: i32) -> Column {
    if !world.loaded(x, z) {
        return Column::UNLOADED;
    }
    let at = |y: i32| facts.get(world.block(x, y, z));
    let still_water = |y: i32| {
        let s = at(y);
        s.fluid == Fluid::Water && !s.fluid_falling
    };
    let mut top = y + SCAN_UP;
    while still_water(top) && top < y + SCAN_UP + SCAN_CLIMB {
        top += 1;
    }
    let bottom = y - SCAN_DOWN;
    let mut water: Option<(f32, bool, i32)> = None;
    let mut bed = bottom as f32;
    for yy in (bottom..=top).rev() {
        let s = at(yy);
        let solid = (s.full_collision || s.legacy_solid) && !s.name.ends_with("_leaves");
        if s.fluid == Fluid::Water && !solid {
            if water.is_none() && !s.fluid_falling {
                // FluidState.getOwnHeight: amount / 9, a source 8 / 9.
                let amount = f32::from(s.fluid_level.clamp(1, 8));
                water = Some((yy as f32 + amount / 9.0, s.fluid_level >= 8, yy));
            }
            continue;
        }
        if solid || water.is_some() {
            bed = (yy + 1) as f32;
            break;
        }
    }
    let light_y = water.map_or(bed as i32, |(_, _, y)| y);
    let light = world.light(x, light_y, z);
    Column {
        bed: Some(bed),
        water: water.map(|(w, _, _)| w),
        source: water.is_some_and(|(_, s, _)| s),
        sky: light >> 4,
        block: light & 15,
    }
}

/// A moving body in the water (the player), world coordinates, velocity in blocks a second.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub position: [f64; 3],
    pub velocity: [f32; 3],
}

/// One frame's input.
#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub camera: [f64; 3],
    /// Seconds since the last frame.
    pub dt: f32,
    /// Rain strength, 0 to 1.
    pub rain: f32,
    pub player: Option<Body>,
}

/// The simulated water around the camera.
pub struct WaterSim {
    grid: Grid,
    foam: Vec<f32>,
    foam_next: Vec<f32>,
    /// The surface a vanilla source holds the cell at, NaN where none does.
    spring: Vec<f32>,
    /// Sky and block light per cell.
    light: Vec<[u8; 2]>,
    /// Vanilla water stood here when the column was last read.
    vanilla: Vec<bool>,
    /// World block X and Z of cell (0, 0)'s corner.
    origin: [i32; 2],
    scan_y: i32,
    scan_row: usize,
    seeded: bool,
    time: f32,
    rain: f32,
    rng: Rng,
    splashes: Splashes,
    /// The player's feet above the surface last frame.
    swimmer: Option<f32>,
    pub style: Style,
    // The frame's output, camera-relative.
    heights: Vec<[f32; 4]>,
    aux: Vec<[u8; 4]>,
    droplets: Vec<[f32; 8]>,
    relative_origin: [f32; 3],
    hide: [[f32; 4]; 2],
}

impl Default for WaterSim {
    fn default() -> Self {
        Self::new()
    }
}

impl WaterSim {
    pub fn new() -> Self {
        let cells = SIZE * SIZE;
        Self {
            grid: Grid::new(SIZE, CELL),
            foam: vec![0.0; cells],
            foam_next: vec![0.0; cells],
            spring: vec![f32::NAN; cells],
            light: vec![[15, 0]; cells],
            vanilla: vec![false; cells],
            origin: [0, 0],
            scan_y: 0,
            scan_row: 0,
            seeded: false,
            time: 0.0,
            rain: 0.0,
            rng: Rng::new(0x9e37_79b9),
            splashes: Splashes::default(),
            swimmer: None,
            style: Style::default(),
            heights: vec![[0.0; 4]; cells],
            aux: vec![[0; 4]; cells],
            droplets: Vec::new(),
            relative_origin: [0.0; 3],
            hide: [[0.0; 4]; 2],
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn origin(&self) -> [i32; 2] {
        self.origin
    }

    /// Advances one frame. `sample(x, z, y)` reads the world's column at `(x, z)` near `y`
    /// ([`scan_column`] in the game).
    pub fn update(&mut self, input: &Input, sample: &mut dyn FnMut(i32, i32, i32) -> Column) {
        let dt = input.dt.clamp(0.0, MAX_FRAME);
        self.time += dt;
        self.rain = input.rain.clamp(0.0, 1.0);
        self.follow(input.camera, sample);
        self.rescan_rows(sample);
        match input.player {
            Some(body) => self.body(body, dt),
            None => self.swimmer = None,
        }
        self.rainfall(dt);
        self.springs(dt);
        self.grid.advance(dt, MAX_SUBSTEPS, MANNING);
        self.plunges(dt);
        self.update_foam(dt);
        self.update_splashes(dt);
        self.build_output(input.camera);
    }

    /// What the renderer draws this frame.
    pub fn surface(&self) -> Surface<'_> {
        Surface {
            size: SIZE as u32,
            cell: CELL,
            origin: self.relative_origin,
            phase: [
                self.origin[0].rem_euclid(4096) as f32,
                self.origin[1].rem_euclid(4096) as f32,
            ],
            time: self.time,
            rain: self.rain,
            heights: &self.heights,
            aux: &self.aux,
            droplets: &self.droplets,
            hide: self.hide,
            style: self.style,
        }
    }

    /// Keeps the region centred on the camera, moving it in whole steps.
    fn follow(&mut self, camera: [f64; 3], sample: &mut dyn FnMut(i32, i32, i32) -> Column) {
        let half = REGION_BLOCKS as i32 / 2;
        let snap = |v: f64| (v.floor() as i32).div_euclid(RECENTER_STEP) * RECENTER_STEP - half;
        let origin = [snap(camera[0]), snap(camera[2])];
        let y = camera[1].floor() as i32;
        if !self.seeded {
            self.seeded = true;
            self.origin = origin;
            self.scan_y = y;
            for bz in 0..REGION_BLOCKS {
                for bx in 0..REGION_BLOCKS {
                    self.read_column(bx, bz, true, sample);
                }
            }
            return;
        }
        if (y - self.scan_y).abs() > RECENTER_STEP {
            self.scan_y = y;
        }
        if origin == self.origin {
            return;
        }
        let shift = [origin[0] - self.origin[0], origin[1] - self.origin[1]];
        self.shift(shift);
        self.origin = origin;
        let n = REGION_BLOCKS as i32;
        for bz in 0..n {
            for bx in 0..n {
                let (ox, oz) = (bx + shift[0], bz + shift[1]);
                if !(0..n).contains(&ox) || !(0..n).contains(&oz) {
                    self.read_column(bx as usize, bz as usize, true, sample);
                }
            }
        }
    }

    /// Moves every cell's state by `blocks` (the region moved that way).
    fn shift(&mut self, blocks: [i32; 2]) {
        let d = blocks.map(|b| b * CELLS_PER_BLOCK as i32);
        fn moved<T: Copy>(v: &mut Vec<T>, d: [i32; 2], fill: T) {
            let n = SIZE as i32;
            let old = std::mem::take(v);
            *v = (0..n * n)
                .map(|i| {
                    let (x, z) = (i % n + d[0], i / n + d[1]);
                    if (0..n).contains(&x) && (0..n).contains(&z) {
                        old[(z * n + x) as usize]
                    } else {
                        fill
                    }
                })
                .collect();
        }
        moved(&mut self.grid.bed, d, WALL);
        moved(&mut self.grid.depth, d, 0.0);
        moved(&mut self.grid.qx, d, 0.0);
        moved(&mut self.grid.qz, d, 0.0);
        moved(&mut self.foam, d, 0.0);
        moved(&mut self.spring, d, f32::NAN);
        moved(&mut self.light, d, [15, 0]);
        moved(&mut self.vanilla, d, false);
        self.splashes.shift(blocks[0] as f32, blocks[1] as f32);
    }

    fn rescan_rows(&mut self, sample: &mut dyn FnMut(i32, i32, i32) -> Column) {
        for _ in 0..ROWS_PER_FRAME {
            let bz = self.scan_row;
            for bx in 0..REGION_BLOCKS {
                self.read_column(bx, bz, false, sample);
            }
            self.scan_row = (self.scan_row + 1) % REGION_BLOCKS;
        }
    }

    /// Reads block column `(bx, bz)` of the region into its cells. `fresh`: the cells are new
    /// to the region and take the world's water as it is; otherwise the solver's water stays,
    /// only following a changed bed.
    fn read_column(
        &mut self,
        bx: usize,
        bz: usize,
        fresh: bool,
        sample: &mut dyn FnMut(i32, i32, i32) -> Column,
    ) {
        let column = sample(
            self.origin[0] + bx as i32,
            self.origin[1] + bz as i32,
            self.scan_y,
        );
        for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let i = self
                .grid
                .index(bx * CELLS_PER_BLOCK + dx, bz * CELLS_PER_BLOCK + dz);
            self.apply_column(i, &column, fresh);
        }
    }

    fn apply_column(&mut self, i: usize, column: &Column, fresh: bool) {
        let g = &mut self.grid;
        let Some(ground) = column.bed else {
            g.bed[i] = WALL;
            g.clear(i);
            self.spring[i] = f32::NAN;
            self.vanilla[i] = false;
            return;
        };
        let bed = column.water.map_or(ground, |w| ground.max(w - MAX_DEPTH));
        if fresh || g.bed[i] >= WALL {
            g.bed[i] = bed;
            g.clear(i);
            g.depth[i] = column.water.map_or(0.0, |w| (w - bed).max(0.0));
            self.foam[i] = 0.0;
        } else if (g.bed[i] - bed).abs() > 1.0e-4 {
            let surface = g.wet(i).then(|| g.surface(i));
            g.bed[i] = bed;
            match surface {
                Some(s) if s > bed + DRY => g.depth[i] = s - bed,
                _ => g.clear(i),
            }
        }
        self.spring[i] = match column.water {
            Some(w) if column.source => w,
            _ => f32::NAN,
        };
        self.vanilla[i] = column.water.is_some();
        self.light[i] = [column.sky, column.block];
    }

    /// Springs restore their level; thin unfed sheets soak away; the sponge damps the edge.
    fn springs(&mut self, dt: f32) {
        let fill = 1.0 - (-SPRING_RATE * dt).exp();
        let n = SIZE;
        for z in 0..n {
            for x in 0..n {
                let i = z * n + x;
                let g = &mut self.grid;
                if g.bed[i] >= WALL {
                    continue;
                }
                let spring = self.spring[i];
                if spring.is_finite() {
                    g.depth[i] = (g.depth[i] + (spring - g.surface(i)) * fill).max(0.0);
                } else if g.depth[i] > 0.0 && g.depth[i] < THIN {
                    g.depth[i] = (g.depth[i] - SOAK_RATE * dt).max(0.0);
                }
                let edge = x.min(z).min(n - 1 - x).min(n - 1 - z);
                if edge < SPONGE {
                    let w = 1.0 - edge as f32 / SPONGE as f32;
                    let damp = (-4.0 * w * dt).exp();
                    g.qx[i] *= damp;
                    g.qz[i] *= damp;
                    // Unfed water reaching the edge leaves the region.
                    if edge < 2 && !spring.is_finite() {
                        g.depth[i] *= (-0.8 * dt).exp();
                    }
                }
                if g.depth[i] <= DRY {
                    g.clear(i);
                }
            }
        }
    }

    /// Region-local block coordinates of a world position.
    fn local(&self, x: f64, z: f64) -> (f32, f32) {
        (
            (x - f64::from(self.origin[0])) as f32,
            (z - f64::from(self.origin[1])) as f32,
        )
    }

    fn body(&mut self, body: Body, dt: f32) {
        let (lx, lz) = self.local(body.position[0], body.position[2]);
        let Some(surface) = surface_at(&self.grid, lx, lz) else {
            self.swimmer = None;
            return;
        };
        let feet = body.position[1] as f32;
        let height = feet - surface;
        let before = self.swimmer.replace(height);
        let [vx, vy, vz] = body.velocity;
        match before {
            Some(b) if b > 0.0 && height <= 0.0 && vy < -1.5 => {
                self.impact(lx, lz, surface, -vy);
            }
            Some(b) if b <= 0.0 && height > 0.0 && vy > 1.5 => {
                self.impact(lx, lz, surface, vy * 0.4);
            }
            _ => {}
        }
        // The body cuts the surface: it drags the water along and throws a bow wave.
        if height < 0.3 && height > -1.9 {
            let speed = (vx * vx + vz * vz).sqrt();
            if speed > 0.2 {
                self.wake(lx, lz, [vx, vz], dt);
                let spray = (speed - 2.5).max(0.0) * 30.0 * dt;
                for _ in 0..self.count(spray) {
                    let (a, r) = (self.rng.range(0.0, std::f32::consts::TAU), 0.4);
                    let drop = Droplet {
                        position: [lx + a.cos() * r, surface, lz + a.sin() * r],
                        velocity: [
                            vx * 0.5 + a.cos() * self.rng.range(0.5, 1.5),
                            self.rng.range(1.5, 3.5),
                            vz * 0.5 + a.sin() * self.rng.range(0.5, 1.5),
                        ],
                        life: 1.2,
                        lifetime: 1.2,
                        size: self.rng.range(0.03, 0.07),
                    };
                    self.splashes.spawn(drop);
                }
            }
        }
    }

    /// A whole number of events for an expected count, carrying the fraction by chance.
    fn count(&mut self, expected: f32) -> usize {
        let whole = expected.floor();
        whole as usize + usize::from(self.rng.unit() < expected - whole)
    }

    /// Calls `f(cell, dx, dz, weight)` for each wet cell within `radius` blocks of `(lx, lz)`,
    /// `dx`, `dz` the direction from the centre and `weight` falling from 1 to 0 at the radius.
    fn around(
        &mut self,
        lx: f32,
        lz: f32,
        radius: f32,
        mut f: impl FnMut(&mut Self, usize, f32, f32, f32),
    ) {
        let (cx, cz) = (lx / CELL, lz / CELL);
        let r = radius / CELL;
        let lo = |c: f32| ((c - r).floor().max(0.0)) as usize;
        let hi = |c: f32| ((c + r).ceil().max(0.0) as usize).min(SIZE - 1);
        for z in lo(cz)..=hi(cz) {
            for x in lo(cx)..=hi(cx) {
                let i = self.grid.index(x, z);
                if !self.grid.wet(i) {
                    continue;
                }
                let (dx, dz) = (x as f32 + 0.5 - cx, z as f32 + 0.5 - cz);
                let d = (dx * dx + dz * dz).sqrt();
                if d >= r {
                    continue;
                }
                let (nx, nz) = if d > 1.0e-4 {
                    (dx / d, dz / d)
                } else {
                    (0.0, 0.0)
                };
                f(self, i, nx, nz, 1.0 - d / r);
            }
        }
    }

    /// An outward push: the ring wave of something hitting the water.
    fn ripple(&mut self, lx: f32, lz: f32, radius: f32, strength: f32, foam: f32) {
        self.around(lx, lz, radius, |s, i, nx, nz, w| {
            let h = s.grid.depth[i];
            s.grid.qx[i] += h * nx * strength * w;
            s.grid.qz[i] += h * nz * strength * w;
            s.foam[i] = (s.foam[i] + foam * w).min(1.0);
        });
    }

    /// Something entered or left the water at `speed` blocks a second.
    fn impact(&mut self, lx: f32, lz: f32, surface: f32, speed: f32) {
        let radius = (0.8 + speed * 0.12).min(2.5);
        self.ripple(lx, lz, radius, speed * 0.9, 0.35 * (speed / 5.0).min(1.0));
        let up = (speed / 6.0).clamp(0.4, 1.6);
        for _ in 0..((speed * 14.0) as usize).min(220) {
            let a = self.rng.range(0.0, std::f32::consts::TAU);
            let out = self.rng.range(0.6, 2.0) * speed.sqrt() * 0.6;
            let r = self.rng.range(0.1, 0.5);
            let life = self.rng.range(0.8, 1.6);
            let drop = Droplet {
                position: [lx + a.cos() * r, surface, lz + a.sin() * r],
                velocity: [a.cos() * out, self.rng.range(2.0, 5.5) * up, a.sin() * out],
                life,
                lifetime: life,
                size: self.rng.range(0.03, 0.09),
            };
            self.splashes.spawn(drop);
        }
    }

    /// A body moving through the surface at `velocity`: the water near it is dragged toward
    /// its speed and pushed aside.
    fn wake(&mut self, lx: f32, lz: f32, velocity: [f32; 2], dt: f32) {
        let gain = 1.0 - (-6.0 * dt).exp();
        let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
        self.around(lx, lz, 0.9, |s, i, nx, nz, w| {
            let h = s.grid.depth[i];
            let [u, v] = s.grid.velocity(i);
            s.grid.qx[i] += h * ((velocity[0] - u) + nx * speed * 0.5) * w * gain;
            s.grid.qz[i] += h * ((velocity[1] - v) + nz * speed * 0.5) * w * gain;
            s.foam[i] = (s.foam[i] + speed * 0.04 * w * gain).min(1.0);
        });
    }

    /// Raindrops ring the surface.
    fn rainfall(&mut self, dt: f32) {
        let drops = self.rain * 160.0 * dt;
        let side = REGION_BLOCKS as f32;
        for _ in 0..self.count(drops) {
            let (x, z) = (self.rng.range(0.0, side), self.rng.range(0.0, side));
            self.ripple(x, z, 0.6, 0.5, 0.0);
        }
    }

    /// Water falling over a ledge into a lower cell: foam and spray where it lands.
    fn plunges(&mut self, dt: f32) {
        let n = SIZE;
        for z in 1..n - 1 {
            for x in 1..n - 1 {
                let i = z * n + x;
                if !self.grid.wet(i) {
                    continue;
                }
                let mut plunge = 0.0;
                for (dx, dz) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let j = (z as i32 + dz) as usize * n + (x as i32 + dx) as usize;
                    let drop = self.grid.bed[j] - self.grid.bed[i];
                    if drop > 0.75 && self.grid.bed[j] < WALL && self.grid.wet(j) {
                        // The neighbour's discharge toward this cell.
                        let toward = -(self.grid.qx[j] * dx as f32 + self.grid.qz[j] * dz as f32);
                        plunge += toward.max(0.0) * drop.min(4.0);
                    }
                }
                if plunge <= 0.0 {
                    continue;
                }
                self.foam[i] = (self.foam[i] + plunge * 0.6 * dt).min(1.0);
                let (lx, lz) = ((x as f32 + 0.5) * CELL, (z as f32 + 0.5) * CELL);
                let surface = self.grid.surface(i);
                for _ in 0..self.count((plunge * 12.0 * dt).min(6.0)) {
                    let life = self.rng.range(0.6, 1.4);
                    let drop = Droplet {
                        position: [
                            lx + self.rng.range(-0.25, 0.25),
                            surface,
                            lz + self.rng.range(-0.25, 0.25),
                        ],
                        velocity: [
                            self.rng.range(-1.5, 1.5),
                            self.rng.range(1.0, 3.5),
                            self.rng.range(-1.5, 1.5),
                        ],
                        life,
                        lifetime: life,
                        size: self.rng.range(0.04, 0.1),
                    };
                    self.splashes.spawn(drop);
                }
            }
        }
    }

    /// Foam: made where the water is fast, shallow and fast, or converging; carried by the
    /// flow; fading away, quickly on dry ground.
    fn update_foam(&mut self, dt: f32) {
        let n = SIZE;
        let g = &self.grid;
        let inv = 1.0 / (2.0 * CELL);
        for z in 0..n {
            for x in 0..n {
                let i = z * n + x;
                if !g.wet(i) {
                    self.foam_next[i] = self.foam[i] * (-3.0 * dt).exp();
                    continue;
                }
                let [u, v] = g.velocity(i);
                // Semi-Lagrangian: the foam that was upstream.
                let back_x = x as f32 - u * dt / CELL;
                let back_z = z as f32 - v * dt / CELL;
                let mut foam = bilinear(&self.foam, back_x, back_z);
                let speed = (u * u + v * v).sqrt();
                let mut made = (speed - 1.0).max(0.0) * 0.15;
                if g.depth[i] < 0.15 && speed > 0.4 {
                    made += 0.4 * speed;
                }
                if (1..n - 1).contains(&x) && (1..n - 1).contains(&z) {
                    let div = (g.velocity(i + 1)[0] - g.velocity(i - 1)[0] + g.velocity(i + n)[1]
                        - g.velocity(i - n)[1])
                        * inv;
                    made += (-div - 1.0).max(0.0) * 0.15;
                }
                foam = (foam + made * dt) * (-0.4 * dt).exp();
                self.foam_next[i] = foam.clamp(0.0, 1.0);
            }
        }
        std::mem::swap(&mut self.foam, &mut self.foam_next);
    }

    fn update_splashes(&mut self, dt: f32) {
        let mut landed = Vec::new();
        let grid = &self.grid;
        self.splashes.update(
            dt,
            |x, z| surface_at(grid, x, z),
            |x, z| {
                let (cx, cz) = ((x / CELL) as isize, (z / CELL) as isize);
                if cx < 0 || cz < 0 || cx >= SIZE as isize || cz >= SIZE as isize {
                    return f32::INFINITY;
                }
                grid.bed[grid.index(cx as usize, cz as usize)]
            },
            |d| landed.push((d.position, d.size)),
        );
        for ([x, _, z], size) in landed {
            self.ripple(x, z, 0.5, size * 4.0, size * 0.5);
        }
    }

    /// The renderer's view of the frame: heights and depths with the surface carried two
    /// cells past the water's edge (so the smooth surface doesn't sag toward dry ground or
    /// into walls), foam, light, droplets and the vanilla water to hide.
    fn build_output(&mut self, camera: [f64; 3]) {
        let n = SIZE;
        let g = &self.grid;
        let cam_y = camera[1];
        let rel = |y: f32| (f64::from(y) - cam_y) as f32;
        // (surface, depth) per cell; NaN surface: not yet known.
        let mut cells: Vec<[f32; 2]> = (0..n * n)
            .map(|i| {
                if g.wet(i) {
                    [g.surface(i), g.depth[i]]
                } else {
                    [f32::NAN, 0.0]
                }
            })
            .collect();
        for _ring in 0..2 {
            let known = cells.clone();
            for z in 0..n {
                for x in 0..n {
                    let i = z * n + x;
                    if !known[i][0].is_nan() {
                        continue;
                    }
                    let (mut sum, mut depth, mut count) = (0.0, 0.0, 0.0);
                    for dz in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let (xx, zz) = (x as i32 + dx, z as i32 + dz);
                            if (0..n as i32).contains(&xx) && (0..n as i32).contains(&zz) {
                                let k = known[zz as usize * n + xx as usize];
                                if !k[0].is_nan() {
                                    sum += k[0];
                                    depth += k[1];
                                    count += 1.0;
                                }
                            }
                        }
                    }
                    if count > 0.0 {
                        let surface = sum / count;
                        // Into a wall the water seems to continue (the wall hides it); onto
                        // dry ground it fades where the ground rises over it.
                        let d = if g.bed[i] >= WALL {
                            depth / count
                        } else {
                            (surface - g.bed[i]).min(0.0)
                        };
                        cells[i] = [surface, d];
                    }
                }
            }
        }
        let mut band = [f32::MAX, f32::MIN];
        for i in 0..n * n {
            let [surface, depth] = cells[i];
            let surface = if surface.is_nan() {
                if g.bed[i] < WALL {
                    g.bed[i]
                } else {
                    cam_y as f32
                }
            } else {
                surface
            };
            let [u, v] = g.velocity(i);
            self.heights[i] = [rel(surface), if depth == 0.0 { -0.05 } else { depth }, u, v];
            let [sky, block] = self.light[i];
            self.aux[i] = [(self.foam[i] * 255.0) as u8, sky * 17, block * 17, 0];
            if (self.vanilla[i] || g.wet(i)) && g.bed[i] < WALL {
                band[0] = band[0].min(g.bed[i] - 1.0);
                band[1] = band[1].max(g.surface(i).max(g.bed[i]) + 1.0);
            }
        }
        let ox = (f64::from(self.origin[0]) - camera[0]) as f32;
        let oz = (f64::from(self.origin[1]) - camera[2]) as f32;
        self.relative_origin = [ox, 0.0, oz];
        let side = REGION_BLOCKS as f32;
        self.hide = if band[0] <= band[1] {
            [
                [
                    ox + HIDE_INSET,
                    oz + HIDE_INSET,
                    ox + side - HIDE_INSET,
                    oz + side - HIDE_INSET,
                ],
                [rel(band[0]), rel(band[1]), 1.0, 0.0],
            ]
        } else {
            [[0.0; 4]; 2]
        };
        self.droplets.clear();
        for d in &self.splashes.drops {
            let [x, y, z] = d.position;
            let fade = (d.life / d.lifetime).clamp(0.0, 1.0);
            self.droplets
                .push([x + ox, rel(y), z + oz, d.size, fade.sqrt(), 0.0, 0.0, 0.0]);
        }
    }
}

/// The water's height at region-local block `(lx, lz)`, smoothly between the wet cells
/// around it; `None` where they are dry.
fn surface_at(g: &Grid, lx: f32, lz: f32) -> Option<f32> {
    let (gx, gz) = (lx / CELL - 0.5, lz / CELL - 0.5);
    let (x0, z0) = (gx.floor(), gz.floor());
    let (tx, tz) = (gx - x0, gz - z0);
    let (mut sum, mut weight) = (0.0, 0.0);
    for (dx, dz, w) in [
        (0, 0, (1.0 - tx) * (1.0 - tz)),
        (1, 0, tx * (1.0 - tz)),
        (0, 1, (1.0 - tx) * tz),
        (1, 1, tx * tz),
    ] {
        let (x, z) = (x0 as i32 + dx, z0 as i32 + dz);
        if x < 0 || z < 0 || x >= g.size as i32 || z >= g.size as i32 {
            continue;
        }
        let i = g.index(x as usize, z as usize);
        if g.wet(i) {
            sum += g.surface(i) * w;
            weight += w;
        }
    }
    (weight >= 0.25).then(|| sum / weight)
}

/// `values` (a `SIZE` square) at fractional cell `(x, z)`, clamped to the grid.
fn bilinear(values: &[f32], x: f32, z: f32) -> f32 {
    let max = (SIZE - 1) as f32;
    let (x, z) = (x.clamp(0.0, max), z.clamp(0.0, max));
    let (x0, z0) = (x.floor() as usize, z.floor() as usize);
    let (x1, z1) = ((x0 + 1).min(SIZE - 1), (z0 + 1).min(SIZE - 1));
    let (tx, tz) = (x - x0 as f32, z - z0 as f32);
    let at = |x: usize, z: usize| values[z * SIZE + x];
    (at(x0, z0) * (1.0 - tx) + at(x1, z0) * tx) * (1.0 - tz)
        + (at(x0, z1) * (1.0 - tx) + at(x1, z1) * tx) * tz
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world of flat ground at 60 with a pond (source blocks to 63 + 8/9) where `pond` says.
    fn world(pond: impl Fn(i32, i32) -> bool) -> impl FnMut(i32, i32, i32) -> Column {
        move |x, z, _| {
            let water = pond(x, z).then_some(63.0 + 8.0 / 9.0);
            Column {
                bed: Some(if water.is_some() { 61.0 } else { 64.0 }),
                water,
                source: water.is_some(),
                sky: 15,
                block: 0,
            }
        }
    }

    fn input(camera: [f64; 3]) -> Input {
        Input {
            camera,
            dt: 1.0 / 60.0,
            rain: 0.0,
            player: None,
        }
    }

    #[test]
    fn a_pond_is_seeded_level_and_stays_put() {
        let mut sim = WaterSim::new();
        let mut sample = world(|x, z| (x - 8).abs() < 6 && (z - 8).abs() < 6);
        for _ in 0..120 {
            sim.update(&input([8.0, 64.0, 8.0]), &mut sample);
        }
        let [ox, oz] = sim.origin();
        let g = sim.grid();
        let center = g.index(((8 - ox) * 2) as usize, ((8 - oz) * 2) as usize);
        assert!((g.surface(center) - (63.0 + 8.0 / 9.0)).abs() < 1.0e-3);
        let dry = g.index(((20 - ox) * 2) as usize, ((8 - oz) * 2) as usize);
        assert!(!g.wet(dry));
    }

    #[test]
    fn a_jump_into_the_pond_throws_spray_and_ripples() {
        let mut sim = WaterSim::new();
        let mut sample = world(|_, _| true);
        sim.update(&input([8.0, 66.0, 8.0]), &mut sample);
        let mut frame = |y: f64, vy: f32| {
            let mut i = input([8.0, y, 8.0]);
            i.player = Some(Body {
                position: [8.0, y, 8.0],
                velocity: [0.0, vy, 0.0],
            });
            sim.update(&i, &mut sample);
        };
        frame(64.5, -8.0);
        frame(63.0, -8.0);
        assert!(!sim.splashes.drops.is_empty());
        let moving = sim
            .grid()
            .qx
            .iter()
            .chain(&sim.grid().qz)
            .any(|q| q.abs() > 0.05);
        assert!(moving, "the impact set the water moving");
    }

    #[test]
    fn moving_the_region_keeps_the_water_that_stays_inside() {
        let mut sim = WaterSim::new();
        let mut sample = world(|_, _| false);
        sim.update(&input([0.0, 64.0, 0.0]), &mut sample);
        // Put a puddle of unfed water at world (4, 4).
        let [ox, oz] = sim.origin();
        let i = sim
            .grid
            .index(((4 - ox) * 2) as usize, ((4 - oz) * 2) as usize);
        sim.grid.depth[i] = 0.5;
        sim.update(&input([16.0, 64.0, 0.0]), &mut sample);
        let [nx, nz] = sim.origin();
        assert_ne!(nx, ox);
        let j = sim
            .grid
            .index(((4 - nx) * 2) as usize, ((4 - nz) * 2) as usize);
        assert!(sim.grid.depth[j] > 0.05, "the puddle moved with the region");
    }
}
