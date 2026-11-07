//! 3D water (Horizon experimental setting `Graphics::volume_water`, single player, off by
//! default): the server's water becomes `horizon_water` volume water. Each level's water blocks
//! are read into the simulation as their chunks load; from then on the simulation owns the
//! water and writes it back as vanilla water blocks at vanilla's eight levels, so swimming,
//! buoyancy, currents, drowning, the client's copy of the world and saving all work as for
//! vanilla water, and the world still opens in vanilla. Vanilla's own spreading is off
//! (`tick_scheduled` skips fluid ticks while this runs).
//!
//! Blocks the game changes (placed, broken, a bucket filled or emptied) reach the simulation
//! through the world's write log (`World::water_writes`).

use super::{Dim, Game};
use horizon_client::chunk::Chunk;
use horizon_water::{COMPRESS, FULL, Section, Water, bits};
use horizon_world::fluid::{FluidKind, FluidState};
use horizon_world::level::Behaviours;
use horizon_world::registry::BlockStateId;
use std::collections::{HashMap, HashSet};

/// The surface of the overworld's seas: water below it from world generation is the endless
/// sea.
const SEA_LEVEL: i32 = 63;

/// What a block state is to the water.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Water can't be here (a solid block, a waterlogged block keeping vanilla water).
    Closed,
    Air,
    /// A water block holding this many eighths (a falling one counts as full).
    Water(u8),
}

fn kind(b: &Behaviours, state: BlockStateId) -> Kind {
    if b.is_air(state) {
        return Kind::Air;
    }
    let fluid = b.fluid(state);
    let water = matches!(fluid.kind(), FluidKind::Water | FluidKind::FlowingWater);
    if water && b.legacy_block(fluid) == state {
        Kind::Water(if fluid.falling() { 8 } else { fluid.amount() })
    } else {
        Kind::Closed
    }
}

/// One level's volume water and the chunks it holds.
#[derive(Default)]
struct LevelWater {
    water: Water,
    chunks: HashSet<(i32, i32)>,
    /// The chunk map's membership revision when `chunks` was last brought up to date.
    revision: u64,
}

/// Every level's volume water.
#[derive(Default)]
pub(super) struct VolumeWater {
    levels: HashMap<Dim, LevelWater>,
    kinds: HashMap<u32, Kind>,
}

impl VolumeWater {
    fn kind(&mut self, b: &Behaviours, state: u32) -> Kind {
        *self
            .kinds
            .entry(state)
            .or_insert_with(|| kind(b, BlockStateId(state)))
    }

    /// Reads a chunk's sections into the water. Deep water is loaded as it rests, pressed a
    /// little by the water over it (`COMPRESS`), so a loaded lake doesn't settle downward; the
    /// sea holds exactly a block per cell.
    fn load_chunk(&mut self, b: &Behaviours, dim: Dim, chunk: &Chunk) {
        let mut states = [0u32; 4096];
        let kinds: Vec<Vec<Kind>> = chunk
            .sections
            .iter()
            .map(|section| {
                section.unpack(&mut states);
                states.iter().map(|&st| self.kind(b, st)).collect()
            })
            .collect();
        let base = chunk.min_y() >> 4;
        let sea_at = |sy: i32, i: usize, k: Kind| {
            dim == Dim::Overworld && sy * 16 + ((i >> 8) as i32) < SEA_LEVEL && k == Kind::Water(8)
        };
        // Full water cells over each column's cell, counted from the top down.
        let mut depth = [0u32; 256];
        let mut sections = Vec::with_capacity(kinds.len());
        for (s, kinds) in kinds.iter().enumerate().rev() {
            let sy = base + s as i32;
            let mut fill: Option<Box<[u16; 4096]>> = None;
            for y in (0..16).rev() {
                for column in 0..256 {
                    let i = y << 8 | column;
                    match kinds[i] {
                        Kind::Water(eighths) => {
                            let mut units = u32::from(eighths) * FULL / 8;
                            if eighths == 8 && !sea_at(sy, i, kinds[i]) {
                                units += depth[column] * COMPRESS;
                            }
                            fill.get_or_insert_with(|| Box::new([0; 4096]))[i] =
                                units.min(u32::from(u16::MAX)) as u16;
                            depth[column] = if eighths == 8 { depth[column] + 1 } else { 0 };
                        }
                        _ => depth[column] = 0,
                    }
                }
            }
            let open = bits(|i| kinds[i] != Kind::Closed);
            let sea = bits(|i| sea_at(sy, i, kinds[i]));
            sections.push((sy, Section::new(open, sea, fill)));
        }
        let level = self.levels.entry(dim).or_default();
        for (sy, section) in sections {
            level.water.insert_section([chunk.x, sy, chunk.z], section);
        }
    }
}

/// The water block for `fill` units, `None` for air. Water with water over it falls, as
/// vanilla's does (a full-height column).
fn water_state(b: &Behaviours, fill: u32, under_water: bool) -> Option<BlockStateId> {
    let eighths = ((fill * 8 + FULL / 2) / FULL).min(8) as u8;
    match eighths {
        0 => None,
        8 => Some(b.legacy_block(FluidState::new(FluidKind::Water, 8, false))),
        e => Some(b.legacy_block(FluidState::new(FluidKind::FlowingWater, e, under_water))),
    }
}

impl Game<'_> {
    /// Whether 3D water runs on this server.
    pub(super) fn volume_water_on(&self) -> bool {
        self.settings.horizon.volume_water && self.serving.is_some()
    }

    /// One tick of 3D water on the level being ticked: new chunks read in, the game's block
    /// changes applied, the water moved, and what changed written back as water blocks.
    pub(super) fn tick_volume_water(&mut self) {
        if !self.volume_water_on() {
            if self.volume_water.take().is_some() {
                self.world.water_writes = None;
            }
            return;
        }
        let b = self.res.behaviours();
        let state = self.volume_water.get_or_insert_with(Default::default);
        let dim = self.dim;
        let writes = self
            .world
            .water_writes
            .replace(Vec::new())
            .unwrap_or_default();

        // Chunks that came and went.
        let revision = self.world.chunks.membership_revision();
        let known = state.levels.get(&dim).map(|l| l.revision);
        if known != Some(revision) {
            let resident: HashSet<(i32, i32)> = self.world.chunks.keys().copied().collect();
            let level = state.levels.entry(dim).or_default();
            let gone: Vec<_> = level.chunks.difference(&resident).copied().collect();
            let new: Vec<_> = resident.difference(&level.chunks).copied().collect();
            for (x, z) in gone {
                let shape = self.world.shape;
                let (lo, hi) = (shape.min_y >> 4, shape.max_y() >> 4);
                for sy in lo..hi {
                    level.water.unload_section([x, sy, z]);
                }
                level.chunks.remove(&(x, z));
            }
            level.revision = revision;
            for key in new {
                if let Some(chunk) = self.world.chunks.get(&key).cloned() {
                    state.load_chunk(b, dim, &chunk);
                    state.levels.entry(dim).or_default().chunks.insert(key);
                }
            }
        }

        // The game's own block changes.
        for ((x, y, z), old, new) in writes {
            let (old, new) = (state.kind(b, old), state.kind(b, new));
            let water = &mut state.levels.entry(dim).or_default().water;
            let p = [x, y, z];
            match (old, new) {
                (_, Kind::Closed) => water.set_open(p, false),
                // A bucket filled (or the water otherwise cleared).
                (Kind::Water(_), Kind::Air) => {
                    water.set_open(p, true);
                    water.take(p, u32::MAX);
                }
                (_, Kind::Air) => water.set_open(p, true),
                // A bucket poured (or water placed): its water joins what is there.
                (_, Kind::Water(eighths)) => {
                    water.set_open(p, true);
                    let want = u32::from(eighths) * FULL / 8;
                    let have = water.fill(p);
                    water.add(p, want.saturating_sub(have));
                }
            }
        }

        let level = state.levels.entry(dim).or_default();
        level.water.tick();

        // What moved, written back as water blocks; these writes aren't the game's.
        let changed = level.water.take_changed();
        for [x, y, z] in changed {
            let current = self.world.block(x, y, z);
            if state
                .kinds
                .get(&current.0)
                .copied()
                .unwrap_or_else(|| kind(b, current))
                == Kind::Closed
            {
                continue;
            }
            let water = &state.levels[&dim].water;
            let want = water_state(b, water.fill([x, y, z]), water.fill([x, y + 1, z]) > 0)
                .unwrap_or(self.world.air);
            if want != current {
                let _ = self.world.set(x, y, z, want);
            }
        }
        if let Some(log) = &mut self.world.water_writes {
            log.clear();
        }
    }
}
