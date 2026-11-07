//! Horizon's volume water (3D water mode, off by default). Water is an amount per block, not a
//! block: it falls, spreads sideways until level, is pushed up by the pressure of the water over
//! it (so it rises in a U-bend), drains through holes, pools in hollows, and thin films resting
//! on the ground evaporate. Its volume is conserved exactly: amounts are integers.
//!
//! The world is sparse: only loaded sections exist, and only cells whose water moved last tick
//! (and their neighbours) are worked on. Water at rest sleeps, so a calm ocean costs nothing.
//! World-generated seas are an endless supply at sea level: a sea cell refills itself, so a hole
//! in the ocean fills back up, but the sea can't be drained.
//!
//! The flow rule is the compressible cellular water of W. Shadow's "Simple fluid simulation with
//! cellular automata": a cell holds a little more than a block when water presses on it from
//! above, which is what lets pressure push water up.

use std::collections::{HashMap, HashSet};

/// One block of water, in units.
pub const FULL: u32 = 1024;
/// How much more a cell holds per block of water over it: water at rest `d` blocks down holds
/// about `FULL + d * COMPRESS` (load deep water so, or it settles by moving water down).
pub const COMPRESS: u32 = FULL / 50;
/// Water this shallow no longer spreads sideways (surface tension holds a puddle's edge): a
/// bucket on a floor makes a puddle an eighth of a block deep, not an invisible film.
pub const SPREAD_MIN: u32 = FULL / 8;
/// Puddles this shallow, resting on the ground with air over them, evaporate.
pub const FILM: u32 = SPREAD_MIN + 1;
/// A film loses a unit every this many ticks (a second).
const EVAPORATE_EVERY: u64 = 20;
/// The most a cell can hold (compressed under a deep column).
const MAX_FILL: u32 = u16::MAX as u32;

/// A block position.
pub type Pos = [i32; 3];
/// A section position: block position divided by 16.
pub type SectionPos = [i32; 3];

fn section_of(p: Pos) -> SectionPos {
    p.map(|v| v >> 4)
}
fn index(p: Pos) -> usize {
    (((p[1] & 15) << 8) | ((p[2] & 15) << 4) | (p[0] & 15)) as usize
}
fn offset(p: Pos, d: [i32; 3]) -> Pos {
    [p[0] + d[0], p[1] + d[1], p[2] + d[2]]
}
const SIDES: [[i32; 3]; 4] = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];
const NEIGHBOURS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 0, 1],
    [0, 0, -1],
    [0, 1, 0],
    [0, -1, 0],
];

/// A 16-cube of the world: each cell's water (stored only once the section holds any),
/// whether water may be there, and whether it is the endless sea.
#[derive(Clone)]
pub struct Section {
    fill: Option<Box<[u16; 4096]>>,
    pub open: [u64; 64],
    pub sea: [u64; 64],
}

fn bit(bits: &[u64; 64], i: usize) -> bool {
    bits[i >> 6] >> (i & 63) & 1 != 0
}
fn set_bit(bits: &mut [u64; 64], i: usize, on: bool) {
    if on {
        bits[i >> 6] |= 1 << (i & 63);
    } else {
        bits[i >> 6] &= !(1 << (i & 63));
    }
}

impl Default for Section {
    fn default() -> Self {
        Self {
            fill: None,
            open: [0; 64],
            sea: [0; 64],
        }
    }
}

impl Section {
    /// A section of `open` cells and `sea` cells holding `fill` (units, `y << 8 | z << 4 | x`).
    pub fn new(open: [u64; 64], sea: [u64; 64], fill: Option<Box<[u16; 4096]>>) -> Self {
        let fill = fill.filter(|f| f.iter().any(|&v| v != 0));
        Self { fill, open, sea }
    }
    /// Whether any cell holds water.
    pub fn is_dry(&self) -> bool {
        self.fill.as_ref().is_none_or(|f| f.iter().all(|&v| v == 0))
    }
    /// The water in cell `i`, units.
    pub fn get(&self, i: usize) -> u32 {
        self.fill.as_ref().map_or(0, |f| u32::from(f[i]))
    }
    fn put(&mut self, i: usize, v: u32) {
        if v == 0 && self.fill.is_none() {
            return;
        }
        self.fill.get_or_insert_with(|| Box::new([0; 4096]))[i] = v as u16;
    }
}

/// Bits from a predicate over the 4096 cells.
pub fn bits(mut f: impl FnMut(usize) -> bool) -> [u64; 64] {
    let mut out = [0u64; 64];
    for i in 0..4096 {
        if f(i) {
            out[i >> 6] |= 1 << (i & 63);
        }
    }
    out
}

fn cell_pos(at: SectionPos, i: usize) -> Pos {
    [
        at[0] * 16 + (i & 15) as i32,
        at[1] * 16 + (i >> 8) as i32,
        at[2] * 16 + ((i >> 4) & 15) as i32,
    ]
}

/// How much of a pair's water belongs below when they share `total` (W. Shadow's
/// `get_stable_state_b`).
fn stable_below(total: u32) -> u32 {
    if total <= FULL {
        FULL
    } else if total < 2 * FULL + COMPRESS {
        (FULL * FULL + total * COMPRESS) / (FULL + COMPRESS)
    } else {
        (total + COMPRESS) / 2
    }
}

/// The volume water of a world.
pub struct Water {
    /// Thin films on the ground evaporate (off: the volume never changes but by `add`/`take`).
    pub evaporate: bool,
    sections: HashMap<SectionPos, Section>,
    /// Cells to work on next tick.
    active: HashSet<Pos>,
    /// Cells whose water changed since [`Self::take_changed`].
    changed: HashSet<Pos>,
    tick: u64,
}

impl Default for Water {
    fn default() -> Self {
        Self {
            evaporate: true,
            sections: HashMap::new(),
            active: HashSet::new(),
            changed: HashSet::new(),
            tick: 0,
        }
    }
}

impl Water {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an empty section where none is (a chunk loaded); cells start closed.
    pub fn load_section(&mut self, at: SectionPos) -> &mut Section {
        self.sections.entry(at).or_default()
    }

    /// Takes a section out (a chunk unloaded), to save it.
    pub fn unload_section(&mut self, at: SectionPos) -> Option<Section> {
        self.active.retain(|p| section_of(*p) != at);
        self.sections.remove(&at)
    }

    /// Puts in a section (a chunk loaded, or a saved section back). Only water that may move
    /// is woken: partly full cells, and full ones over open space or beside a partly full or
    /// empty open cell, so a loaded ocean at rest costs nothing.
    pub fn insert_section(&mut self, at: SectionPos, section: Section) {
        let mut wake = Vec::new();
        if section.fill.is_some() {
            for i in 0..4096 {
                let v = section.get(i);
                if v == 0 {
                    continue;
                }
                let restless = v < FULL && !bit(&section.sea, i)
                    || [
                        (-1i32, 0i32, 0i32),
                        (1, 0, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                        (0, -1, 0),
                    ]
                    .into_iter()
                    .any(|(dx, dy, dz)| {
                        let (x, y, z) = (
                            (i & 15) as i32 + dx,
                            (i >> 8) as i32 + dy,
                            ((i >> 4) & 15) as i32 + dz,
                        );
                        if !(0..16).contains(&x) || !(0..16).contains(&y) || !(0..16).contains(&z) {
                            return false;
                        }
                        let j = ((y << 8) | (z << 4) | x) as usize;
                        bit(&section.open, j) && section.get(j) < FULL && !bit(&section.sea, j)
                    });
                if restless {
                    wake.push(cell_pos(at, i));
                }
            }
        }
        self.sections.insert(at, section);
        for p in wake {
            self.wake(p);
        }
    }

    pub fn section(&self, at: SectionPos) -> Option<&Section> {
        self.sections.get(&at)
    }

    /// The water at `p`, in units (0 where unloaded).
    pub fn fill(&self, p: Pos) -> u32 {
        self.sections
            .get(&section_of(p))
            .map_or(0, |s| s.get(index(p)))
    }

    /// The water at `p` as a share of the block, 0 to 1 (more under pressure is clamped).
    pub fn level(&self, p: Pos) -> f32 {
        (self.fill(p) as f32 / FULL as f32).min(1.0)
    }

    /// Whether water may be at `p` (not a solid block, loaded).
    pub fn is_open(&self, p: Pos) -> bool {
        self.sections
            .get(&section_of(p))
            .is_some_and(|s| bit(&s.open, index(p)))
    }

    fn is_sea(&self, p: Pos) -> bool {
        self.sections
            .get(&section_of(p))
            .is_some_and(|s| bit(&s.sea, index(p)))
    }

    /// Opens or closes `p` (a block was broken or placed). Water in a cell that closes is
    /// pushed out: up if it can go, else to the first open side; it is never lost.
    pub fn set_open(&mut self, p: Pos, open: bool) {
        let Some(s) = self.sections.get_mut(&section_of(p)) else {
            return;
        };
        let i = index(p);
        if bit(&s.open, i) == open {
            return;
        }
        set_bit(&mut s.open, i, open);
        let held = s.get(i);
        if !open {
            set_bit(&mut s.sea, i, false);
            s.put(i, 0);
            if held > 0 {
                let to = std::iter::once([0, 1, 0])
                    .chain(SIDES)
                    .map(|d| offset(p, d))
                    .find(|&q| self.is_open(q));
                if let Some(q) = to {
                    self.add(q, held);
                }
            }
            self.changed.insert(p);
        }
        self.wake(p);
    }

    /// Marks `p` as the endless sea: it refills itself whenever it is short.
    pub fn set_sea(&mut self, p: Pos, sea: bool) {
        if let Some(s) = self.sections.get_mut(&section_of(p)) {
            set_bit(&mut s.sea, index(p), sea);
        }
        self.wake(p);
    }

    /// Adds `amount` units at `p` (a bucket poured); returns what didn't fit (closed or
    /// unloaded).
    pub fn add(&mut self, p: Pos, amount: u32) -> u32 {
        if !self.is_open(p) {
            return amount;
        }
        let s = self
            .sections
            .get_mut(&section_of(p))
            .expect("open means loaded");
        let i = index(p);
        let now = (s.get(i) + amount).min(MAX_FILL);
        let left = s.get(i) + amount - now;
        s.put(i, now);
        self.changed.insert(p);
        self.wake(p);
        left
    }

    /// Takes up to `amount` units from `p` (a bucket filled); returns how much it took.
    pub fn take(&mut self, p: Pos, amount: u32) -> u32 {
        let Some(s) = self.sections.get_mut(&section_of(p)) else {
            return 0;
        };
        let i = index(p);
        let took = s.get(i).min(amount);
        s.put(i, s.get(i) - took);
        if took > 0 {
            self.changed.insert(p);
            self.wake(p);
        }
        took
    }

    /// Works on `p` and its neighbours next tick.
    pub fn wake(&mut self, p: Pos) {
        self.active.insert(p);
        for d in NEIGHBOURS {
            self.active.insert(offset(p, d));
        }
    }

    /// Cells being worked on: how busy the water is.
    pub fn active_cells(&self) -> usize {
        self.active.len()
    }

    /// The cells whose water changed since the last call (for the renderer and saving).
    pub fn take_changed(&mut self) -> HashSet<Pos> {
        std::mem::take(&mut self.changed)
    }

    /// Total water held in loaded sections, in units.
    pub fn volume(&self) -> u64 {
        self.sections
            .values()
            .filter_map(|s| s.fill.as_ref())
            .flat_map(|f| f.iter())
            .map(|&f| u64::from(f))
            .sum()
    }

    /// Advances one game tick.
    pub fn tick(&mut self) {
        self.tick += 1;
        let active = std::mem::take(&mut self.active);
        let mut delta: HashMap<Pos, i64> = HashMap::new();
        for &p in &active {
            if !self.is_open(p) {
                continue;
            }
            let held = self.fill(p);
            if self.is_sea(p) && held < FULL {
                *delta.entry(p).or_default() += i64::from(FULL - held);
                continue;
            }
            if held == 0 {
                continue;
            }
            let mut left = held;
            // Down: as much as the cell below takes.
            let below = offset(p, [0, -1, 0]);
            if self.is_open(below) {
                let under = self.fill(below);
                // The sea holds exactly a block: nothing presses more into it.
                let room = if self.is_sea(below) {
                    FULL.saturating_sub(under)
                } else {
                    stable_below(left + under).saturating_sub(under)
                };
                let flow = room.min(left);
                if flow > 0 {
                    left -= flow;
                    *delta.entry(below).or_default() += i64::from(flow);
                }
            }
            // Sideways: toward each lower neighbour, a quarter of the difference.
            if left > SPREAD_MIN {
                let start = left;
                for d in SIDES {
                    let q = offset(p, d);
                    if !self.is_open(q) {
                        continue;
                    }
                    let there = self.fill(q);
                    if start > there + 4 {
                        let flow = ((start - there) / 5).min(left);
                        left -= flow;
                        *delta.entry(q).or_default() += i64::from(flow);
                    }
                }
            }
            // Up: what pressure pushes past what this cell should hold under the one above.
            if left > FULL && !self.is_sea(p) {
                let above = offset(p, [0, 1, 0]);
                if self.is_open(above) {
                    let over = self.fill(above);
                    let keep = stable_below(left + over);
                    if left > keep {
                        let flow = left - keep;
                        left -= flow;
                        *delta.entry(above).or_default() += i64::from(flow);
                    }
                }
            }
            // A film on the ground under open air evaporates, a unit at a time.
            if self.evaporate
                && left > 0
                && left < FILM
                && self.tick % EVAPORATE_EVERY == 0
                && !self.is_open(below)
                && self.fill(offset(p, [0, 1, 0])) == 0
            {
                left -= 1;
            }
            if left != held {
                *delta.entry(p).or_default() -= i64::from(held - left);
            }
        }
        for (p, d) in delta {
            if d == 0 {
                continue;
            }
            let Some(s) = self.sections.get_mut(&section_of(p)) else {
                continue;
            };
            let i = index(p);
            let now = (i64::from(s.get(i)) + d).clamp(0, i64::from(MAX_FILL));
            s.put(i, now as u32);
            self.changed.insert(p);
            self.wake(p);
        }
        // Films stay awake until they're gone.
        for p in active {
            let f = self.fill(p);
            if f > 0 && f < FILM && self.is_open(p) && !self.is_open(offset(p, [0, -1, 0])) {
                self.active.insert(p);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world with open air over a floor at y = 0, from -r to r in X and Z, walled around.
    fn floor(r: i32) -> Water {
        let mut w = Water::new();
        for y in -16..32 {
            for z in -r - 1..=r + 1 {
                for x in -r - 1..=r + 1 {
                    w.load_section(section_of([x, y, z]));
                    let inside = x.abs() <= r && z.abs() <= r && y >= 1;
                    w.set_open([x, y, z], inside);
                }
            }
        }
        w
    }

    fn run(w: &mut Water, ticks: usize) {
        for _ in 0..ticks {
            w.tick();
        }
    }

    #[test]
    fn water_falls_and_spreads_level() {
        let mut w = floor(3);
        w.add([0, 10, 0], 9 * FULL);
        run(&mut w, 400);
        // Nine blocks over a 7x7 floor: about 0.18 deep everywhere.
        let levels: Vec<u32> = (-3..=3)
            .flat_map(|x| (-3..=3).map(move |z| [x, 1, z]))
            .map(|p| w.fill(p))
            .collect();
        let (lo, hi) = (levels.iter().min().unwrap(), levels.iter().max().unwrap());
        // Level to within 2% of a block: smaller steps are left to rest.
        assert!(hi - lo <= FULL / 50, "level: {lo}..{hi}");
        assert_eq!(w.fill([0, 2, 0]), 0, "nothing hangs in the air");
    }

    #[test]
    fn a_closed_basin_keeps_every_drop() {
        let mut w = floor(4);
        w.evaporate = false;
        w.add([2, 6, -1], 20 * FULL + 17);
        w.add([-3, 3, 3], 5 * FULL);
        let before = w.volume();
        run(&mut w, 300);
        assert_eq!(w.volume(), before);
    }

    #[test]
    fn calm_water_sleeps() {
        let mut w = floor(3);
        w.add([0, 5, 0], 49 * FULL * 2);
        run(&mut w, 600);
        assert_eq!(w.active_cells(), 0, "a settled pool does no work");
    }

    #[test]
    fn water_drains_through_a_hole() {
        let mut w = floor(2);
        for y in -10..1 {
            w.set_open([0, y, 0], true);
        }
        w.add([1, 1, 1], 4 * FULL);
        // What doesn't reach the hole is left as a thin wet layer, which dries.
        run(&mut w, 20 * 200);
        let floor_water: u32 = (-2..=2)
            .flat_map(|x| (-2..=2).map(move |z| [x, 1, z]))
            .map(|p| w.fill(p))
            .sum();
        assert!(
            floor_water < FULL / 10,
            "it ran down the hole: {floor_water}"
        );
    }

    #[test]
    fn pressure_lifts_water_up_a_u_bend() {
        // Two shafts joined at the bottom: water poured down one rises in the other.
        let mut w = Water::new();
        for y in -1..20 {
            for x in -1..=3 {
                w.load_section(section_of([x, y, 0]));
            }
        }
        for y in 0..12 {
            w.set_open([0, y, 0], true);
            w.set_open([2, y, 0], true);
        }
        w.set_open([1, 0, 0], true);
        w.add([0, 11, 0], 12 * FULL);
        run(&mut w, 3000);
        let height = |x| (0..12).filter(|&y| w.fill([x, y, 0]) > FULL / 2).count();
        let (a, b) = (height(0), height(2));
        assert!(a.abs_diff(b) <= 1, "both shafts fill alike: {a} and {b}");
    }

    #[test]
    fn a_bucket_on_a_wide_floor_makes_a_puddle_not_a_film() {
        let mut w = floor(8);
        w.evaporate = false;
        w.add([0, 1, 0], FULL);
        run(&mut w, 400);
        let wet = (-8..=8)
            .flat_map(|x| (-8..=8).map(move |z| [x, 1, z]))
            .filter(|&p| w.fill(p) >= FULL / 16)
            .count();
        assert!((4..=16).contains(&wet), "a puddle of a few blocks: {wet}");
    }

    #[test]
    fn a_puddle_dries_up() {
        let mut w = floor(6);
        w.add([0, 1, 0], FULL / 2);
        run(&mut w, 20 * 200);
        assert_eq!(w.volume(), 0, "it evaporated");
    }

    #[test]
    fn the_sea_refills_a_hole() {
        let mut w = floor(4);
        for x in -4..=4 {
            for z in -4..=4 {
                for y in 1..=3 {
                    w.add([x, y, z], FULL);
                    w.set_sea([x, y, z], true);
                }
            }
        }
        run(&mut w, 50);
        w.take([0, 3, 0], FULL);
        w.take([0, 2, 0], FULL);
        run(&mut w, 50);
        // Full again (the lower cell a little over, pressed by the water on it).
        assert!(w.fill([0, 2, 0]) >= FULL);
        assert!(w.fill([0, 3, 0]) >= FULL - 8);
    }

    #[test]
    fn a_deep_sea_at_rest_stays_asleep() {
        let mut w = floor(4);
        for x in -4..=4 {
            for z in -4..=4 {
                for y in 1..=12 {
                    w.add([x, y, z], FULL);
                    w.set_sea([x, y, z], true);
                }
            }
        }
        run(&mut w, 20);
        assert_eq!(w.active_cells(), 0, "a full sea does no work");
        assert!((1..=12).all(|y| w.fill([0, y, 0]) == FULL));
    }

    #[test]
    fn a_block_placed_in_water_pushes_it_out() {
        let mut w = floor(3);
        w.add([0, 1, 0], FULL);
        let before = w.volume();
        w.set_open([0, 1, 0], false);
        assert_eq!(w.volume(), before);
    }
}
