// Simulated water (gfx/sim_water.rs). The surface is a grid mesh finer than the solver's
// cells, its heights a cubic B-spline of the cells' (C2: no cell edges show), plus waves where
// the water is deep; the fragment stage adds flowing detail normals, Fresnel reflection of the
// sky, the sun's glint, colour by depth and foam. Droplets are camera-facing soft discs.

struct Uniforms {
    view_proj: mat4x4<f32>,
    // xyz cell (0, 0)'s corner, camera-relative; w the cell size in blocks.
    origin: vec4<f32>,
    // Cells along a side, seconds, rain, wave height.
    grid: vec4<f32>,
    // World X and Z of the corner (wrapped), edge fade in blocks, mesh vertices per cell.
    phase: vec4<f32>,
    // Sky colour; w the daylight.
    sky: vec4<f32>,
    fog_color: vec4<f32>,
    // Environmental start and end, render-distance start and end.
    fog: vec4<f32>,
    sun: vec4<f32>,
    // Colours; w: opacity per block, reflectance, glint sharpness.
    shallow: vec4<f32>,
    deep: vec4<f32>,
    foam: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
// Per cell: surface height (camera-relative), depth (negative: dry), velocity X, Z.
@group(0) @binding(1) var heights: texture_2d<f32>;
// Per cell: foam, sky light, block light, 1 where a block stands out of the water.
@group(0) @binding(2) var aux: texture_2d<f32>;

// Uniform cubic B-spline weights of the four taps around t, and their derivatives.
fn bspline(t: f32) -> vec4<f32> {
    let s = 1.0 - t;
    return vec4(s * s * s, 4.0 - 6.0 * t * t + 3.0 * t * t * t,
        1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t * t * t, t * t * t) / 6.0;
}
fn bspline_slope(t: f32) -> vec4<f32> {
    return vec4(-0.5 * (1.0 - t) * (1.0 - t), 1.5 * t * t - 2.0 * t,
        0.5 + t - 1.5 * t * t, 0.5 * t * t);
}

struct Field {
    surface: f32,
    // d(surface)/dx, d(surface)/dz, per block.
    slope: vec2<f32>,
    depth: f32,
    velocity: vec2<f32>,
    foam: f32,
    light: vec2<f32>,
}

// The cells' values at `g` (in cells from the corner), smoothed.
fn field(g: vec2<f32>) -> Field {
    let n = i32(u.grid.x);
    let p = g - 0.5;
    let base = vec2<i32>(floor(p));
    let t = p - floor(p);
    let wx = bspline(t.x);
    let wz = bspline(t.y);
    let sx = bspline_slope(t.x);
    let sz = bspline_slope(t.y);
    var f: Field;
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let c = clamp(base + vec2(i - 1, j - 1), vec2(0), vec2(n - 1));
            let h = textureLoad(heights, c, 0);
            let a = textureLoad(aux, c, 0);
            let w = wx[i] * wz[j];
            f.surface += h.x * w;
            f.slope += h.x * vec2(sx[i] * wz[j], wx[i] * sz[j]);
            f.depth += h.y * w;
            f.velocity += h.zw * w;
            f.foam += a.x * w;
            f.light += a.yz * w;
        }
    }
    f.slope /= u.origin.w;
    return f;
}

// One travelling wave (deep-water dispersion): height and its X and Z slopes.
fn wave(p: vec2<f32>, t: f32, angle: f32, length: f32, height: f32, offset: f32) -> vec3<f32> {
    let d = vec2(cos(angle), sin(angle));
    let k = 6.2831853 / length;
    let phase = dot(d, p) * k - sqrt(9.81 * k) * t + offset;
    return vec3(height * sin(phase), height * k * cos(phase) * d);
}
// Wind sea: twelve waves of unrelated lengths spread around the wind's heading (so they never
// line up into a repeating pattern), their heights following the spectrum's fall toward short
// waves, under a slow drifting envelope that gathers them into groups, as on open water.
const WIND: f32 = 0.6;
const WAVE_COUNT: i32 = 12;
fn waves(p: vec2<f32>, t: f32) -> vec3<f32> {
    var sum = vec3(0.0);
    var total = 0.0;
    for (var i = 0; i < WAVE_COUNT; i++) {
        let fi = f32(i);
        // Lengths from 9 down to about 1.2 blocks, by an irrational ratio.
        let length = 9.0 * pow(0.833, fi);
        // Golden-angle spread within +-70 degrees of the wind.
        let spread = (fract(fi * 0.618034) - 0.5) * 2.44;
        let height = pow(length / 9.0, 1.2);
        sum += wave(p, t, WIND + spread, length, height, fi * 2.399);
        total += height;
    }
    let envelope = 0.55 + 0.45 * noise(p * 0.045 + vec2(t * 0.021, -t * 0.013)).x * 2.0;
    return sum / total * 2.0 * clamp(envelope, 0.15, 1.0);
}

struct SurfaceOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // World X and Z (wrapped).
    @location(2) world: vec2<f32>,
    @location(3) velocity: vec2<f32>,
    // Depth, foam, the edge's fade.
    @location(4) water: vec3<f32>,
    // Sky and block light, 0 to 1.
    @location(5) light: vec2<f32>,
}

@vertex fn vs_surface(@builtin(vertex_index) vertex: u32) -> SurfaceOut {
    let per_cell = u.phase.w;
    let row = u32(u.grid.x * per_cell) + 1u;
    let g = vec2<f32>(f32(vertex % row), f32(vertex / row)) / per_cell;
    let f = field(g);
    let local = g * u.origin.w;
    let world = u.phase.xy + local;
    let height = u.grid.w * (1.0 + u.grid.z) * smoothstep(0.15, 2.0, f.depth);
    let w = waves(world, u.grid.y) * height;
    let side = u.grid.x * u.origin.w;
    let edge = min(min(local.x, local.y), side - max(local.x, local.y));
    var out: SurfaceOut;
    out.position = vec3(u.origin.x + local.x, f.surface + w.x, u.origin.z + local.y);
    out.clip = u.view_proj * vec4(out.position, 1.0);
    out.normal = normalize(vec3(-(f.slope.x + w.y), 1.0, -(f.slope.y + w.z)));
    out.world = world;
    out.velocity = f.velocity;
    out.water = vec3(f.depth, f.foam, smoothstep(0.0, u.phase.z, edge));
    out.light = f.light;
    return out;
}

// A hash of an integer lattice point to a gradient in [-1, 1)^2.
fn gradient(i: vec2<i32>) -> vec2<f32> {
    let h = bitcast<vec2<u32>>(i) * vec2(1597334673u, 3812015801u);
    let n = (h.x ^ h.y) * 1597334673u;
    let m = vec2(n, n * 3812015801u);
    return vec2<f32>(m >> vec2(8u)) / 8388608.0 - 1.0;
}
// Gradient noise: its value and its derivative.
fn noise(p: vec2<f32>) -> vec3<f32> {
    let cell = floor(p);
    let i = vec2<i32>(cell);
    let f = p - cell;
    let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let ds = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let ga = gradient(i);
    let gb = gradient(i + vec2(1, 0));
    let gc = gradient(i + vec2(0, 1));
    let gd = gradient(i + vec2(1, 1));
    let va = dot(ga, f);
    let vb = dot(gb, f - vec2(1.0, 0.0));
    let vc = dot(gc, f - vec2(0.0, 1.0));
    let vd = dot(gd, f - vec2(1.0, 1.0));
    let k = va - vb - vc + vd;
    let value = va + s.x * (vb - va) + s.y * (vc - va) + s.x * s.y * k;
    let d = ga + s.x * (gb - ga) + s.y * (gc - ga) + s.x * s.y * (ga - gb - gc + gd)
        + ds * (s.yx * k + vec2(vb, vc) - va);
    return vec3(value, d);
}
// Small ripples' slope at `p`: three octaves drifting with the wind.
fn ripple_layer(p: vec2<f32>, t: f32) -> vec2<f32> {
    return noise(p * 1.1 + vec2(t * 0.11, t * 0.07)).yz * 1.1
        + noise(p * 2.7 - vec2(t * 0.13, -t * 0.05)).yz * 2.7 * 0.35
        + noise(p * 6.3 + vec2(-t * 0.2, t * 0.17)).yz * 6.3 * 0.12;
}
// The ripples carried by the flow: two layers, each advected for a period and reset while
// the other shows, so they move with the water without stretching.
fn ripples(p: vec2<f32>, flow: vec2<f32>, t: f32) -> vec2<f32> {
    let period = 2.0;
    let a = fract(t / period);
    let b = fract(t / period + 0.5);
    let wa = 1.0 - abs(2.0 * a - 1.0);
    let la = ripple_layer(p - flow * a * period, t);
    let lb = ripple_layer(p - flow * b * period + vec2(0.37, 0.71), t);
    return la * wa + lb * (1.0 - wa);
}

fn linear_fog(d: f32, start: f32, end: f32) -> f32 {
    if d <= start { return 0.0; }
    if d >= end { return 1.0; }
    return (d - start) / (end - start);
}
fn fogged(color: vec3<f32>, position: vec3<f32>) -> vec3<f32> {
    let spherical = length(position);
    let cylindrical = max(length(position.xz), abs(position.y));
    let f = max(linear_fog(spherical, u.fog.x, u.fog.y), linear_fog(cylindrical, u.fog.z, u.fog.w));
    return mix(color, u.fog_color.rgb, f * u.fog_color.a);
}
// How lit the water is: the sky's light by day, block light at any time.
fn brightness(light: vec2<f32>) -> f32 {
    let l = max(light.x * mix(0.15, 1.0, u.sky.w), light.y * 0.85);
    return 0.12 + 0.88 * l * l;
}

@fragment fn fs_surface(in: SurfaceOut) -> @location(0) vec4<f32> {
    let depth = in.water.x;
    let wet = smoothstep(0.0, 0.05, depth) * in.water.z;
    if wet < 0.01 { discard; }
    // Cut along block edges: the nearest cell, not the smoothed field, says where blocks are.
    let n_cells = i32(u.grid.x);
    let cell = vec2<i32>(floor((in.position.xz - u.origin.xz) / u.origin.w));
    if all(cell >= vec2(0)) && all(cell < vec2(n_cells)) && textureLoad(aux, cell, 0).w > 0.5 {
        discard;
    }
    let speed = length(in.velocity);
    let detail = ripples(in.world, in.velocity, u.grid.y) * (0.015 + 0.015 * min(speed, 3.0) + 0.03 * u.grid.z);
    var n = normalize(in.normal - vec3(detail.x, 0.0, detail.y));
    let view = normalize(-in.position);
    // The camera under this part of the surface sees it from below.
    let below = in.position.y > 0.0;
    if below { n = -n; }
    let facing = max(dot(n, view), 0.0);
    let f0 = u.deep.w;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - facing, 5.0);
    let lit = brightness(in.light);
    let r = reflect(-view, n);
    let sky = mix(u.fog_color.rgb, u.sky.rgb, smoothstep(-0.05, 0.45, r.y));
    let reflection = sky * mix(0.12, 1.0, in.light.x * u.sky.w);
    let sun = normalize(u.sun.xyz);
    let glint = pow(max(dot(r, sun), 0.0), u.foam.w) * u.sky.w * in.light.x
        * smoothstep(-0.05, 0.1, sun.y) * 3.0;
    let clear = exp(-max(depth, 0.0) * u.shallow.w);
    let body = mix(u.deep.rgb, u.shallow.rgb, clear) * lit;
    var color = mix(body, reflection, fresnel) + vec3(glint);
    var alpha = mix(clamp(1.0 - clear * 0.7, 0.3, 0.95), 1.0, fresnel);
    // Foam: the field's amount as the coverage of a flowing pattern.
    let coverage = clamp(in.water.y * 1.6, 0.0, 1.0);
    let pattern = 0.5 + 0.5 * (noise(in.world * 2.3 - in.velocity * 0.4).x
        + 0.5 * noise(in.world * 5.9 + vec2(u.grid.y * 0.1)).x);
    let foam = smoothstep(1.0 - coverage, 1.2 - coverage, pattern) * step(0.001, coverage);
    color = mix(color, u.foam.rgb * lit, foam);
    alpha = max(alpha, foam);
    if below { alpha *= 0.6; }
    return vec4(fogged(color, in.position), alpha * wet);
}

struct DropletOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) opacity: f32,
    @location(2) position: vec3<f32>,
}

@vertex fn vs_droplet(@builtin(vertex_index) vertex: u32, @location(0) at: vec4<f32>,
    @location(1) extra: vec4<f32>) -> DropletOut {
    let corner = vec2(f32(vertex & 1u), f32(vertex >> 1u)) * 2.0 - 1.0;
    var out: DropletOut;
    out.position = at.xyz + (u.right.xyz * corner.x + u.up.xyz * corner.y) * at.w;
    out.clip = u.view_proj * vec4(out.position, 1.0);
    out.corner = corner;
    out.opacity = extra.x;
    return out;
}

@fragment fn fs_droplet(in: DropletOut) -> @location(0) vec4<f32> {
    let r = dot(in.corner, in.corner);
    if r > 1.0 { discard; }
    let color = mix(u.foam.rgb, u.sky.rgb, 0.25) * mix(0.25, 1.0, u.sky.w);
    return vec4(fogged(color, in.position), (1.0 - r) * in.opacity * 0.85);
}
