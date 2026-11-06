
fn post(input: PostInput) -> vec4<f32> {
    // Nothing to occlude in the sky.
    if (input.depth <= 0.0) {
        return input.color;
    }

    let size = vec2<f32>(textureDimensions(scene_depth));
    let pixel = vec2<i32>(
        clamp(input.uv * size, vec2<f32>(0.0), size - 1.0)
    );

    let surface = surface_at(pixel);

    // Start with terrain only.
    if (!surface.present || surface.data > 0.5) {
        return input.color;
    }

    let position = view_position(input.uv, input.depth);
    let normal = normalize(surface.normal);

    // Twelve surrounding depth samples.
    let offsets = array<vec2<f32>, 12>(
        vec2<f32>(6.0, 0.0),
        vec2<f32>(-6.0, 0.0),
        vec2<f32>(0.0, 6.0),
        vec2<f32>(0.0, -6.0),

        vec2<f32>(5.0, 5.0),
        vec2<f32>(-5.0, 5.0),
        vec2<f32>(5.0, -5.0),
        vec2<f32>(-5.0, -5.0),

        vec2<f32>(14.0, 0.0),
        vec2<f32>(-14.0, 0.0),
        vec2<f32>(0.0, 14.0),
        vec2<f32>(0.0, -14.0)
    );

    var occlusion = 0.0;

    // Expand the sampling pattern slightly with distance.
    // Reduce the screen-space sampling radius with distance.
    // Wider AO nearby, gradually smaller farther away.
    let scale = mix(
        1.7,
        0.65,
        smoothstep(6.0, 30.0, length(position))
    );

    for (var i = 0; i < 12; i = i + 1) {
        let uv = input.uv + offsets[i] * scale / size;

        if (any(uv <= vec2<f32>(0.0)) ||
            any(uv >= vec2<f32>(1.0))) {
            continue;
        }

        let depth = depth_at(uv);

        if (depth <= 0.0) {
            continue;
        }

        let sample_pixel = vec2<i32>(
            clamp(uv * size, vec2<f32>(0.0), size - 1.0)
        );

        let sample_surface = surface_at(sample_pixel);

        // Don't treat grass and leaves as solid occluders.
        if (!sample_surface.present || sample_surface.data > 0.5) {
            continue;
        }

        let sample_pos = view_position(uv, depth);
        let delta = sample_pos - position;
        let dist = length(delta);

        // Ignore the same surface and distant geometry.
        if (dist < 0.02 || dist > 2.5) {
            continue;
        }

        // Geometry in the hemisphere above the surface
        // contributes to ambient occlusion.
        let angle = dot(normal, delta / dist);

        let direction_weight = smoothstep(
            0.05, 0.5, angle
        );

        let distance_weight = 1.0 - smoothstep(
            0.2, 2.5, dist
        );

        occlusion += direction_weight * distance_weight;
    }

    // Gentle contact darkening.
    // AO is most useful for nearby contact details.
    let distance = length(position);

    // Full strength nearby, fading out over distance.
    let distance_fade = 1.0 - smoothstep(
        12.0,
        40.0,
        distance
    );

    let ao = clamp(
        occlusion / 12.0 * 0.85 * distance_fade,
        0.0,
        0.25
    );

    return vec4<f32>(
        input.color.rgb * (1.0 - ao),
        input.color.a
    );
}
