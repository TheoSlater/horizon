
fn daylight() -> f32 {
    return select(
        0.0,
        smoothstep(-0.1, 0.12, frame.sun_direction.y),
        frame.dimension == 0
    );
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Create a detailed normal for each surface.

fn surface_normal(input: MaterialInput) -> vec3<f32> {
    let n = normalize(input.normal);

    // Keep vanilla surfaces flat unless real normal maps exist.
    if (input.translucent || !block_pbr_maps()) {
        return n;
    }

    // Read the real LabPBR normal map.
    let map = block_normal_map(input.uv);

    // Reduce the intensity for natural-looking detail.
    let xy = (map.xy * 2.0 - vec2<f32>(1.0)) * 0.55;
    let z = sqrt(max(1.0 - dot(xy, xy), 0.0001));

    let local = normalize(vec3<f32>(xy, z));

    // Convert texture-space normals into world space.
    let t = normalize(input.tangent.xyz);
    let b = normalize(cross(n, t)) * input.tangent.w;

    return normalize(
        t * local.x +
        b * local.y +
        n * local.z
    );
}


// Keep our existing soft shadows.
fn soft_shadow(position: vec3<f32>) -> f32 {
    let p = shadow_position(position);

    if (p.x < 0.0 || p.x > 1.0 ||
        p.y < 0.0 || p.y > 1.0 ||
        p.z < 0.0) {
        return 1.0;
    }

    let texel = 1.5 /
        vec2<f32>(textureDimensions(shadow_map));

    var visibility = 0.0;

    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;

            visibility += textureSampleCompareLevel(
                shadow_map,
                shadow_sampler,
                p.xy + offset,
                p.z - 0.0007
            );
        }
    }

    return visibility / 9.0;
}

fn material(input: MaterialInput) -> Surface {
    let base = input.albedo * input.tint *
        vec4<f32>(input.lightmap, 1.0);

    let n = normalize(input.normal);
    let detail = surface_normal(input);
    let sun = normalize(frame.sun_direction);

    let day = daylight();

    // Light changes caused by the surface's fine detail.
    let flat_light = max(dot(n, sun), 0.0);
    let detailed_light = max(dot(detail, sun), 0.0);

    let detail_brightness = clamp(
        1.0 + (detailed_light - flat_light)
            * 0.85 * day * input.light.y,
        0.75,
        1.25
    );

    var shadow_brightness = 1.0;

    if (frame.has_shadow != 0u) {
        let position = input.position + n * 0.05;
        let visibility = soft_shadow(position);

        let strength = 0.40 * day * input.light.y;
        shadow_brightness =
            1.0 - (1.0 - visibility) * strength;
    }

    let color = clamp(
        base.rgb * detail_brightness * shadow_brightness,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    let vegetation = (input.tags & 1u) != 0u;

    set_surface(
        input.normal,
        0.0,
        select(0.0, 1.0, vegetation)
    );

    return Surface(vec4<f32>(color, base.a));
}
