
fn daylight() -> f32 {
    return select(
        0.0,
        smoothstep(-0.1, 0.12, frame.sun_direction.y),
        frame.dimension == 0
    );
}

// Sample the shadow map multiple times to soften edges.
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

    // 3x3 percentage-closer filtering
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(
                f32(x), f32(y)
            ) * texel;

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

    var brightness = 1.0;

    if (frame.has_shadow != 0u) {
        let position =
            input.position + input.normal * 0.05;

        let visibility = soft_shadow(position);

        // Softer, less intense shadows
        let strength =
            0.40 * daylight() * input.light.y;

        brightness =
            1.0 - (1.0 - visibility) * strength;
    }

    return Surface(vec4<f32>(
        base.rgb * brightness,
        base.a
    ));
}
