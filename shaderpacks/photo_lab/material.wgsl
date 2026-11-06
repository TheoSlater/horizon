fn material(input: MaterialInput) -> Surface {
    let base = input.albedo * input.tint *
        vec4<f32>(input.lightmap, 1.0);

    let warmth = vec3<f32>(1.15, 1.05, 0.90);

    return Surface(vec4<f32>(
        min(base.rgb * warmth, vec3<f32>(1.0)),
        base.a
    ));
}
