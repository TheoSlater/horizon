# Realistic renderer

Phase 1 keeps Horizon's existing terrain, entity and shader-pack renderers. The
renderer mode and quality preset live in `graphics::Graphics` and persist in
`horizon-graphics.txt`. Vanilla is the default and does not create Realistic
pipelines or targets. Realistic mode lazily creates `gfx/realistic/` resources
and asks `world_post::Post` to finish the world through its identity pass. That
boundary is after world rendering and before the hand, HUD and UI. Existing
resolution scaling, FXAA and TAA remain in the same order; when FXAA is active,
the Realistic pass reads a resize-aware intermediate, and TAA feeds it from the
existing history target.

Horizon's and imported shader packs own their world/post pipeline while active.
They take precedence over Realistic mode; the Realistic pass is released and
the renderer logs when it is suspended. Removing the pack allows the selected
mode to resume on the next frame. The public shader-pack API is unchanged.

Future Realistic WGSL belongs beside `gfx/realistic/mod.rs`. Phase 1 reuses
`world_post.wgsl`'s copy entry point instead of keeping a second copy shader.
Each pass should own only its pipelines and non-shared targets, with size-based
targets recreated by that owner. Add passes to the Realistic finishing path in
dependency order, leaving world geometry and depth production in the existing
renderer.

Phase 2 can add development-only hot reloading by watching those WGSL files,
compiling replacement modules and pipelines on the render thread, then swapping
them in only after validation succeeds. Release builds should keep the current
build-time `include_str!` path. Keep the watcher and reload controls internal;
they do not require a new shader-pack language or public API.
