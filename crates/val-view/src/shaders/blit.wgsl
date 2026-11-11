@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_smp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
    // Generate the fullscreen triangle directly from vertex_index.
    // This avoids dynamically indexing a function-local array, which
    // is rejected by the Naga version bundled with wgpu 0.20.
    let p = vec2<f32>(
        f32((vid << 1u) & 2u) * 2.0 - 1.0,
        f32(vid & 2u) * 2.0 - 1.0
    );
    var out: VOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    return textureSample(scene_tex, scene_smp, in.uv);
}
