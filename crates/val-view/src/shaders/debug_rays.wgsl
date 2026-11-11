struct Cam {
    pos: vec4<f32>,
    look_at: vec4<f32>,
    up: vec4<f32>,
    fov: f32,
    aspect: f32,
    screen: vec2<f32>,
};

struct DebugRaySegment {
    start: vec4<f32>, // xyz = world-space start, w = normalized packet energy
    end: vec4<f32>,   // xyz = world-space end, w = valid flag
    color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> debug_segments: array<DebugRaySegment>;
@group(0) @binding(1) var<uniform> cam: Cam;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn offscreen() -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    return output;
}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let segment = debug_segments[vertex_index / 2u];
    if (segment.end.w < 0.5) {
        return offscreen();
    }

    let forward_delta = cam.look_at.xyz - cam.pos.xyz;
    if (dot(forward_delta, forward_delta) < 0.00000001) {
        return offscreen();
    }
    let forward = normalize(forward_delta);
    let right_delta = cross(forward, cam.up.xyz);
    if (dot(right_delta, right_delta) < 0.00000001) {
        return offscreen();
    }
    let right = normalize(right_delta);
    let up = normalize(cross(right, forward));
    let start_relative = segment.start.xyz - cam.pos.xyz;
    let end_relative = segment.end.xyz - cam.pos.xyz;
    let start_depth = dot(start_relative, forward);
    let end_depth = dot(end_relative, forward);
    // Skip the entire segment if either endpoint is behind the near plane.
    // This avoids a false diagonal from the screen corner into the visible view.
    if (start_depth <= 0.001 || end_depth <= 0.001) {
        return offscreen();
    }
    let start_ndc = vec2<f32>(
        dot(start_relative, right) / (start_depth * cam.aspect),
        dot(start_relative, up) / (start_depth * cam.fov),
    );
    let end_ndc = vec2<f32>(
        dot(end_relative, right) / (end_depth * cam.aspect),
        dot(end_relative, up) / (end_depth * cam.fov),
    );
    // Naga/WGSL in the project's wgpu 0.20 toolchain does not expose
    // isNan/isInf. NaN is the only value not caught by the magnitude
    // bounds below, so detect it with the IEEE self-inequality property.
    // Infinite values are rejected by the same magnitude bounds.
    if (any(abs(start_ndc) > vec2<f32>(4.0))
        || any(abs(end_ndc) > vec2<f32>(4.0))
        || any(start_ndc != start_ndc)
        || any(end_ndc != end_ndc)) {
        return offscreen();
    }
    let is_end = (vertex_index & 1u) == 1u;
    let ndc = select(start_ndc, end_ndc, is_end);

    var output: VertexOutput;
    output.position = vec4<f32>(ndc, 0.5, 1.0);
    let energy_brightness = 0.24 + 0.76 * sqrt(clamp(segment.start.w, 0.0, 1.0));
    output.color = vec4<f32>(
        segment.color.rgb * energy_brightness,
        segment.color.a * energy_brightness,
    );
    return output;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
