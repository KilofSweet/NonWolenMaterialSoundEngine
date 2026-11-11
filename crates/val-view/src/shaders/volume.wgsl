struct Cam {
    pos: vec4<f32>,
    look_at: vec4<f32>,
    up: vec4<f32>,
    fov: f32,
    aspect: f32,
    screen: vec2<f32>,
};
struct Dim {
    dims: vec4<u32>,
    spacing: vec4<f32>,
};
struct Play {
    t_us: f32,
    freq: f32,
    shell_thickness_m: f32,
    fade_us: f32,
    source: vec4<f32>,
    visual: vec4<f32>, // air gain, material energy gain, base opacity, source glow gain
    emission: vec4<f32>, // xyz = source direction, w = cone cosine
    history: vec4<f32>, // x = bin width us, z = inverse fixed-point scale, w = max felt alpha
    visual2: vec4<f32>, // xyz = energy hot color (linear RGB), w = max air-wave alpha
    history_counts: vec4<u32>, // x = exact number of time bins
};
struct MaterialBounds {
    min: vec4<u32>,
    max: vec4<u32>,
    properties: vec4<f32>, // effective density, mean absorption, mean scattering, mean sound speed
};

@group(0) @binding(0) var live_energy_tex: texture_3d<f32>;
@group(0) @binding(1) var<uniform> cam: Cam;
@group(0) @binding(2) var<uniform> dim: Dim;
@group(0) @binding(3) var<uniform> play: Play;
@group(0) @binding(4) var medium_tex: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> material_visuals: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> material_bounds: array<MaterialBounds>;

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    switch i32(i) % 6 {
        case 0: { return vec3<f32>(v, t, p); }
        case 1: { return vec3<f32>(q, v, p); }
        case 2: { return vec3<f32>(p, v, t); }
        case 3: { return vec3<f32>(p, q, v); }
        case 4: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

fn ray_box(ro: vec3<f32>, rd: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
    let eps = 0.0000001;
    if (abs(rd.x) < eps && (ro.x < lo.x || ro.x >= hi.x)) { return vec2<f32>(1.0e30, -1.0e30); }
    if (abs(rd.y) < eps && (ro.y < lo.y || ro.y >= hi.y)) { return vec2<f32>(1.0e30, -1.0e30); }
    if (abs(rd.z) < eps && (ro.z < lo.z || ro.z >= hi.z)) { return vec2<f32>(1.0e30, -1.0e30); }
    let safe_rd = select(vec3<f32>(eps), rd, abs(rd) >= vec3<f32>(eps));
    let a = (lo - ro) / safe_rd;
    let b = (hi - ro) / safe_rd;
    let near_v = min(a, b);
    let far_v = max(a, b);
    let near_t = max(max(near_v.x, near_v.y), max(near_v.z, 0.0));
    let far_t = min(far_v.x, min(far_v.y, far_v.z));
    return vec2<f32>(near_t, far_t);
}

// Read the same full-world time histogram for both air and material voxels.
// A voxel only becomes visible after packets have actually reached its arrival bin.
fn voxel_energy(cell: vec3<u32>) -> f32 {
    let dims = dim.dims.xyz;
    if (any(cell >= dims)) { return 0.0; }
    return max(textureLoad(live_energy_tex, vec3<i32>(cell), 0).x, 0.0);
}

fn is_inside_material(position: vec3<f32>) -> bool {
    let material_count = arrayLength(&material_bounds);
    for (var i = 0u; i < material_count; i = i + 1u) {
        let region = material_bounds[i];
        let lo = vec3<f32>(region.min.xyz) * dim.spacing.x;
        let hi = vec3<f32>(region.max.xyz) * dim.spacing.x;
        if (all(position >= lo) && all(position < hi)) {
            return true;
        }
    }
    return false;
}

fn air_energy_at(position: vec3<f32>) -> f32 {
    let world_max = vec3<f32>(dim.dims.xyz) * dim.spacing.x;
    if (any(position < vec3<f32>(0.0)) || any(position >= world_max)) {
        return 0.0;
    }
    let cell = vec3<u32>(floor(position / dim.spacing.x));
    return voxel_energy(cell);
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(
        f32((vid << 1u) & 2u) * 2.0 - 1.0,
        f32(vid & 2u) * 2.0 - 1.0
    );
    return vec4<f32>(p, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(
        frag_coord.x / cam.screen.x * 2.0 - 1.0,
        1.0 - frag_coord.y / cam.screen.y * 2.0
    );
    let forward = normalize(cam.look_at.xyz - cam.pos.xyz);
    let right = normalize(cross(forward, cam.up.xyz));
    let up = cross(right, forward);
    let ray_dir = normalize(forward + ndc.x * right * cam.aspect + ndc.y * up * cam.fov);

    let voxel_size = dim.spacing.x;
    let cell = vec3<f32>(voxel_size);
    let material_count = arrayLength(&material_bounds);
    let frequency_position = clamp((log2(max(play.freq, 100.0) / 1000.0) + 1.0) / 6.0, 0.0, 1.0);
    let shell_hue = fract(0.54 + 0.48 * frequency_position);
    let air_wave_col = hsv2rgb(shell_hue, 0.78, 1.0);

    let bg = vec3<f32>(0.008, 0.012, 0.026);
    var color = vec3<f32>(0.0);

    // The air front is sampled from the same time-resolved packet histogram as
    // the felt. It therefore cannot run through a slab or illuminate a second
    // insert before transmitted packet energy has actually entered that air voxel.
    let source_pos = play.source.xyz;
    let to_source = source_pos - cam.pos.xyz;
    let closest_t = max(dot(to_source, ray_dir), 0.0);
    let closest_point = cam.pos.xyz + ray_dir * closest_t;
    let source_delta = closest_point - source_pos;
    let source_distance = length(source_delta);
    let source_core_radius = max(play.source.w, voxel_size * 1.25);
    let source_core = 1.0 - smoothstep(
        source_core_radius * 0.5,
        source_core_radius,
        length(to_source - ray_dir * closest_t)
    );
    let source_halo = exp(-source_distance / (source_core_radius * 4.0));
    color += vec3<f32>(0.035, 0.30, 0.72) * source_halo * 0.20 * play.visual.w;
    color += vec3<f32>(1.0, 0.78, 0.48) * source_core * 1.8 * play.visual.w;

    // Raymarch the actual air portion of the 3D world. Samples read the global
    // packet histogram, so a front behind a material exists only after packets
    // have crossed its entry/exit interfaces and deposited energy in those cells.
    let world_max = vec3<f32>(dim.dims.xyz) * voxel_size;
    let world_hit = ray_box(cam.pos.xyz, ray_dir, vec3<f32>(0.0), world_max);
    let air_enter = max(world_hit.x, 0.0);
    let air_exit = world_hit.y;
    if (play.t_us > 0.0 && air_exit > air_enter) {
        let air_span = air_exit - air_enter;
        let air_step = min(max(max(air_span / 96.0, voxel_size), play.shell_thickness_m), air_span);
        var air_transmittance = 1.0;
        for (var air_sample = 0u; air_sample < 96u; air_sample = air_sample + 1u) {
            let distance = air_enter + (f32(air_sample) + 0.5) * air_step;
            if (distance >= air_exit || air_transmittance < 0.035) { break; }
            let p = cam.pos.xyz + ray_dir * distance;
            if (!is_inside_material(p)) {
                let energy = air_energy_at(p);
                // air_wave_gain is the direct response scale for relative packet energy.
                let intensity = 1.0 - exp(-energy * play.visual.x);
                if (intensity > 0.001) {
                    let air_alpha = clamp(intensity * play.visual2.w, 0.0, play.visual2.w);
                    let air_color = air_wave_col * (0.32 + 0.68 * intensity);
                    color += air_color * air_alpha * air_transmittance;
                    air_transmittance *= 1.0 - air_alpha;
                }
            }
        }
    }

    // Find the nearest material region, process it, then repeat. This orders
    // multiple translucent felt instances front-to-back without scanning air.
    var processed_mask = 0u;
    var scene_transmittance = 1.0;
    for (var order = 0u; order < material_count; order = order + 1u) {
        var nearest_index = material_count;
        var nearest_enter = 1.0e30;
        var nearest_exit = -1.0;

        for (var candidate = 0u; candidate < material_count; candidate = candidate + 1u) {
            let bit = 1u << candidate;
            if ((processed_mask & bit) == 0u) {
                let bounds = material_bounds[candidate];
                let lo = vec3<f32>(bounds.min.xyz) * voxel_size;
                let hi = vec3<f32>(bounds.max.xyz) * voxel_size;
                let hit = ray_box(cam.pos.xyz, ray_dir, lo, hi);
                if (hit.y > hit.x && hit.x < nearest_enter) {
                    nearest_index = candidate;
                    nearest_enter = hit.x;
                    nearest_exit = hit.y;
                }
            }
        }

        if (nearest_index >= material_count) { break; }
        processed_mask = processed_mask | (1u << nearest_index);

        let bounds = material_bounds[nearest_index];
        let lower_cell = vec3<i32>(bounds.min.xyz);
        let upper_cell = vec3<i32>(bounds.max.xyz) - vec3<i32>(1);
        let p0 = cam.pos.xyz + ray_dir * (nearest_enter + voxel_size * 0.0001);
        var vi = clamp(vec3<i32>(floor(p0 / voxel_size)), lower_cell, upper_cell);
        let step_i = select(vec3<i32>(-1), vec3<i32>(1), ray_dir >= vec3<f32>(0.0));
        let safe_dir = select(
            vec3<f32>(0.0000001), ray_dir,
            abs(ray_dir) > vec3<f32>(0.0000001)
        );
        let t_delta = abs(cell / safe_dir);
        let next_cell = vec3<f32>(vi) + select(
            vec3<f32>(0.0), vec3<f32>(1.0), ray_dir >= vec3<f32>(0.0)
        );
        var t_max = (next_cell * cell - cam.pos.xyz) / safe_dir;
        t_max = select(vec3<f32>(1.0e30), t_max, abs(ray_dir) > vec3<f32>(0.0000001));
        let extent = bounds.max.xyz - bounds.min.xyz;
        let step_limit = extent.x + extent.y + extent.z + 4u;
        let material_visual = material_visuals[nearest_index];
        var local_transmittance = scene_transmittance;

        for (var step = 0u; step < step_limit; step = step + 1u) {
            if (any(vi < lower_cell) || any(vi > upper_cell)) { break; }
            let med = textureLoad(medium_tex, vi, 0);

            // Semi-transparent material body with deterministic cell-scale grain.
            let cell_hash = (u32(vi.x) * 1973u)
                ^ (u32(vi.y) * 9277u)
                ^ (u32(vi.z) * 26699u)
                ^ (nearest_index * 31847u);
            let hash_mix = cell_hash ^ (cell_hash >> 16u);
            let fiber = 0.86 + 0.14 * (f32(hash_mix & 255u) / 255.0);
            let density_variation = clamp(med.y / 85.0, 0.0, 1.4);
            let base_alpha = clamp(
                material_visual.a * play.visual.z * fiber,
                0.0,
                0.16
            );
            let material_base = material_visual.rgb * (0.78 + 0.28 * density_variation);

            // Energy is the common driver for both color and opacity. Empty
            // cells retain only their configured base translucency; cells with
            // packet energy become brighter, warmer and more opaque together.
            let accumulated_energy = voxel_energy(vec3<u32>(vi));
            // felt_energy_gain directly maps the normalized packet-history value
            // to a 0..1 response. Material attenuation/scattering already changed
            // accumulated_energy in the transport solve; do not amplify it again
            // based on coefficients here. This keeps color and alpha tied to the
            // energy that actually reached this voxel.
            let response = clamp(
                1.0 - exp(-accumulated_energy * play.visual.y),
                0.0,
                1.0
            );
            let hot = clamp(response * 0.92, 0.0, 1.0);
            let energy_color = mix(
                material_base,
                play.visual2.xyz,
                hot
            );
            let maximum_energy_alpha = clamp(play.history.w, base_alpha, 0.95);
            let energy_alpha = mix(base_alpha, maximum_energy_alpha, response);
            color += energy_color * energy_alpha * local_transmittance;
            local_transmittance *= 1.0 - energy_alpha;

            if (local_transmittance < 0.035) { break; }
            let t_next = min(t_max.x, min(t_max.y, t_max.z));
            if (t_next > nearest_exit) { break; }
            if (t_max.x <= t_next + 0.000001) {
                vi.x += step_i.x;
                t_max.x += t_delta.x;
            }
            if (t_max.y <= t_next + 0.000001) {
                vi.y += step_i.y;
                t_max.y += t_delta.y;
            }
            if (t_max.z <= t_next + 0.000001) {
                vi.z += step_i.z;
                t_max.z += t_delta.z;
            }
        }
        scene_transmittance = local_transmittance;
    }

    return vec4<f32>(bg + color, 1.0);
}
