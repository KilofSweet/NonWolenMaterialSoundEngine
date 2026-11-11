struct SolveParams {
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    spacing: f32,
    src: vec4<f32>,                  // xyz = source position, w = transport step (m)
    source_dir: vec4<f32>,           // xyz = cone axis, w = cos(half angle)
    max_steps: u32,
    num_rays: u32,
    material_count: u32,
    seed: u32,
    source_energy: f32,              // relative total source energy, divided among packets
    air_speed_mps: f32,
    air_density_kg_m3: f32,
    air_absorption_m_inv: f32,
    time_bin_us: f32,
    time_bin_count: u32,
    energy_scale: f32,
    _padding: u32,
    debug_ray_count: u32,
    debug_segment_count: u32,
    debug_segment_offset: u32,
    _debug_padding: u32,
    debug_color: vec4<f32>,
};

struct DebugRaySegment {
    start: vec4<f32>,
    end: vec4<f32>,
    color: vec4<f32>,
};

struct MaterialBounds {
    min: vec4<u32>,                  // xyz = inclusive minimum voxel coordinate
    max: vec4<u32>,                  // xyz = exclusive maximum voxel coordinate
    properties: vec4<f32>,           // x = effective density; y/z = mean sigma_a/s; w = mean c
};

struct BoundaryResult {
    direction: vec3<f32>,
    transmitted: bool,
};

@group(0) @binding(0) var medium_tex: texture_3d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<storage, read_write> energy_hist: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: SolveParams;
@group(0) @binding(4) var<storage, read> material_bounds: array<MaterialBounds>;
@group(0) @binding(5) var<storage, read_write> debug_segments: array<DebugRaySegment>;

const EPS: f32 = 0.000002;
const TINY: f32 = 0.0000001;

fn ray_box(ro: vec3<f32>, rd: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
    if (abs(rd.x) < TINY && (ro.x < lo.x || ro.x >= hi.x)) { return vec2<f32>(1.0e30, -1.0e30); }
    if (abs(rd.y) < TINY && (ro.y < lo.y || ro.y >= hi.y)) { return vec2<f32>(1.0e30, -1.0e30); }
    if (abs(rd.z) < TINY && (ro.z < lo.z || ro.z >= hi.z)) { return vec2<f32>(1.0e30, -1.0e30); }
    let safe_rd = select(vec3<f32>(TINY), rd, abs(rd) >= vec3<f32>(TINY));
    let a = (lo - ro) / safe_rd;
    let b = (hi - ro) / safe_rd;
    let near_v = min(a, b);
    let far_v = max(a, b);
    let near_t = max(max(near_v.x, near_v.y), max(near_v.z, 0.0));
    let far_t = min(far_v.x, min(far_v.y, far_v.z));
    return vec2<f32>(near_t, far_t);
}

// Normal points from the current medium toward the medium being entered.
fn box_entry_normal(ro: vec3<f32>, rd: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let safe_rd = select(vec3<f32>(TINY), rd, abs(rd) >= vec3<f32>(TINY));
    let a = (lo - ro) / safe_rd;
    let b = (hi - ro) / safe_rd;
    let near_v = min(a, b);
    if (near_v.x >= near_v.y && near_v.x >= near_v.z) {
        return vec3<f32>(select(-1.0, 1.0, rd.x >= 0.0), 0.0, 0.0);
    }
    if (near_v.y >= near_v.z) {
        return vec3<f32>(0.0, select(-1.0, 1.0, rd.y >= 0.0), 0.0);
    }
    return vec3<f32>(0.0, 0.0, select(-1.0, 1.0, rd.z >= 0.0));
}

fn box_exit_normal(ro: vec3<f32>, rd: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let safe_rd = select(vec3<f32>(TINY), rd, abs(rd) >= vec3<f32>(TINY));
    let a = (lo - ro) / safe_rd;
    let b = (hi - ro) / safe_rd;
    let far_v = max(a, b);
    if (far_v.x <= far_v.y && far_v.x <= far_v.z) {
        return vec3<f32>(select(-1.0, 1.0, rd.x >= 0.0), 0.0, 0.0);
    }
    if (far_v.y <= far_v.z) {
        return vec3<f32>(0.0, select(-1.0, 1.0, rd.y >= 0.0), 0.0);
    }
    return vec3<f32>(0.0, 0.0, select(-1.0, 1.0, rd.z >= 0.0));
}

fn find_material(p: vec3<f32>) -> i32 {
    for (var m = 0u; m < params.material_count; m = m + 1u) {
        let region = material_bounds[m];
        let lo = vec3<f32>(region.min.xyz) * params.spacing;
        let hi = vec3<f32>(region.max.xyz) * params.spacing;
        if (all(p >= lo) && all(p < hi)) { return i32(m); }
    }
    return -1;
}

fn sample_medium(p: vec3<f32>) -> vec4<f32> {
    let dims = vec3<u32>(params.dim_x, params.dim_y, params.dim_z);
    let cell = clamp(vec3<i32>(floor(p / params.spacing)), vec3<i32>(0), vec3<i32>(dims) - vec3<i32>(1));
    return textureLoad(medium_tex, cell, 0);
}

fn interface_event(
    direction: vec3<f32>,
    normal_from_a_to_b: vec3<f32>,
    c_a: f32,
    rho_a: f32,
    c_b: f32,
    rho_b: f32,
    random_value: f32,
) -> BoundaryResult {
    let n = normalize(normal_from_a_to_b);
    let cos_i = clamp(dot(direction, n), 0.0, 1.0);
    let eta = c_b / max(c_a, 1.0);
    let sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i);
    if (sin2_t >= 1.0) {
        return BoundaryResult(normalize(direction - 2.0 * cos_i * n), false);
    }

    let cos_t = sqrt(max(0.0, 1.0 - sin2_t));
    let z_a = max(rho_a * c_a, 0.001);
    let z_b = max(rho_b * c_b, 0.001);
    let denom = max(z_b * cos_i + z_a * cos_t, 0.001);
    let pressure_r = (z_b * cos_i - z_a * cos_t) / denom;
    let reflect_probability = clamp(pressure_r * pressure_r, 0.0, 1.0);

    if (random_value < reflect_probability) {
        return BoundaryResult(normalize(direction - 2.0 * cos_i * n), false);
    }

    let tangent = direction - cos_i * n;
    let refracted = normalize(eta * tangent + cos_t * n);
    return BoundaryResult(refracted, true);
}

fn advance_rng(state: u32) -> u32 {
    return state * 1664525u + 1013904223u;
}

fn random_from_state(state: u32) -> f32 {
    return f32(state & 0x00ffffffu) / 16777216.0;
}

// One histogram spans the complete world, including air. This is intentionally
// indexed by global voxel so the visualized air field is the actual packet path.
fn deposit_voxel_energy(
    position: vec3<f32>,
    packet_energy: f32,
    arrival_time_us: f32,
    segment_length: f32,
) {
    let dims = vec3<u32>(params.dim_x, params.dim_y, params.dim_z);
    let cell = clamp(
        vec3<u32>(floor(position / params.spacing)),
        vec3<u32>(0u),
        dims - vec3<u32>(1u),
    );
    let cell_index = cell.x + cell.y * params.dim_x + cell.z * params.dim_x * params.dim_y;
    let bin_f = floor(max(arrival_time_us, 0.0) / params.time_bin_us);
    // Avoid comparing against f32(params.time_bin_count), which loses exactness
    // for large integer counts. Conversion is safe only inside the u32 range.
    if (bin_f >= 0.0 && bin_f < 4294967296.0) {
        let bin = u32(bin_f);
        if (bin < params.time_bin_count) {
            let contribution = packet_energy * (segment_length / params.spacing) * params.energy_scale;
            if (contribution > 0.5) {
                let fixed_point = u32(min(round(contribution), 4294967040.0));
                let histogram_index = cell_index * params.time_bin_count + bin;
                atomicAdd(&energy_hist[histogram_index], fixed_point);
            }
        }
    }
}

fn record_debug_segment(
    ray_index: u32,
    segment_index: u32,
    start: vec3<f32>,
    end: vec3<f32>,
    normalized_energy: f32,
) {
    if (ray_index < params.debug_ray_count && segment_index < params.debug_segment_count) {
        let index = params.debug_segment_offset
            + ray_index * params.debug_segment_count
            + segment_index;
        debug_segments[index] = DebugRaySegment(
            vec4<f32>(start, clamp(normalized_energy, 0.0, 1.0)),
            vec4<f32>(end, 1.0),
            params.debug_color,
        );
    }
}

fn scatter_direction(direction: vec3<f32>, g: f32, u: f32, phi: f32) -> vec3<f32> {
    var cos_theta = 1.0 - 2.0 * u;
    if (abs(g) > 0.001) {
        let q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        cos_theta = clamp((1.0 + g * g - q * q) / (2.0 * g), -1.0, 1.0);
    }
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(direction.y) > 0.92);
    let tangent = normalize(cross(helper, direction));
    let bitangent = cross(direction, tangent);
    return normalize(direction * cos_theta + sin_theta * (tangent * cos(phi) + bitangent * sin(phi)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_rays) { return; }

    // Spread debug captures over the complete packet index range. Capturing
    // only the first N indices would cluster around one edge of the cone because
    // the deterministic cone sampler derives its polar coordinate from idx.
    var debug_ray_index = 0u;
    var capture_debug_ray = false;
    if (params.debug_ray_count > 0u) {
        let debug_stride = (params.num_rays - 1u) / params.debug_ray_count + 1u;
        debug_ray_index = idx / debug_stride;
        capture_debug_ray = (idx % debug_stride == 0u)
            && (debug_ray_index < params.debug_ray_count);
    }

    // Deterministic, uniformly distributed rays inside the configured source cone.
    let n = f32(idx);
    let u = (n + 0.5) / f32(params.num_rays);
    let cos_theta = mix(params.source_dir.w, 1.0, u);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.39996323 * n + f32(params.seed & 0xffffu) * (6.28318530718 / 65536.0);
    let forward = normalize(params.source_dir.xyz);
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(forward.y) > 0.92);
    let tangent = normalize(cross(helper, forward));
    let bitangent = cross(forward, tangent);
    var dir = normalize(forward * cos_theta + sin_theta * (tangent * cos(phi) + bitangent * sin(phi)));

    var rng = (idx * 747796405u + 2891336453u) ^ params.seed;
    var pos = params.src.xyz;
    var current_material: i32 = -1;
    var time_us = 0.0;
    var packet_energy = max(params.source_energy, 0.0) / f32(params.num_rays);
    let step_m = params.src.w;
    let world_extent = vec3<f32>(f32(params.dim_x), f32(params.dim_y), f32(params.dim_z)) * params.spacing;

    for (var iteration = 0u; iteration < params.max_steps; iteration = iteration + 1u) {
        if (any(pos < vec3<f32>(0.0)) || any(pos >= world_extent) || packet_energy < 0.000000001) { break; }

        if (current_material < 0) {
            // Air is explicitly marched and accumulated, not skipped. Steps stop
            // exactly at the nearest material interface or at the world boundary.
            let world_hit = ray_box(pos, dir, vec3<f32>(0.0), world_extent);
            let world_exit = max(world_hit.y, 0.0);
            var boundary_distance = world_exit;
            var nearest_index: i32 = -1;
            for (var m = 0u; m < params.material_count; m = m + 1u) {
                let region = material_bounds[m];
                let lo = vec3<f32>(region.min.xyz) * params.spacing;
                let hi = vec3<f32>(region.max.xyz) * params.spacing;
                let hit = ray_box(pos, dir, lo, hi);
                let entry = max(hit.x, 0.0);
                if (hit.y > entry && entry < boundary_distance) {
                    boundary_distance = entry;
                    nearest_index = i32(m);
                }
            }

            if (boundary_distance <= EPS) { break; }
            let segment_start = pos;
            let segment_length = min(step_m, boundary_distance);
            let segment_end = segment_start + dir * segment_length;
            if (capture_debug_ray) {
                record_debug_segment(
                    debug_ray_index,
                    iteration,
                    segment_start,
                    segment_end,
                    packet_energy / max(params.source_energy / f32(params.num_rays), 0.000000001),
                );
            }
            let sample_position = segment_start + dir * (segment_length * 0.5);
            let midpoint_time = time_us + segment_length / (2.0 * max(params.air_speed_mps, 1.0)) * 1000000.0;
            let energy_at_midpoint = packet_energy
                * exp(-max(params.air_absorption_m_inv, 0.0) * segment_length * 0.5);
            deposit_voxel_energy(sample_position, energy_at_midpoint, midpoint_time, segment_length);

            time_us += segment_length / max(params.air_speed_mps, 1.0) * 1000000.0;
            packet_energy *= exp(-max(params.air_absorption_m_inv, 0.0) * segment_length);
            pos = segment_start + dir * segment_length;

            if (segment_length >= boundary_distance - EPS) {
                if (nearest_index < 0) {
                    // The ray reached the outer world boundary through air.
                    break;
                }
                let region = material_bounds[u32(nearest_index)];
                let lo = vec3<f32>(region.min.xyz) * params.spacing;
                let hi = vec3<f32>(region.max.xyz) * params.spacing;
                let normal = box_entry_normal(segment_start, dir, lo, hi);
                let material_probe = pos + dir * EPS;
                let medium_b = sample_medium(material_probe);
                rng = advance_rng(rng);
                let event = interface_event(
                    dir, normal,
                    params.air_speed_mps, params.air_density_kg_m3,
                    max(medium_b.w, 20.0), region.properties.x,
                    random_from_state(rng),
                );
                dir = event.direction;
                pos += dir * EPS;
                current_material = select(-1, nearest_index, event.transmitted);
            }
            continue;
        }

        let material_index = u32(current_material);
        let region = material_bounds[material_index];
        let lo = vec3<f32>(region.min.xyz) * params.spacing;
        let hi = vec3<f32>(region.max.xyz) * params.spacing;
        let hit = ray_box(pos, dir, lo, hi);
        let distance_to_exit = max(hit.y, 0.0);
        if (distance_to_exit <= EPS) {
            pos += dir * EPS;
            current_material = find_material(pos);
            continue;
        }

        // Each material segment is clipped to the exact exit distance. No step
        // can jump over a thin block or skip a second material behind the first.
        let segment_length = min(step_m, distance_to_exit);
        let segment_start = pos;
        let segment_end = segment_start + dir * segment_length;
        if (capture_debug_ray) {
            record_debug_segment(
                debug_ray_index,
                iteration,
                segment_start,
                segment_end,
                packet_energy / max(params.source_energy / f32(params.num_rays), 0.000000001),
            );
        }
        let sample_position = pos + dir * (segment_length * 0.5);
        let medium = sample_medium(sample_position);
        let local_speed = max(medium.w, 20.0);
        let energy_at_midpoint = packet_energy * exp(-max(medium.x, 0.0) * segment_length * 0.5);
        let midpoint_time = time_us + segment_length / (2.0 * local_speed) * 1000000.0;
        deposit_voxel_energy(sample_position, energy_at_midpoint, midpoint_time, segment_length);

        time_us += segment_length / local_speed * 1000000.0;
        packet_energy *= exp(-max(medium.x, 0.0) * segment_length);
        pos += dir * segment_length;

        if (segment_length >= distance_to_exit - EPS) {
            // Resolve A->B, A->air, or A->A' at the actual boundary.
            let normal = box_exit_normal(sample_position, dir, lo, hi);
            let next_probe = pos + dir * EPS;
            let next_material = find_material(next_probe);
            var next_speed = params.air_speed_mps;
            var next_density = params.air_density_kg_m3;
            if (next_material >= 0) {
                let next_region = material_bounds[u32(next_material)];
                let next_medium = sample_medium(next_probe);
                next_speed = max(next_medium.w, 20.0);
                next_density = next_region.properties.x;
            }

            rng = advance_rng(rng);
            let event = interface_event(
                dir, normal,
                local_speed, region.properties.x,
                next_speed, next_density,
                random_from_state(rng),
            );
            dir = event.direction;
            pos += dir * EPS;
            current_material = select(i32(material_index), next_material, event.transmitted);
            continue;
        }

        // Scattering redirects packets inside the current material; absorption
        // has already reduced their energy through the local sigma_a coefficient.
        let scatter_probability = 1.0 - exp(-max(medium.y, 0.0) * segment_length);
        if (scatter_probability > 0.0) {
            rng = advance_rng(rng);
            let r0 = random_from_state(rng);
            if (r0 < scatter_probability) {
                rng = advance_rng(rng);
                let r1 = random_from_state(rng);
                rng = advance_rng(rng);
                let r2 = random_from_state(rng);
                dir = scatter_direction(dir, clamp(medium.z, -0.95, 0.95), r1, 6.28318530718 * r2);
            }
        }
    }
}
