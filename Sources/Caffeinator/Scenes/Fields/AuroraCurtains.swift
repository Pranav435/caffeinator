import Foundation
import Metal

/// The aurora borealis from the shore of a northern lake, driven by the real planetary Kp index from NOAA.
/// Each curtain is a thin vertical sheet of light hanging along a folded east–west line; a pixel's ray is marched
/// to find where it crosses each sheet, and the sheet's glow is integrated there, coloured by altitude: violet
/// nitrogen at the lower border near 100 km, oxygen's 557.7 nm green above it, and oxygen's 630 nm red high up.
/// Quiet nights get a soft arc low in the north; storms bring bright, fast, folded curtains overhead.
@MainActor final class AuroraCurtains: Scene {
    static let feed = URL(string: "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")!

    /// The latest planetary Kp and the start of its three-hour window.
    struct Reading: Equatable {
        let kp: Double
        let time: Date
    }

    private let table: MTLComputePipelineState
    private let sky: MTLRenderPipelineState
    private let finish: MTLRenderPipelineState
    private let bloom: Bloom
    private let curtains: MTLTexture
    private var frames: [MTLTexture] = []
    private var reading: Reading?
    private var loaded = false
    private var fetched = Date.distantPast
    private var override: Double?
    private var level: Float = -1
    private var settled = false
    private var clock: Float = 0

    private var surgeStart: Float = -1000
    private var nextSurge: Float = 45
    private var brightness: Float = 0
    private var gust: Float = 0

    var scale: Float { 0.5 }
    var ready: Bool { loaded }
    var facts: String? {
        let science = " Oxygen glows green at 557.7 nanometres 100 to 150 kilometres up, and red at 630 nanometres above that."
        if let override { return "Showing a Kp index of \(Self.format(override)), \(Self.describe(override))." + science }
        guard let reading else {
            return loaded ? "The Kp index is unavailable offline, so this is a quiet night." + science : "Fetching the planetary Kp index."
        }
        let clock = DateFormatter()
        clock.dateFormat = "HH:mm"
        clock.timeZone = TimeZone(identifier: "UTC")
        return "Planetary Kp index \(Self.format(reading.kp)), \(Self.describe(reading.kp)), for the three hours from \(clock.string(from: reading.time)) UTC." + science
    }

    private var kp: Double { override ?? reading?.kp ?? 1.5 }

    init(gpu: GPU) throws {
        let library = try gpu.library(FieldKit.stars + Self.source)
        table = try gpu.kernel(library, "auroraTable")
        sky = try gpu.pipeline(library, "auroraSky")
        finish = try gpu.pipeline(library, "auroraFinish")
        bloom = try Bloom(gpu: gpu)
        curtains = gpu.texture(4096, 4)
        // --variant n shows a night with Kp n, for testing.
        override = Snapshot.variant.map { Double(min(max($0, 0), 9)) }
        fetch()
    }

    /// Fetches the Kp index, at most every half hour (NOAA publishes a new value every three hours).
    private func fetch() {
        fetched = .now
        Task { @MainActor in
            if let json = await Feed.object(Self.feed, maxAge: 30 * 60), let latest = Self.parse(json) { reading = latest }
            loaded = true
        }
    }

    /// The last entry with a Kp value. The feed has been both an array of objects and an array of rows with a header,
    /// so both are accepted.
    nonisolated static func parse(_ json: Any) -> Reading? {
        guard let rows = json as? [Any] else { return nil }
        let stamp = DateFormatter()
        stamp.locale = Locale(identifier: "en_US_POSIX")
        stamp.timeZone = TimeZone(identifier: "UTC")
        for row in rows.reversed() {
            var kp: Double?
            var tag: String?
            if let o = row as? [String: Any] {
                kp = (o["Kp"] as? NSNumber)?.doubleValue ?? (o["Kp"] as? String).flatMap(Double.init)
                tag = o["time_tag"] as? String
            } else if let a = row as? [Any], a.count >= 2 {
                kp = (a[1] as? NSNumber)?.doubleValue ?? (a[1] as? String).flatMap(Double.init)
                tag = a[0] as? String
            }
            guard let kp, kp.isFinite, (0...9).contains(kp), let tag else { continue }
            for format in ["yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd HH:mm:ss.SSS", "yyyy-MM-dd HH:mm:ss"] {
                stamp.dateFormat = format
                if let time = stamp.date(from: tag) { return Reading(kp: kp, time: time) }
            }
        }
        return nil
    }

    nonisolated static func describe(_ kp: Double) -> String {
        switch kp {
        case ..<2: "a quiet night"
        case ..<4: "an unsettled night"
        case ..<5: "an active night"
        case ..<6: "a minor geomagnetic storm, G1"
        case ..<7: "a moderate geomagnetic storm, G2"
        case ..<8: "a strong geomagnetic storm, G3"
        case ..<9: "a severe geomagnetic storm, G4"
        default: "an extreme geomagnetic storm, G5"
        }
    }

    nonisolated static func format(_ kp: Double) -> String { String(format: "%.1f", kp) }

    /// How lively the aurora is, 0.12 for Kp 0 (still a soft arc: it's a screensaver) to 1 for Kp 7 and up.
    nonisolated static func activity(_ kp: Double) -> Float { Float(min(max((kp + 0.8) / 7.5, 0.12), 1)) }

    func draw(_ f: Frame) {
        let w = f.target.width, h = f.target.height
        if frames.first?.width != w || frames.first?.height != h { frames = [f.gpu.texture(w, h), f.gpu.texture(w, h)] }
        let t = f.time
        if f.now.timeIntervalSince(fetched) > 30 * 60 { fetch() }
        // Activity eases to a new Kp over half a minute rather than jumping.
        let goal = Self.activity(kp)
        if level < 0 || loaded != settled {
            level = goal
            settled = loaded
        }
        level += (goal - level) * min(f.dt / 30, 1)
        let act = level

        // Brightness breathes over a minute and a half; substorms flare up every few minutes, more often and more
        // strongly on active nights.
        if t > nextSurge {
            surgeStart = t
            nextSurge = t + .random(in: 150...360) / (0.5 + act)
        }
        let since = t - surgeStart
        let flare = since < 20 ? smooth(0, 20, since) : exp(-(since - 20) / 60)
        brightness = (0.5 + 0.8 * act) * (0.8 + 0.2 * sin(t * 2 * .pi / 90 + f.seed)) * (1 + flare * act)

        // Curtains sit far to the north on quiet nights, low over the horizon, and close in overhead in storms.
        let z0 = -(430 - 390 * act)
        let distances = SIMD4<Float>(z0, z0 - 70 - 50 * act, z0 - 190, z0 + 95)
        let strengths = SIMD4<Float>(1, 0.55, 0.25 + 0.35 * act, 0.75 * smooth(0.35, 0.8, act)) * brightness
        let count: Float = act > 0.35 ? 4 : 3
        clock += f.dt * (0.5 + 1.6 * act)
        f.cmd.compute(table, 4096, 4, uniforms: f.uniforms(distances, [act, 20 + 100 * act, clock + f.seed * 50, 0], strengths),
                      textures: [curtains])

        let yaw = 0.12 * sin(t / 520 + f.seed)
        let pitch: Float = 0.08, zoom: Float = 1.5
        let jitter = SIMD2<Float>(FieldKit.halton(Int(f.index) % 32 + 1, 2), FieldKit.halton(Int(f.index) % 32 + 1, 3)) - 0.5
        let blend: Float = f.index == 0 ? 1 : 1 - exp(-f.dt / 0.2)
        let profile = SIMD4<Float>(105 - 12 * act, 0.04 + 0.7 * pow(act, 2.5), 0.12 * act + 0.55 * act * act, 30 + 22 * act)
        let camera = SIMD4<Float>(yaw, pitch, zoom, 0)
        let now = frames[Int(f.index & 1)], past = frames[Int((f.index + 1) & 1)]
        f.cmd.draw(sky, into: now, uniforms: f.uniforms(camera, [jitter.x, jitter.y, blend, brightness], profile,
                                                           [f.seed, act, count, 0]), textures: [past, curtains])
        let glow = bloom.glow(f.cmd, now)
        f.cmd.draw(finish, into: f.target, uniforms: f.uniforms(camera, [0, 0, 0, brightness], profile, [f.seed, act, 0, 0]),
                   textures: [now, glow])

        if let synth = f.sound { play(synth, act, f) }
    }

    private func smooth(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let t = min(max((x - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// Wind over the lake, a low pad, and a high shimmer of slightly detuned pairs that swells with the aurora.
    private func play(_ synth: Synth, _ act: Float, _ f: Frame) {
        gust += (Float.random(in: 0...1) - gust) * min(f.dt / 6, 1)
        let swell = min(brightness * (0.4 + 0.6 * act), 1.4)
        synth.noise(0.05 + 0.035 * gust, cutoff: 320 + 380 * gust)
        synth.drone(0, 110, 0.035, pan: -0.2)
        synth.drone(1, 164.8, 0.02, pan: 0.2)
        synth.drone(2, 880, 0.006 + 0.02 * swell, pan: -0.5)
        synth.drone(3, 882.4, 0.006 + 0.02 * swell, pan: 0.5)
        synth.drone(4, 1318.5, 0.003 + 0.014 * swell, pan: -0.3)
        synth.drone(5, 1321.6, 0.003 + 0.014 * swell, pan: 0.3)
        synth.drone(6, 1760 * (1 + 0.002 * sin(f.time * 0.7)), 0.008 * swell * act, pan: 0)
    }

    private static let source = """
    constant float EARTH = 6371.0;

    // The curtain table: one row per curtain, 4096 samples from 1600 km west to 1600 km east of the lake, each holding
    // the curtain's distance north (km, negative), its brightness there, and its fine vertical rays.
    // a: the curtains' mean distances   b: activity, fold amplitude (km), clock   c: strengths
    kernel void auroraTable(constant U &u [[buffer(0)]], texture2d<float, access::write> out [[texture(0)]],
                            uint2 g [[thread_position_in_grid]]) {
        float x = (float(g.x) + 0.5) / 4096 * 3200 - 1600;
        int k = int(g.y);
        float t = u.b.z, act = u.b.x;
        float meander = (fbm(float2(x / 450 + k * 7.1, t * 0.004 + k * 3.3), 3) - 0.5) * 2;
        float wobble = noise(float2(x / 110 + k * 1.7, t * 0.02 + k));
        float fold = sin(x / 48 + t * 0.06 + 3 * wobble) * smoothstep(0.2, 0.7, noise(float2(x / 260 + 3 + k, t * 0.006)));
        float z = u.a[k] + meander * (90 + 140 * act) + fold * u.b.y;
        float patch = smoothstep(0.15, 0.75, fbm(float2(x / 300 + k * 4.7 + t * 0.01, t * 0.005 + k), 3));
        float rays = noise(float2(x / 1.7 + t * 0.25, k * 10 + t * 0.02)) * 0.7 + noise(float2(x / 6 - t * 0.1, k * 3 + t * 0.01)) * 0.3;
        rays = mix(1.0, 0.2 + 1.9 * pow(rays, 2.2), 0.45 + 0.4 * act);
        float ends = smoothstep(1600.0, 1100.0, abs(x));
        out.write(float4(z, patch * u.c[k] * ends, rays, 0), g);
    }

    // Light emitted per unit of a sheet at altitude h (km): violet N2+ at the lower border, green oxygen (557.7 nm)
    // above it, red oxygen (630 nm) high up. c: lower border, red, violet, green scale height.
    inline float3 emission(float h, constant U &u) {
        float above = h - u.c.x;
        float base = smoothstep(-3.0, 1.5, above);
        float green = base * (0.65 * exp(-max(above, 0.0) / 9) + 0.35 * exp(-max(above, 0.0) / u.c.w));
        float red = smoothstep(u.c.x + 70, u.c.x + 150, h) * exp(-max(h - u.c.x - 150, 0.0) / 100) * u.c.y;
        float violet = exp(-(above + 2) * (above + 2) / 14) * u.c.z;
        return float3(0.26, 1.0, 0.45) * green + float3(1.0, 0.07, 0.14) * red + float3(0.5, 0.2, 1.0) * violet;
    }

    // Marches the ray between 80 and 420 km altitude (the Earth curving away below it), looks for each place it
    // crosses a curtain, and adds the sheet's light there. A sheet seen edge-on is brighter, as real folds are.
    inline float3 aurora(float3 d, float jitter, constant U &u, texture2d<float> table) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float dy = d.y;
        float c = (1 - dy * dy) / (2 * EARTH);
        float s0 = (-dy + sqrt(dy * dy + 4 * c * 80)) / (2 * c);
        float s1 = min((-dy + sqrt(dy * dy + 4 * c * 420)) / (2 * c), 2600.0);
        const int N = 24;
        float ds = (s1 - s0) / N;
        float3 acc = 0;
        for (int k = 0; k < 4; k++) {
            if (u.d.z < k + 0.5) break;
            float v = (k + 0.5) / 4;
            float sp = s0;
            float4 cp = table.sample(s, float2((sp * d.x + 1600) / 3200, v));
            float gp = sp * d.z - cp.x;
            // Samples sit at a jittered offset that changes every frame, so edge-on folds average out instead of
            // beading; the first bracket still starts at the bottom of the range so the lower border is never missed.
            for (int i = 1; i <= N + 1; i++) {
                float sc = min(s0 + (i - 1 + jitter) * ds, s1);
                float4 cc = table.sample(s, float2((sc * d.x + 1600) / 3200, v));
                float gc = sc * d.z - cc.x;
                if (gp * gc <= 0 && gp != gc) {
                    // One secant step on the bracket, so the crossing (and the sharp lower border) lands exactly.
                    float t = gp / (gp - gc);
                    float sm = mix(sp, sc, t);
                    float gm = sm * d.z - table.sample(s, float2((sm * d.x + 1600) / 3200, v)).x;
                    float lo = gp - gm, hi = gm - gc;
                    t = gp * gm < 0 ? t * gp / (abs(lo) < 1e-5 ? 1e-5 : lo) : t + (1 - t) * gm / (abs(hi) < 1e-5 ? 1e-5 : hi);
                    t = saturate(t);
                    float sx = mix(sp, sc, t);
                    float h = sx * dy + c * sx * sx;
                    float slope = abs(gc - gp) / ds;
                    float4 at = table.sample(s, float2((sx * d.x + 1600) / 3200, v));
                    acc += emission(h, u) * at.y * at.z / max(slope, 0.25);
                }
                sp = sc; cp = cc; gp = gc;
            }
        }
        return acc;
    }

    // Camera looking north (-z) with east to the right. cam: yaw, pitch, zoom.
    inline float3 view(float2 p, float4 cam) {
        float cy = cos(cam.x), sy = sin(cam.x), cp = cos(cam.y), sp = sin(cam.y);
        float3 f = float3(sy * cp, sp, -cy * cp), r = float3(cy, 0, sy);
        return normalize(f * cam.z + p.x * r + p.y * cross(r, f));
    }
    inline float2 toUV(float3 d, float4 cam, float2 res) {
        float cy = cos(cam.x), sy = sin(cam.x), cp = cos(cam.y), sp = sin(cam.y);
        float3 f = float3(sy * cp, sp, -cy * cp), r = float3(cy, 0, sy);
        float2 p = float2(dot(d, r), dot(d, cross(r, f))) * cam.z / max(dot(d, f), 1e-3);
        return float2((p.x * res.y / res.x + 1) * 0.5, (1 - p.y) * 0.5);
    }

    inline float ridged(float x, float seed) {
        float n = 0, a = 0.5;
        for (int i = 0; i < 5; i++) { n += a * (1 - abs(noise(float2(x, seed + i * 7.3)) * 2 - 1)); x *= 2.1; a *= 0.5; }
        return n;
    }

    // Elevations (radians) of the far range, the near hills and the spruce forest along the far shore, at azimuth az.
    inline float3 skyline(float az, float seed) {
        float far = 0.022 + 0.17 * pow(ridged(az * 2.2 + seed, 1.0), 2.6);
        float near = 0.008 + 0.024 * fbm(float2(az * 3.5 + seed * 3, 5.5), 4);
        // Spruce: a ragged band of narrow spires.
        float spires = abs(fract(az * 260 + noise(float2(az * 90, 2.0)) * 2) - 0.5) * 2;
        float forest = 0.003 + 0.003 * noise(float2(az * 40, 7.0)) + 0.005 * (1 - spires) * noise(float2(az * 150, 3.0));
        return float3(far, near, forest);
    }

    // Everything above the water: sky, stars, aurora, mountains and trees. Accumulated over frames with sub-pixel
    // jitter, which smooths edges and the march. b: jitter, blend, brightness   d: seed, activity.
    fragment float4 auroraSky(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> past [[texture(0)]],
                              texture2d<float> table [[texture(1)]]) {
        float2 p = centered(in.uv + u.b.xy / u.res, u.res);
        float3 d = view(p, u.a);
        float3 lit = float3(0.26, 1.0, 0.45) * u.b.w;
        float e = d.y;
        float az = atan2(d.x, -d.z);
        float3 line = skyline(az, u.d.x);
        float3 col;
        if (e < line.z || e < 0) {
            col = float3(0.0003, 0.0004, 0.0005) + lit * 0.0008;
        } else if (e < line.y) {
            col = float3(0.004, 0.006, 0.009) + lit * 0.005;
        } else if (e < line.x) {
            // The far range, hazy with distance and lit by the aurora, with snow lying in streaks down the gullies.
            float below = line.x - e;
            float gullies = smoothstep(0.5, 0.85, noise(float2(az * 70, e * 9)) * 0.7 + noise(float2(az * 160, e * 20)) * 0.3);
            float snow = gullies * smoothstep(0.07, 0.01, below) + smoothstep(0.012, 0.0, below) * 0.6;
            col = float3(0.011, 0.015, 0.023) * (0.7 + 0.5 * smoothstep(0.08, 0.0, below)) + lit * (0.008 + 0.035 * snow);
        } else {
            // Sky: deep blue overhead, a little lighter toward the horizon, tinted by the aurora's own light.
            float3 base = mix(float3(0.006, 0.009, 0.015), float3(0.0018, 0.0026, 0.0062), smoothstep(0.0, 0.45, e));
            col = base + lit * 0.012 * exp(-e * 6);
            // The stars turn about the celestial pole, 68.5 degrees up in the north, once a sidereal day.
            float3 pole = float3(0, 0.9304, -0.3665);
            float th = u.time * 7.2921e-5 + u.d.x;
            float3 q = d * cos(th) + cross(pole, d) * sin(th) + pole * dot(pole, d) * (1 - cos(th));
            float fp = 2 / (u.a.z * u.res.y);
            float3 stars = starLayer(q, 420, 0.05, fp, 0.35, 2.3, 0.0022) + starLayer(q, 110, 0.035, fp, 3.0, 5.7, 0.0022);
            float twinkle = 1 + 0.2 * sin(u.time * 1.3 + hash12(floor(q.xy * 400)) * 40);
            col += stars * twinkle * smoothstep(0.0, 0.18, e);
            float jit = 0.02 + 0.96 * hash12(in.pos.xy + float2(u.frame % 64, u.frame % 61) * 17.3);
            col += aurora(d, jit, u, table) * 0.7;
        }
        float3 prev = past.read(uint2(in.pos.xy)).rgb;
        return float4(mix(prev, col, u.b.z), 1);
    }

    // Slope of the lake surface at q (metres): a slow swell and wind ripples stretched across the view. Each octave
    // fades out once a pixel covers more than about a wavelength of water, so distant water doesn't shimmer.
    inline float2 waves(float2 q, float t, float footprint) {
        float2 g = 0;
        float wavelength = 2.4, a = 0.012;
        for (int i = 0; i < 4; i++) {
            float k = TAU / wavelength;
            float2 p = float2(q.x * 0.35, q.y) * k + float2(t * 0.3, t * 0.9) * (1 + i * 0.6) + i * 17.1;
            float e = 0.25;
            float2 slope = float2(noise(p + float2(e, 0)) - noise(p - float2(e, 0)), noise(p + float2(0, e)) - noise(p - float2(0, e))) / (2 * e);
            g += slope * float2(0.35, 1) * a * smoothstep(1.2, 0.3, footprint / wavelength);
            wavelength *= 0.45;
            a *= 0.85;
        }
        return g;
    }

    // The sky at full size, and the lake below it: the sky reflected off rippled water, by Fresnel's law, so the
    // reflection is bright near the far shore and the water at your feet is dark.
    fragment float4 auroraFinish(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> sky [[texture(0)]],
                                 texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 d = view(centered(in.uv, u.res), u.a);
        float3 c;
        if (d.y >= 0) {
            c = sky.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.35;
        } else {
            float dist = 3 / -d.y;
            float footprint = dist * 2 / (u.a.z * u.res.y) / -d.y;
            float2 g = waves(d.xz * dist, u.time, footprint);
            float3 n = normalize(float3(-g.x, 1, -g.y));
            float3 r = reflect(d, n);
            r.y = max(r.y, 0.0006);
            float2 uv = toUV(r, u.a, u.res);
            float3 seen = sky.sample(s, uv).rgb + glow.sample(s, uv).rgb * 0.35;
            float fresnel = 0.02 + 0.98 * pow(1 - saturate(dot(-d, n)), 5.0);
            c = seen * fresnel + float3(0.0005, 0.0008, 0.0011);
        }
        float2 v = in.uv - 0.5;
        c *= 1 - 0.3 * dot(v, v);
        return float4(film(c), 1);
    }
    """
}
