import Foundation
import Metal
import simd

/// Vibration modes of a square and a circular plate, for the Chladni scene and its tests.
enum PlateMode: Equatable {
    /// cos(nπx)cos(mπy) ± cos(mπx)cos(nπy) on the unit square.
    case square(Int, Int, Bool)
    /// J_m(k r) cos(mθ) on a disc of radius ½, with k at the n-th turning point of J_m so the rim moves freely.
    case circle(Int, Int)

    /// Bessel function of the first kind by its integral, J_m(x) = (1/π) ∫₀^π cos(mτ − x sin τ) dτ.
    static func bessel(_ m: Int, _ x: Double) -> Double {
        let steps = 400
        var sum = 0.0
        for i in 0...steps {
            let t = Double.pi * Double(i) / Double(steps)
            let w = i == 0 || i == steps ? 0.5 : 1.0
            sum += w * cos(Double(m) * t - x * sin(t))
        }
        return sum / Double(steps)
    }

    /// The n-th positive zero of J_m′ (n counts from 1), found by scanning and bisection.
    static func turningPoint(_ m: Int, _ n: Int) -> Double {
        func slope(_ x: Double) -> Double { (bessel(m, x + 1e-4) - bessel(m, x - 1e-4)) / 2e-4 }
        var found = 0
        var x = 0.05
        var previous = slope(x)
        while x < 60 {
            let next = x + 0.05
            let s = slope(next)
            if previous * s < 0 {
                var a = x, b = next
                for _ in 0..<50 {
                    let c = (a + b) / 2
                    if slope(a) * slope(c) <= 0 { b = c } else { a = c }
                }
                found += 1
                if found == n { return (a + b) / 2 }
            }
            previous = s
            x = next
        }
        return x
    }

    /// Wavenumber squared in units where the plate is one unit across, and the frequency it gives: bending waves
    /// in a plate have frequency proportional to k², here set so a 24 cm plate's (1, 2) mode is near 180 hertz.
    var k2: Double {
        switch self {
        case let .square(m, n, _): Double.pi * Double.pi * Double(m * m + n * n)
        case let .circle(m, n): pow(2 * Self.turningPoint(m, n), 2)
        }
    }

    var hertz: Double { k2 * 36 / (Double.pi * Double.pi) }

    var isSquare: Bool { if case .square = self { true } else { false } }
}

/// A Chladni plate filmed in a studio: a brushed brass square or a lathe-turned aluminium disc, 24 cm across, bolted
/// through its centre to a vibration driver and sprinkled with half a million grains of fine white sand. The plate rings in
/// one mode; wherever it moves hard enough to throw a grain (more than one g), grains hop, and the plate's slope nudges
/// each hop downhill in |w|, so the sand gathers on the nodal lines where the plate stands still. The drive steps from
/// mode to mode, so the figure dissolves and re-forms. A softbox lights it from the upper left; the camera looks down at
/// a slight angle through a lens with shallow depth of field, and sways slowly so no highlight stays put.
@MainActor final class Chladni: Scene {
    static let modes: [PlateMode] = [
        .square(1, 3, false), .square(2, 3, true), .square(3, 5, false), .square(1, 4, true), .square(2, 5, true),
        .square(3, 4, false), .circle(2, 2), .circle(3, 2), .circle(1, 3), .circle(4, 2),
        .square(4, 5, true), .square(1, 6, false), .square(3, 7, true), .square(2, 7, false), .square(5, 6, false),
        .square(4, 7, true), .circle(6, 2), .circle(2, 4), .circle(5, 3), .square(3, 8, false),
    ]

    /// Per mode: its frequency, its wavenumber and the normaliser that scales its peak displacement to one,
    /// worked out once (finding Bessel turning points takes a moment).
    static let tuning: [(hertz: Double, k: Double, scale: Double)] = modes.map { mode in
        switch mode {
        case .square: return (mode.hertz, 0, 1)
        case let .circle(m, n):
            let k = 2 * PlateMode.turningPoint(m, n)
            let peak = stride(from: 0.0, through: k / 2, by: k / 800).map { abs(PlateMode.bessel(m, $0)) }.max() ?? 1
            return (k * k * 36 / (Double.pi * Double.pi), k, 1 / max(peak, 1e-3))
        }
    }

    static let grains = 1 << 19
    static let side = 512
    /// The sand is counted into a map of the plate this many texels across, about one grain per texel.
    static let sandSide = 1024
    static let hold: Float = 42
    static let glide: Float = 7

    private let fieldKernel: MTLComputePipelineState
    private let hop: MTLComputePipelineState
    private let scatter: MTLComputePipelineState
    private let splat: MTLComputePipelineState
    private let heights: MTLRenderPipelineState
    private let shade: MTLRenderPipelineState
    private let focus: MTLRenderPipelineState
    private let flying: MTLRenderPipelineState
    private let lens: Lens
    private let sand: MTLBuffer
    private let field: MTLTexture
    private let counts: MTLBuffer
    private let sandMap: MTLTexture
    private var light: MTLTexture?
    private var sharp: MTLTexture?
    private var modeIndex: Int
    private var previousIndex: Int
    private var since: Float = 0
    private var started = false

    var scale: Float { 0.7 }
    var mode: PlateMode { Self.modes[modeIndex] }
    var previous: PlateMode { Self.modes[previousIndex] }

    var facts: String? {
        let hz = Int((Self.tuning[modeIndex].hertz / 10).rounded() * 10).formatted()
        let settling = since < Self.glide ? "stepping to " : ""
        return switch mode {
        case let .square(m, n, _): "Square brass plate, \(settling)mode (\(m), \(n)) at about \(hz) hertz; sand gathers on the still lines"
        case let .circle(m, n):
            "Round aluminium plate, \(settling)mode with \(m) nodal diameter\(m == 1 ? "" : "s") and \(n - 1) nodal circle\(n == 2 ? "" : "s"), at about \(hz) hertz"
        }
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(LabShaders.common + Self.source)
        fieldKernel = try gpu.kernel(library, "plateField")
        hop = try gpu.kernel(library, "plateHop")
        scatter = try gpu.kernel(library, "plateScatter")
        splat = try gpu.kernel(library, "plateSplat")
        heights = try gpu.pipeline(library, "plateHeights")
        shade = try gpu.pipeline(library, "plateShade")
        focus = try gpu.pipeline(library, "plateFocus")
        flying = try gpu.pipeline(library, "sprite", vertex: "plateFlying", blend: .add)
        lens = try Lens(gpu: gpu)
        modeIndex = (Snapshot.variant ?? Int.random(in: 0..<Self.modes.count)) % Self.modes.count
        previousIndex = modeIndex
        // Each grain: (x, y, height, vertical speed), (horizontal velocity, tint, glint).
        var g: [SIMD4<Float>] = []
        g.reserveCapacity(Self.grains * 2)
        var rng = LabRandom()
        let round = !Self.modes[modeIndex].isSquare
        for _ in 0..<Self.grains {
            var p = SIMD2<Float>(Float(rng.uniform(-0.49...0.49)), Float(rng.uniform(-0.49...0.49)))
            while round && simd_length(p) > 0.49 { p = SIMD2(Float(rng.uniform(-0.49...0.49)), Float(rng.uniform(-0.49...0.49))) }
            g.append(SIMD4(p.x, p.y, 0, 0))
            g.append(SIMD4(0, 0, Float(rng.uniform()), Float(rng.uniform())))
        }
        sand = gpu.buffer(g)
        field = gpu.texture(Self.side, Self.side, .rgba16Float)
        counts = gpu.buffer(Self.sandSide * Self.sandSide * 4)
        sandMap = gpu.texture(Self.sandSide, Self.sandSide, .rgba16Float)
    }

    /// Writes the displacement of the old mode (r) and the new one (g), each scaled to peak at one.
    private func bake(_ f: Frame) {
        let u = f.uniforms(Self.describe(previousIndex), Self.describe(modeIndex))
        f.cmd.compute(fieldKernel, Self.side, Self.side, uniforms: u, textures: [field])
    }

    /// (kind, m, n or k, sign or normaliser) for the kernel.
    private static func describe(_ i: Int) -> SIMD4<Float> {
        switch modes[i] {
        case let .square(a, b, plus): SIMD4(0, Float(a), Float(b), plus ? 1 : -1)
        case let .circle(a, _): SIMD4(1, Float(a), Float(tuning[i].k), Float(tuning[i].scale))
        }
    }

    func draw(_ f: Frame) {
        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            sharp = f.gpu.texture(f.target.width, f.target.height)
        }
        since += f.dt
        if !started {
            started = true
            bake(f)
            since = Self.glide
            // Let the first figure mostly form before the first frame is shown.
            var warm = hopUniforms(f, dt: 1.0 / 30)
            for i in 0..<240 {
                warm.frame = UInt32(i) &+ 7919
                f.cmd.compute(hop, Self.grains, uniforms: warm, textures: [field], buffers: [sand, counts])
                if i % 4 == 3 { count(f) }
            }
        } else if since > Self.hold {
            previousIndex = modeIndex
            modeIndex = (modeIndex + 1) % Self.modes.count
            since = 0
            bake(f)
        }
        // A new plate shape means a fresh plate: swap it while the studio lights are down and sprinkle the sand again.
        let swap = previous.isSquare != mode.isSquare
        if swap, since >= Self.glide / 2, since - f.dt < Self.glide / 2 {
            f.cmd.compute(scatter, Self.grains, uniforms: hopUniforms(f, dt: f.dt), buffers: [sand])
        }
        let dim: Float = swap ? min(abs(since / Self.glide - 0.5) * 2, 1) : 1
        let hopU = hopUniforms(f, dt: f.dt)
        f.cmd.compute(hop, Self.grains, uniforms: hopU, textures: [field], buffers: [sand, counts])

        // The camera sways slowly round the plate and tilts a little, as if on a motion-control head.
        let t = f.time + f.seed * 41
        let azimuth = 0.14 * sin(t * 0.0041), tilt = 0.42 + 0.05 * sin(t * 0.0029 + 1)
        let distance: Float = 640
        let eye = SIMD3<Float>(sin(tilt) * sin(azimuth), cos(tilt), -sin(tilt) * cos(azimuth)) * distance
        let target = SIMD3<Float>(0, 0, -6)
        let blur = 60 * f.size.y / 1000
        let round: Float = (since < Self.glide / 2 ? previous : mode).isSquare ? 0 : 1
        var u = f.uniforms(SIMD4(eye, 3.9), SIMD4(target, blur), SIMD4(simd_length(target - eye), round, 0, 0),
                           SIMD4(14 * f.size.y / 1000, 0, 0, 0))
        count(f)
        f.cmd.draw(heights, into: sandMap, uniforms: f.uniforms(res: SIMD2(Float(Self.sandSide), Float(Self.sandSide))), buffers: [counts])
        f.cmd.draw(shade, into: light!, uniforms: u, textures: [sandMap])
        f.cmd.draw(focus, into: sharp!, uniforms: u, textures: [light!])
        u.pad = SIMD2(Float(f.size.y), 0)
        f.cmd.draw(flying, into: sharp!, clear: false, uniforms: u, buffers: [sand], vertices: Self.grains, primitive: .point)
        let fade = dim * dim * (3 - 2 * dim)
        lens.develop(f, sharp!, Lens.Look(exposure: 0.85 * fade, white: [1.0, 0.99, 0.97], bloom: 0.04, halation: 0.03,
                                          vignette: 0.4, aberration: 0.8, grain: 0.45))

        if let synth = f.sound { play(synth, f) }
    }

    /// Counts the grains in each texel of the sand map, for the height of the sand and for the next step's sliding.
    private func count(_ f: Frame) {
        if let blit = f.cmd.makeBlitCommandEncoder() {
            blit.fill(buffer: counts, range: 0..<counts.length, value: 0)
            blit.endEncoding()
        }
        f.cmd.compute(splat, Self.grains, uniforms: f.uniforms(res: SIMD2(Float(Self.sandSide), Float(Self.sandSide))),
                      buffers: [sand, counts])
    }

    /// a = (old mode amplitude, new mode amplitude, peak acceleration in g, time step), b = (round plate, shake in g).
    /// Mid-step the driver crosses many weak resonances at once; that broadband shake loosens the old figure so the
    /// new one gathers sand from the whole plate.
    private func hopUniforms(_ f: Frame, dt: Float) -> U {
        let t = min(since / Self.glide, 1)
        let oldAmp = max(1 - t * 2.2, 0)
        let newAmp = t < 0.5 ? 0 : pow((t - 0.5) * 2, 2)
        let shape: Float = (since < Self.glide / 2 ? previous : mode).isSquare ? 0 : 1
        let swap = previous.isSquare != mode.isSquare
        let shake: Float = swap ? 0 : 12 * sin(.pi * min(max((t - 0.12) / 0.6, 0), 1))
        return f.uniforms(SIMD4(oldAmp, newAmp, 40, min(dt, 1.0 / 20)), SIMD4(shape, shake, 0, 0), SIMD4(Float(Self.sandSide), 0, 0, 0))
    }

    /// The plate's own tone, its frequency squeezed into a gentle range (square root of the ratio, so intervals halve),
    /// with a fifth and an octave above, swelling with the vibration; sand rattles in proportion to the grains in flight.
    private func play(_ synth: Synth, _ f: Frame) {
        let t = min(since / Self.glide, 1)
        let eased = t * t * (3 - 2 * t)
        let (from, to) = (Self.tuning[previousIndex].hertz, Self.tuning[modeIndex].hertz)
        let hz = Float(exp(log(from) + (log(to) - log(from)) * Double(eased)))
        let pitch = 110 * (hz / 180).squareRoot()
        let swell = t < 0.5 ? max(1 - t * 2.2, 0) : pow((t - 0.5) * 2, 2)
        synth.drone(0, pitch, 0.08 * (0.35 + 0.65 * swell), pan: -0.15)
        synth.drone(1, pitch * 1.5, 0.025 * swell, pan: 0.2)
        synth.drone(2, pitch * 2, 0.03 * swell, pan: 0.1)
        synth.drone(3, pitch * 3.01, 0.008 * swell)
        let g = sand.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        var flying = 0
        for i in stride(from: Int(f.index % 64), to: Self.grains, by: 1024) where g[2 * i].z > 0 { flying += 1 }
        let airborne = Float(flying) / Float(Self.grains / 1024)
        synth.noise(0.012, cutoff: 900, crackle: 30 * airborne, crackleTone: 5200)
    }

    private static let source = """
    // a, b: old and new modes as (kind, m, n or k, sign or normaliser). Square modes live on the unit square, circular
    // ones on a disc of radius ½, both centred at the origin.
    inline float besselJ(int m, float x) {
        float s = 0;
        for (int i = 0; i <= 48; i++) {
            float t = PI * float(i) / 48;
            s += (i == 0 || i == 48 ? 0.5 : 1.0) * cos(float(m) * t - x * sin(t));
        }
        return s / 48;
    }

    inline float mode(float4 d, float2 p) {
        if (d.x < 0.5) {
            float2 q = p + 0.5;
            float m = d.y, n = d.z;
            return 0.5 * (cos(n * PI * q.x) * cos(m * PI * q.y) + d.w * cos(m * PI * q.x) * cos(n * PI * q.y));
        }
        float r = length(p);
        return besselJ(int(d.y), d.z * r) * cos(d.y * atan2(p.y, p.x)) * d.w;
    }

    kernel void plateField(constant U &u [[buffer(0)]], texture2d<float, access::write> out [[texture(0)]],
                           uint2 g [[thread_position_in_grid]]) {
        float2 p = (float2(g) + 0.5) / float2(out.get_width(), out.get_height()) - 0.5;
        out.write(float4(mode(u.a, p), mode(u.b, p), 0, 0), g);
    }

    // One step of every grain. Resting grains are thrown when the plate's peak acceleration there beats gravity;
    // flying grains fall back. a = (old amplitude, new amplitude, peak g, dt), b.x = 1 on the round plate, c.x the
    // sand map's size, for last frame's grain counts.
    kernel void plateHop(constant U &u [[buffer(0)]], texture2d<float> field [[texture(0)]], device float4 *sand [[buffer(1)]],
                         const device uint *counts [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float4 p = sand[2 * i], q = sand[2 * i + 1];
        float dt = u.a.w;
        const float gravity = 1.0;
        if (p.z <= 0) {
            float h = 1.0 / 512;
            float2 c = p.xy + 0.5;
            float2 w = field.sample(s, c, level(0)).rg;
            float v = w.x * u.a.x + w.y * u.a.y;
            float a = abs(v) * u.a.z;
            uint r = pcg(i * 2654435761u + u.frame * 97u);
            float shake = u.b.y;
            if (a > 0.02 || a + shake > 1) {
                float2 dx = field.sample(s, c + float2(h, 0), level(0)).rg - field.sample(s, c - float2(h, 0), level(0)).rg;
                float2 dy = field.sample(s, c + float2(0, h), level(0)).rg - field.sample(s, c - float2(0, h), level(0)).rg;
                float2 slope = float2(dx.x * u.a.x + dx.y * u.a.y, dy.x * u.a.x + dy.y * u.a.y) / (2 * h);
                float2 downhill = -sign(v) * slope / max(length(slope), 1e-4);
                float total = a + shake;
                if (total > 1) {
                    if (rnd(r) < min(dt * 15, 1.0)) {
                        float up = sqrt(min(total - 1, 30.0) / 30) * 0.24 * (0.4 + 0.6 * rnd(r));
                        float ang = rnd(r) * TAU;
                        float mix = a / total;
                        p.z = 1e-4;
                        p.w = up;
                        q.xy = float2(cos(ang), sin(ang)) * up * (1.4 - 1.05 * mix) + downhill * up * 0.22 * mix;
                    }
                } else {
                    // Below one g the grain cannot leave the plate, but the shaking still lets it creep downhill,
                    // until it is within its own small distance of the node: sand piles have width.
                    float far = abs(v) / max(length(slope), 1e-4);
                    // A few grains in a hundred catch on the brushing marks and stay where they land. The rest jostle
                    // as they creep, so they settle in a loose drift rather than on neat contours.
                    float keep = q.z * q.z * 0.006;
                    if (q.w > 0.03) {
                        if (far > keep) p.xy += downhill * min(far - keep, (0.004 + 0.03 * a) * dt);
                        p.xy += (float2(rnd(r), rnd(r)) - 0.5) * 0.03 * min(a * 3, 1.0) * dt;
                    }
                }
            }
            // Sand can't pile steeper than its angle of repose, about 33 degrees: where last frame's grain counts rise
            // faster than that, a grain slides down the side of the pile, so ridges keep a natural width.
            int side = int(u.c.x);
            int2 cell = clamp(int2((p.xy + 0.5) * float(side)), int2(1), int2(side - 2));
            float nl = float(counts[cell.y * side + cell.x - 1] & 0xFFFFu), nr = float(counts[cell.y * side + cell.x + 1] & 0xFFFFu);
            float nd = float(counts[(cell.y - 1) * side + cell.x] & 0xFFFFu), nu = float(counts[(cell.y + 1) * side + cell.x] & 0xFFFFu);
            float2 grad = float2(nr - nl, nu - nd);
            float steep = length(grad);
            if (steep > 1.4 && rnd(r) < 0.5) p.xy -= grad / steep * (0.6 + rnd(r)) / float(side);
        } else {
            p.xy += q.xy * dt;
            p.w -= gravity * dt;
            p.z += p.w * dt;
            if (p.z <= 0) { p.z = 0; p.w = 0; q.xy = 0; }
        }
        // Keep grains on the plate, bouncing off the rim.
        if (u.b.x > 0.5) {
            float r = length(p.xy);
            if (r > 0.495) { p.xy *= 0.495 / r; q.xy = reflect(q.xy, p.xy / r) * 0.5; }
        } else {
            if (abs(p.x) > 0.495) { p.x = clamp(p.x, -0.495, 0.495); q.x = -q.x * 0.5; }
            if (abs(p.y) > 0.495) { p.y = clamp(p.y, -0.495, 0.495); q.y = -q.y * 0.5; }
        }
        sand[2 * i] = p;
        sand[2 * i + 1] = q;
    }

    kernel void plateScatter(constant U &u [[buffer(0)]], device float4 *sand [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        uint r = pcg(i * 747796405u + u.frame);
        float2 p;
        if (u.b.x > 0.5) {
            float a = rnd(r) * TAU, d = sqrt(rnd(r)) * 0.49;
            p = float2(cos(a), sin(a)) * d;
        } else {
            p = float2(rnd(r), rnd(r)) * 0.98 - 0.49;
        }
        sand[2 * i] = float4(p, 0, 0);
        sand[2 * i + 1].xy = 0;
    }

    // The studio. Lengths in millimetres, y up; the plate's top is y = 0, its centre the origin, and plate coordinates
    // (sim x, sim y) are world (x, z) over the plate's size.
    constant float PLATE = 240.0;
    constant float THICK = 2.0;
    constant float TABLE = -150.0;
    // The softbox, up and to the left behind the plate, and a narrow strip light low on the right.
    constant float3 KEY = float3(-0.42, 0.78, 0.46);
    constant float3 STRIP = float3(0.905, 0.302, 0.302);

    // Counts the grains resting in each texel of the sand map (low 16 bits), and the shadows of grains in flight, cast
    // along the softbox's direction (high 16 bits).
    kernel void plateSplat(constant U &u [[buffer(0)]], const device float4 *sand [[buffer(1)]],
                           device atomic_uint *counts [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        float4 p = sand[2 * i];
        bool flying = p.z > 0;
        float2 xy = flying ? p.xy - KEY.xz / KEY.y * p.z : p.xy;
        int2 c = int2((xy + 0.5) * u.res);
        if (any(c < 0) || any(c >= int2(u.res))) return;
        atomic_fetch_add_explicit(&counts[c.y * int(u.res.x) + c.x], flying ? 65536u : 1u, memory_order_relaxed);
    }

    // What a mirror-like surface sees in direction r: a dark studio, the softbox and the strip light. `spread` widens
    // and dims them for rougher metal.
    inline float3 studio(float3 r, float spread) {
        // A dark studio with a pale ceiling that bounces a little light back down.
        float3 c = float3(0.010, 0.010, 0.012) + float3(0.11, 0.105, 0.10) * smoothstep(0.3, 1.0, r.y) * (0.7 + 0.3 * r.z);
        float3 key = normalize(KEY);
        float3 ax = normalize(cross(float3(0, 1, 0), key)), ay = cross(key, ax);
        float z = dot(r, key);
        if (z > 0.2) {
            float2 s = float2(dot(r, ax), dot(r, ay)) / z;
            float2 extent = float2(0.24, 0.34);
            float2 e = smoothstep(extent + spread + 0.02, extent - spread * 0.7 - 0.02, abs(s));
            float centre = 1 - 0.3 * dot(s / extent, s / extent);
            c += float3(1.0, 0.97, 0.93) * 4.5 * e.x * e.y * max(centre, 0.0) / (1 + spread * 10);
        }
        float3 sx = normalize(cross(float3(0, 1, 0), STRIP)), sy = cross(STRIP, sx);
        float zs = dot(r, STRIP);
        if (zs > 0.2) {
            float2 s = float2(dot(r, sx), dot(r, sy)) / zs;
            float2 e = smoothstep(float2(0.05, 0.5) + spread + 0.01, float2(0.05, 0.5) - spread * 0.7 - 0.01, abs(s));
            c += float3(0.92, 0.96, 1.0) * 2.2 * e.x * e.y / (1 + spread * 10);
        }
        return c;
    }

    // Light reaching a matte surface with normal n from the softbox, the strip and the room.
    inline float3 irradiance(float3 n, float shadow) {
        return float3(1.0, 0.97, 0.93) * 2.4 * max(dot(n, normalize(KEY)), 0.0) * shadow
             + float3(0.92, 0.96, 1.0) * 0.35 * max(dot(n, STRIP), 0.0) + 0.12 * (0.6 + 0.4 * n.y);
    }

    inline float grainsAt(const device uint *counts, int2 c, int side) {
        c = clamp(c, int2(0), int2(side - 1));
        return float(counts[c.y * side + c.x] & 0xFFFFu);
    }

    // The sand map: height in millimetres (a grain is about a quarter of a millimetre, and piles slump), the grain count,
    // and the count of flying grains' shadows.
    fragment float4 plateHeights(V in [[stage_in]], constant U &u [[buffer(0)]], const device uint *counts [[buffer(1)]]) {
        int side = int(u.res.x);
        int2 c = int2(in.pos.xy);
        uint raw = counts[c.y * side + c.x];
        float n = float(raw & 0xFFFFu);
        float around = grainsAt(counts, c + int2(1, 0), side) + grainsAt(counts, c - int2(1, 0), side) +
                       grainsAt(counts, c + int2(0, 1), side) + grainsAt(counts, c - int2(0, 1), side);
        return float4(0.22 * (n * 0.5 + around * 0.125), n, float(raw >> 16), 1);
    }

    inline float sandHeight(texture2d<float> h, float2 uv) {
        constexpr sampler s(filter::linear, address::clamp_to_zero);
        return h.sample(s, uv).r;
    }

    inline float plateEdge(float2 p, float round) {
        return round > 0.5 ? length(p) - PLATE / 2 : max(abs(p.x), abs(p.y)) - PLATE / 2;
    }

    // a: eye, zoom; b: target, blur (circle of confusion in pixels per unit of relative defocus); c: focus distance,
    // round plate. Writes linear light, with the distance along the view in alpha for the depth of field.
    fragment float4 plateShade(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> hmap [[texture(0)]]) {
        constexpr sampler ls(filter::linear, address::clamp_to_zero);
        float2 q = centered(in.uv, u.res);
        float3 eye = u.a.xyz;
        float3 fwd = normalize(u.b.xyz - eye);
        float3 rd = ray(q, eye, u.b.xyz, u.a.w);
        float round = u.c.y;
        float3 key = normalize(KEY);
        float3 c;
        float depth;
        float t0 = -eye.y / rd.y;
        float3 p = eye + rd * t0;
        float edge = plateEdge(p.xz, round);
        float tb = (-THICK - eye.y) / rd.y;
        float3 pb = eye + rd * tb;
        if (edge < 0) {
            depth = t0 * dot(rd, fwd);
            float2 uv = p.xz / PLATE + 0.5;
            float2 texel = 1.0 / float2(hmap.get_width(), hmap.get_height());
            float4 m = hmap.sample(ls, uv);
            float h = sandHeight(hmap, uv);
            float mm = PLATE * texel.x;
            float hx = sandHeight(hmap, uv + float2(texel.x, 0)) - sandHeight(hmap, uv - float2(texel.x, 0));
            float hz = sandHeight(hmap, uv + float2(0, texel.y)) - sandHeight(hmap, uv - float2(0, texel.y));
            float3 n = normalize(float3(-hx / (2 * mm), 1, -hz / (2 * mm)));
            // March toward the softbox through the sand's height field for soft shadows, on the sand and the metal.
            float2 toward = normalize(key.xz) * texel;
            float rise = mm * key.y / length(key.xz);
            float occ = 0;
            for (int k = 1; k <= 9; k++) {
                float hk = sandHeight(hmap, uv + toward * float(k) * 1.3);
                occ = max(occ, (hk - h - rise * float(k) * 1.3) / (0.35 + 0.12 * float(k)));
            }
            float flyShadow = exp(-m.b * 0.6);
            float shadow = (1 - saturate(occ)) * flyShadow;
            float cover = 1 - exp(-m.g * 1.3);
            float2 grainCell = floor(uv * float2(hmap.get_width(), hmap.get_height()));
            float tint = hash12(grainCell * 1.71);
            float glint = step(0.985, hash12(grainCell + 17.3)) * saturate(m.g);

            // The metal: brushed brass on the square plate, lathe-turned aluminium on the round one.
            float2 pc = p.xz;
            float r = length(pc);
            float3 tangent = round > 0.5 ? normalize(float3(-pc.y, 0, pc.x) + 1e-4) : float3(1, 0, 0);
            float groove = round > 0.5 ? r : pc.y;
            float streak = (noise(float2(groove * 9.0, 0.5)) - 0.5) * 0.6 + (noise(float2(groove * 37.0, 3.5)) - 0.5) * 0.4;
            float wear = fbm(pc * 0.02 + round * 7.0, 4);
            float3 nm = n;
            // Scratches: fine straight lines in random directions that catch the light differently.
            float2 cell = floor(pc / 26.0);
            float2 hsh = hash22(cell);
            if (hsh.x > 0.72) {
                float ang = hsh.y * PI;
                float2 dir = float2(cos(ang), sin(ang));
                float2 local = pc - (cell + 0.5) * 26.0 - (hash22(cell + 3.1) - 0.5) * 14.0;
                float across = dot(local, float2(-dir.y, dir.x));
                float along = dot(local, dir);
                float len = 4 + 14 * hash12(cell + 9.7);
                float mark = exp(-sq(across / 0.05)) * smoothstep(len, len * 0.3, abs(along)) * (0.4 + 0.6 * hash12(cell + 1.3));
                nm = normalize(nm + float3(-dir.y, 0, dir.x) * mark * 0.12);
            }
            float3 spec = 0;
            float rough = round > 0.5 ? 0.035 : 0.05;
            for (int k = -2; k <= 2; k++) {
                float3 rr = reflect(rd, nm);
                rr = normalize(rr + tangent * (float(k) * 0.045 + streak * 0.05) * (round > 0.5 ? 1.0 : 0.8));
                spec += studio(rr, rough + wear * 0.03);
            }
            spec /= 5;
            float3 f0 = round > 0.5 ? float3(0.86, 0.87, 0.88) : float3(0.91, 0.74, 0.40);
            f0 *= 0.82 + 0.18 * wear;
            float3 patina = round > 0.5 ? float3(0.05, 0.05, 0.05) : float3(0.09, 0.06, 0.03);
            float3 metal = f0 * spec * (0.35 + 0.65 * shadow) + patina * irradiance(float3(0, 1, 0), shadow) * (0.3 + wear);
            // White sand, each grain a little different, sparkling where there's quartz.
            float3 albedo = mix(float3(0.62, 0.58, 0.52), float3(0.92, 0.9, 0.85), tint);
            float around = 0.25 * (sandHeight(hmap, uv + texel * float2(3, 1)) + sandHeight(hmap, uv - texel * float2(3, 1)) +
                                   sandHeight(hmap, uv + texel * float2(-1, 3)) + sandHeight(hmap, uv - texel * float2(-1, 3)));
            float ao = 1 - saturate((around - h) * 0.8);
            float3 grain = albedo * irradiance(n, shadow) * ao + glint * float3(1.0, 0.97, 0.9) * 2.0 * shadow * saturate(dot(n, key));
            c = mix(metal, grain, cover);
            // The bevelled top edge catches a line of light.
            float bevel = smoothstep(-0.9, -0.1, edge);
            if (bevel > 0) {
                float2 outward = round > 0.5 ? normalize(pc) : (abs(pc.x) > abs(pc.y) ? float2(sign(pc.x), 0) : float2(0, sign(pc.y)));
                float3 nb = normalize(float3(outward.x, 1.2, outward.y));
                c = mix(c, f0 * studio(reflect(rd, nb), 0.08) + patina * 0.1, bevel);
            }
            // The drive bolt: a steel nut on a washer, and the shadow they cast.
            float2 sp = pc + key.xz / key.y * 4.0;
            float under = round > 0.5 ? 0 : 0;
            float washer = r - 8.5;
            float2 hp = abs(pc);
            float hex = max(hp.x * 0.866 + hp.y * 0.5, hp.y) - 5.0;
            float shadowNut = smoothstep(1.5, -1.5, length(sp) - 7.5 + under);
            c *= 1 - 0.6 * shadowNut * step(0.0, washer);
            if (washer < 0) {
                float3 steel = float3(0.62, 0.63, 0.65);
                float3 nw = hex < 0 ? normalize(float3(0, 1, 0) + float3(pc.x, 0, pc.y) * smoothstep(-1.2, 0.0, hex) * 0.18)
                                    : normalize(float3(pc.x, 0, pc.y) * smoothstep(-1.5, 0.0, washer) * 0.12 + float3(0, 1, 0));
                float swirl = hex < 0 ? noise(float2(atan2(pc.y, pc.x) * 6, r * 4)) : noise(float2(r * 30, 0.5));
                float3 rr = normalize(reflect(rd, nw) + tangent * (swirl - 0.5) * 0.15);
                c = steel * studio(rr, 0.08) + 0.04 * irradiance(nw, 1.0);
                if (hex < 0 && r < 3.2) c = steel * 0.25 * studio(reflect(rd, float3(0, 1, 0)), 0.2) + 0.01;
            }
        } else if (plateEdge(pb.xz, round) < 0) {
            // The plate's edge face.
            depth = tb * dot(rd, fwd);
            float2 pc = pb.xz;
            float2 outward = round > 0.5 ? normalize(pc) : (abs(pc.x) > abs(pc.y) ? float2(sign(pc.x), 0) : float2(0, sign(pc.y)));
            float3 n = float3(outward.x, 0, outward.y);
            float3 f0 = round > 0.5 ? float3(0.86, 0.87, 0.88) : float3(0.91, 0.74, 0.40);
            c = f0 * studio(normalize(reflect(rd, n) + float3(0, 0.2, 0)), 0.15) * 0.8;
        } else {
            // The table far below: dark cloth, softly lit, with the plate's blurred shadow.
            float tt = (TABLE - eye.y) / rd.y;
            float3 tp = eye + rd * tt;
            depth = tt * dot(rd, fwd);
            float2 up = tp.xz + key.xz / key.y * (-TABLE);
            float e = plateEdge(up, round);
            float shadow = smoothstep(-25.0, 30.0, e);
            float weave = 0.85 + 0.15 * noise(tp.xz * 0.8);
            float pool = exp(-dot(tp.xz - float2(-90, 60), tp.xz - float2(-90, 60)) / 160000.0);
            c = float3(0.05, 0.047, 0.045) * weave * (irradiance(float3(0, 1, 0), shadow) * (0.4 + 0.8 * pool));
            // The driver's dark body under the plate.
            float body = length(tp.xz) - 45;
            c *= mix(0.15, 1.0, smoothstep(-2.0, 2.0, body));
        }
        return float4(c, depth);
    }

    inline float circleOfConfusion(float depth, constant U &u) { return u.b.w * abs(1 - u.c.x / max(depth, 1.0)); }

    // Depth of field by gathering: each pixel takes the light of neighbours whose blur reaches it, but a blurred
    // background never spills over a sharper foreground. d.x is the largest blur radius in pixels.
    fragment float4 plateFocus(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float4 c0 = light.sample(s, in.uv);
        float coc0 = min(circleOfConfusion(c0.a, u), u.d.x);
        if (coc0 < 0.6) return c0;
        float3 acc = c0.rgb;
        float total = 1;
        for (int i = 0; i < 14; i++) {
            float rr = sqrt((float(i) + 0.5) / 14) * coc0;
            float a = float(i) * 2.39996323;
            float4 t = light.sample(s, in.uv + float2(cos(a), sin(a)) * rr / u.res);
            float cocT = min(circleOfConfusion(t.a, u), u.d.x);
            float reach = t.a > c0.a ? min(cocT, coc0) : cocT;
            float w = saturate(reach - rr + 1);
            acc += t.rgb * w;
            total += w;
        }
        return float4(acc / total, c0.a);
    }

    // Grains in flight, drawn after the depth of field with their own blur. pad.x is the target height in pixels.
    vertex P plateFlying(uint id [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *sand [[buffer(1)]]) {
        float4 p = sand[2 * id], q = sand[2 * id + 1];
        P o;
        o.pos = float4(2, 2, 0, 1); o.size = 0; o.color = 0;
        if (p.z <= 0) return o;
        float3 w = float3(p.x * PLATE, p.z * PLATE + 0.15, p.y * PLATE);
        float4 clip = project(w, u.a.xyz, u.b.xyz, u.a.w, u.res.x / u.res.y);
        o.pos = clip;
        float px = pixelsPer(clip.w, u.a.w, u.res.y);
        float coc = min(circleOfConfusion(clip.w, u), u.d.x);
        float sharp = max(0.3 * px, 1.2);
        o.size = sqrt(sharp * sharp + coc * coc);
        float3 albedo = mix(float3(0.62, 0.58, 0.52), float3(0.92, 0.9, 0.85), q.z);
        float3 lit = albedo * (float3(1.0, 0.97, 0.93) * 2.4 * 0.7 + 0.25);
        o.color = float4(lit * sq(1.2 / o.size) * 0.9, 0);
        return o;
    }
    """
}
