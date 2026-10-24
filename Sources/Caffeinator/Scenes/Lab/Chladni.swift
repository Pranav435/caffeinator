import Foundation
import Metal

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
    /// in a plate have frequency proportional to k², here set so a 24 cm steel plate's (1, 2) mode is near 180 hertz.
    var k2: Double {
        switch self {
        case let .square(m, n, _): Double.pi * Double.pi * Double(m * m + n * n)
        case let .circle(m, n): pow(2 * Self.turningPoint(m, n), 2)
        }
    }

    var hertz: Double { k2 * 36 / (Double.pi * Double.pi) }

    var isSquare: Bool { if case .square = self { true } else { false } }
}

/// A Chladni plate: a dark metal plate seen from above, sprinkled with a quarter of a million grains of sand.
/// The plate rings in one mode; wherever it moves hard enough to throw a grain (more than one g), grains hop, and the
/// plate's slope nudges each hop downhill in |w|, so the sand gathers on the nodal lines where the plate stands still.
/// The drive frequency glides from mode to mode, so the figure dissolves and re-forms.
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

    static let grains = 1 << 18
    static let side = 512
    static let hold: Float = 42
    static let glide: Float = 7

    private let fieldKernel: MTLComputePipelineState
    private let hop: MTLComputePipelineState
    private let scatter: MTLComputePipelineState
    private let splat: MTLRenderPipelineState
    private let shade: MTLRenderPipelineState
    private let tone: MTLRenderPipelineState
    private let bloom: Bloom
    private let sand: MTLBuffer
    private let field: MTLTexture
    private var grainLight: MTLTexture?
    private var light: MTLTexture?
    private var modeIndex: Int
    private var previousIndex: Int
    private var since: Float = 0
    private var started = false

    var scale: Float { 1 }
    var mode: PlateMode { Self.modes[modeIndex] }
    var previous: PlateMode { Self.modes[previousIndex] }

    var facts: String? {
        let hz = Int((Self.tuning[modeIndex].hertz / 10).rounded() * 10).formatted()
        let settling = since < Self.glide ? "gliding to " : ""
        return switch mode {
        case let .square(m, n, _): "Square plate, \(settling)mode (\(m), \(n)) at about \(hz) hertz; sand gathers on the still lines"
        case let .circle(m, n):
            "Round plate, \(settling)mode with \(m) nodal diameter\(m == 1 ? "" : "s") and \(n - 1) nodal circle\(n == 2 ? "" : "s"), at about \(hz) hertz"
        }
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        fieldKernel = try gpu.kernel(library, "plateField")
        hop = try gpu.kernel(library, "plateHop")
        scatter = try gpu.kernel(library, "plateScatter")
        splat = try gpu.pipeline(library, "sprite", vertex: "plateGrain", blend: .add)
        shade = try gpu.pipeline(library, "plateShade")
        tone = try gpu.pipeline(library, "plateTone")
        bloom = try Bloom(gpu: gpu)
        modeIndex = (Snapshot.variant ?? Int.random(in: 0..<Self.modes.count)) % Self.modes.count
        previousIndex = modeIndex
        // Each grain: (x, y, height, vertical speed), (horizontal velocity, tint, glint).
        var g: [SIMD4<Float>] = []
        g.reserveCapacity(Self.grains * 2)
        for _ in 0..<Self.grains {
            g.append(SIMD4(.random(in: -0.5...0.5), .random(in: -0.5...0.5), 0, 0))
            g.append(SIMD4(0, 0, .random(in: 0...1), .random(in: 0...1)))
        }
        sand = gpu.buffer(g)
        field = gpu.texture(Self.side, Self.side, .rgba16Float)
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
            grainLight = f.gpu.texture(f.target.width, f.target.height)
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
                f.cmd.compute(hop, Self.grains, uniforms: warm, textures: [field], buffers: [sand])
            }
        } else if since > Self.hold {
            previousIndex = modeIndex
            modeIndex = (modeIndex + 1) % Self.modes.count
            since = 0
            bake(f)
        }
        // A new plate shape means a fresh plate: swap it while the screen is dark and sprinkle the sand again.
        let swap = previous.isSquare != mode.isSquare
        if swap, since >= Self.glide / 2, since - f.dt < Self.glide / 2 {
            f.cmd.compute(scatter, Self.grains, uniforms: hopUniforms(f, dt: f.dt), buffers: [sand])
        }
        let dim: Float = swap ? min(abs(since / Self.glide - 0.5) * 2, 1) : 1
        f.cmd.compute(hop, Self.grains, uniforms: hopUniforms(f, dt: f.dt), textures: [field], buffers: [sand])

        let px = Float(f.target.height) * 0.88
        let round: Float = (shapeMix > 0.5 ? mode : previous).isSquare ? 0 : 1
        var u = f.uniforms(SIMD4(px, round, 0, 0), hopUniforms(f, dt: f.dt).a)
        u.c = SIMD4(-0.55, -0.45, 0.7, 0)
        u.d = SIMD4(dim * dim * (3 - 2 * dim), 0, 0, 0)
        f.cmd.draw(splat, into: grainLight!, uniforms: u, buffers: [sand], vertices: Self.grains * 2, primitive: .point)
        f.cmd.draw(shade, into: light!, uniforms: u, textures: [grainLight!, field])
        let glow = bloom.glow(f.cmd, light!)
        f.cmd.draw(tone, into: f.target, uniforms: u, textures: [light!, glow])

        if let synth = f.sound { play(synth, f) }
    }

    /// 0 while the plate is the old shape, 1 once it is the new one; shapes swap while the plate is still.
    private var shapeMix: Float { since < Self.glide / 2 ? 0 : 1 }

    /// a = (old mode amplitude, new mode amplitude, peak acceleration in g, time step), b = (round plate, shake in g).
    /// Mid-glide the sweep crosses many weak resonances at once; that broadband shake loosens the old figure so the
    /// new one gathers sand from the whole plate.
    private func hopUniforms(_ f: Frame, dt: Float) -> U {
        let t = min(since / Self.glide, 1)
        let oldAmp = max(1 - t * 2.2, 0)
        let newAmp = t < 0.5 ? 0 : pow((t - 0.5) * 2, 2)
        let shape: Float = (shapeMix > 0.5 ? mode : previous).isSquare ? 0 : 1
        let swap = previous.isSquare != mode.isSquare
        let shake: Float = swap ? 0 : 12 * sin(.pi * min(max((t - 0.12) / 0.6, 0), 1))
        return f.uniforms(SIMD4(oldAmp, newAmp, 40, min(dt, 1.0 / 20)), SIMD4(shape, shake, 0, 0))
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
        for i in stride(from: Int(f.index % 64), to: Self.grains, by: 512) where g[2 * i].z > 0 { flying += 1 }
        let airborne = Float(flying) / Float(Self.grains / 512)
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
    // flying grains fall back. a = (old amplitude, new amplitude, peak g, dt), b.x = 1 on the round plate.
    kernel void plateHop(constant U &u [[buffer(0)]], texture2d<float> field [[texture(0)]], device float4 *sand [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
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
                    float keep = q.z * q.z * 0.0055;
                    if (far > keep) p.xy += downhill * min(far - keep, (0.004 + 0.03 * a) * dt);
                }
            }
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

    // Grains as tiny points: even vertices are the grain (r: sand, g: glints), odd ones its shadow (a), cast away
    // from the lamp by its height. a.x is the plate's size in pixels; c is the direction to the lamp.
    vertex P plateGrain(uint id [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *sand [[buffer(1)]]) {
        uint i = id >> 1;
        float4 p = sand[2 * i], q = sand[2 * i + 1];
        bool shadow = id & 1;
        float2 xy = p.xy;
        if (shadow) xy -= u.c.xy / u.c.z * p.z;
        float2 px = u.res * 0.5 + xy * u.a.x;
        P o;
        o.pos = float4(px / u.res * float2(2, -2) + float2(-1, 1), 0, 1);
        float lift = p.z * 40;
        // Grains at rest are single pixels and cast no shadow worth drawing; only flying ones grow and cast one.
        o.size = p.z > 0 ? (1.6 + lift) * u.res.y / 1000 : 1;
        if (shadow) {
            if (p.z <= 0) o.pos = float4(-2, -2, 0, 1);
            o.color = float4(0, 0, 0, 0.5 / (1 + lift));
        } else {
            float glint = q.w > 0.985 ? pow(0.5 + 0.5 * sin(q.w * 9000 + (p.x + p.y) * 300), 8) : 0;
            o.color = float4(0.55 + 0.45 * q.z, glint * 3, 0, 0) * (1 + lift * 0.4) * (p.z > 0 ? 0.6 : 1.0);
        }
        return o;
    }

    inline float plateMask(float2 p, float round, float px) {
        float d = round > 0.5 ? length(p) - 0.5 : max(abs(p.x), abs(p.y)) - 0.5;
        return smoothstep(1.5 / px, -1.5 / px, d);
    }

    // Warm sand on dark brushed steel under a lamp at the upper left. a.x: plate size in pixels, a.y: plate shape.
    fragment float4 plateShade(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> grains [[texture(0)]],
                               texture2d<float> field [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float px = u.a.x;
        float2 p = (in.uv - 0.5) * u.res / px;
        float2 texel = 1 / u.res;
        float3 lamp = normalize(float3(-0.55, 0.45, 0.7));
        // Sand height from grain density, softened a touch.
        float4 g = grains.sample(s, in.uv);
        float d = g.r * 0.5;
        float dl = grains.sample(s, in.uv - float2(texel.x, 0)).r, dr = grains.sample(s, in.uv + float2(texel.x, 0)).r;
        float du = grains.sample(s, in.uv - float2(0, texel.y)).r, dd = grains.sample(s, in.uv + float2(0, texel.y)).r;
        d += (dl + dr + du + dd) * 0.125;
        float cover = 1 - exp(-d * 0.9);
        float3 n = normalize(float3((dl - dr) * 0.35, (dd - du) * 0.35, 1));
        float toward = grains.sample(s, in.uv + float2(-0.55, -0.45) * texel * 3).r;
        float self = clamp(1 - (toward - d) * 0.25, 0.55, 1.0);
        float diffuse = max(dot(n, lamp), 0.0);
        float3 grain = float3(0.78, 0.56, 0.32) * (0.18 + 0.95 * diffuse) * self;

        // Brushed steel: fine streaks along x, a broad soft reflection of the lamp, darker toward the far corner.
        float brush = noise(float2(p.x * 900, p.y * 6)) * 0.5 + noise(float2(p.x * 300, p.y * 2.5)) * 0.5;
        float sheen = exp(-dot(p - float2(-0.22, -0.2), p - float2(-0.22, -0.2)) * 5);
        float3 metal = float3(0.012, 0.013, 0.016) * (0.85 + 0.3 * brush) + float3(0.04, 0.042, 0.048) * sheen * (0.75 + 0.25 * brush);
        float shade = exp(-g.a * 1.2);
        // Ridges of sand cast a soft shadow on the steel on the side away from the lamp.
        float ridge = grains.sample(s, in.uv + float2(-0.55, -0.45) * texel * 5).r + grains.sample(s, in.uv + float2(-0.55, -0.45) * texel * 2.5).r;
        metal *= 1 - 0.55 * (1 - exp(-ridge * 0.35));
        float3 c = mix(metal, grain, cover) * shade + float3(1.0, 0.92, 0.78) * g.g * cover;

        float inside = plateMask(p, u.a.y, px);
        // The plate's edge catches the light on the lamp side; beyond it, a dark table with the plate's soft shadow.
        float edge = u.a.y > 0.5 ? length(p) - 0.5 : max(abs(p.x), abs(p.y)) - 0.5;
        float rim = exp(-abs(edge) * px / 1.5) * (0.5 + 0.5 * dot(normalize(p + 1e-5), float2(-0.7, -0.7)));
        float drop = u.a.y > 0.5 ? length(p - float2(0.012, 0.012)) - 0.5 : max(abs(p.x - 0.012), abs(p.y - 0.012)) - 0.5;
        float3 table = float3(0.004, 0.0042, 0.005) * (1 - 0.6 * smoothstep(0.03, -0.01, drop));
        c = mix(table, c, inside) + float3(0.12, 0.12, 0.13) * max(rim, 0.0) * 0.35;
        // The drive bolt at the centre: a hex nut with bevelled edges catching the lamp, and its shadow.
        float2 hp = abs(p);
        float hex = max(hp.x * 0.866 + hp.y * 0.5, hp.y) - 0.017;
        float under = max(abs(p.x - 0.004) * 0.866 + abs(p.y - 0.004) * 0.5, abs(p.y - 0.004)) - 0.018;
        c *= 1 - 0.6 * smoothstep(0.008, -0.002, under);
        float bevel = smoothstep(-0.0045, 0.0, hex);
        float facing = dot(normalize(p + 1e-5), float2(-0.7, -0.7));
        float3 face = float3(0.045, 0.046, 0.05) * (1 + 0.6 * dot(p, float2(-1, -1)) / 0.017);
        float3 rimNut = float3(0.03, 0.031, 0.035) + float3(0.16, 0.16, 0.17) * max(facing, 0.0) * 0.6;
        float3 nut = mix(face, rimNut, bevel) * (1 - 0.6 * smoothstep(0.006, 0.0045, length(p)));
        c = mix(c, nut, smoothstep(1.2 / px, -1.2 / px, hex));
        return float4(c, 1);
    }

    fragment float4 plateTone(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                              texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = light.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.25;
        return float4(film(c * 1.3) * u.d.x, 1);
    }
    """
}
