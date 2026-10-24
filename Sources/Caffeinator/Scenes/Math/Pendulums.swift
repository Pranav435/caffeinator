import Foundation
import Metal

/// Two thousand double pendulums released a millionth of a radian apart, integrated in double precision with RK4.
/// They swing as one bright pendulum until chaos amplifies the difference and fans them into a rainbow.
@MainActor final class Pendulums: Scene {
    private static let count = 2048
    /// Trail points recorded per pendulum each frame.
    private static let samples = 4
    /// Spread of the starting angles across the whole bundle, in radians.
    private static let spread = 1e-6
    /// Gravity over arm length: a slow, heavy swing.
    private static let g = 1.8

    private let lines: MTLRenderPipelineState
    private let trailLines: MTLRenderPipelineState
    private let dots: MTLRenderPipelineState
    private let fade: MTLRenderPipelineState
    private let compose: MTLRenderPipelineState
    private let bloom: Bloom
    private var light: MTLTexture?
    private var trails: MTLTexture?
    /// Per frame in flight: bob positions (two float2 per pendulum), then trail points.
    private var arms: [MTLBuffer]
    private var paths: [MTLBuffer]

    private var t1: [Double]
    private var t2: [Double]
    private var w1: [Double]
    private var w2: [Double]
    private var last: [SIMD2<Float>]
    private var side: [Int8]

    private var rng: UInt64
    private var simTime = 0.0
    private var runLength = 90.0
    private var split: Double?
    private var pivot = SIMD2<Float>(0, 0.08)
    private var hue: Float = 0
    private var level: Float = 0
    private var resetting = false
    private var spreadPx: Float = 0
    private var mixed: Float = 0
    private var tokens: Float = 0
    private var chimeCool: Float = 0
    private var startAngles = (0.0, 0.0)

    var scale: Float { 0.6 }

    var facts: String? {
        let n = Self.count.formatted()
        let start = "\(n) double pendulums released from \(Self.degrees(startAngles.0)) and \(Self.degrees(startAngles.1)), their starting angles spread across one millionth of a radian"
        if let split { return start + "; chaos split them apart after \(Int(split.rounded())) seconds" }
        return start + "; still swinging as one after \(Int(simTime)) seconds"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        lines = try gpu.pipeline(library, "glowLine", vertex: "armQuad", format: .rgba32Float, blend: .add)
        trailLines = try gpu.pipeline(library, "glowLine", vertex: "trailQuad", format: .rgba32Float, blend: .add)
        dots = try gpu.pipeline(library, "sprite", vertex: "bobDot", format: .rgba32Float, blend: .add)
        fade = try gpu.pipeline(library, "fadeOut", format: .rgba32Float, blend: .fade)
        compose = try gpu.pipeline(library, "pendulumCompose")
        bloom = try Bloom(gpu: gpu)
        let n = Self.count
        arms = (0..<3).map { _ in gpu.buffer(n * 2 * MemoryLayout<SIMD2<Float>>.stride) }
        paths = (0..<3).map { _ in gpu.buffer(n * (Self.samples + 1) * MemoryLayout<SIMD2<Float>>.stride) }
        t1 = Array(repeating: 0, count: n)
        t2 = t1
        w1 = t1
        w2 = t1
        last = Array(repeating: .zero, count: n)
        side = Array(repeating: 0, count: n)
        rng = UInt64.random(in: 1...UInt64.max)
        if let v = Snapshot.variant { rng = UInt64(v + 1) &* 0x9E37_79B9_7F4A_7C15 }
        reset()
    }

    private func random() -> Double {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return Double(rng >> 11) / Double(1 << 53)
    }

    /// New starting angles high enough above the horizontal to be chaotic, and a new place for the pivot.
    private func reset() {
        let a = (1.75 + random() * 1.05) * (random() < 0.5 ? -1 : 1)
        let b = (1.6 + random() * 1.3) * (random() < 0.5 ? -1 : 1)
        startAngles = (a, b)
        for i in 0..<Self.count {
            let e = Self.spread * Double(i) / Double(Self.count - 1)
            t1[i] = a + e
            t2[i] = b + e
            w1[i] = 0
            w2[i] = 0
            side[i] = 0
        }
        simTime = 0
        split = nil
        mixed = 0
        runLength = 60 + random() * 25
        pivot = SIMD2(Float(random() - 0.5) * 0.3, 0.06 + Float(random()) * 0.08)
        hue = Float(random())
        for i in 0..<Self.count { last[i] = bob(i).1 }
    }

    private func bob(_ i: Int) -> (SIMD2<Float>, SIMD2<Float>) {
        let a = SIMD2(Float(sin(t1[i])), Float(-cos(t1[i])))
        return (a, a + SIMD2(Float(sin(t2[i])), Float(-cos(t2[i]))))
    }

    /// Advances every pendulum by `dt` seconds in `steps` RK4 steps, recording `samples` trail points along the way.
    private func advance(_ dt: Double, into path: UnsafeMutablePointer<SIMD2<Float>>) {
        // RK4 steps of at most about 4.5 ms, a whole number per trail sample.
        let steps = Self.samples * max(1, Int((dt / (0.0045 * Double(Self.samples))).rounded(.up)))
        let h = dt / Double(steps)
        let g = Self.g
        let every = steps / Self.samples
        t1.withUnsafeMutableBufferPointer { t1 in
        t2.withUnsafeMutableBufferPointer { t2 in
        w1.withUnsafeMutableBufferPointer { w1 in
        w2.withUnsafeMutableBufferPointer { w2 in
            // Equal masses and arms, in units where the arm length is one.
            @inline(__always) func accel(_ a: Double, _ b: Double, _ p: Double, _ q: Double) -> (Double, Double) {
                let d = a - b, sd = sin(d), cd = cos(d)
                let den = 3 - (2 * cd * cd - 1)
                let x = (-3 * g * sin(a) - g * sin(a - 2 * b) - 2 * sd * (q * q + p * p * cd)) / den
                let y = 2 * sd * (2 * p * p + 2 * g * cos(a) + q * q * cd) / den
                return (x, y)
            }
            for i in 0..<Self.count {
                var a = t1[i], b = t2[i], p = w1[i], q = w2[i]
                let base = path + i * (Self.samples + 1)
                base[0] = last[i]
                for s in 0..<steps {
                    let (k1p, k1q) = accel(a, b, p, q)
                    let (k2p, k2q) = accel(a + 0.5 * h * p, b + 0.5 * h * q, p + 0.5 * h * k1p, q + 0.5 * h * k1q)
                    let (k3p, k3q) = accel(a + 0.5 * h * (p + 0.5 * h * k1p), b + 0.5 * h * (q + 0.5 * h * k1q),
                                           p + 0.5 * h * k2p, q + 0.5 * h * k2q)
                    let (k4p, k4q) = accel(a + h * (p + 0.5 * h * k2p), b + h * (q + 0.5 * h * k2q), p + h * k3p, q + h * k3q)
                    a += h * (p + h / 6 * (k1p + k2p + k3p))
                    b += h * (q + h / 6 * (k1q + k2q + k3q))
                    p += h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
                    q += h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
                    if (s + 1) % every == 0 {
                        let x1 = sin(a), y1 = -cos(a)
                        base[(s + 1) / every] = SIMD2(Float(x1 + sin(b)), Float(y1 - cos(b)))
                    }
                }
                t1[i] = a; t2[i] = b; w1[i] = p; w2[i] = q
                last[i] = base[Self.samples]
            }
        }}}}
    }

    func draw(_ f: Frame) {
        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height, .rgba32Float)
            trails = f.gpu.texture(f.target.width, f.target.height, .rgba32Float)
        }
        let slot = Int(f.index % 3)
        let path = paths[slot].contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let arm = arms[slot].contents().assumingMemoryBound(to: SIMD2<Float>.self)

        // Fade out near the end of a run, start again with new angles, fade back in.
        if simTime > runLength { resetting = true }
        level = resetting ? max(level - f.dt / 2.5, 0) : min(level + f.dt / 2, 1)
        if resetting, level == 0 {
            reset()
            resetting = false
        }
        let dt = Double(f.dt)
        advance(dt, into: path)
        simTime += dt

        // How far the lower bobs have fanned out, in pixels, sets the exposure: together they share one line's
        // worth of pixels, apart each needs its own light.
        var mean = SIMD2<Float>.zero
        for i in 0..<Self.count {
            let (a, b) = bob(i)
            arm[2 * i] = a
            arm[2 * i + 1] = b
            mean += b
        }
        mean /= Float(Self.count)
        var variance: Float = 0
        for i in stride(from: 0, to: Self.count, by: 8) { let d = arm[2 * i + 1] - mean; variance += (d * d).sum() }
        let rms = sqrt(variance / Float(Self.count / 8))
        // Neighbours in the bundle stay neighbours while it fans out in order; once they're scattered it is fully mixed.
        var gap: Float = 0
        for i in stride(from: 0, to: Self.count - 1, by: 8) {
            let d = arm[2 * i + 3] - arm[2 * i + 1]
            gap += (d * d).sum().squareRoot()
        }
        gap /= Float(Self.count / 8)
        mixed += ((gap > 0.25 ? 1 : 0) - mixed) * min(f.dt / 6, 1)
        if split == nil, rms > 0.03, simTime > 1 { split = simTime }
        let reach = 0.21 * Float(f.target.height)
        spreadPx += (rms * reach - spreadPx) * min(f.dt * 4, 1)
        let exposure = 4 * min(max(spreadPx / 2.5, 1), Float(Self.count) / 40) / Float(Self.count)

        // While the bundle is tight its arms lie on top of each other: draw every few, each standing for its neighbours,
        // with enough of them that the fan never shows gaps as it opens.
        var stride = 1
        while stride < 64, Float(Self.count / (stride * 2)) > spreadPx * 4 + 32 { stride *= 2 }
        let u = f.uniforms([pivot.x, pivot.y, 0.42, exposure], [hue, level, Float(Self.count), Float(Self.samples)],
                           [1.1, 1 - 0.85 * mixed, 2 + 0.4 * mixed, Float(stride)])
        f.cmd.draw(fade, into: trails!, clear: f.index == 0, uniforms: f.uniforms([pow(0.94, f.dt * 30), 0, 0, 0]))
        f.cmd.draw(trailLines, into: trails!, clear: false, uniforms: u, buffers: [paths[slot]], vertices: 6,
                   instances: Self.count * Self.samples)
        f.cmd.draw(lines, into: light!, uniforms: u, buffers: [arms[slot]], vertices: 6, instances: Self.count / stride * 2)
        f.cmd.draw(dots, into: light!, clear: false, uniforms: u, buffers: [arms[slot]], vertices: Self.count * 2,
                   primitive: .point)
        let glow = bloom.glow(f.cmd, light!)
        f.cmd.draw(compose, into: f.target, uniforms: u, textures: [light!, trails!, glow])

        if let synth = f.sound { play(synth, f, arm) }
    }

    /// Each time a lower bob swings through the bottom it can sound: together, the bundle rings one soft chime per
    /// swing; fanned out, the crossings become a scatter of plucks, pitched by the pendulum's colour.
    private func play(_ synth: Synth, _ f: Frame, _ arm: UnsafeMutablePointer<SIMD2<Float>>) {
        var crossed: [Int] = []
        for i in 0..<Self.count {
            let b = arm[2 * i + 1]
            let s: Int8 = b.y > 0 ? 0 : (b.x < 0 ? -1 : 1)
            if s != 0, side[i] != 0, s != side[i] { crossed.append(i) }
            if s != 0 { side[i] = s }
        }
        let together = max(0, 1 - spreadPx / 40)
        let root: Float = 146.8
        let scale: [Float] = [1, 9.0 / 8, 5.0 / 4, 3.0 / 2, 5.0 / 3, 2, 9.0 / 4, 5.0 / 2, 3, 10.0 / 3]
        chimeCool -= f.dt
        tokens = min(tokens + f.dt * 5, 3)
        if crossed.count > Self.count / 3, chimeCool < 0 {
            let right = arm[2 * crossed[0] + 1].x > 0
            synth.play(.bell, root * (right ? 4 : 3), amp: 0.07 * level, pan: right ? 0.3 : -0.3, length: 3.5)
            chimeCool = 0.5
        } else if !crossed.isEmpty, tokens >= 1, together < 0.8 {
            tokens -= 1
            let i = crossed[Int(random() * Double(crossed.count))]
            let note = scale[min(Int(Float(i) / Float(Self.count) * Float(scale.count)), scale.count - 1)]
            let amp = (0.025 + 0.035 * min(Float(crossed.count) / 40, 1)) * level
            synth.play(.pluck, root * 2 * note, amp: amp, pan: max(-0.8, min(0.8, arm[2 * i].x)), length: 1.4)
        }
        synth.drone(0, root / 2, 0.08 * level, pan: -0.2)
        synth.drone(1, root * 0.75, 0.05 * level, pan: 0.2)
        let shimmer = (1 - together) * level
        synth.drone(2, root * 2 * 1.003, 0.025 * shimmer, pan: -0.5)
        synth.drone(3, root * 3 * 0.998, 0.018 * shimmer, pan: 0.5)
        synth.drone(4, root, 0.04 * level * together)
    }

    static func degrees(_ radians: Double) -> String {
        let d = Int((abs(radians) * 180 / .pi).rounded())
        return "\(d) degrees \(radians < 0 ? "left" : "right")"
    }

    private static let source = """
    // a: pivot (centered coords), arm length (fraction of height), exposure per pendulum
    // b: hue offset, fade, count, trail samples   c: line half-width in pixels, arm and bob brightness, arm stride
    struct L { float4 pos [[position]]; float across; float hub; float4 color; };

    inline float4 placed(float2 q, constant U &u) {
        float2 p = u.a.xy + q * u.a.z;
        return float4(p.x * u.res.y / u.res.x, p.y, 0, 1);
    }

    // A screen-aligned quad around the segment a→b, wide enough for a soft glow.
    inline L segment(float2 a, float2 b, uint corner, float4 color, constant U &u) {
        float4 pa = placed(a, u), pb = placed(b, u);
        float2 sa = pa.xy * u.res * 0.5, sb = pb.xy * u.res * 0.5;
        float2 d = sb - sa;
        float len = length(d);
        float2 dir = len > 1e-4 ? d / len : float2(1, 0);
        float2 nrm = float2(-dir.y, dir.x);
        float w = u.c.x * 1.7;
        const uint corners[6] = {0, 1, 2, 2, 1, 3};
        uint k = corners[corner];
        float2 s = (k & 1) ? sb : sa;
        float side = (k & 2) ? 1.0 : -1.0;
        s += nrm * side * w;
        L o;
        o.pos = float4(s / (u.res * 0.5), 0, 1);
        o.across = side * w;
        o.hub = float(k & 1);
        o.color = color;
        return o;
    }

    inline float3 hueOf(float i, constant U &u) {
        return spectrum(fract(i / u.b.z * 0.85 + u.b.x));
    }

    vertex L armQuad(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                     const device float2 *bobs [[buffer(1)]]) {
        uint p = (iid >> 1) * uint(u.c.w), seg = iid & 1;
        float2 a = seg == 0 ? float2(0) : bobs[2 * p];
        float2 b = bobs[2 * p + seg];
        float e = u.a.w * u.b.y * u.c.y * u.c.w;
        float4 c = float4(hueOf(float(p), u) * e * (seg == 0 ? 0.4 : 1.0), 1);
        L o = segment(a, b, vid, c, u);
        // Every upper arm meets the others at the pivot; fading them there keeps it from burning a bright star in.
        if (seg == 1) o.hub = 1;
        return o;
    }

    vertex L trailQuad(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                       const device float2 *path [[buffer(1)]]) {
        uint n = uint(u.b.w);
        uint p = iid / n, s = iid % n;
        float2 a = path[p * (n + 1) + s], b = path[p * (n + 1) + s + 1];
        float e = u.a.w * u.b.y * 0.25 * u.c.z;
        L o = segment(a, b, vid, float4(hueOf(float(p), u) * e, 1), u);
        o.hub = 1;
        return o;
    }

    fragment float4 glowLine(L in [[stage_in]], constant U &u [[buffer(0)]]) {
        float x = in.across / u.c.x;
        float core = exp(-x * x * 2.2);
        return float4(in.color.rgb * core * pow(saturate(in.hub * 1.6), 2.5), 0);
    }

    vertex P bobDot(uint vid [[vertex_id]], constant U &u [[buffer(0)]], const device float2 *bobs [[buffer(1)]]) {
        uint p = vid >> 1;
        bool lower = vid & 1;
        P o;
        o.pos = placed(bobs[vid], u);
        o.size = lower ? 9 : 5;
        float e = u.a.w * u.b.y * (lower ? 2.2 * u.c.z : 0.8 * u.c.y);
        o.color = float4(hueOf(float(p), u) * e, 0);
        return o;
    }

    fragment float4 pendulumCompose(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                                    texture2d<float> trails [[texture(1)]], texture2d<float> glow [[texture(2)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float2 q = p - u.a.xy;
        // Upper arms all meet at the pivot: hold back the light there so it can't burn a star into the screen.
        float3 c = light.sample(s, in.uv).rgb + trails.sample(s, in.uv).rgb;
        c += glow.sample(s, in.uv).rgb * 0.6;
        float r = length(p * float2(0.8, 1));
        float3 bg = mix(float3(0.010, 0.012, 0.024), float3(0.002, 0.002, 0.005), smoothstep(0.0, 1.4, r));
        bg += float3(0.012, 0.010, 0.020) * exp(-dot(q, q) * 3) * u.b.y;
        return float4(film(c + bg), 1);
    }
    """
}
