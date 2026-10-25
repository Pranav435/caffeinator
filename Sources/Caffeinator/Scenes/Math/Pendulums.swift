import Foundation
import Metal

/// A long-exposure film of a real double pendulum: machined aluminium arms on ball bearings, an LED on the tip, in a
/// dark studio under a softbox. Eight takes, each released a millionth of a radian further over and lit by a different
/// colour of LED, are overlaid as one multiple exposure. Integrated in double precision with RK4, the eight swing as one
/// white-hot trail until chaos splits it into colours and ghostly arms.
@MainActor final class Pendulums: Scene {
    /// LED peak wavelengths in nanometres, one per take: red to royal blue. Together they make white.
    static let leds: [Float] = [650, 620, 595, 570, 530, 505, 470, 445]
    private static let count = leds.count
    /// Positions recorded per pendulum each frame, for motion blur and smooth trails.
    private static let samples = 8
    /// Spread of the starting angles across all the takes, in radians.
    private static let spread = 1e-6
    /// Gravity over arm length as played back: 30 cm arms (g/L = 33 per second squared) filmed at 120 frames a second
    /// and shown at a quarter speed.
    private static let g = 2.05
    /// How long the long exposure remembers light, in seconds.
    private static let memory: Float = 3.5

    private let arms: MTLRenderPipelineState
    private let streaks: MTLRenderPipelineState
    private let fade: MTLRenderPipelineState
    private let compose: MTLRenderPipelineState
    private let lens: Lens
    private var light: MTLTexture?
    private var ghosts: MTLTexture?
    private var exposure: MTLTexture?
    /// Per frame in flight: elbow and tip for each take at each sample, starting with where the last frame ended.
    private var paths: [MTLBuffer]

    private var t1: [Double]
    private var t2: [Double]
    private var w1: [Double]
    private var w2: [Double]
    private var last: [SIMD4<Float>]
    private var side: [Int8]

    private var rng: UInt64
    private var simTime = 0.0
    private var runLength = 90.0
    private var split: Double?
    private var pivot = SIMD2<Float>(0, 0.1)
    private var level: Float = 0
    private var resetting = false
    private var spreadPx: Float = 0
    private var tokens: Float = 0
    private var chimeCool: Float = 0
    private var startAngles = (0.0, 0.0)
    private var cleared = false

    var scale: Float { 0.75 }

    var facts: String? {
        let start = "Eight takes of one double pendulum with 30-centimetre arms, overlaid, each lit by its own LED from 445 to 650 nanometres; released from \(Self.degrees(startAngles.0)) and \(Self.degrees(startAngles.1)), their starting angles a millionth of a radian apart from first to last"
        if let split { return start + "; chaos split the takes apart after \(Int(split.rounded())) seconds" }
        return start + "; still swinging as one after \(Int(simTime)) seconds"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(MathOptics.source + Self.source)
        arms = try gpu.pipeline(library, "pendulumArm", vertex: "pendulumArmQuad", blend: .add)
        streaks = try gpu.pipeline(library, "ledStreak", vertex: "ledStreakQuad", blend: .add)
        fade = try gpu.pipeline(library, "fadeOut", blend: .fade)
        compose = try gpu.pipeline(library, "pendulumStudio")
        lens = try Lens(gpu: gpu)
        let n = Self.count
        paths = (0..<3).map { _ in gpu.buffer(n * (Self.samples + 1) * MemoryLayout<SIMD4<Float>>.stride) }
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

    /// New starting angles high enough above the horizontal to be chaotic, and the camera reframed a little.
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
            last[i] = joints(i)
        }
        simTime = 0
        split = nil
        runLength = 60 + random() * 25
        pivot = SIMD2(Float(random() - 0.5) * 0.36, 0.1 + Float(random()) * 0.06)
    }

    /// Elbow and tip of take `i`, in arm lengths from the pivot, y up.
    private func joints(_ i: Int) -> SIMD4<Float> {
        let x1 = sin(t1[i]), y1 = -cos(t1[i])
        return SIMD4(Float(x1), Float(y1), Float(x1 + sin(t2[i])), Float(y1 - cos(t2[i])))
    }

    /// Advances every take by `dt` seconds in RK4 steps, recording `samples` positions along the way.
    private func advance(_ dt: Double, into path: UnsafeMutablePointer<SIMD4<Float>>) {
        // RK4 steps of at most about 4 ms, a whole number per sample.
        let steps = Self.samples * max(1, Int((dt / (0.004 * Double(Self.samples))).rounded(.up)))
        let h = dt / Double(steps)
        let g = Self.g
        let every = steps / Self.samples
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
                    base[(s + 1) / every] = SIMD4(Float(x1), Float(y1), Float(x1 + sin(b)), Float(y1 - cos(b)))
                }
            }
            t1[i] = a; t2[i] = b; w1[i] = p; w2[i] = q
            last[i] = base[Self.samples]
        }
    }

    func draw(_ f: Frame) {
        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            ghosts = f.gpu.texture(f.target.width, f.target.height)
            exposure = f.gpu.texture(f.target.width, f.target.height)
        }
        let slot = Int(f.index % 3)
        let path = paths[slot].contents().assumingMemoryBound(to: SIMD4<Float>.self)

        // Between takes the film fades to black, the pendulum is lifted back and the camera reframed.
        if simTime > runLength { resetting = true }
        level = resetting ? max(level - f.dt / 2.5, 0) : min(level + f.dt / 2, 1)
        if resetting, level == 0 {
            reset()
            resetting = false
            cleared = true
        }
        let dt = Double(f.dt)
        advance(dt, into: path)
        simTime += dt

        // How far the tips have fanned out, in pixels.
        var mean = SIMD2<Float>.zero
        for i in 0..<Self.count { mean += SIMD2(last[i].z, last[i].w) }
        mean /= Float(Self.count)
        var variance: Float = 0
        for i in 0..<Self.count { let d = SIMD2(last[i].z, last[i].w) - mean; variance += (d * d).sum() }
        let rms = sqrt(variance / Float(Self.count))
        if split == nil, rms > 0.03, simTime > 1 { split = simTime }
        let reach = 0.2 * Float(f.target.height)
        spreadPx += (rms * reach - spreadPx) * min(f.dt * 4, 1)

        let n = Self.count, s = Self.samples
        let u = f.uniforms([pivot.x, pivot.y, 0.4, f.aspect], [level, Float(n), Float(s), f.dt],
                           [-0.9, 1.25, 1.1, 0])
        f.cmd.draw(fade, into: exposure!, clear: f.index == 0 || cleared, uniforms: f.uniforms([exp(-f.dt / Self.memory), 0, 0, 0]))
        cleared = false
        f.cmd.draw(streaks, into: exposure!, clear: false, uniforms: u, buffers: [paths[slot]], vertices: 6, instances: n * s)
        f.cmd.draw(arms, into: ghosts!, uniforms: u, buffers: [paths[slot]], vertices: 6, instances: n * s)
        f.cmd.draw(compose, into: light!, uniforms: u, textures: [ghosts!, exposure!])
        lens.develop(f, light!, Lens.Look(exposure: 1.1, bloom: 0.035, halation: 0.05, vignette: 0.45, aberration: 1.0, grain: 0.6))

        if let synth = f.sound { play(synth, f) }
    }

    /// Each time a tip swings through the bottom it can sound: together, the takes ring one soft chime per swing;
    /// split apart, the crossings become a scatter of plucks, each take's note set by the colour of its LED.
    private func play(_ synth: Synth, _ f: Frame) {
        var crossed: [Int] = []
        for i in 0..<Self.count {
            let b = SIMD2(last[i].z, last[i].w)
            let s: Int8 = b.y > 0 ? 0 : (b.x < 0 ? -1 : 1)
            if s != 0, side[i] != 0, s != side[i] { crossed.append(i) }
            if s != 0 { side[i] = s }
        }
        let together = max(0, 1 - spreadPx / 40)
        let root: Float = 146.8
        let scale: [Float] = [1, 9.0 / 8, 5.0 / 4, 3.0 / 2, 5.0 / 3, 2, 9.0 / 4, 5.0 / 2, 3, 10.0 / 3]
        chimeCool -= f.dt
        tokens = min(tokens + f.dt * 5, 3)
        if crossed.count > Self.count / 2, together > 0.5, chimeCool < 0 {
            let right = last[crossed[0]].z > 0
            synth.play(.bell, root * (right ? 4 : 3), amp: 0.07 * level, pan: right ? 0.3 : -0.3, length: 3.5)
            chimeCool = 0.5
        } else if !crossed.isEmpty, tokens >= 1, together < 0.8 {
            tokens -= 1
            let i = crossed[Int(random() * Double(crossed.count))]
            let amp = (0.035 + 0.02 * Float(crossed.count) / Float(Self.count)) * level
            synth.play(.pluck, root * 2 * scale[i], amp: amp, pan: max(-0.8, min(0.8, last[i].x)), length: 1.4)
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
    // a: pivot (centered coords), arm length (fraction of half-height), aspect   b: fade, takes, samples, dt
    // c: softbox position (centered coords) and height in front of the backdrop
    constant float LEDS[8] = {650, 620, 595, 570, 530, 505, 470, 445};
    // Drive currents balanced so the eight colours add up to white.
    constant float DRIVE[8] = {0.775, 0.769, 0.774, 0.834, 1.124, 1.168, 1.174, 1.113};

    inline float2 toPixels(float2 p, constant U &u) {
        return float2((p.x / u.a.w * 0.5 + 0.5) * u.res.x, (0.5 - p.y * 0.5) * u.res.y);
    }

    // The LED on each tip: its light spread along where it went this sample, into the long exposure.
    vertex Streak ledStreakQuad(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                const device float4 *path [[buffer(1)]]) {
        uint n = uint(u.b.z);
        uint take = iid / n, s = iid % n;
        float2 a = u.a.xy + path[take * (n + 1) + s].zw * u.a.z;
        float2 b = u.a.xy + path[take * (n + 1) + s + 1].zw * u.a.z;
        // Power so the trail of a tip sweeping 300 pixels a second is just overexposed at its core.
        float power = 1500 * u.res.y / 1000 * u.b.w / float(n);
        return streakCorner(toPixels(a, u), toPixels(b, u), 0.6, vid, float4(ledColor(LEDS[take]) * DRIVE[take] * power, 0), u.res);
    }

    fragment float4 ledStreak(Streak in [[stage_in]]) {
        return float4(in.color.rgb * streakLight(in.local, in.len, in.blur), 0);
    }

    // The arms, once per take per sample, each weighted so the takes average like a multiple exposure and the samples
    // blur like a 1/30 s shutter. Out goes premultiplied colour and coverage.
    struct Arm { float4 pos [[position]]; float2 local; float2 world; float2 dir; float len; uint lower; };

    vertex Arm pendulumArmQuad(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                               const device float4 *path [[buffer(1)]]) {
        uint n = uint(u.b.z);
        uint lower = iid & 1, k = iid >> 1;
        uint open = n / 2;
        uint take = k / open, s = open + k % open;
        float4 e = path[take * (n + 1) + s + 1];
        float2 a = u.a.xy + (lower ? e.xy : float2(0)) * u.a.z;
        float2 b = u.a.xy + (lower ? e.zw : e.xy) * u.a.z;
        float2 d = b - a;
        float len = length(d);
        float2 dir = len > 1e-5 ? d / len : float2(1, 0);
        float2 nrm = float2(-dir.y, dir.x);
        const uint corners[6] = {0, 1, 2, 2, 1, 3};
        uint c = corners[vid];
        float m = 0.03;
        float along = (c & 1) ? len + m : -m;
        float across = (c & 2) ? m : -m;
        float2 p = a + dir * along + nrm * across;
        Arm o;
        o.pos = float4(p.x / u.a.w, p.y, 0, 1);
        o.local = float2(along, across);
        o.world = p;
        o.dir = dir;
        o.len = len;
        o.lower = lower;
        return o;
    }

    inline float stadium(float2 q, float len, float r) {
        return length(float2(q.x - clamp(q.x, 0.0, len), q.y)) - r;
    }

    // Light from the softbox reaching a point on the rig, which hangs 0.6 in front of the backdrop.
    inline float3 softbox(float2 p, float depth, constant U &u, thread float3 &toLight) {
        float3 d = float3(u.c.xy - p, u.c.z - depth);
        float r2 = dot(d, d);
        toLight = d * rsqrt(r2);
        return float3(1.0, 0.96, 0.9) * 2.2 / (r2 + 0.4);
    }

    // Machined, bead-blasted aluminium with bevelled edges that catch the softbox.
    inline float3 aluminium(float d, float2 grad, float2 p, float albedo, float depth, constant U &u) {
        float bevel = 0.0035;
        float t = saturate(-d / bevel);
        float3 n = normalize(float3(grad * (1 - t) * 1.6, 0.25 + t));
        float3 l;
        float3 e = softbox(p, depth, u, l);
        float3 h = normalize(l + float3(0, 0, 1));
        float diffuse = saturate(dot(n, l));
        float spec = pow(saturate(dot(n, h)), 48) * 1.6 + pow(saturate(dot(n, h)), 8) * 0.12;
        return e * (albedo * (0.06 + diffuse) + spec * 0.5);
    }

    fragment float4 pendulumArm(Arm in [[stage_in]], constant U &u [[buffer(0)]]) {
        float2 q = in.local;
        float len = in.len;
        float px = 2.0 / u.res.y;
        float d, shade = 0.62;
        float2 grad;
        if (in.lower == 0) {
            // Upper arm: a bar with bosses round both bearings and a lightening slot down its middle.
            d = min(stadium(q, len, 0.0125), min(length(q) - 0.024, length(q - float2(len, 0)) - 0.021));
            d = max(d, -stadium(q - float2(len * 0.27, 0), len * 0.46, 0.0055));
        } else {
            // Lower arm, in front: a thinner bar with a boss on the elbow axle and a block for the LED at the tip.
            d = min(stadium(q, len, 0.0105), min(length(q) - 0.019, length(q - float2(len, 0)) - 0.016));
            d = max(d, -stadium(q - float2(len * 0.3, 0), len * 0.4, 0.0045));
        }
        float cover = saturate(0.5 - d / px);
        if (cover <= 0) return 0;
        // Screen-space gradient of the distance, rotated back into the picture's axes.
        grad = normalize(float2(dfdx(d), -dfdy(d)) + 1e-6);
        float3 c = aluminium(d, grad, in.world, shade, 0.6 + 0.02 * float(in.lower), u);
        // Bearings and axle caps: a polished steel race, and a dark socket-head bolt in the middle.
        float joint = in.lower == 0 ? length(q - float2(len, 0)) : length(q);
        if (joint < 0.015) {
            float race = smoothstep(0.0155, 0.0145, joint) * smoothstep(0.0085, 0.0095, joint);
            float3 l;
            float3 e = softbox(in.world, 0.62, u, l);
            float2 rn = (in.lower == 0 ? q - float2(len, 0) : q) / max(joint, 1e-4);
            float2 rd = float2(rn.x * in.dir.x - rn.y * in.dir.y, rn.x * in.dir.y + rn.y * in.dir.x);
            float glint = pow(saturate(dot(rd, normalize(l.xy)) * 0.5 + 0.5), 6);
            float3 steel = e * (0.15 + 1.4 * glint);
            float socket = smoothstep(0.0045, 0.0035, joint);
            c = mix(c, steel, race);
            c = mix(c, e * 0.03, socket);
        }
        if (in.lower == 1) {
            // The LED's black housing on the tip.
            float led = length(q - float2(len, 0));
            c = mix(c, float3(0.004), smoothstep(0.0075, 0.0065, led));
        }
        float w = 2 / (u.b.y * u.b.z);
        return float4(c * cover * w, cover * w);
    }

    // The studio: a charcoal paper backdrop lit by a softbox high to the left, a black anodised T-slot post holding
    // the bearing block, faint haze, then the ghosts of the arms and the long exposure of the LEDs over it all.
    fragment float4 pendulumStudio(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> ghosts [[texture(0)]],
                                   texture2d<float> exposure [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float px = 2.0 / u.res.y;
        float3 l;
        float3 c = softbox(p, 0.0, u, l) * 0.1 * (0.94 + 0.06 * fbm(p * 1.5 + 3.1, 3));
        // Haze in the softbox's beam, drifting slowly.
        float2 beam = p - u.c.xy;
        float haze = fbm(float3(p * 1.3 + float2(u.time * 0.012, 0), u.time * 0.02), 4);
        c += float3(0.010, 0.0095, 0.009) * haze * haze * exp(-length(beam) * 0.9);

        // The post: 40 mm black anodised extrusion with two T-slots, up to the bearing block at the pivot.
        float2 q = p - u.a.xy;
        float pw = 0.027;
        if (abs(q.x) < pw + px && q.y < 0.0) {
            float edge = saturate((pw - abs(q.x)) / px);
            float x = q.x / pw;
            float2 g = float2(sign(x) * smoothstep(0.75, 1.0, abs(x)), 0);
            float slot = smoothstep(0.08, 0.04, abs(abs(x) - 0.42));
            float3 post = aluminium(-(pw - abs(q.x)), g, p, 0.035, 0.55, u) * (1 - 0.85 * slot);
            c = mix(c, post, edge);
        }
        float2 bq = abs(q - float2(0, 0.004)) - float2(0.05, 0.042);
        float block = length(max(bq, 0.0)) + min(max(bq.x, bq.y), 0.0) - 0.006;
        if (block < px) {
            float2 g = normalize(float2(dfdx(block), -dfdy(block)) + 1e-6);
            float3 b = aluminium(block, g, p, 0.04, 0.57, u);
            // Four socket screws in the corners.
            float2 sq = abs(q - float2(0, 0.004)) - float2(0.034, 0.027);
            float screw = length(sq);
            b *= 0.4 + 0.6 * smoothstep(0.004, 0.006, screw);
            c = mix(c, b, saturate(0.5 - block / px));
        }

        // The arms: an average over the takes, so where they part each is a ghost.
        float4 arm = ghosts.sample(s, in.uv);
        float a = arm.a;
        c = c * (1 - min(a, 1.0)) + arm.rgb / max(a, 1.0);

        // The axle cap on the pivot, in front of every take's upper arm.
        float cap = length(q);
        if (cap < 0.012) {
            float3 e = softbox(p, 0.63, u, l);
            float ring = smoothstep(0.012, 0.0105, cap);
            float3 steel = e * (0.2 + 1.2 * pow(saturate(dot(normalize(q + 1e-5), normalize(l.xy)) * 0.5 + 0.5), 5));
            c = mix(c, steel, ring);
            c = mix(c, e * 0.03, smoothstep(0.0045, 0.0035, cap));
        }

        c += exposure.sample(s, in.uv).rgb;
        return float4(c * u.b.x, 1);
    }
    """
}
