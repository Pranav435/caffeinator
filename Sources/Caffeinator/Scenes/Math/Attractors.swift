import Metal

/// A light painting of a strange attractor: two small lights carried along the attractor's flow, RK4 in the GPU, are
/// photographed in a dark, slightly smoky studio as the camera slowly circles. The long exposure keeps the last half
/// minute of their paths as glowing filaments, out of focus where they swing near or far, reflected in a polished
/// black floor and lighting the smoke. Every two and a half minutes the equations change.
@MainActor final class Attractors: Scene {
    private struct System {
        let name: String
        let equation: String
        let center: SIMD3<Float>
        let size: Float
        /// Largest RK4 step, and model time per second of film: how fast the lights travel.
        let step: Float
        let pace: Float
        /// Colour of the second light, an LED wavelength in nanometres; the first is a warm white.
        let led: Float
        /// Drone root in hertz.
        let root: Float
    }

    private static let systems = [
        System(name: "Lorenz", equation: "σ = 10, ρ = 28, β = 8/3", center: [0, 0, 25], size: 25, step: 0.002, pace: 0.42, led: 505, root: 110),
        System(name: "Aizawa", equation: "a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1", center: [0, 0, 0.55], size: 1.3,
               step: 0.008, pace: 0.8, led: 592, root: 98),
        System(name: "Thomas", equation: "b = 0.208186", center: [0, 0, 0], size: 3.6, step: 0.04, pace: 8, led: 470, root: 123.5),
        System(name: "Halvorsen", equation: "a = 1.4", center: [-3.5, -3.5, -3.5], size: 10, step: 0.004, pace: 0.5, led: 625, root: 87.3),
        System(name: "Chen", equation: "a = 35, b = 3, c = 28", center: [0, 0, 22], size: 21, step: 0.0012, pace: 0.22, led: 528, root: 103.8),
        System(name: "Rössler", equation: "a = 0.2, b = 0.2, c = 5.7", center: [0, -1, 5], size: 11, step: 0.02, pace: 2.2, led: 450, root: 92.5),
        System(name: "Dadras", equation: "a = 3, b = 2.7, c = 1.7, d = 2, e = 9", center: [0, 0, 0], size: 11, step: 0.003, pace: 0.55, led: 612,
               root: 116.5),
    ]

    /// Lights, and positions remembered per light: 34 seconds of exposure at 240 a second.
    private static let lights = 2
    private static let length = 8192
    private static let perSecond: Float = 240

    private let history: MTLBuffer
    private var head = 0
    private var budget: Float = 0
    private let trace: MTLComputePipelineState
    private let trails: MTLRenderPipelineState
    private let mirrored: MTLRenderPipelineState
    private let lamps: MTLRenderPipelineState
    private let studio: MTLRenderPipelineState
    private let bloom: Bloom
    private let lens: Lens
    private var light: MTLTexture?
    private var reflection: MTLTexture?
    private var hdr: MTLTexture?
    private var system: Int
    private var since: Float = 0
    private var level: Float = 0
    private var fresh = true
    private var wing: Float = 0
    private var rang: Float = -10

    var scale: Float { 0.62 }
    var facts: String? {
        let s = Self.systems[system]
        return "Light painting of the \(s.name) attractor, \(s.equation); a warm white lamp and a \(Int(s.led)) nanometre LED trace its flow, each trail the last 34 seconds of a long exposure"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(MathOptics.source + Self.source)
        trace = try gpu.kernel(library, "attractorTrace")
        trails = try gpu.pipeline(library, "paintedLight", vertex: "paintedTrail", blend: .add)
        mirrored = try gpu.pipeline(library, "paintedLight", vertex: "paintedReflection", blend: .add)
        lamps = try gpu.pipeline(library, "paintedLight", vertex: "paintedLamp", blend: .add)
        studio = try gpu.pipeline(library, "paintingStudio")
        bloom = try Bloom(gpu: gpu)
        lens = try Lens(gpu: gpu)
        system = (Snapshot.variant ?? Int.random(in: 0..<Self.systems.count)) % Self.systems.count
        history = gpu.buffer(Self.lights * Self.length * MemoryLayout<SIMD4<Float>>.stride)
    }

    func draw(_ f: Frame) {
        since += f.dt
        if since > 150 { level = max(level - f.dt / 2.5, 0) } else { level = min(level + f.dt / 2, 1) }
        if since > 150, level == 0 {
            since = 0
            system = (system + 1) % Self.systems.count
            fresh = true
        }
        let s = Self.systems[system]
        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            reflection = f.gpu.texture(f.target.width / 2, f.target.height / 2)
            hdr = f.gpu.texture(f.target.width, f.target.height)
        }

        // Each stored position is a fixed slice of model time, taken in RK4 steps no longer than the system allows.
        let slice = s.pace / Self.perSecond
        let sub = max(1, Int((slice / s.step).rounded(.up)))
        var u = f.uniforms([Float(system), slice / Float(sub), Float(sub), 0], [s.center.x, s.center.y, s.center.z, s.size],
                           [0, 0, Float(Self.length), Float(head)])
        if fresh {
            // Start each painting already exposed: run the lights from scattered points onto the attractor, then
            // through a whole history.
            fresh = false
            u.a.w = 1
            u.c.x = 2000
            f.cmd.compute(trace, Self.lights, uniforms: u, buffers: [history])
            u.a.w = 0
            u.c.x = Float(Self.length)
            f.cmd.compute(trace, Self.lights, uniforms: u, buffers: [history])
            budget = 0
        }
        budget += f.dt * Self.perSecond
        let n = Int(budget)
        budget -= Float(n)
        u.c.x = Float(n)
        f.cmd.compute(trace, Self.lights, uniforms: u, buffers: [history])
        head = (head + n) % Self.length

        let angle = f.time * 0.045 + f.seed
        let eye = SIMD3<Float>(sin(angle) * 2.9, 0.35 + 0.25 * sin(f.time * 0.027 + f.seed), cos(angle) * 2.9)
        let v = f.uniforms([eye.x, eye.y, eye.z, 2.4], [Float(Self.length), Float(head), s.led, level],
                           [-1.25, 0.1, Float(Self.lights), f.aspect])
        let segments = Self.lights * ((Self.length - 1) / 8 * 8)
        f.cmd.draw(trails, into: light!, uniforms: v, buffers: [history], vertices: 6, instances: segments)
        f.cmd.draw(lamps, into: light!, clear: false, uniforms: v, buffers: [history], vertices: 6, instances: Self.lights)
        var r = v
        r.res = SIMD2(Float(reflection!.width), Float(reflection!.height))
        f.cmd.draw(mirrored, into: reflection!, uniforms: r, buffers: [history], vertices: 6, instances: segments)
        let glow = bloom.glow(f.cmd, light!)
        f.cmd.draw(studio, into: hdr!, uniforms: v, textures: [light!, reflection!, glow], buffers: [history])
        lens.develop(f, hdr!, Lens.Look(exposure: 1, bloom: 0.02, halation: 0.04, vignette: 0.5, aberration: 1.2, grain: 0.7))

        if let synth = f.sound { play(synth, s, f) }
    }

    /// The first light is the tracer: its height bends the chord, and crossing between the wings rings a bell.
    private func play(_ synth: Synth, _ s: System, _ f: Frame) {
        let p = history.contents().assumingMemoryBound(to: SIMD4<Float>.self)[(head + Self.length - 1) % Self.length]
        let side: Float = p.x < 0 ? -1 : 1
        if side != wing {
            if wing != 0, f.time - rang > 4 {
                synth.play(.bell, s.root * (side < 0 ? 6 : 8), amp: 0.07 * level, pan: side * 0.5, length: 3)
                rang = f.time
            }
            wing = side
        }
        let bend = 1 + p.z * 0.03
        synth.drone(0, s.root, 0.12 * level, pan: -0.3)
        synth.drone(1, s.root * (side < 0 ? 1.2 : 1.25) * bend, 0.07 * level, pan: 0.3)
        synth.drone(2, s.root * 1.5 * bend, 0.06 * level)
        synth.drone(3, s.root * 2 + p.y * 4, (0.025 + min(p.w, 1) * 0.03) * level, pan: p.x * 0.6)
        synth.noise(0.05, cutoff: 300)
    }

    private static let source = """
    float3 field(int k, float3 q) {
        float x = q.x, y = q.y, z = q.z;
        switch (k) {
        case 0: return float3(10 * (y - x), x * (28 - z) - y, x * y - 8.0 / 3.0 * z);
        case 1: return float3((z - 0.7) * x - 3.5 * y, 3.5 * x + (z - 0.7) * y,
                              0.6 + 0.95 * z - z * z * z / 3 - (x * x + y * y) * (1 + 0.25 * z) + 0.1 * z * x * x * x);
        case 2: return float3(sin(y) - 0.208186 * x, sin(z) - 0.208186 * y, sin(x) - 0.208186 * z);
        case 3: return float3(-1.4 * x - 4 * y - 4 * z - y * y, -1.4 * y - 4 * z - 4 * x - z * z, -1.4 * z - 4 * x - 4 * y - x * x);
        case 4: return float3(35 * (y - x), -7 * x - x * z + 28 * y, x * y - 3 * z);
        case 5: return float3(-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7));
        default: return float3(y - 3 * x + 2.7 * y * z, 1.7 * y - x * z + z, 2 * x * y - 9 * z);
        }
    }

    // a: system, RK4 step, steps per stored position, start over   b: center, size   c: positions to add, history, head
    // Each light keeps its positions (scaled to about ±1) in a ring buffer, with its speed in w.
    kernel void attractorTrace(constant U &u [[buffer(0)]], device float4 *hist [[buffer(1)]], uint t [[thread_position_in_grid]]) {
        int k = int(u.a.x);
        float h = u.a.y;
        float3 c = u.b.xyz;
        float s = u.b.w;
        uint n = uint(u.c.z), head = uint(u.c.w);
        device float4 *mine = hist + t * n;
        float3 q = u.a.w > 0 ? c + s * float3(0.1 + 0.07 * float(t), 0.05, -0.04 * float(t))
                             : mine[(head + n - 1) % n].xyz * s + c;
        float3 v = 0;
        for (uint i = 0; i < uint(u.c.x); i++) {
            for (int j = 0; j < int(u.a.z); j++) {
                float3 k1 = field(k, q), k2 = field(k, q + 0.5 * h * k1), k3 = field(k, q + 0.5 * h * k2), k4 = field(k, q + h * k3);
                v = (k1 + 2 * k2 + 2 * k3 + k4) / 6;
                q += h * v;
            }
            float3 w = (q - c) / s;
            if (any(isnan(w)) || length(w) > 3) { w = float3(0.1 + 0.05 * t, 0.1, 0); q = w * s + c; }
            mine[(head + i) % n] = float4(w, length(v) * h * float(u.a.z) / (s * 0.02));
        }
        // Warming up from scattered starts, fill the whole history with the settled position.
        if (u.a.w > 0) for (uint i = 0; i < n; i++) mine[i] = float4((q - c) / s, 0);
    }

    // a: eye, zoom   b: history length, head, LED wavelength, fade   c: floor height, aperture, lights, aspect
    inline float3 worldOf(float4 h) { return float3(h.x, h.z, h.y); }

    inline float3 lampColor(uint light, constant U &u) {
        return light == 0 ? blackbody(3100) * float3(1.0, 0.93, 0.85) : ledColor(u.b.z) * 0.9;
    }

    // Screen pixels (y down) and depth of a world point.
    inline float3 shot(float3 q, constant U &u) {
        float4 c = project(q, u.a.xyz, float3(0), u.a.w, u.c.w);
        float z = max(c.w, 0.05);
        return float3((c.x / z * 0.5 + 0.5) * u.res.x, (0.5 - c.y / z * 0.5) * u.res.y, z);
    }

    // Blur disc radius in pixels for a point at depth z, focused on the attractor's centre.
    inline float defocus(float z, constant U &u) {
        float focus = length(u.a.xyz);
        return u.c.y * u.a.w * abs(1 / focus - 1 / z) * u.res.y * 0.5;
    }

    inline float3 positionAt(const device float4 *hist, uint light, uint age, constant U &u, bool mirror) {
        uint n = uint(u.b.x), head = uint(u.b.y);
        float3 q = worldOf(hist[light * n + (head + 2 * n - 1 - min(age, n - 1)) % n]);
        if (mirror) q.y = 2 * u.c.x - q.y;
        return q;
    }

    // Segments go in groups of eight positions. A group whose blur is wide is drawn as two or one longer segments,
    // since the blur hides the corners, which saves filling the same soft band eight times over.
    inline Streak segmentOf(uint vid, uint iid, constant U &u, const device float4 *hist, bool mirror) {
        uint n = uint(u.b.x);
        uint per = (n - 1) / 8 * 8;
        uint light = iid / per, k = iid % per;
        uint group = k / 8, sub = k % 8;
        float floorY = u.c.x;
        float3 start = shot(positionAt(hist, light, group * 8, u, mirror), u);
        float wide = defocus(start.z, u) * (mirror ? 0.5 : 1.0) * 1000 / u.res.y;
        uint stride = wide < 6 ? 1 : wide < 20 ? 4 : 8;
        if (sub * stride >= 8) return streakCorner(0, 0, 0, vid, 0, u.res);
        uint age = group * 8 + sub * stride;
        float3 p = positionAt(hist, light, age, u, mirror), q = positionAt(hist, light, age + stride, u, mirror);
        float gloss = mirror ? floorY - p.y : 0.0;
        float3 a = shot(p, u), b = shot(q, u);
        float blur = defocus(a.z, u);
        // The floor's reflection is smeared more the further the light hung above it.
        if (mirror) blur = blur * 0.5 + gloss * 0.03 * u.res.y + 1.5;
        float older = float(age) / float(n);
        float fade = smoothstep(1.0, 0.7, older) * u.b.w;
        float energy = 14 * u.res.y / 1000 * fade * float(stride);
        if (mirror) energy *= 0.25;
        return streakCorner(a.xy, b.xy, min(blur, 60.0), vid, float4(lampColor(light, u) * energy, 0), u.res);
    }

    vertex Streak paintedTrail(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                               const device float4 *hist [[buffer(1)]]) {
        return segmentOf(vid, iid, u, hist, false);
    }

    vertex Streak paintedReflection(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                    const device float4 *hist [[buffer(1)]]) {
        return segmentOf(vid, iid, u, hist, true);
    }

    // The lamps themselves, where they are now: a brighter point at the head of each trail.
    vertex Streak paintedLamp(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                              const device float4 *hist [[buffer(1)]]) {
        uint n = uint(u.b.x), head = uint(u.b.y);
        float3 a = shot(worldOf(hist[iid * n + (head + n - 1) % n]), u);
        float blur = defocus(a.z, u);
        return streakCorner(a.xy, a.xy, min(blur, 60.0), vid, float4(lampColor(iid, u) * 600 * u.res.y / 1000 * u.b.w, 0), u.res);
    }

    fragment float4 paintedLight(Streak in [[stage_in]]) {
        return float4(in.color.rgb * streakLight(in.local, in.len, in.blur), 0);
    }

    // The studio: black, but for a polished concrete floor lit by everywhere the lamps have been, smoke catching the
    // light, and the reflection.
    fragment float4 paintingStudio(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                                   texture2d<float> mirror [[texture(1)]], texture2d<float> glow [[texture(2)]],
                                   const device float4 *hist [[buffer(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float3 rd = ray(p, u.a.xyz, float3(0), u.a.w);
        float3 c = 0;
        float floorY = u.c.x;
        if (rd.y < 0) {
            float t = (floorY - u.a.y) / rd.y;
            float3 fp = u.a.xyz + rd * t;
            // Light that reached this patch of floor over the exposure, from samples along both lamps' paths.
            uint n = uint(u.b.x), head = uint(u.b.y);
            float3 e = 0;
            for (uint light = 0; light < uint(u.c.z); light++) {
                float3 col = lampColor(light, u);
                for (uint i = 0; i < 16; i++) {
                    uint age = (i * 389u + 101u) % (n * 7 / 10);
                    float3 q = worldOf(hist[light * n + (head + 2 * n - 1 - age) % n]);
                    float3 d = q - fp;
                    float r2 = dot(d, d);
                    e += col * max(d.y, 0.0) * rsqrt(r2) / r2;
                }
            }
            float grain = fbm(fp.xz * 9, 4);
            float albedo = 0.05 * (0.75 + 0.5 * grain) * (0.85 + 0.3 * noise(fp.xz * 1.3));
            c = e * albedo * 0.2 * u.b.w;
            // Fresnel: the polished floor mirrors more at grazing angles.
            float fres = 0.04 + 0.96 * pow(1 - abs(rd.y), 5);
            c += mirror.sample(s, in.uv).rgb * fres * 12 * (0.8 + 0.4 * grain);
            c *= exp(-t * 0.08);
        }
        // Smoke: the glow of the lights, caught in slowly drifting wisps.
        float3 g = glow.sample(s, in.uv).rgb;
        float smoke = fbm(float3(p * 1.4 + float2(u.time * 0.01, -u.time * 0.006), u.time * 0.015), 5);
        c += g * (0.04 + 1.0 * smoothstep(0.4, 0.85, smoke));
        c += light.sample(s, in.uv).rgb;
        return float4(c, 1);
    }
    """
}
