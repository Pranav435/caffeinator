import Metal

/// A million particles integrating a strange attractor with RK4, drawn as glowing points while the camera circles.
/// Every two and a half minutes the equations change and the cloud reflows into the next attractor.
@MainActor final class Attractors: Scene {
    private struct System {
        let name: String
        let equation: String
        let center: SIMD3<Float>
        let size: Float
        let step: Float
        let speed: Float
        /// Cosine palette (offset, amplitude, frequency, phase), slow to fast.
        let colors: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)
        /// Drone root in hertz.
        let root: Float
        /// How many particles to draw and how long their trails last. Smooth attractors read better as a few long threads.
        var drawn = 1 << 18
        var trail: Float = 0.7
        /// Long single trajectories drawn over the cloud, the classic way attractors are plotted.
        var threads = 3
    }

    private static let systems = [
        System(name: "Lorenz", equation: "σ = 10, ρ = 28, β = 8/3", center: [0, 0, 25], size: 25, step: 0.002, speed: 180,
               colors: ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 1], [0.0, 0.1, 0.2]), root: 110, drawn: 60_000, trail: 0.9, threads: 12),
        System(name: "Aizawa", equation: "a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1", center: [0, 0, 0.55], size: 1.3, step: 0.008,
               speed: 3, colors: ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 1], [0.3, 0.2, 0.2]), root: 98),
        System(name: "Thomas", equation: "b = 0.208186", center: [0, 0, 0], size: 3.6, step: 0.06, speed: 1.6,
               colors: ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 0.7, 0.4], [0, 0.15, 0.2]), root: 123.5),
        System(name: "Halvorsen", equation: "a = 1.4", center: [-3.5, -3.5, -3.5], size: 10, step: 0.004, speed: 40,
               colors: ([0.8, 0.5, 0.4], [0.2, 0.4, 0.2], [2, 1, 1], [0, 0.25, 0.25]), root: 87.3),
        System(name: "Chen", equation: "a = 35, b = 3, c = 28", center: [0, 0, 22], size: 21, step: 0.0012, speed: 260,
               colors: ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 0.5], [0.8, 0.9, 0.3]), root: 103.8),
        System(name: "Rössler", equation: "a = 0.2, b = 0.2, c = 5.7", center: [0, -1, 5], size: 11, step: 0.02, speed: 20,
               colors: ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 1], [0.6, 0.7, 0.8]), root: 92.5),
        System(name: "Dadras", equation: "a = 3, b = 2.7, c = 1.7, d = 2, e = 9", center: [0, 0, 0], size: 11, step: 0.003, speed: 50,
               colors: ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [2, 1, 0], [0.5, 0.2, 0.25]), root: 116.5),
    ]

    private let count = 1 << 20
    private let length = 8192
    private let particles: MTLBuffer
    private let history: MTLBuffer
    private var head = 0
    private let step: MTLComputePipelineState
    private let trace: MTLComputePipelineState
    private let points: MTLRenderPipelineState
    private let lines: MTLRenderPipelineState
    private let tone: MTLRenderPipelineState
    private let fade: MTLRenderPipelineState
    private let bloom: Bloom
    private var light: MTLTexture?
    private var system: Int
    private var since: Float = 0
    private var wing: Float = 0
    private var rang: Float = -10

    var scale: Float { 1 }
    var facts: String? {
        let s = Self.systems[system]
        return "\(s.name) attractor, \(s.equation)"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        step = try gpu.kernel(library, "advance")
        points = try gpu.pipeline(library, "solid", vertex: "streak", blend: .add)
        trace = try gpu.kernel(library, "trace")
        lines = try gpu.pipeline(library, "solid", vertex: "threadLine", blend: .add)
        tone = try gpu.pipeline(library, "expose")
        fade = try gpu.pipeline(library, "fadeOut", blend: .fade)
        bloom = try Bloom(gpu: gpu)
        system = Snapshot.variant ?? Int.random(in: 0..<Self.systems.count)
        // Start as a random cloud; the flow pulls it onto the attractor within seconds.
        // Each particle is two float4s: position and speed, then where it was a frame ago.
        particles = gpu.buffer((0..<count).flatMap { _ in
            let p = SIMD4<Float>(.random(in: -1...1), .random(in: -1...1), .random(in: -1...1), 0)
            return [p, p]
        })
        history = gpu.buffer((0..<12 * length).map { _ in SIMD4<Float>(.random(in: -0.5...0.5), .random(in: -0.5...0.5), .random(in: -0.5...0.5), 0) })
    }

    func draw(_ f: Frame) {
        since += f.dt
        if since > 150 {
            since = 0
            system = (system + 1) % Self.systems.count
        }
        let s = Self.systems[system]
        if light?.width != f.target.width || light?.height != f.target.height { light = f.gpu.texture(f.target.width, f.target.height) }

        let angle = f.time * 0.05 + f.seed
        let eye = SIMD3<Float>(sin(angle) * 2.7, 0.5 + 0.35 * sin(f.time * 0.031), cos(angle) * 2.7)
        var u = f.uniforms([Float(system), s.step * min(f.dt * 60, 2), 6, s.speed],
                           [s.center.x, s.center.y, s.center.z, s.size],
                           [eye.x, eye.y, eye.z, 2.5],
                           [Float(count), Float(count) / Float(s.drawn) * (1 - s.trail) / 0.3, 0, 0])
        if f.index == 0 {
            // Let the random starting cloud settle onto the attractor before the first frame shows,
            // and give each thread a full history.
            var warm = u
            warm.a.z = 400
            for _ in 0..<10 { f.cmd.compute(step, count, uniforms: warm, buffers: [particles]) }
            warm.a.z = Float(length)
            warm.d.z = Float(length)
            f.cmd.compute(trace, s.threads, uniforms: warm, buffers: [history])
        }
        f.cmd.compute(step, count, uniforms: u, buffers: [particles])
        u.d.z = Float(length)
        u.d.w = Float(head)
        f.cmd.compute(trace, s.threads, uniforms: u, buffers: [history])
        head = (head + 6) % length
        u.d.w = Float(head)
        // Short trails smooth the sparkle and show which way the flow runs.
        f.cmd.draw(fade, into: light!, clear: f.index == 0, uniforms: f.uniforms([pow(s.trail, f.dt * 60), 0, 0, 0]))
        f.cmd.draw(points, into: light!, clear: false, uniforms: u, buffers: [particles], vertices: s.drawn * 2, primitive: .line)
        u.b.w = 1 - pow(s.trail, f.dt * 60)
        f.cmd.draw(lines, into: light!, clear: false, uniforms: u, buffers: [history], vertices: s.threads * (length - 1) * 2, primitive: .line)
        let glow = bloom.glow(f.cmd, light!)
        let (o, a, fr, ph) = s.colors
        u = f.uniforms(SIMD4(o, 0), SIMD4(a, 0), SIMD4(fr, 0), SIMD4(ph, 0))
        f.cmd.draw(tone, into: f.target, uniforms: u, textures: [light!, glow])

        if let synth = f.sound { play(synth, s, f) }
    }

    /// The first particle is the tracer: its height bends the chord, and crossing between the wings rings a bell.
    private func play(_ synth: Synth, _ s: System, _ f: Frame) {
        let p = particles.contents().assumingMemoryBound(to: SIMD4<Float>.self)[0]
        let side: Float = p.x < 0 ? -1 : 1
        if side != wing {
            if wing != 0, f.time - rang > 4 {
                synth.play(.bell, s.root * (side < 0 ? 6 : 8), amp: 0.07, pan: side * 0.5, length: 3)
                rang = f.time
            }
            wing = side
        }
        let bend = 1 + p.z * 0.03
        synth.drone(0, s.root, 0.12, pan: -0.3)
        synth.drone(1, s.root * (side < 0 ? 1.2 : 1.25) * bend, 0.07, pan: 0.3)
        synth.drone(2, s.root * 1.5 * bend, 0.06)
        synth.drone(3, s.root * 2 + p.y * 4, 0.025 + min(p.w, 1) * 0.03, pan: p.x * 0.6)
        synth.noise(0.05, cutoff: 300)
    }

    private static let source = """
    // a: system, step, substeps, speed scale   b: center, size   c: eye, zoom   d: count
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

    kernel void advance(constant U &u [[buffer(0)]], device float4 *ps [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        int k = int(u.a.x);
        float h = u.a.y;
        float3 c = u.b.xyz;
        float s = u.b.w;
        float3 was = ps[2 * i].xyz;
        float3 q = was * s + c;
        float3 v = 0;
        for (int n = 0; n < int(u.a.z); n++) {
            float3 k1 = field(k, q), k2 = field(k, q + 0.5 * h * k1), k3 = field(k, q + 0.5 * h * k2), k4 = field(k, q + h * k3);
            v = (k1 + 2 * k2 + 2 * k3 + k4) / 6;
            q += h * v;
        }
        float3 w = (q - c) / s;
        if (any(isnan(w)) || any(isinf(w)) || length(w) > 3) {
            // Escaped: restart beside another particle, which is already on the attractor.
            uint other = pcg(i + u.frame * 7919u) % uint(u.d.x);
            uint r = pcg(i ^ (u.frame * 2654435761u));
            w = ps[2 * other].xyz + (float3(r & 1023, (r >> 10) & 1023, (r >> 20) & 1023) / 1023 - 0.5) * 0.01;
            was = w;
        }
        ps[2 * i] = float4(w, length(v) / u.a.w);
        ps[2 * i + 1] = float4(was, 0);
    }

    // Threads keep their last d.z positions in a ring buffer; d.w is where the next one goes.
    kernel void trace(constant U &u [[buffer(0)]], device float4 *hist [[buffer(1)]], uint t [[thread_position_in_grid]]) {
        int k = int(u.a.x);
        float h = u.a.y;
        float3 c = u.b.xyz;
        float s = u.b.w;
        uint n = uint(u.d.z), head = uint(u.d.w);
        device float4 *mine = hist + t * n;
        float3 q = mine[(head + n - 1) % n].xyz * s + c;
        for (uint i = 0; i < uint(u.a.z); i++) {
            float3 k1 = field(k, q), k2 = field(k, q + 0.5 * h * k1), k3 = field(k, q + 0.5 * h * k2), k4 = field(k, q + h * k3);
            float3 v = (k1 + 2 * k2 + 2 * k3 + k4) / 6;
            q += h * v;
            float3 w = (q - c) / s;
            if (any(isnan(w)) || length(w) > 3) { w = float3(0.1 + 0.05 * t, 0.1, 0); q = w * s + c; }
            mine[(head + i) % n] = float4(w, length(v) / u.a.w);
        }
    }

    vertex P threadLine(uint id [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *hist [[buffer(1)]]) {
        uint n = uint(u.d.z);
        uint t = id / ((n - 1) * 2);
        uint age = (id / 2) % (n - 1) + (id & 1);
        float4 s = hist[t * n + (uint(u.d.w) + 2 * n - 1 - age) % n];
        P o;
        o.pos = project(s.xzy, u.c.xyz, float3(0), u.c.w, u.res.x / u.res.y);
        o.size = 1;
        float fade = 1 - float(age) / float(n);
        float b = fade * fade * 0.3 * u.b.w;
        o.color = float4(b, b, b, b * clamp(s.w, 0.0, 1.0));
        return o;
    }

    // Each particle is a short line from where it was to where it is, like a long exposure: the light is spread
    // along the line, so fast particles leave faint streaks and slow ones bright points.
    vertex P streak(uint id [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *ps [[buffer(1)]]) {
        uint i = id >> 1;
        float4 s = ps[2 * i];
        float aspect = u.res.x / u.res.y;
        float4 head = project(s.xzy, u.c.xyz, float3(0), u.c.w, aspect);
        float4 tail = project(ps[2 * i + 1].xzy, u.c.xyz, float3(0), u.c.w, aspect);
        float pixels = length((head.xy / max(head.w, 0.01) - tail.xy / max(tail.w, 0.01)) * u.res * 0.5);
        P o;
        o.pos = (id & 1) ? tail : head;
        o.size = 1;
        float t = clamp(s.w, 0.0, 1.0);
        float b = clamp(2.2 / max(head.w, 0.1), 0.4, 1.6) * 0.012 * u.d.y / max(pixels * 0.6, 1.0) * ((id & 1) ? 0.2 : 1.0);
        o.color = float4(b, b, b, b * t);
        return o;
    }

    // Light carries brightness in rgb and speed-weighted brightness in alpha, so their ratio is the average speed.
    fragment float4 expose(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                           texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear);
        float4 l = light.sample(s, in.uv) + glow.sample(s, in.uv) * 0.35;
        float speed = clamp(l.a / max(l.r, 1e-6), 0.0, 1.0);
        float3 c = palette(speed, u.a.xyz, u.b.xyz, u.c.xyz, u.d.xyz) * l.r * 2.5;
        float2 p = in.uv - 0.5;
        float3 bg = float3(0.006, 0.007, 0.012) * (1 - dot(p, p));
        return float4(film(c + bg), 1);
    }
    """
}
