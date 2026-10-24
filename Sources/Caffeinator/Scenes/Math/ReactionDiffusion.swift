import Foundation
import Metal

/// Gray–Scott reaction–diffusion: two chemicals diffuse and react, U + 2V → 3V, while U is fed in and V removed.
/// The feed and kill rates drift slowly between Pearson's named regimes, so spots divide, corals branch, worms crawl
/// and mazes fold for hours. Lit as a glossy relief from the concentration.
@MainActor final class ReactionDiffusion: Scene {
    struct Regime {
        let name: String
        let feed: Float
        let kill: Float
        /// Drone chord root in hertz.
        let root: Float
    }

    static let regimes = [
        Regime(name: "mitosis, spots that grow and divide", feed: 0.0367, kill: 0.0649, root: 98),
        Regime(name: "coral growth", feed: 0.0545, kill: 0.062, root: 110),
        Regime(name: "worms", feed: 0.078, kill: 0.061, root: 87.3),
        Regime(name: "fingerprints", feed: 0.037, kill: 0.060, root: 103.8),
        Regime(name: "a maze", feed: 0.029, kill: 0.057, root: 92.5),
        Regime(name: "holes in a sheet", feed: 0.039, kill: 0.058, root: 82.4),
        Regime(name: "solitons, spots that hold still", feed: 0.030, kill: 0.062, root: 116.5),
        Regime(name: "pulsating solitons", feed: 0.025, kill: 0.060, root: 110),
    ]

    private static let rows = 400
    private static let stepsPerSecond: Float = 330
    private static let dwell: Float = 200
    private static let drift: Float = 70

    private let step: MTLComputePipelineState
    private let flow: MTLComputePipelineState
    private let seed: MTLComputePipelineState
    private let prepare: MTLComputePipelineState
    private let render: MTLRenderPipelineState
    private let bloom: Bloom
    private let stats: MTLBuffer
    private var grids: [MTLTexture] = []
    private var look: MTLTexture?
    private var last: MTLTexture?
    private var hdr: MTLTexture?
    private var gw = 0, gh = 0
    private var current = 0
    private var budget: Float = 0

    private var index: Int
    private var phase: Float = 0
    private var feed: Float
    private var kill: Float
    private var palette: Float = 0
    private var change: Float = 0
    private var coverage: Float = 0.1
    private var texture: Float = 0.1
    private var sinceSeed: Float = 0
    private var rng: UInt64

    var scale: Float { 0.75 }

    var facts: String? {
        let r = Self.regimes[index]
        let f = String(format: "%.4f", feed), k = String(format: "%.4f", kill)
        if phase > Self.dwell {
            let n = Self.regimes[(index + 1) % Self.regimes.count]
            return "Gray–Scott reaction–diffusion, feed \(f), kill \(k): drifting from \(r.name) toward \(n.name)"
        }
        return "Gray–Scott reaction–diffusion, feed \(f), kill \(k): \(r.name)"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        step = try gpu.kernel(library, "grayScott")
        flow = try gpu.kernel(library, "rdFlow")
        seed = try gpu.kernel(library, "rdSeed")
        prepare = try gpu.kernel(library, "rdLook")
        render = try gpu.pipeline(library, "rdRender")
        bloom = try Bloom(gpu: gpu)
        stats = gpu.buffer(16)
        rng = UInt64.random(in: 1...UInt64.max)
        if let v = Snapshot.variant { rng = UInt64(v + 1) &* 0x9E37_79B9_7F4A_7C15 }
        index = (Snapshot.variant ?? Int.random(in: 0..<Self.regimes.count)) % Self.regimes.count
        feed = Self.regimes[index].feed
        kill = Self.regimes[index].kill
        palette = Float(index % 3)
    }

    private func random() -> Float {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return Float(rng >> 40) / Float(1 << 24)
    }

    /// Drops patches of V into the U sea, three per pass, with ragged edges so they don't grow as perfect discs.
    /// With `holes`, punches bare patches instead, to break up a sheet that has gone uniform.
    private func sprinkle(_ f: Frame, count: Int, clear: Bool, holes: Bool = false) {
        for i in 0..<max((count + 2) / 3, 1) {
            let d = (0..<3).map { _ in SIMD4(random() * Float(gw), random() * Float(gh), 4 + random() * 8, 1) }
            f.cmd.compute(seed, gw, gh, uniforms: f.uniforms([clear && i == 0 ? 1 : 0, holes ? 1 : 0, random() * 1000, 0],
                                                             [Float(gw), Float(gh), d[2].x, d[2].y], d[0], d[1]),
                          textures: [grids[current]])
        }
    }

    func draw(_ f: Frame) {
        if gw == 0 || hdr?.width != f.target.width || hdr?.height != f.target.height {
            hdr = f.gpu.texture(f.target.width, f.target.height)
            if gw == 0 {
                gh = Self.rows
                gw = Int((Float(gh) * f.aspect).rounded())
                grids = (0..<2).map { _ in f.gpu.texture(gw, gh, .rg32Float) }
                look = f.gpu.texture(gw, gh, .rgba16Float)
                last = f.gpu.texture(gw, gh, .r32Float)
                sprinkle(f, count: 30, clear: true)
                // Start already grown, so the first minutes aren't empty.
                for _ in 0..<12 { simulate(f, steps: 150) }
            }
        }

        // Dwell on a regime, then drift to the next.
        phase += f.dt
        if phase > Self.dwell + Self.drift {
            phase = 0
            index = (index + 1) % Self.regimes.count
        }
        let a = Self.regimes[index], b = Self.regimes[(index + 1) % Self.regimes.count]
        let t = phase < Self.dwell ? 0 : (phase - Self.dwell) / Self.drift
        let s = t * t * (3 - 2 * t)
        feed = a.feed + (b.feed - a.feed) * s
        kill = a.kill + (b.kill - a.kill) * s
        let target = Float(index % 3) + s
        palette += (target - palette) * min(f.dt / 4, 1)

        // Keep it alive: when most of the pattern has died back, seed it again; and now and then add a drop anyway.
        sinceSeed += f.dt
        let counts = stats.contents().assumingMemoryBound(to: UInt32.self)
        if f.index > 2 {
            coverage = Float(counts[0]) / Float(max(gw * gh / 16, 1))
            texture = Float(counts[2]) / Float(max(gw * gh / 16, 1))
            let rate = Float(counts[1]) / 1000 / max(f.dt, 1e-3)
            change += (rate - change) * min(f.dt * 2, 1)
        }
        if (coverage < 0.02 && sinceSeed > 6) || sinceSeed > 90 {
            sprinkle(f, count: coverage < 0.02 ? 24 : 3, clear: false)
            sinceSeed = 0
        } else if texture < 0.03, sinceSeed > 6 {
            // A flat sheet of V: break it with bare holes and fresh spots.
            sprinkle(f, count: 18, clear: false, holes: true)
            sprinkle(f, count: 6, clear: false)
            sinceSeed = 0
        }

        budget += f.dt * Self.stepsPerSecond
        let n = Int(budget)
        budget -= Float(n)
        simulate(f, steps: n)
        // A slow, swirling current keeps even the still patterns moving, so nothing stands in one place for hours.
        f.cmd.compute(flow, gw, gh, uniforms: f.uniforms([Float(n) * 0.012, f.time * 0.004, 0, 0]),
                      textures: [grids[current], grids[1 - current]])
        current = 1 - current

        counts.update(repeating: 0, count: 4)
        let u = f.uniforms([palette, Float(gw), Float(gh), 0])
        f.cmd.compute(prepare, gw, gh, uniforms: u, textures: [grids[current], look!, last!], buffers: [stats])
        f.cmd.draw(render, into: hdr!, uniforms: u, textures: [look!, look!, look!])
        let glow = bloom.glow(f.cmd, hdr!)
        var final = u
        final.a.w = 1
        f.cmd.draw(render, into: f.target, uniforms: final, textures: [look!, hdr!, glow])

        if let synth = f.sound { play(synth, a, b, s) }
    }

    private func simulate(_ f: Frame, steps: Int) {
        let u = f.uniforms([feed, kill, 1, 0.5])
        for _ in 0..<steps {
            f.cmd.compute(step, gw, gh, uniforms: u, textures: [grids[current], grids[1 - current]])
            current = 1 - current
        }
    }

    /// A pad on the regime's chord, cross-fading as the parameters drift, under filtered noise and soft crackle that
    /// rise with how fast the pattern is changing.
    private func play(_ synth: Synth, _ a: Regime, _ b: Regime, _ s: Float) {
        // Total change per second runs from about 600 for still spots to 20,000 for pulsing ones.
        let busy = min(max(log2(max(change, 1) / 600) / 5, 0), 1)
        let ratios: [Float] = [1, 1.5, 2, 2.5]
        for (i, r) in ratios.enumerated() {
            let amp: Float = [0.07, 0.045, 0.03, 0.018][i] * (0.6 + 0.4 * busy)
            synth.drone(i, a.root * r, amp * (1 - s), pan: Float(i % 2 == 0 ? -0.3 : 0.3))
            synth.drone(4 + i, b.root * r, amp * s, pan: Float(i % 2 == 0 ? 0.3 : -0.3))
        }
        synth.noise(0.02 + 0.05 * busy, cutoff: 300 + 1500 * busy, crackle: 2 + 10 * busy, crackleTone: 2400)
    }

    private static let source = """
    // a: feed, kill, Du, Dv. The 3×3 Laplacian from Karl Sims: neighbours 0.2, corners 0.05, centre -1.
    kernel void grayScott(constant U &u [[buffer(0)]], texture2d<float, access::read> a [[texture(0)]],
                          texture2d<float, access::write> b [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
        int w = a.get_width(), h = a.get_height();
        if (int(g.x) >= w || int(g.y) >= h) return;
        int x0 = (int(g.x) + w - 1) % w, x1 = (int(g.x) + 1) % w, y0 = (int(g.y) + h - 1) % h, y1 = (int(g.y) + 1) % h;
        float2 c = a.read(g).rg;
        float2 lap = -c
            + 0.2 * (a.read(uint2(x0, g.y)).rg + a.read(uint2(x1, g.y)).rg + a.read(uint2(g.x, y0)).rg + a.read(uint2(g.x, y1)).rg)
            + 0.05 * (a.read(uint2(x0, y0)).rg + a.read(uint2(x1, y0)).rg + a.read(uint2(x0, y1)).rg + a.read(uint2(x1, y1)).rg);
        float uvv = c.x * c.y * c.y;
        float nu = c.x + u.a.z * lap.x - uvv + u.a.x * (1 - c.x);
        float nv = c.y + u.a.w * lap.y + uvv - (u.a.x + u.a.y) * c.y;
        b.write(float4(clamp(nu, 0.0, 1.0), clamp(nv, 0.0, 1.0), 0, 0), g);
    }

    // Carries the chemicals along a slowly turning curl-noise current. a: distance in cells, time
    kernel void rdFlow(constant U &u [[buffer(0)]], texture2d<float, access::sample> a [[texture(0)]],
                       texture2d<float, access::write> b [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
        int w = a.get_width(), h = a.get_height();
        if (int(g.x) >= w || int(g.y) >= h) return;
        constexpr sampler s(filter::linear, address::repeat);
        float2 size = float2(w, h);
        float2 p = float2(g) + 0.5;
        float2 v = curl(p / size.y * 1.6 + u.a.y) * 0.25 + float2(0.35, 0.1);
        b.write(a.sample(s, (p - v * u.a.x) / size), g);
    }

    // Sets V in up to three discs (c, d and b.zw with radius 6) with ragged edges, or with a.y = 1 clears them back to
    // bare U; a.x = 1 first resets the grid.
    kernel void rdSeed(constant U &u [[buffer(0)]], texture2d<float, access::read_write> t [[texture(0)]],
                       uint2 g [[thread_position_in_grid]]) {
        if (g.x >= t.get_width() || g.y >= t.get_height()) return;
        float2 c = u.a.x > 0.5 ? float2(1, 0) : t.read(g).rg;
        float2 p = float2(g);
        float2 size = u.b.xy;
        float jag = hash12(p + u.a.z) * 3;
        float4 discs[3] = {u.c, u.d, float4(u.b.zw, 6, 1)};
        for (int i = 0; i < 3; i++) {
            if (discs[i].w <= 0) continue;
            float2 d = abs(p - discs[i].xy);
            d = min(d, size - d);
            if (length(d) < discs[i].z + jag) c = u.a.y > 0.5 ? float2(1, 0) : float2(0.5, 0.25 + 0.1 * hash12(p * 1.7 + u.a.z));
        }
        t.write(float4(c, 0, 0), g);
    }

    // Height (V), its gradient and the change since last frame, for lighting. Counts covered and textured cells (one in
    // sixteen) and the total change.
    kernel void rdLook(constant U &u [[buffer(0)]], texture2d<float, access::read> a [[texture(0)]],
                       texture2d<float, access::write> look [[texture(1)]], texture2d<float, access::read_write> last [[texture(2)]],
                       device atomic_uint *stats [[buffer(1)]],
                       uint2 g [[thread_position_in_grid]]) {
        int w = a.get_width(), h = a.get_height();
        bool valid = int(g.x) < w && int(g.y) < h;
        float v = 0, moved = 0;
        float2 grad = 0;
        if (valid) {
            int x0 = (int(g.x) + w - 1) % w, x1 = (int(g.x) + 1) % w, y0 = (int(g.y) + h - 1) % h, y1 = (int(g.y) + 1) % h;
            v = a.read(g).g;
            grad = float2(a.read(uint2(x1, g.y)).g - a.read(uint2(x0, g.y)).g, a.read(uint2(g.x, y1)).g - a.read(uint2(g.x, y0)).g) * 0.5;
            moved = abs(v - last.read(g).r);
            last.write(float4(v), g);
            look.write(float4(v, grad, moved), g);
        }
        bool sampled = valid && (g.x % 4 == 0) && (g.y % 4 == 0);
        uint covered = simd_sum(uint(sampled && v > 0.1));
        uint textured = simd_sum(uint(sampled && length(grad) > 0.008));
        uint delta = simd_sum(uint(moved * 1000));
        if (simd_is_first()) {
            atomic_fetch_add_explicit(&stats[0], covered, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[1], delta, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[2], textured, memory_order_relaxed);
        }
    }

    inline float4 bspline(texture2d<float> t, float2 p) {
        constexpr sampler s(filter::linear, address::repeat);
        float2 size = float2(t.get_width(), t.get_height());
        float2 q = p - 0.5, i = floor(q), f = q - i;
        float2 w0 = (1 - f) * (1 - f) * (1 - f) / 6, w1 = (4 - 6 * f * f + 3 * f * f * f) / 6;
        float2 w2 = (1 + 3 * f + 3 * f * f - 3 * f * f * f) / 6, w3 = f * f * f / 6;
        float2 g0 = w0 + w1, g1 = w2 + w3;
        float2 h0 = (w1 / g0 - 1 + i + 0.5) / size, h1 = (w3 / g1 + 1 + i + 0.5) / size;
        return (t.sample(s, float2(h0.x, h0.y)) * g0.x + t.sample(s, float2(h1.x, h0.y)) * g1.x) * g0.y
             + (t.sample(s, float2(h0.x, h1.y)) * g0.x + t.sample(s, float2(h1.x, h1.y)) * g1.x) * g1.y;
    }

    // Three materials, blended by a.x: molten copper, deep sea, obsidian with gold veins.
    // Each is low, high and specular colours in linear light, then (glow, diffuse, specular) strengths.
    constant float3 mats[3][4] = {
        {float3(0.010, 0.003, 0.002), float3(0.900, 0.300, 0.060), float3(1.00, 0.75, 0.45), float3(0.30, 0.90, 0.8)},
        {float3(0.001, 0.004, 0.012), float3(0.020, 0.300, 0.420), float3(0.50, 0.85, 1.00), float3(0.10, 0.90, 0.8)},
        {float3(0.004, 0.003, 0.006), float3(0.420, 0.280, 0.080), float3(1.00, 0.86, 0.60), float3(0.00, 0.45, 2.2)},
    };

    inline float3 mat(float m, int k) {
        int i = int(floor(m)) % 3, j = (i + 1) % 3;
        return mix(mats[i][k], mats[j][k], smoothstep(0.0, 1.0, fract(m)));
    }

    // Pass 1 (a.w = 0) lights the relief; pass 2 (a.w = 1) adds glow and tone maps. a: material, grid size
    fragment float4 rdRender(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> look [[texture(0)]],
                             texture2d<float> hdr [[texture(1)]], texture2d<float> glow [[texture(2)]]) {
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        if (u.a.w > 0.5) {
            float3 c = hdr.sample(ls, in.uv).rgb + glow.sample(ls, in.uv).rgb * 0.35;
            float2 q = in.uv - 0.5;
            c *= 1 - dot(q, q) * 0.9;
            return float4(film(c), 1);
        }
        float2 p = in.uv * u.a.yz;
        float4 f = bspline(look, p);
        float v = saturate(f.x * 2.6);
        float2 grad = f.yz * 2.6;
        float3 n = normalize(float3(-grad * 7, 1));
        float3 l = normalize(float3(-0.45, 0.55, 0.7));
        float3 h = normalize(l + float3(0, 0, 1));
        float diffuse = saturate(dot(n, l));
        float spec = pow(saturate(dot(n, h)), 60);
        float sheen = pow(1 - n.z, 2);
        // Materials change with a soft front sweeping across the screen rather than a blend, which would pass
        // through grey.
        float sweep = in.uv.x * 0.75 + (1 - in.uv.y) * 0.25 + (noise(in.uv * 4) - 0.5) * 0.25;
        float m = floor(u.a.x) + smoothstep(sweep - 0.06, sweep + 0.06, fract(u.a.x) * 1.4 - 0.2);
        float3 lowc = mat(m, 0), highc = mat(m, 1), specc = mat(m, 2), k = mat(m, 3);
        float3 base = mix(lowc, highc, smoothstep(0.15, 0.95, v));
        float3 c = base * (0.12 + k.y * diffuse) * (0.2 + 0.8 * v);
        c += specc * spec * (0.15 + v) * k.z + highc * sheen * 0.15 * v;
        // A soft reflection of an overhead light, and the crests glowing as if hot.
        float3 r = reflect(float3(0, 0, -1), n);
        c += specc * smoothstep(0.25, 0.9, dot(r, l)) * 0.08 * k.z * v;
        c += highc * pow(v, 5) * k.x;
        // A faint inner glow where the reaction front is moving.
        c += highc * saturate(f.w * 120) * 0.4;
        return float4(c, 1);
    }
    """
}
