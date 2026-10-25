import AppKit
import Metal

/// Jeff Jones's 2010 Physarum polycephalum model: a million agents each sense a chemical trail at three points ahead,
/// turn toward the strongest, step forward and deposit more trail, while the trail map diffuses and decays. Out of
/// those rules grow veiny transport networks that keep rearranging. The parameters drift slowly between regimes, and
/// every so often food appears at 36 cities around Tokyo while Tokyo Bay and the ocean repel, as in Tero and
/// colleagues' 2010 experiment, where the real slime mold rebuilt something very like the Tokyo rail network.
@MainActor final class SlimeMold: Scene {
    static let width = 1536, height = 960
    static let agents = 1 << 20

    /// Sensor angle and turn in degrees, sensor offset and step in cells, and the fraction of trail lost per step.
    struct Regime {
        let name: String
        let sensor: Float, turn: Float, offset: Float, step: Float, decay: Float
        /// Root of the pad chord, in hertz.
        let root: Float
    }

    static let regimes = [
        Regime(name: "a fine network", sensor: 22.5, turn: 45, offset: 9, step: 1, decay: 0.1, root: 110),
        Regime(name: "a wide mesh", sensor: 45, turn: 45, offset: 22, step: 1.3, decay: 0.06, root: 98),
        Regime(name: "broad veins with feathery fans", sensor: 15, turn: 25, offset: 30, step: 1.5, decay: 0.05, root: 130.8),
        Regime(name: "a fine honeycomb", sensor: 60, turn: 30, offset: 16, step: 1.1, decay: 0.08, root: 87.3),
    ]

    static let rail = Regime(name: "a rail-like network", sensor: 22.5, turn: 45, offset: 14, step: 1, decay: 0.08, root: 110)

    private let seed: MTLComputePipelineState
    private let move: MTLComputePipelineState
    private let diffuse: MTLComputePipelineState
    private let zero: MTLComputePipelineState
    private let show: MTLRenderPipelineState
    private let lens: Lens
    private let agents: MTLBuffer
    private let counts: MTLBuffer
    private let turns: MTLBuffer
    private var trail: [MTLTexture]
    private let sea: MTLTexture
    private let food: MTLTexture
    private var light: MTLTexture?

    private var regime: Int
    private var held: Float = 0
    private var shifted = false
    private var params: SIMD4<Float>
    private var decay: Float
    private var steps: Float = 0
    private var stepIndex: UInt32 = 0
    private var tokyo: Float = 0
    private var tokyoClock: Float
    private var activity: Float = 0
    private var labelAlpha: Float = -1

    var scale: Float { 1 }
    var facts: String? {
        let r = Self.regimes[regime]
        let before = Self.regimes[(regime + Self.regimes.count - 1) % Self.regimes.count]
        let stage = tokyo > 0.5 ? "Now growing \(Self.rail.name)"
            : shifted && held < 60 ? "Now shifting from \(before.name) to \(r.name)" : "Now growing \(r.name)"
        var s = "1,048,576 agents; sensor angle \(Self.number(params.x * 180 / .pi)) degrees, turning \(Self.number(params.y * 180 / .pi)) degrees, "
            + "sensing \(Self.number(params.z)) cells ahead; the trail fades \(Int((decay * 100).rounded())) percent a step. \(stage)."
        if tokyo > 0.5 {
            s += " Food has appeared at 36 cities around Tokyo, with the bay and the ocean off limits: in a 2010 experiment, "
                + "slime mold fed oat flakes at such positions grew a network much like the Tokyo rail system."
        }
        return s
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        seed = try gpu.kernel(library, "slimeSeed")
        move = try gpu.kernel(library, "slimeMove")
        diffuse = try gpu.kernel(library, "slimeDiffuse")
        zero = try gpu.kernel(library, "slimeZero")
        show = try gpu.pipeline(library, "slimeShow")
        lens = try Lens(gpu: gpu)
        agents = gpu.buffer(Self.agents * MemoryLayout<SIMD4<Float>>.stride)
        counts = gpu.buffer(Self.width * Self.height * 4)
        turns = gpu.buffer(3 * 64 * 4)
        trail = [gpu.texture(Self.width, Self.height, .rg16Float), gpu.texture(Self.width, Self.height, .rg16Float)]
        let w = Self.width, h = Self.height
        // The coast needn't be sharper than a few cells, so the sea mask is a quarter the size and sampled smoothly.
        var mask = SlimeTokyo.sea(w / 4, h / 4)
        sea = mask.withUnsafeMutableBytes { gpu.texture(w / 4, h / 4, .r8Unorm, bytes: $0.baseAddress!, bytesPerRow: w / 4) }
        var flakes = SlimeTokyo.food(w, h)
        food = flakes.withUnsafeMutableBytes { gpu.texture(w, h, .r8Unorm, bytes: $0.baseAddress!, bytesPerRow: w) }
        mask = []
        flakes = []

        let start = Snapshot.variant ?? Int.random(in: 0..<Self.regimes.count)
        regime = start % Self.regimes.count
        let r = Self.regimes[regime]
        params = Self.vector(r)
        decay = r.decay
        // The first Tokyo evening comes after a few minutes, or at once with --variant 4 and up.
        tokyoClock = start >= Self.regimes.count ? 0 : -.random(in: 150...300)
    }

    private static func vector(_ r: Regime) -> SIMD4<Float> {
        SIMD4(r.sensor * .pi / 180, r.turn * .pi / 180, r.offset, r.step)
    }

    private static func number(_ x: Float) -> String {
        abs(x - x.rounded()) < 0.05 ? String(Int(x.rounded())) : String(format: "%.1f", x)
    }

    func draw(_ f: Frame) {
        advanceParameters(f)
        if light?.width != f.target.width || light?.height != f.target.height { light = f.gpu.texture(f.target.width, f.target.height) }

        guard let e = f.cmd.makeComputeCommandEncoder() else { return }
        let run = FieldDispatch(encoder: e)
        let w = Self.width, h = Self.height
        var u = f.uniforms(params, [decay, tokyo, Float(Self.agents), 0], [Float(w), Float(h), 0, 0])
        if f.index == 0 {
            run(seed, w * h, 1, u, [trail[0], trail[1]], buffers: [agents, counts])
        }
        // A fixed 40 steps a second, so growth looks the same at any frame rate.
        steps += f.dt * 40
        let n = min(Int(steps), 3)
        steps -= Float(n)
        let slot = Int(f.index % 3)
        u.c.z = Float(slot)
        run(zero, 64, 1, u, [], buffers: [turns])
        for _ in 0..<n {
            u.frame = stepIndex
            stepIndex &+= 1
            run(move, Self.agents, 1, u, [trail[0], sea], buffers: [agents, counts, turns])
            run(diffuse, w, h, u, [trail[0], sea, food, trail[1]], buffers: [counts])
            trail.swapAt(0, 1)
        }
        e.endEncoding()

        f.cmd.draw(show, into: light!, uniforms: f.uniforms([tokyo, 0, 0, 0]), textures: [trail[0], sea, food])
        lens.develop(f, light!, Lens.Look(exposure: 1.1, white: [1, 0.98, 0.95], bloom: 0.04, halation: 0.03, vignette: 0.35, aberration: 1.0, grain: 0.4))

        readActivity(slot: (slot + 1) % 3, steps: n)
        labels(f)
        if let synth = f.sound { play(synth) }
    }

    /// Holds each regime for a few minutes, then eases the parameters to the next. Every nine minutes or so the
    /// Tokyo map fades in for three and a half.
    private func advanceParameters(_ f: Frame) {
        held += f.dt
        if held > 200 {
            held = 0
            shifted = true
            regime = (regime + 1) % Self.regimes.count
        }
        let r = Self.regimes[regime]
        let ease = 1 - exp(-f.dt / 20)
        // The Tokyo map always runs on a fine network like the one Jones used to model road and rail networks.
        let goal = Self.vector(r) + (Self.vector(Self.rail) - Self.vector(r)) * tokyo
        params += (goal - params) * ease
        decay += (r.decay + (Self.rail.decay - r.decay) * tokyo - decay) * ease

        tokyoClock += f.dt
        if tokyoClock > 210 { tokyoClock = -.random(in: 300...420) }
        let target: Float = tokyoClock >= 0 ? 1 : 0
        tokyo += (target - tokyo) * (1 - exp(-f.dt / (target > tokyo ? 8 : 15)))
        if tokyo < 0.001 { tokyo = 0 }
    }

    /// The share of agents off the network, exploring: high while the web is reorganizing, low once it settles.
    private func readActivity(slot: Int, steps: Int) {
        let p = turns.contents().assumingMemoryBound(to: UInt32.self)
        var total: UInt32 = 0
        for i in 0..<64 { total &+= p[slot * 64 + i] }
        let fraction = Float(total) / Float(Self.agents * max(steps, 1))
        if fraction.isFinite, fraction <= 1 { activity += (fraction - activity) * 0.1 }
    }

    /// City names, faintly, while the Tokyo map is up. Redrawn only as it fades.
    private func labels(_ f: Frame) {
        let alpha = (tokyo * 10).rounded() / 10
        guard alpha != labelAlpha else { return }
        labelAlpha = alpha
        guard alpha > 0 else { f.overlay.clear(); return }
        f.overlay.draw { ctx, size in
            for city in SlimeTokyo.cities where city.label {
                let p = SlimeTokyo.position(city, size.width, size.height)
                ctx.text(city.name, at: CGPoint(x: p.x + 7, y: p.y - 6), size: 11, weight: .medium,
                         color: NSColor(white: 1, alpha: CGFloat(alpha) * 0.42), tracking: 0.6)
            }
        }
    }

    /// A low pad on the regime's chord, with crackle whose density follows how many agents are off exploring:
    /// a few clicks a second over a settled web, a busy fizz while it rebuilds.
    private func play(_ synth: Synth) {
        let root = Self.regimes[regime].root
        let busy = min(activity.squareRoot(), 0.5)
        synth.noise(0.012, cutoff: 500, crackle: 1 + 60 * busy, crackleTone: 2400 + 4000 * busy)
        synth.drone(0, root / 2, 0.07, pan: -0.2)
        synth.drone(1, root * 0.75, 0.035, pan: 0.25)
        synth.drone(2, root, 0.03 + 0.02 * busy, pan: -0.35)
        synth.drone(3, root * 1.5 * (1 + 0.002 * sin(Float(stepIndex) * 0.01)), 0.018, pan: 0.4)
        // Tokyo adds a bright fifth above, like a station chime held under the hum.
        synth.drone(4, root * 3, 0.02 * tokyo, pan: 0.1)
    }

    private static let source = """
    // a: sensor angle, turn angle (radians), sensor offset, step (cells)   b: decay, Tokyo strength, agent count
    // c: map width, height, counter slot

    kernel void slimeSeed(constant U &u [[buffer(0)]], device float4 *ag [[buffer(1)]], device atomic_uint *counts [[buffer(2)]],
                          texture2d<float, access::write> t0 [[texture(0)]], texture2d<float, access::write> t1 [[texture(1)]],
                          uint i [[thread_position_in_grid]]) {
        uint s = pcg(i * 3u + 17u + uint(u.seed * 1000));
        if (i < uint(u.b.z)) ag[i] = float4(rnd(s) * u.c.x, rnd(s) * u.c.y, rnd(s) * TAU, 0);
        uint w = uint(u.c.x);
        atomic_store_explicit(&counts[i], 0u, memory_order_relaxed);
        t0.write(float4(0), uint2(i % w, i / w));
        t1.write(float4(0), uint2(i % w, i / w));
    }

    kernel void slimeZero(constant U &u [[buffer(0)]], device atomic_uint *turns [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        atomic_store_explicit(&turns[uint(u.c.z) * 64 + i], 0u, memory_order_relaxed);
    }

    inline float2 wrapped(float2 p, float2 size) { return p - floor(p / size) * size; }

    constexpr sampler smooth(filter::linear, address::clamp_to_edge);

    inline float sense(texture2d<float, access::read> trail, texture2d<float> sea, float2 p, float h, constant U &u) {
        float2 q = wrapped(p + float2(cos(h), sin(h)) * u.a.z, u.c.xy);
        uint2 c = uint2(q);
        float t = trail.read(c).x;
        return u.b.y > 0 ? t - u.b.y * smoothstep(0.45, 0.9, sea.sample(smooth, q / u.c.xy, level(0)).x) * 400 : t;
    }

    // Sense ahead, left and right; keep going if ahead is strongest, turn toward the stronger side, or pick a side
    // at random when ahead is weakest. Then step forward and leave a mark.
    kernel void slimeMove(constant U &u [[buffer(0)]], device float4 *ag [[buffer(1)]], device atomic_uint *counts [[buffer(2)]],
                          device atomic_uint *turns [[buffer(3)]], texture2d<float, access::read> trail [[texture(0)]],
                          texture2d<float> sea [[texture(1)]], uint i [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
        float4 a = ag[i];
        float h = a.z;
        float f = sense(trail, sea, a.xy, h, u), l = sense(trail, sea, a.xy, h + u.a.x, u), r = sense(trail, sea, a.xy, h - u.a.x, u);
        uint s = pcg(i ^ (u.frame * 2654435761u));
        // Agents with the trail weakest straight ahead are off the network, exploring; their share is the activity.
        bool lost = false;
        if (f > l && f > r) {}
        else if (f < l && f < r) { h += (s & 1) ? u.a.y : -u.a.y; lost = true; }
        else if (l > r) h += u.a.y;
        else if (r > l) h -= u.a.y;
        h += (float(s >> 8 & 1023) / 1023 - 0.5) * 0.06;
        float2 p = wrapped(a.xy + float2(cos(h), sin(h)) * u.a.w, u.c.xy);
        // Now and then an agent wanders off somewhere new, so fresh branches keep sprouting in the empty cells.
        // While the Tokyo map is up, agents caught at sea give up and start again somewhere else.
        float wet = u.b.y > 0 ? u.b.y * sea.sample(smooth, p / u.c.xy, level(0)).x : 0;
        if ((s >> 20) < 2u || float(s >> 22) < wet * 40) {
            uint k = pcg(s);
            p = float2(float(k & 0xFFFF) / 65536 * u.c.x, float(k >> 16) / 65536 * u.c.y);
            h = float(pcg(k) & 1023) / 1023 * TAU;
        }
        ag[i] = float4(p, h, 0);
        uint2 c = min(uint2(p), uint2(u.c.xy) - 1);
        atomic_fetch_add_explicit(&counts[c.y * uint(u.c.x) + c.x], 1u, memory_order_relaxed);
        uint n = simd_sum(lost ? 1u : 0u);
        if (lane == 0) atomic_fetch_add_explicit(&turns[uint(u.c.z) * 64 + (i >> 5) % 64], n, memory_order_relaxed);
    }

    // Deposit, then diffusion and decay. Four bilinear taps at the corners of a cell make a 3 × 3 binomial blur, close
    // to Jones's mean filter. The second channel keeps a running average of how many agents pass through each cell:
    // the flow along the veins. Food keeps topping up the trail; the sea drains it.
    kernel void slimeDiffuse(constant U &u [[buffer(0)]], device uint *counts [[buffer(1)]],
                             texture2d<float> trail [[texture(0)]], texture2d<float> sea [[texture(1)]],
                             texture2d<float, access::read> food [[texture(2)]], texture2d<float, access::write> out [[texture(3)]],
                             uint2 g [[thread_position_in_grid]]) {
        constexpr sampler s(filter::linear, address::repeat);
        float2 uv = (float2(g) + 0.5) / u.c.xy, d = 0.5 / u.c.xy;
        float sum = (trail.sample(s, uv + float2(-d.x, -d.y)).x + trail.sample(s, uv + float2(d.x, -d.y)).x
                   + trail.sample(s, uv + float2(-d.x, d.y)).x + trail.sample(s, uv + float2(d.x, d.y)).x) * 2.25;
        float2 here = trail.read(g).xy;
        uint index = g.y * uint(u.c.x) + g.x;
        float n = float(counts[index]);
        counts[index] = 0;
        // Jones's agents can't share a cell; capping the mark a crowded cell leaves has much the same effect, and keeps
        // the network from collapsing into a few thick tubes.
        float t = (sum / 9) * (1 - u.b.x) + min(n, 1.5) * 5;
        float wet = u.b.y > 0 ? u.b.y * sea.sample(smooth, uv, level(0)).x : 0;
        if (u.b.y > 0) t = (t + u.b.y * food.read(g).x * 40) * (1 - wet * 0.3);
        float flow = mix(here.y, n, 0.06) * (1 - wet * 0.2);
        out.write(float4(min(t, 2000.0), flow, 0, 0), g);
    }

    // Slime mold on agar in a petri dish, from above under a macro lens. The trail field becomes a height field of
    // yellow tubes: thick where the flow is strong, thin sheets at the growing front. Light comes from above left.
    inline float slimeBody(texture2d<float> trail, float2 uv) {
        constexpr sampler s(filter::linear, address::repeat);
        float2 tf = trail.sample(s, uv).xy;
        float tube = smoothstep(7.0, 34.0, tf.x);
        float vein = 1 - exp(-tf.y / 8.0);
        return tube * (0.6 + 0.55 * vein);
    }

    fragment float4 slimeShow(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> trail [[texture(0)]],
                              texture2d<float> sea [[texture(1)]], texture2d<float> food [[texture(2)]]) {
        constexpr sampler s(filter::linear, address::repeat);
        float2 R = u.res, p = in.pos.xy, uv = in.uv;
        float tokyo = u.a.x;
        float2 mid = R * 0.5;
        float rd = mix(0.42 * R.x, 0.62 * R.x, smoothstep(0.0, 1.0, tokyo));
        float rw = 0.03 * R.y;
        float r = length(p - mid);
        float3 lamp = float3(1.0, 0.93, 0.8);

        // The table outside the dish: nearly black, with the dish's glow on it.
        float3 col = float3(0.010, 0.008, 0.006) * (1.2 - 0.6 * smoothstep(rd, rd * 1.35, r));
        float edge = rd - r;

        if (edge > 0) {
            // Agar: pale, translucent amber, mottled, a little darker toward the walls where it is shallower.
            float mott = fbm(uv * float2(9, 5.6) * 2.0 + u.seed, 5);
            float fine = noise(p / 2.3);
            float3 agar = float3(0.56, 0.42, 0.23) * (0.88 + 0.28 * mott) * (0.96 + 0.06 * fine);
            // A soft ring lamp reflected in the wet surface, and a broad pool of light.
            float2 q = (p - mid) / R.y;
            agar *= 0.55 + 0.75 * exp(-dot(q - float2(-0.08, -0.06), q - float2(-0.08, -0.06)) * 1.6);
            float2 rq = (p - (mid + float2(-0.14, -0.2) * R.y)) / R.y;
            float gloss = exp(-pow(abs(length(rq) - 0.18), 2) / 0.0009) * 0.05 + exp(-dot(rq, rq) / 0.03) * 0.025;
            // Beads of condensation.
            float cell = 0.017 * R.y;
            float2 id = floor(p / cell);
            float3 hh = hash33(float3(id, 5.3));
            if (hh.x < 0.07) {
                float2 c = (id + 0.2 + 0.6 * hh.yz) * cell;
                float d = length(p - c) / (0.0014 * R.y * (1 + 2 * hh.y));
                agar *= 1 - 0.3 * smoothstep(1.2, 0.8, d) * smoothstep(0.3, 0.9, d);
                gloss += 0.5 * exp(-dot(p - c + float2(1.2), p - c + float2(1.2)) / 1.3) * step(d, 1.0);
            }
            // Meniscus: the agar creeps up the glass wall and catches the light.
            float men = smoothstep(rw * 1.9, rw * 0.6, rd - r - rw * 0.2);
            men = rd - r < rw * 2.2 ? smoothstep(rw * 2.2, rw, rd - r) : 0.0;
            agar *= 1 - 0.35 * men;
            col = agar * lamp;
            col += lamp * gloss;
            float seaMask = sea.sample(s, uv).x * tokyo;
            col *= 1 - 0.25 * seaMask;

            // Contact shadow of the slime on the agar, soft and offset away from the light.
            float2 toward = float2(-0.0035, -0.0055);
            float sh = slimeBody(trail, uv + toward) * 0.5 + slimeBody(trail, uv + toward * 2.4) * 0.5;
            col *= 1 - 0.5 * saturate(sh);

            // The slime. A height field from the trail; normals from its gradient; subsurface colour from thickness.
            float h = slimeBody(trail, uv);
            // Shuttle streaming: the tubes swell and slacken in slow waves that travel along the network.
            h *= 1.0 + 0.1 * sin(18.0 * noise(uv * float2(7.0, 4.4) + 3.1) + 4.0 * noise(uv * float2(19, 12)) - u.time * 0.62);
            float2 e = 1.6 / R;
            float hx = slimeBody(trail, uv + float2(e.x, 0)) - slimeBody(trail, uv - float2(e.x, 0));
            float hy = slimeBody(trail, uv + float2(0, e.y)) - slimeBody(trail, uv - float2(0, e.y));
            float3 n = normalize(float3(-hx, -hy, 0.35));
            float3 L = normalize(float3(-0.45, -0.55, 0.7));
            float diff = saturate(dot(n, L)) * 0.6 + 0.55;
            float spec = pow(saturate(dot(reflect(-L, n), float3(0, 0, 1))), 24.0) * 0.5;
            float thick = saturate(h);
            // Thin sheets are pale lemon; thick tubes deepen to saturated yellow-orange as the light travels further.
            float3 thin = float3(0.95, 0.82, 0.20), deep = float3(0.92, 0.55, 0.03);
            float3 body = mix(thin, deep, smoothstep(0.35, 1.0, thick) * 0.8);
            body *= 0.55 + 0.45 * smoothstep(0.0, 0.5, noise(uv * float2(700, 440)) + 0.2);
            float3 slime = body * diff * lamp * 0.9 + lamp * spec * 0.7;
            float alpha = smoothstep(0.03, 0.4, h) * 0.97;
            col = mix(col, slime, alpha);

            // Oat flakes at the food sources.
            float fo = food.sample(s, uv).x * tokyo;
            float ragged = fo + 0.35 * (noise(uv * float2(900, 560)) - 0.5);
            float flake = smoothstep(0.1, 0.17, ragged);
            if (flake > 0) {
                float tex = 0.85 + 0.3 * noise(uv * float2(2200, 1400));
                float3 oat = float3(0.70, 0.56, 0.36) * tex * (0.7 + 0.5 * smoothstep(0.1, 0.5, fo));
                float lit = 0.8;
                col = mix(col, oat * lit * lamp, flake * 0.8);
            }
        }

        // The dish: a glass wall seen from above, refracting its own ring of light, with bright inner and outer edges.
        float wall = (rd - r) / rw;
        if (wall > -0.05 && wall < 1.0) {
            float k = saturate(wall);
            col = mix(col, float3(0.05, 0.075, 0.07) * lamp, smoothstep(0.0, 0.3, 1 - k) * 0.0);
            float line = exp(-pow(wall - 0.0, 2) / 0.004) * 0.9 + exp(-pow(wall - 1.0, 2) / 0.006) * 0.35;
            float arc = 0.35 + 0.65 * pow(0.5 + 0.5 * dot(normalize(p - mid + 1e-3), normalize(float2(-1, -1.2))), 2.0);
            col += lamp * line * arc * 0.35;
            col = mix(col, col * 0.45 + float3(0.02, 0.03, 0.025), smoothstep(0.9, 0.5, wall) * smoothstep(0.0, 0.2, wall) * 0.6);
        }
        return float4(max(col, 0.0), 1);
    }
    """
}

/// The Greater Tokyo map for the slime mold: 36 cities as food, and the sea as a mask the agents avoid. The screen
/// shows a window 338 km wide and 211 km tall around Tokyo, so the map keeps its true proportions on a 16:10 grid.
enum SlimeTokyo {
    struct City {
        let name: String
        let lat: Double, lon: Double
        var label = false
    }

    static let north = 36.70, south = 34.80, centerLon = 139.75
    /// Kilometres per degree of latitude, and of longitude at 35.75° N.
    static let kmLat = 111.0, kmLon = 90.3

    static let cities: [City] = [
        City(name: "Tokyo", lat: 35.681, lon: 139.767, label: true), City(name: "Yokohama", lat: 35.466, lon: 139.622, label: true),
        City(name: "Kawasaki", lat: 35.531, lon: 139.697), City(name: "Saitama", lat: 35.906, lon: 139.624, label: true),
        City(name: "Chiba", lat: 35.613, lon: 140.113, label: true), City(name: "Funabashi", lat: 35.702, lon: 139.985),
        City(name: "Kashiwa", lat: 35.862, lon: 139.971), City(name: "Matsudo", lat: 35.784, lon: 139.901),
        City(name: "Hachiōji", lat: 35.656, lon: 139.339, label: true), City(name: "Tachikawa", lat: 35.698, lon: 139.414),
        City(name: "Machida", lat: 35.542, lon: 139.445), City(name: "Sagamihara", lat: 35.594, lon: 139.353),
        City(name: "Fujisawa", lat: 35.339, lon: 139.490), City(name: "Hiratsuka", lat: 35.328, lon: 139.350),
        City(name: "Odawara", lat: 35.256, lon: 139.155, label: true), City(name: "Yokosuka", lat: 35.281, lon: 139.672),
        City(name: "Atsugi", lat: 35.441, lon: 139.365), City(name: "Kawagoe", lat: 35.908, lon: 139.485),
        City(name: "Tokorozawa", lat: 35.787, lon: 139.473), City(name: "Ōme", lat: 35.790, lon: 139.258),
        City(name: "Kumagaya", lat: 36.139, lon: 139.390, label: true), City(name: "Kasukabe", lat: 35.980, lon: 139.753),
        City(name: "Tsukuba", lat: 36.083, lon: 140.111, label: true), City(name: "Toride", lat: 35.911, lon: 140.050),
        City(name: "Narita", lat: 35.776, lon: 140.318, label: true), City(name: "Sakura", lat: 35.723, lon: 140.224),
        City(name: "Kisarazu", lat: 35.381, lon: 139.925, label: true), City(name: "Ichihara", lat: 35.498, lon: 140.116),
        City(name: "Mobara", lat: 35.428, lon: 140.288), City(name: "Tsuchiura", lat: 36.078, lon: 140.204),
        City(name: "Koga", lat: 36.178, lon: 139.755), City(name: "Oyama", lat: 36.315, lon: 139.800),
        City(name: "Utsunomiya", lat: 36.555, lon: 139.883, label: true), City(name: "Kamakura", lat: 35.319, lon: 139.547),
        City(name: "Hannō", lat: 35.851, lon: 139.318), City(name: "Mito", lat: 36.371, lon: 140.476, label: true),
    ]

    /// Coastlines as (longitude, latitude) polygons of sea: the Pacific with Tokyo and Sagami bays, and Suruga Bay.
    static let seas: [[(Double, Double)]] = [
        [(141.70, 36.75), (140.66, 36.75), (140.62, 36.45), (140.58, 36.30), (140.62, 36.10), (140.70, 35.93), (140.85, 35.74),
         (140.70, 35.66), (140.55, 35.55), (140.40, 35.38), (140.33, 35.20), (140.10, 35.10), (139.95, 34.92), (139.87, 34.90),
         (139.82, 34.98), (139.84, 35.12), (139.80, 35.31), (139.90, 35.37), (140.02, 35.48), (140.10, 35.58), (140.00, 35.67),
         (139.86, 35.65), (139.78, 35.62), (139.76, 35.55), (139.70, 35.49), (139.65, 35.44), (139.63, 35.37), (139.67, 35.29),
         (139.74, 35.25), (139.67, 35.17), (139.62, 35.13), (139.60, 35.21), (139.57, 35.29), (139.48, 35.31), (139.35, 35.32),
         (139.20, 35.28), (139.15, 35.24), (139.12, 35.14), (139.07, 35.08), (139.10, 34.98), (139.05, 34.75), (141.70, 34.75)],
        [(138.76, 34.75), (138.78, 34.95), (138.84, 35.05), (138.78, 35.12), (138.65, 35.13), (138.52, 35.04), (138.40, 34.97),
         (138.30, 34.85), (138.26, 34.75)],
    ]

    /// Where a place falls on a map of the given size, top-left origin.
    static func position(lat: Double, lon: Double, _ w: Double, _ h: Double) -> CGPoint {
        let kmHigh = (north - south) * kmLat
        let kmWide = kmHigh * w / h
        let x = ((lon - centerLon) * kmLon / kmWide + 0.5) * w
        let y = (north - lat) / (north - south) * h
        return CGPoint(x: x, y: y)
    }

    static func position(_ c: City, _ w: Double, _ h: Double) -> CGPoint { position(lat: c.lat, lon: c.lon, w, h) }

    /// The sea as an 8-bit mask of the given size, filled by even-odd scanlines and softened a little at the coast.
    static func sea(_ w: Int, _ h: Int) -> [UInt8] {
        var mask = [Float](repeating: 0, count: w * h)
        let polygons = seas.map { $0.map { position(lat: $0.1, lon: $0.0, Double(w), Double(h)) } }
        for y in 0..<h {
            let py = Double(y) + 0.5
            for poly in polygons {
                var xs: [Double] = []
                for i in poly.indices {
                    let a = poly[i], b = poly[(i + 1) % poly.count]
                    if (a.y <= py) != (b.y <= py) { xs.append(a.x + (py - a.y) / (b.y - a.y) * (b.x - a.x)) }
                }
                xs.sort()
                var i = 0
                while i + 1 < xs.count {
                    let from = max(Int(xs[i].rounded()), 0), to = min(Int(xs[i + 1].rounded()), w)
                    if from < to { for x in from..<to { mask[y * w + x] = 1 } }
                    i += 2
                }
            }
        }
        // Every city stands on land, whatever the coarse coastline says.
        for c in cities {
            let p = position(c, Double(w), Double(h))
            for y in max(Int(p.y) - 2, 0)...min(Int(p.y) + 2, h - 1) {
                for x in max(Int(p.x) - 2, 0)...min(Int(p.x) + 2, w - 1) { mask[y * w + x] = 0 }
            }
        }
        // A box blur, five cells wide, for a soft shore.
        for _ in 0..<1 {
            var next = mask
            for y in 0..<h {
                for x in 0..<w {
                    var s: Float = 0
                    for d in -2...2 { s += mask[y * w + min(max(x + d, 0), w - 1)] }
                    next[y * w + x] = s / 5
                }
            }
            for y in 0..<h {
                for x in 0..<w {
                    var s: Float = 0
                    for d in -2...2 { s += next[min(max(y + d, 0), h - 1) * w + x] }
                    mask[y * w + x] = s / 5
                }
            }
        }
        return mask.map { UInt8(min(max($0, 0), 1) * 255) }
    }

    /// Food: a small Gaussian heap at each city.
    static func food(_ w: Int, _ h: Int) -> [UInt8] {
        var grid = [Float](repeating: 0, count: w * h)
        for c in cities {
            let p = position(c, Double(w), Double(h))
            let cx = Int(p.x), cy = Int(p.y)
            for y in max(cy - 12, 0)..<min(cy + 13, h) {
                for x in max(cx - 12, 0)..<min(cx + 13, w) {
                    let dx = Double(x) + 0.5 - p.x, dy = Double(y) + 0.5 - p.y
                    grid[y * w + x] = max(grid[y * w + x], Float(exp(-(dx * dx + dy * dy) / 18)))
                }
            }
        }
        return grid.map { UInt8(min(max($0, 0), 1) * 255) }
    }
}
