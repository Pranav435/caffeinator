import Metal

/// Stable fluids (Jos Stam, 1999) on the GPU: semi-Lagrangian advection, vorticity confinement and a pressure
/// projection by Jacobi iteration, on a 512 × 320 grid. Two slow stirrers and a faint drifting current keep the water
/// moving; now and then a drop of indigo, vermilion or gold falls in with a little momentum and curls into a vortex pair.
/// The ink rides on a finer 1024 × 640 grid with MacCormack advection so filaments stay sharp, and it is drawn as
/// pigment, with Kubelka–Munk absorption and scattering, so mixed inks darken like paint instead of adding like light.
@MainActor final class InkInWater: Scene {
    static let width = 512, height = 320
    static let iterations = 40
    static let names = ["indigo", "vermilion", "gold"]

    private let velocityStep: MTLComputePipelineState
    private let curlStep: MTLComputePipelineState
    private let confine: MTLComputePipelineState
    private let divergenceStep: MTLComputePipelineState
    private let jacobi: MTLComputePipelineState
    private let project: MTLComputePipelineState
    private let advectDye: MTLComputePipelineState
    private let correctDye: MTLComputePipelineState
    private let measure: MTLComputePipelineState
    private let clearAll: MTLComputePipelineState
    private let shade: MTLRenderPipelineState
    private let show: MTLRenderPipelineState
    private let bloom: Bloom
    private let lit: MTLTexture
    private let velocity: [MTLTexture]
    private let pressure: [MTLTexture]
    private let divergence: MTLTexture
    private let curl: MTLTexture
    private let dye: [MTLTexture]
    private let stats: MTLBuffer
    private static let blocks = (64, 40)
    private var v = 0
    private var d = 0

    private var nextDrop: Float = 0.2
    private var drops = 0
    private var lastInk = -1
    private var energy: Float = 0
    private var amounts = SIMD3<Float>(repeating: 0)

    var scale: Float { 1 }
    var facts: String? {
        var s = "Navier–Stokes fluid on a 512 by 320 grid, \(Self.iterations) pressure iterations a frame, 3 inks: indigo, vermilion and gold."
        guard drops > 0 else { return s }
        s += " \(drops) \(drops == 1 ? "drop" : "drops") so far"
        if amounts.sum() > 0.002 {
            let i = amounts.x >= amounts.y && amounts.x >= amounts.z ? 0 : amounts.y >= amounts.z ? 1 : 2
            s += ", the water mostly \(Self.names[i]) now"
        }
        return s + "."
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        velocityStep = try gpu.kernel(library, "inkVelocity")
        curlStep = try gpu.kernel(library, "inkCurl")
        confine = try gpu.kernel(library, "inkConfine")
        divergenceStep = try gpu.kernel(library, "inkDivergence")
        jacobi = try gpu.kernel(library, "inkJacobi")
        project = try gpu.kernel(library, "inkProject")
        advectDye = try gpu.kernel(library, "inkAdvect")
        correctDye = try gpu.kernel(library, "inkCorrect")
        measure = try gpu.kernel(library, "inkMeasure")
        clearAll = try gpu.kernel(library, "inkClear")
        shade = try gpu.pipeline(library, "inkShade")
        show = try gpu.pipeline(library, "inkShow")
        bloom = try Bloom(gpu: gpu)
        let w = Self.width, h = Self.height
        velocity = [gpu.texture(w, h, .rg32Float), gpu.texture(w, h, .rg32Float)]
        pressure = [gpu.texture(w, h, .r32Float), gpu.texture(w, h, .r32Float)]
        divergence = gpu.texture(w, h, .r32Float)
        curl = gpu.texture(w, h, .r32Float)
        dye = (0..<3).map { _ in gpu.texture(2 * w, 2 * h, .rgba16Float) }
        lit = gpu.texture(2 * w, 2 * h)
        stats = gpu.buffer(Self.blocks.0 * Self.blocks.1 * MemoryLayout<SIMD4<Float>>.stride)
    }

    func draw(_ f: Frame) {
        let t = f.time
        readStats()

        // Two stirrers wander on slow, incommensurate loops, each fading in and out so the water has calm spells.
        var stir: [SIMD4<Float>] = []
        var active: [Float] = []
        for k in 0..<2 {
            let p = Float(k) * 2.4 + f.seed
            func at(_ t: Float) -> SIMD2<Float> {
                SIMD2(256 + 170 * sin(t / 47 + p) + 50 * sin(t / 19 + 2 * p), 160 + 95 * sin(t / 61 + 1.7 * p) + 30 * cos(t / 23 + p))
            }
            let here = at(t)
            let speed = (at(t + 0.5) - at(t - 0.5)) * 1.6
            stir.append(SIMD4(here.x, here.y, speed.x, speed.y))
            active.append(pow(sin(t / 37 + 1.9 * p) * 0.5 + 0.5, 2))
        }

        // Now and then a drop of ink, pushed in a random direction so it rolls up into a pair of vortices.
        var drop = SIMD4<Float>(-1000, -1000, 0, 0)
        var dropInfo = SIMD4<Float>(0, 0, 0, 0)
        if t >= nextDrop {
            // Three drops in quick succession to open, then one every few seconds.
            nextDrop = t + (drops < 2 ? 1.1 : .random(in: 3...7.5))
            var ink = Int.random(in: 0..<3)
            if ink == lastInk { ink = (ink + Int.random(in: 1...2)) % 3 }
            lastInk = ink
            drops += 1
            let x = Float.random(in: 70...442), y = Float.random(in: 50...270)
            let angle = Float.random(in: 0..<(2 * .pi))
            let push = Float.random(in: 40...75)
            let radius = Float.random(in: 9...15)
            drop = SIMD4(x, y, cos(angle) * push, sin(angle) * push)
            dropInfo = SIMD4(radius, Float(ink), .random(in: 0..<100), 1)
            if let synth = f.sound {
                let notes: [Float] = [587.3, 659.3, 740.0, 880.0, 987.8, 1174.7]
                let note = notes[(ink * 2 + Int.random(in: 0...1)) % notes.count]
                synth.play(.pluck, note * (radius > 12 ? 0.5 : 1), amp: 0.07, pan: (x / 256 - 1) * 0.7, length: 1.4)
            }
        }

        let w = Self.width, h = Self.height
        var u = f.uniforms(stir[0], stir[1], drop, dropInfo)
        u.dt = min(f.dt, 1.0 / 30)
        u.pad = SIMD2(active[0], active[1])
        guard let e = f.cmd.makeComputeCommandEncoder() else { return }
        let run = FieldDispatch(encoder: e)
        let (v0, v1) = (velocity[v], velocity[1 - v])
        if f.index == 0 {
            run(clearAll, w, h, u, [v0, pressure[0], pressure[1]])
            run(clearAll, 2 * w, 2 * h, u, [dye[0], dye[1], dye[2]])
        }
        run(velocityStep, w, h, u, [v0, v1])
        run(curlStep, w, h, u, [v1, curl])
        run(confine, w, h, u, [v1, curl, v0])
        run(divergenceStep, w, h, u, [v0, divergence])
        for i in 0..<Self.iterations {
            run(jacobi, w, h, u, [pressure[i & 1], divergence, pressure[(i + 1) & 1]])
        }
        run(project, w, h, u, [v0, pressure[Self.iterations & 1], v1])
        let (d0, d1, d2) = (dye[d], dye[(d + 1) % 3], dye[(d + 2) % 3])
        run(advectDye, 2 * w, 2 * h, u, [d0, v1, d1])
        run(correctDye, 2 * w, 2 * h, u, [d0, d1, v1, d2])
        run(measure, Self.blocks.0, Self.blocks.1, u, [v1, d2], buffers: [stats])
        e.endEncoding()
        v = 1 - v
        d = (d + 2) % 3
        f.cmd.draw(shade, into: lit, uniforms: f.uniforms(res: SIMD2(Float(2 * w), Float(2 * h))), textures: [d2])
        f.cmd.draw(show, into: f.target, uniforms: f.uniforms(), textures: [lit, bloom.glow(f.cmd, lit)])

        if let synth = f.sound { play(synth) }
    }

    /// Sums kinetic energy and ink from the last finished frame. The buffer is shared, so this may read a frame
    /// that is still in flight; for a sound level that doesn't matter.
    private func readStats() {
        let n = Self.blocks.0 * Self.blocks.1
        let p = stats.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        var sum = SIMD4<Float>(repeating: 0)
        for i in 0..<n { sum += p[i] }
        let cells = Float(Self.width * Self.height)
        let e = sum.x / cells
        energy = e.isFinite ? e : 0
        let ink = SIMD3(sum.y, sum.z, sum.w) / Float(n * 16)
        amounts = ink.x.isFinite ? ink : .zero
    }

    /// Water noise that brightens with the flow's kinetic energy, and a soft chord whose notes follow how much of each
    /// ink is in the water: indigo the low note, vermilion the middle, gold the top.
    private func play(_ synth: Synth) {
        // Mean kinetic energy runs from about 14 (cells per second, squared) in a calm spell to 45 while stirred.
        let flow = min(max((energy.squareRoot() - 3) / 5, 0), 1)
        synth.noise(0.03 + 0.07 * flow, cutoff: 250 + 900 * flow)
        let notes: [Float] = [146.8, 220.0, 293.7]
        for i in 0..<3 {
            let a = min(amounts[i] / 0.02, 1)
            synth.drone(i, notes[i], 0.01 + 0.035 * a, pan: Float(i - 1) * 0.4)
        }
        synth.drone(3, 73.4, 0.03)
    }

    private static let source = """
    constant int2 GRID = int2(512, 320);

    // Bilinear read at continuous cell coordinates (cell centres at +0.5), by hand so the weights keep full precision.
    inline float4 bilerp(texture2d<float, access::read> t, float2 x) {
        float2 sz = float2(t.get_width(), t.get_height());
        x = clamp(x - 0.5, float2(0), sz - 1.001);
        uint2 i = uint2(x);
        float2 f = x - float2(i);
        float4 a = t.read(i), b = t.read(i + uint2(1, 0)), c = t.read(i + uint2(0, 1)), d = t.read(i + uint2(1, 1));
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    inline int2 cell(int2 p) { return clamp(p, int2(0), GRID - 1); }

    kernel void inkClear(texture2d<float, access::write> a [[texture(0)]], texture2d<float, access::write> b [[texture(1)]],
                         texture2d<float, access::write> c [[texture(2)]], uint2 g [[thread_position_in_grid]]) {
        a.write(float4(0), g); b.write(float4(0), g); c.write(float4(0), g);
    }

    // Advects the velocity along itself (midpoint back-trace), damps it, and adds the stirrers, a faint drifting
    // current and the push of a new drop. a, b: stirrers (position, velocity); pad: their strengths;
    // c: drop position and push; d.x: drop radius.
    kernel void inkVelocity(constant U &u [[buffer(0)]], texture2d<float, access::read> vel [[texture(0)]],
                            texture2d<float, access::write> out [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
        float2 x = float2(g) + 0.5;
        float2 v = vel.read(g).xy;
        float2 mid = bilerp(vel, x - 0.5 * u.dt * v).xy;
        float2 w = bilerp(vel, x - u.dt * mid).xy * exp(-u.dt / 10.0);
        // A touch of viscosity, which keeps grid-scale ripples from building up.
        int2 p = int2(g);
        float2 around = vel.read(uint2(cell(p + int2(1, 0)))).xy + vel.read(uint2(cell(p - int2(1, 0)))).xy
                      + vel.read(uint2(cell(p + int2(0, 1)))).xy + vel.read(uint2(cell(p - int2(0, 1)))).xy;
        w = mix(w, around * 0.25, min(u.dt * 3.0, 0.2));
        for (int k = 0; k < 2; k++) {
            float4 s = k == 0 ? u.a : u.b;
            float2 dd = x - s.xy;
            float fall = exp(-dot(dd, dd) / (2 * 30.0 * 30.0));
            w += (s.zw - w) * fall * min(u.dt * 1.5, 1.0) * (k == 0 ? u.pad.x : u.pad.y);
        }
        w += curl(x / 160 + float2(u.time * 0.004, 7.3 + u.seed)) * u.dt * 0.9;
        float2 dd = x - u.c.xy;
        w += u.c.zw * exp(-dot(dd, dd) / (2 * u.d.x * u.d.x * 2.2));
        if (g.x == 0 || g.x == uint(GRID.x - 1)) w.x = 0;
        if (g.y == 0 || g.y == uint(GRID.y - 1)) w.y = 0;
        out.write(float4(w, 0, 0), g);
    }

    kernel void inkCurl(texture2d<float, access::read> vel [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                        uint2 g [[thread_position_in_grid]]) {
        int2 p = int2(g);
        float w = 0.5 * (vel.read(uint2(cell(p + int2(1, 0)))).y - vel.read(uint2(cell(p - int2(1, 0)))).y
                       - vel.read(uint2(cell(p + int2(0, 1)))).x + vel.read(uint2(cell(p - int2(0, 1)))).x);
        out.write(float4(w), g);
    }

    // Vorticity confinement (Fedkiw, Stam and Jensen 2001): pushes swirls back up that numerical diffusion smears out.
    kernel void inkConfine(constant U &u [[buffer(0)]], texture2d<float, access::read> vel [[texture(0)]],
                           texture2d<float, access::read> vort [[texture(1)]], texture2d<float, access::write> out [[texture(2)]],
                           uint2 g [[thread_position_in_grid]]) {
        int2 p = int2(g);
        float l = abs(vort.read(uint2(cell(p - int2(1, 0)))).x), r = abs(vort.read(uint2(cell(p + int2(1, 0)))).x);
        float t = abs(vort.read(uint2(cell(p - int2(0, 1)))).x), b = abs(vort.read(uint2(cell(p + int2(0, 1)))).x);
        float w = vort.read(g).x;
        float2 n = float2(r - l, b - t);
        n /= length(n) + 1e-5;
        float2 v = vel.read(g).xy + u.dt * 0.12 * float2(n.y * w, -n.x * w);
        out.write(float4(v, 0, 0), g);
    }

    // Walls on all four sides: a neighbour past the edge mirrors the normal velocity, so none flows through.
    kernel void inkDivergence(texture2d<float, access::read> vel [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                              uint2 g [[thread_position_in_grid]]) {
        int2 p = int2(g);
        float2 c = vel.read(g).xy;
        float l = p.x > 0 ? vel.read(uint2(p - int2(1, 0))).x : -c.x;
        float r = p.x < GRID.x - 1 ? vel.read(uint2(p + int2(1, 0))).x : -c.x;
        float t = p.y > 0 ? vel.read(uint2(p - int2(0, 1))).y : -c.y;
        float b = p.y < GRID.y - 1 ? vel.read(uint2(p + int2(0, 1))).y : -c.y;
        out.write(float4(0.5 * (r - l + b - t)), g);
    }

    kernel void inkJacobi(texture2d<float, access::read> pin [[texture(0)]], texture2d<float, access::read> div [[texture(1)]],
                          texture2d<float, access::write> pout [[texture(2)]], uint2 g [[thread_position_in_grid]]) {
        int2 p = int2(g);
        float s = pin.read(uint2(cell(p - int2(1, 0)))).x + pin.read(uint2(cell(p + int2(1, 0)))).x
                + pin.read(uint2(cell(p - int2(0, 1)))).x + pin.read(uint2(cell(p + int2(0, 1)))).x;
        pout.write(float4((s - div.read(g).x) * 0.25), g);
    }

    kernel void inkProject(texture2d<float, access::read> vel [[texture(0)]], texture2d<float, access::read> pr [[texture(1)]],
                           texture2d<float, access::write> out [[texture(2)]], uint2 g [[thread_position_in_grid]]) {
        int2 p = int2(g);
        float l = pr.read(uint2(cell(p - int2(1, 0)))).x, r = pr.read(uint2(cell(p + int2(1, 0)))).x;
        float t = pr.read(uint2(cell(p - int2(0, 1)))).x, b = pr.read(uint2(cell(p + int2(0, 1)))).x;
        float2 v = vel.read(g).xy - 0.5 * float2(r - l, b - t);
        if (p.x == 0 || p.x == GRID.x - 1) v.x = 0;
        if (p.y == 0 || p.y == GRID.y - 1) v.y = 0;
        out.write(float4(v, 0, 0), g);
    }

    // Ink lives on a grid twice as fine as the velocity. First the plain semi-Lagrangian step...
    kernel void inkAdvect(constant U &u [[buffer(0)]], texture2d<float, access::read> dye [[texture(0)]],
                          texture2d<float, access::read> vel [[texture(1)]], texture2d<float, access::write> out [[texture(2)]],
                          uint2 g [[thread_position_in_grid]]) {
        float2 x = float2(g) + 0.5;
        float2 v = bilerp(vel, x * 0.5).xy * 2;
        out.write(bilerp(dye, x - u.dt * v), g);
    }

    // ...then MacCormack's correction: trace the result back, compare with where it started, and fix half the error,
    // clamped to the neighbourhood it came from so it can't overshoot. Then the ink fades a little, spreads a little,
    // and a new drop goes in: d.x radius, d.y which ink, d.z a seed for its ragged edge.
    kernel void inkCorrect(constant U &u [[buffer(0)]], texture2d<float, access::read> dye [[texture(0)]],
                           texture2d<float, access::read> hat [[texture(1)]], texture2d<float, access::read> vel [[texture(2)]],
                           texture2d<float, access::write> out [[texture(3)]], uint2 g [[thread_position_in_grid]]) {
        float2 x = float2(g) + 0.5;
        float2 v = bilerp(vel, x * 0.5).xy * 2;
        float4 h = hat.read(g);
        float4 c = h + 0.5 * (dye.read(g) - bilerp(hat, x + u.dt * v));
        float2 sz = float2(dye.get_width(), dye.get_height());
        float2 q = clamp(x - u.dt * v - 0.5, float2(0), sz - 1.001);
        uint2 i = uint2(q);
        float4 a = dye.read(i), b = dye.read(i + uint2(1, 0)), e = dye.read(i + uint2(0, 1)), f = dye.read(i + uint2(1, 1));
        c = clamp(c, min(min(a, b), min(e, f)), max(max(a, b), max(e, f)));
        // Ink fades slowly, faint haze faster, and the walls swallow what reaches them, so the water never silts up.
        float2 edge = min(x, sz - x) / 24;
        float wall = smoothstep(0.0, 1.0, min(edge.x, edge.y));
        c = max(c * exp(-u.dt / 70.0) - u.dt * 0.0022, 0.0) * mix(1 - u.dt * 2, 1.0, wall);
        float2 dd = x - u.c.xy * 2;
        float r = u.d.x * 2;
        // Concentric rings, the way suminagashi marblers float ink, so the flow folds them into nested lines.
        float far = length(dd) / max(r, 1e-3);
        float blob = 0.8 * exp(-far * far * 30) + 0.3 * smoothstep(1.05, 0.75, far);
        for (int k = 0; k < 3; k++) {
            float rk = 0.33 + 0.3 * k;
            blob += exp(-(far - rk) * (far - rk) / 0.01);
        }
        if (blob > 0.001) {
            float ragged = 0.75 + 0.5 * noise(x / (r * 0.6) + u.d.z);
            float amount = 2.4 * blob * ragged * u.d.w;
            int k = int(u.d.y);
            c += float4(k == 0 ? amount : 0, k == 1 ? amount : 0, k == 2 ? amount : 0, 0);
        }
        out.write(max(c, 0.0), g);
    }

    // Per 8 × 8 block of the velocity grid: summed kinetic energy, and the ink in it (sampled every fourth cell).
    kernel void inkMeasure(texture2d<float, access::read> vel [[texture(0)]], texture2d<float, access::read> dye [[texture(1)]],
                           device float4 *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
        float e = 0;
        for (uint j = 0; j < 8; j++) for (uint i = 0; i < 8; i++) { float2 v = vel.read(g * 8 + uint2(i, j)).xy; e += dot(v, v); }
        float3 ink = 0;
        for (uint j = 0; j < 4; j++) for (uint i = 0; i < 4; i++) ink += dye.read(g * 16 + uint2(i, j) * 4).rgb;
        out[g.y * 64 + g.x] = float4(e, ink);
    }

    // Kubelka–Munk reflectance of a pigment layer over black, per channel, from absorption K and scattering S.
    inline float3 kubelkaMunk(float3 K, float3 S) {
        float3 bs = sqrt(K * K + 2 * K * S);
        float3 t = tanh(bs);
        return S * t / max((K + S) * t + bs, 1e-5);
    }

    // Cubic B-spline filtering from four bilinear taps, so the ink grid never shows when scaled up.
    inline float4 smoothSample(texture2d<float> t, float2 uv) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 sz = float2(t.get_width(), t.get_height());
        float2 p = uv * sz - 0.5, i = floor(p), f = p - i;
        float2 w0 = (1 - f) * (1 - f) * (1 - f) / 6, w1 = (4 - 6 * f * f + 3 * f * f * f) / 6;
        float2 w3 = f * f * f / 6, w2 = 1 - w0 - w1 - w3;
        float2 s0 = w0 + w1, s1 = w2 + w3;
        float2 c0 = (i - 0.5 + w1 / s0) / sz, c1 = (i + 1.5 + w3 / s1) / sz;
        return (t.sample(s, float2(c0.x, c0.y)) * s0.x + t.sample(s, float2(c1.x, c0.y)) * s1.x) * s0.y
             + (t.sample(s, float2(c0.x, c1.y)) * s0.x + t.sample(s, float2(c1.x, c1.y)) * s1.x) * s1.y;
    }

    // The ink as pigment, lit from the upper left. Light reaching a point has already passed through the ink between
    // it and the light, so clouds shade themselves and their folds read in depth.
    fragment float4 inkShade(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> dye [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = dye.sample(s, in.uv).rgb;
        // Absorption and scattering of indigo, vermilion and gold, per red, green and blue.
        const float3 K0 = float3(2.3, 1.8, 0.75), S0 = float3(0.25, 0.3, 0.55);
        const float3 K1 = float3(0.12, 1.7, 3.6), S1 = float3(1.0, 0.45, 0.3);
        const float3 K2 = float3(0.08, 0.5, 3.4), S2 = float3(1.0, 0.85, 0.35);
        float3 K = c.x * K0 + c.y * K1 + c.z * K2;
        float3 S = c.x * S0 + c.y * S1 + c.z * S2;
        float3 R = kubelkaMunk(K, S);

        float2 texel = 1 / float2(dye.get_width(), dye.get_height());
        float2 toward = normalize(float2(-0.55, -0.85)) * texel * 3.5;
        float3 depth = 0;
        for (int i = 1; i <= 6; i++) depth += dye.sample(s, in.uv + toward * (float(i) + 0.5 * hash12(in.pos.xy))).rgb;
        float3 shadow = exp(-(depth.x * (K0 + S0) + depth.y * (K1 + S1) + depth.z * (K2 + S2)) * 0.07);

        float hl = dot(dye.sample(s, in.uv - float2(texel.x * 1.5, 0)).rgb, 1), hr = dot(dye.sample(s, in.uv + float2(texel.x * 1.5, 0)).rgb, 1);
        float ht = dot(dye.sample(s, in.uv - float2(0, texel.y * 1.5)).rgb, 1), hb = dot(dye.sample(s, in.uv + float2(0, texel.y * 1.5)).rgb, 1);
        float3 n = normalize(float3(hl - hr, hb - ht, 1.6));
        float3 L = normalize(float3(-0.5, 0.65, 0.6));
        float spec = pow(max(dot(reflect(-L, n), float3(0, 0, 1)), 0.0), 30.0) * 0.08 * smoothstep(0.1, 0.8, dot(c, 1));

        float2 q = in.uv - float2(0.4, 0.35);
        float light = 1.6 * (1 - 0.55 * dot(q, q));
        float3 water = float3(0.004, 0.008, 0.012) * (1.2 - dot(q, q));
        float3 col = R * light * (0.18 + 0.82 * shadow) * (0.8 + 0.35 * dot(n, L)) + spec * shadow + water * exp(-(K + S) * 0.6);
        return float4(col, 1);
    }

    fragment float4 inkShow(V in [[stage_in]], texture2d<float> lit [[texture(0)]], texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = smoothSample(lit, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.18;
        return float4(film(c), 1);
    }
    """
}
