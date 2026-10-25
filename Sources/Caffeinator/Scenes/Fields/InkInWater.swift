import Metal

/// A glass tank of still water seen from the side, backlit by a dim warm softbox, with drops of ink falling in. Stable
/// fluids (Jos Stam, 1999) on the GPU in the vertical plane: semi-Lagrangian advection, vorticity confinement and a
/// pressure projection by Jacobi iteration on a 512 × 320 grid, with buoyancy: ink is denser than water, so each drop
/// sinks, rolls into a vortex ring and sheds plumes. The ink rides on a finer 1024 × 640 grid with MacCormack advection.
/// It is drawn by Beer–Lambert absorption of the backlight, so mixed inks darken and blue with yellow turns green.
@MainActor final class InkInWater: Scene {
    static let width = 512, height = 320
    static let iterations = 40
    static let names = ["Prussian blue", "vermilion", "gold"]

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
    private var lit: MTLTexture?
    private let gpu: GPU
    private let lens: Lens
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

    var scale: Float { 0.85 }
    var facts: String? {
        var s = "Navier–Stokes fluid on a 512 by 320 grid, \(Self.iterations) pressure iterations a frame, 3 inks: Prussian blue, vermilion and gold, falling through water in a glass tank."
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
        lens = try Lens(gpu: gpu)
        self.gpu = gpu
        let w = Self.width, h = Self.height
        velocity = [gpu.texture(w, h, .rg32Float), gpu.texture(w, h, .rg32Float)]
        pressure = [gpu.texture(w, h, .r32Float), gpu.texture(w, h, .r32Float)]
        divergence = gpu.texture(w, h, .r32Float)
        curl = gpu.texture(w, h, .r32Float)
        dye = (0..<3).map { _ in gpu.texture(2 * w, 2 * h, .rgba16Float) }
        stats = gpu.buffer(Self.blocks.0 * Self.blocks.1 * MemoryLayout<SIMD4<Float>>.stride)
    }

    func draw(_ f: Frame) {
        let t = f.time
        readStats()

        // A drop of ink every few seconds, released just under the surface with a downward push.
        var drop = SIMD4<Float>(-1000, -1000, 0, 0)
        var dropInfo = SIMD4<Float>(0, 0, 0, 0)
        if t >= nextDrop {
            nextDrop = t + (drops < 2 ? 3.5 : .random(in: 7...14))
            var ink = Int.random(in: 0..<3)
            if ink == lastInk { ink = (ink + Int.random(in: 1...2)) % 3 }
            lastInk = ink
            drops += 1
            let x = Float.random(in: 90...422), y = Float.random(in: 14...30)
            let push = Float.random(in: 26...48)
            let radius = Float.random(in: 9...15)
            drop = SIMD4(x, y, Float.random(in: -6...6), push)
            dropInfo = SIMD4(radius, Float(ink), .random(in: 0..<100), 1)
            if let synth = f.sound {
                let notes: [Float] = [587.3, 659.3, 740.0, 880.0, 987.8, 1174.7]
                let note = notes[(ink * 2 + Int.random(in: 0...1)) % notes.count]
                synth.play(.pluck, note * (radius > 12 ? 0.5 : 1), amp: 0.07, pan: (x / 256 - 1) * 0.7, length: 1.4)
            }
        }

        let w = Self.width, h = Self.height
        var u = f.uniforms(.zero, .zero, drop, dropInfo)
        u.dt = min(f.dt, 1.0 / 30)
        guard let e = f.cmd.makeComputeCommandEncoder() else { return }
        let run = FieldDispatch(encoder: e)
        let (v0, v1) = (velocity[v], velocity[1 - v])
        if f.index == 0 {
            run(clearAll, w, h, u, [v0, pressure[0], pressure[1]])
            run(clearAll, 2 * w, 2 * h, u, [dye[0], dye[1], dye[2]])
        }
        let (d0, d1, d2) = (dye[d], dye[(d + 1) % 3], dye[(d + 2) % 3])
        run(velocityStep, w, h, u, [v0, v1])
        run(curlStep, w, h, u, [v1, curl])
        run(confine, w, h, u, [v1, curl, v0, d0])
        run(divergenceStep, w, h, u, [v0, divergence])
        for i in 0..<Self.iterations {
            run(jacobi, w, h, u, [pressure[i & 1], divergence, pressure[(i + 1) & 1]])
        }
        run(project, w, h, u, [v0, pressure[Self.iterations & 1], v1])
        run(advectDye, 2 * w, 2 * h, u, [d0, v1, d1])
        run(correctDye, 2 * w, 2 * h, u, [d0, d1, v1, d2])
        run(measure, Self.blocks.0, Self.blocks.1, u, [v1, d2], buffers: [stats])
        e.endEncoding()
        v = 1 - v
        d = (d + 2) % 3
        let size = Int(f.size.x), rows = Int(f.size.y)
        if lit == nil || lit!.width != size || lit!.height != rows { lit = gpu.texture(size, rows) }
        f.cmd.draw(shade, into: lit!, uniforms: f.uniforms(), textures: [d2])
        lens.develop(f, lit!, Lens.Look(exposure: 1.0, white: [1, 1, 1], bloom: 0.05, halation: 0.03, vignette: 0.3, aberration: 0.8, grain: 0.45))

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
        let flow = min(max(energy.squareRoot() / 6, 0), 1)
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

    // Advects the velocity along itself (midpoint back-trace), damps it, and adds the push of a new drop.
    // c: drop position and push; d.x: drop radius.
    kernel void inkVelocity(constant U &u [[buffer(0)]], texture2d<float, access::read> vel [[texture(0)]],
                            texture2d<float, access::write> out [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
        float2 x = float2(g) + 0.5;
        float2 v = vel.read(g).xy;
        float2 mid = bilerp(vel, x - 0.5 * u.dt * v).xy;
        float2 w = bilerp(vel, x - u.dt * mid).xy * exp(-u.dt / 7.0);
        // A touch of viscosity, which keeps grid-scale ripples from building up.
        int2 p = int2(g);
        float2 around = vel.read(uint2(cell(p + int2(1, 0)))).xy + vel.read(uint2(cell(p - int2(1, 0)))).xy
                      + vel.read(uint2(cell(p + int2(0, 1)))).xy + vel.read(uint2(cell(p - int2(0, 1)))).xy;
        w = mix(w, around * 0.25, min(u.dt * 3.0, 0.2));
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
                           texture2d<float, access::read> ink [[texture(3)]], uint2 g [[thread_position_in_grid]]) {
        int2 p = int2(g);
        float l = abs(vort.read(uint2(cell(p - int2(1, 0)))).x), r = abs(vort.read(uint2(cell(p + int2(1, 0)))).x);
        float t = abs(vort.read(uint2(cell(p - int2(0, 1)))).x), b = abs(vort.read(uint2(cell(p + int2(0, 1)))).x);
        float w = vort.read(g).x;
        float2 n = float2(r - l, b - t);
        n /= length(n) + 1e-5;
        float2 v = vel.read(g).xy + u.dt * 0.1 * float2(n.y * w, -n.x * w);
        // Buoyancy: ink is denser than water, so it pulls its own water down (y runs down the grid).
        float4 c = bilerp(ink, (float2(g) + 0.5) * 2);
        v.y += u.dt * 20.0 * (c.x + c.y + c.z);
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
        // Over minutes the tank clears: ink thins out into faint haze that is slowly drained, so it never silts up.
        c = max(c * exp(-u.dt / 45.0) - u.dt * 0.0016, 0.0);
        // Ink that has sunk to the floor of the tank settles out and drains off faster.
        c *= 1 - u.dt * 0.12 * smoothstep(0.8, 0.96, x.y / sz.y);
        float2 dd = x - u.c.xy * 2;
        float r = u.d.x * 2;
        float far = length(dd) / max(r, 1e-3);
        float blob = smoothstep(1.0, 0.35, far);
        if (blob > 0.001) {
            float ragged = 0.7 + 0.6 * noise(x / (r * 0.45) + u.d.z);
            float amount = 6.0 * blob * ragged * u.d.w;
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

    // Backlight: a dim, warm softbox behind the tank, a little brighter in the middle and falling off at its edges.
    inline float3 softbox(float2 p, float2 R, float2 c, float2 hsz) {
        float2 q = (p - c) / hsz;
        float edge = smoothstep(1.0, 0.55, max(abs(q.x), abs(q.y) * 1.05));
        float centre = 1.0 - 0.28 * dot(q, q) * 0.5;
        return float3(1.0, 0.76, 0.52) * (0.004 + 0.19 * edge * centre);
    }

    // Beer–Lambert transmission of the three inks (per red, green and blue) at a dye concentration.
    inline float3 inkTransmission(float3 c) {
        const float3 K0 = float3(3.0, 2.0, 0.75);    // Prussian blue: passes blue, kills red
        const float3 K1 = float3(0.25, 2.1, 4.6);    // vermilion: passes red
        const float3 K2 = float3(0.15, 0.75, 5.2);   // gold: passes red and green
        return exp(-2.1 * (c.x * K0 + c.y * K1 + c.z * K2));
    }

    // Looking through the front of the glass tank: ink in water, the glass edges, the meniscus, bubbles and dust.
    fragment float4 inkShade(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> dye [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 R = u.res, p = in.pos.xy;
        float W = min(0.80 * R.x, 0.72 * R.y * 1.6), H = W / 1.6;
        float2 mid = R * float2(0.5, 0.5);
        float air = 0.09 * H, total = H + air;
        float x0 = mid.x - W / 2, x1 = mid.x + W / 2;
        float yt = mid.y - total / 2 - 0.02 * R.y, ys = yt + air, yb = yt + total;
        float gt = 0.013 * H;
        float2 boxC = float2(mid.x, (yt + yb) / 2), boxHalf = float2(W * 0.66, total * 0.72);
        float3 L = softbox(p, R, boxC, boxHalf);
        float3 col = L;

        float ground = max(p.y - (yb + 1.9 * gt), 0.0);
        if (p.y > yb + 1.9 * gt) {
            // The table: dark, with a faint blurred reflection of the softbox.
            float refl = exp(-ground / (0.05 * R.y)) * smoothstep(x0 - gt - 0.1 * W, x0 + 0.1 * W, p.x) * smoothstep(x1 + gt + 0.1 * W, x1 - 0.1 * W, p.x);
            col = float3(1.0, 0.76, 0.52) * (0.003 + 0.05 * refl);
        }

        float wall = min(p.x - x0, x1 - p.x);
        float menisc = 0.012 * H * exp(-max(wall, 0.0) / (0.012 * H));
        float surf = ys - menisc;
        bool insideX = p.x >= x0 && p.x <= x1;
        float2 uvw = (p - float2(x0, ys)) / float2(W, H);

        if (insideX && p.y >= yt && p.y <= yb) {
            float3 water = L * 0.94;
            if (p.y >= surf) {
                // Refraction bends the backlight by the gradient of ink density, a little.
                float2 tx = 1.5 / float2(dye.get_width(), dye.get_height());
                float dl = dot(dye.sample(s, uvw - float2(tx.x, 0)).rgb, 1), dr = dot(dye.sample(s, uvw + float2(tx.x, 0)).rgb, 1);
                float dt = dot(dye.sample(s, uvw - float2(0, tx.y)).rgb, 1), db = dot(dye.sample(s, uvw + float2(0, tx.y)).rgb, 1);
                float2 bend = float2(dl - dr, dt - db) * 0.004;
                float3 c = dye.sample(s, clamp(uvw + bend, 0.0, 1.0)).rgb;
                float3 T = inkTransmission(c);
                // Ink also scatters a little backlight forward, so thin veils glow rather than go grey.
                float thin = 1 - dot(T, float3(0.33));
                col = water * T + L * 0.05 * thin * float3(0.9, 0.8, 0.7) * (1 - 0.5 * thin);
                // Brighter silvered band under the surface, and the bright line of the surface itself.
                float below = p.y - surf;
                col *= 1 + 0.35 * exp(-below / (0.012 * H));
                col += L * 1.3 * exp(-below * below / 3.0);
                // Bubbles stuck on the glass: a dark rim with a bright pinhole of focused backlight.
                float cell = 0.05 * H;
                float2 g = p / cell, id = floor(g);
                float3 h = hash33(float3(id, 3.7));
                if (h.x < 0.09) {
                    float2 pos = (id + 0.25 + 0.5 * h.yz) * cell;
                    float rr = (0.003 + 0.007 * h.y * h.y) * H;
                    float d = length(p - pos) / rr;
                    float rim = smoothstep(0.55, 0.95, d) * smoothstep(1.35, 1.0, d);
                    float2 offs = (p - pos) / rr - float2(-0.3, -0.3);
                    float spot = exp(-dot(offs, offs) * 9) * step(d, 1.0);
                    col = col * (1 - 0.6 * rim) + L * (0.55 * spot + 0.08 * step(d, 1.0));
                }
            } else {
                col = L * 0.98;
            }
            // Dust adrift in the light.
            float2 dg = p / (0.09 * H);
            float2 did = floor(dg);
            float3 dh = hash33(float3(did, 9.1));
            if (dh.x < 0.3) {
                float2 pos = (did + 0.5 + 0.35 * sin(u.time * (0.05 + 0.04 * dh.y) + dh.yz * 40.0)) * 0.09 * H;
                float2 dd = p - pos;
                col += L * 0.22 * dh.z * exp(-dot(dd, dd) / 1.2);
            }
            // The softbox mirrored faintly in the front glass.
            float sx = (p.x - x0) / W + 0.55 * (p.y - yt) / total;
            col += float3(1.0, 0.85, 0.65) * 0.018 * (exp(-(sx - 0.55) * (sx - 0.55) / 0.003) + 0.5 * exp(-(sx - 0.66) * (sx - 0.66) / 0.0004));
        } else if (p.x >= x0 - gt && p.x <= x1 + gt && p.y >= yt && p.y <= yb + 1.9 * gt) {
            // Glass seen edge-on: thick, faintly green, lit from within by the backlight, with bright rims.
            bool base = p.y > yb;
            float inner = base ? p.y - yb : (p.x < x0 ? x0 - p.x : p.x - x1);
            float thick = base ? 1.9 * gt : gt;
            float k = inner / thick;
            float3 edge = L * float3(0.5, 0.78, 0.66) * 0.85 + L * 0.2;
            if (base) {
                // The base shows the tank's contents dimly, upside down.
                float3 c = dye.sample(s, float2(clamp(uvw.x, 0.0, 1.0), 1 - 0.5 * k)).rgb;
                edge = mix(edge, L * float3(0.5, 0.78, 0.66) * inkTransmission(c), 0.55);
            }
            col = edge * (0.8 + 0.4 * k);
            col += L * (0.9 * exp(-inner * inner / 2.0) + 1.1 * exp(-(thick - inner) * (thick - inner) / 1.5));
            if (p.y < ys) col *= 0.9;
        }
        return float4(max(col, 0.0), 1);
    }
    """
}
