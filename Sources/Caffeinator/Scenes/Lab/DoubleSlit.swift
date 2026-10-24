import Foundation
import Metal

/// The two-slit experiment, solved rather than drawn: the 2D time-dependent Schrödinger equation runs on the GPU
/// with Visscher's staggered leapfrog (ħ = m = 1, one grid cell per unit length). Gaussian wave packets enter from
/// the left, hit a barrier with slits, and a detector screen on the right turns |ψ|² into single electron hits,
/// so the fringes build up dot by dot as in Tonomura's 1989 experiment.
@MainActor final class DoubleSlit: Scene {
    /// One arrangement of slits and beam. Lengths are grid cells.
    struct Setup {
        let slits: Int
        let spacing: Float
        let width: Float
        let k: Float

        /// The share of a packet with vertical spread `sigma` that lines up with the open slits.
        func opening(_ sigma: Float) -> Float {
            let s = sigma * Float(2).squareRoot()
            var sum: Float = 0
            for i in 0..<slits {
                let y = (Float(i) - Float(slits - 1) / 2) * spacing
                sum += (erf((y + width / 2) / s) - erf((y - width / 2) / s)) / 2
            }
            return sum
        }
    }

    static let setups = [
        Setup(slits: 2, spacing: 50, width: 8, k: 0.6),
        Setup(slits: 2, spacing: 84, width: 8, k: 0.6),
        Setup(slits: 1, spacing: 0, width: 26, k: 0.6),
        Setup(slits: 3, spacing: 44, width: 7, k: 0.6),
        Setup(slits: 2, spacing: 50, width: 7, k: 0.85),
        Setup(slits: 2, spacing: 64, width: 9, k: 0.45),
    ]

    // Grid cells are square and the grid fills the height; how much of it shows to the left depends on the display's
    // shape (see `left`), and packets are born just off screen. The screen sits at `screen`, the slits 402 cells before it.
    static let nx = 896, ny = 480
    static let screen: Float = 872
    static let barrier: Float = 470
    static let step: Float = 0.25
    /// Simulated time units per second of scene time.
    static let pace: Float = 80
    static let packetsPerSetup = 8
    private static let sigma = SIMD2<Float>(28, 50)

    private let re: MTLBuffer
    private let im: MTLBuffer
    private let column: MTLBuffer
    private let hits: MTLBuffer
    private let capacity = 8192
    private let stepRe: MTLComputePipelineState
    private let stepIm: MTLComputePipelineState
    private let launch: MTLComputePipelineState
    private let probe: MTLComputePipelineState
    private let field: MTLRenderPipelineState
    private let dots: MTLRenderPipelineState
    private let tone: MTLRenderPipelineState
    private let bloom: Bloom
    private var light: MTLTexture?

    private var setupIndex: Int
    private var simTime: Float = 0
    private var carry: Float = 0
    private var nextPacket: Float = 0
    private var packets = 0
    private var count = 0
    private var cleared: Float = -100
    private var lastClick: Float = -1
    private var frameSim: Float = 0
    private var rng = SystemRandomNumberGenerator()

    var scale: Float { 1 }
    var setup: Setup { Self.setups[setupIndex] }

    var facts: String? {
        let s = setup
        let lambda = 2 * Float.pi / s.k
        let what = switch s.slits {
        case 1: String(format: "one slit %.1f wavelengths wide", s.width / lambda)
        case 2: String(format: "two slits %.1f wavelengths apart", s.spacing / lambda)
        default: String(format: "three slits %.1f wavelengths apart", s.spacing / lambda)
        }
        return "Electron wave packet through \(what); \(count.formatted()) detection\(count == 1 ? "" : "s") so far"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        stepRe = try gpu.kernel(library, "slitStepRe")
        stepIm = try gpu.kernel(library, "slitStepIm")
        launch = try gpu.kernel(library, "slitLaunch")
        probe = try gpu.kernel(library, "slitProbe")
        field = try gpu.pipeline(library, "slitField")
        dots = try gpu.pipeline(library, "sprite", vertex: "slitHit", blend: .add)
        tone = try gpu.pipeline(library, "slitTone")
        bloom = try Bloom(gpu: gpu)
        re = gpu.buffer([Float](repeating: 0, count: Self.nx * Self.ny))
        im = gpu.buffer([Float](repeating: 0, count: Self.nx * Self.ny))
        column = gpu.buffer([Float](repeating: 0, count: Self.ny))
        hits = gpu.buffer([SIMD4<Float>](repeating: SIMD4(0, 0, -1000, 0), count: capacity))
        setupIndex = (Snapshot.variant ?? Int.random(in: 0..<Self.setups.count)) % Self.setups.count
    }

    /// The first grid column on screen: the detector plate takes about a fifth of the width, the tank the rest.
    private func left(_ f: Frame) -> Float {
        let cell = f.size.y / Float(Self.ny)
        let plate = max(0.18 * f.size.x, 0.2 * f.size.y)
        return max(Self.screen - (f.size.x - plate) / cell, 150)
    }

    /// Simulation constants for the kernels: a = (dt, k, energy, damping), b = barrier (x, half thickness, height, slits),
    /// c = slits (spacing, half width, centre y), d = packet (x, y, sigma x, sigma y).
    private func simUniforms(_ f: Frame) -> U {
        let s = setup
        let energy = 1 - cos(s.k)
        return U(res: SIMD2(Float(Self.nx), Float(Self.ny)), time: simTime, dt: Self.step, frame: f.index, seed: f.seed,
                 a: SIMD4(Self.step, s.k, energy, 0.35),
                 b: SIMD4(Self.barrier, 3, 2, Float(s.slits)),
                 c: SIMD4(s.spacing, s.width / 2, Float(Self.ny) / 2, Self.screen),
                 d: SIMD4(left(f) - 72, Float(Self.ny) / 2, Self.sigma.x, Self.sigma.y))
    }

    func draw(_ f: Frame) {
        if light?.width != f.target.width || light?.height != f.target.height { light = f.gpu.texture(f.target.width, f.target.height) }
        if f.index > 0 { detect(f) }

        if simTime >= nextPacket {
            if packets >= Self.packetsPerSetup {
                packets = 0
                setupIndex = (setupIndex + 1) % Self.setups.count
                count = 0
                cleared = f.time
            }
            f.cmd.compute(launch, Self.nx, Self.ny, uniforms: simUniforms(f), buffers: [re, im])
            packets += 1
            // The next electron leaves as this one's centre reaches the screen.
            nextPacket = simTime + (Self.screen - left(f) + 72) / sin(setup.k) * 0.9 + 100
        }

        carry += f.dt * Self.pace / Self.step
        var steps = min(Int(carry), 40)
        carry -= Float(steps)
        // Open with the first electron already on its way to the slits.
        if f.index == 0 { steps += 1200 }
        frameSim = Float(steps) * Self.step
        for _ in 0..<steps {
            let u = simUniforms(f)
            f.cmd.compute(stepRe, Self.nx, Self.ny, uniforms: u, buffers: [re, im])
            f.cmd.compute(stepIm, Self.nx, Self.ny, uniforms: u, buffers: [re, im])
            simTime += Self.step
        }
        f.cmd.compute(probe, Self.ny, uniforms: simUniforms(f), buffers: [re, im, column])

        var u = simUniforms(f)
        u.res = f.size
        u.time = f.time
        u.pad = SIMD2(simTime, max(f.time - cleared, 0))
        u.d = SIMD4(left(f), 0, 0, 0)
        f.cmd.draw(field, into: light!, uniforms: u, buffers: [re, im, column])
        f.cmd.draw(dots, into: light!, clear: false, uniforms: u, buffers: [hits], vertices: capacity, primitive: .point)
        let glow = bloom.glow(f.cmd, light!)
        f.cmd.draw(tone, into: f.target, uniforms: u, textures: [light!, glow])

        if let synth = f.sound {
            let root = 98 * setup.k / 0.6
            synth.drone(0, root, 0.07, pan: -0.2)
            synth.drone(1, root * 1.5, 0.035, pan: 0.2)
            synth.drone(2, root * 2, 0.015)
            synth.noise(0.02, cutoff: 260)
        }
    }

    /// Turns the probability arriving at the screen during the last frame into individual hits.
    private func detect(_ f: Frame) {
        let p = column.contents().assumingMemoryBound(to: Float.self)
        var total: Float = 0
        for y in 0..<Self.ny { total += max(p[y], 0) }
        guard total.isFinite, total > 0 else { return }
        // Each packet carries 2π σx σy of probability, about two thirds of what lines up with the slits reaches the
        // screen, and each packet should make about 150 hits whatever the slits.
        let norm = 2 * Float.pi * Self.sigma.x * Self.sigma.y
        let mean = total * sin(setup.k) * frameSim / norm * 150 / (0.65 * setup.opening(Self.sigma.y))
        let n = min(poisson(mean), 40)
        guard n > 0 else { return }
        let h = hits.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        var lastY: Float = 0.5
        for _ in 0..<n {
            var r = Float.random(in: 0..<total, using: &rng)
            var y = 0
            while y < Self.ny - 1, r > max(p[y], 0) { r -= max(p[y], 0); y += 1 }
            let v = (Float(y) + Float.random(in: 0..<1, using: &rng)) / Float(Self.ny)
            h[count % capacity] = SIMD4(Float.random(in: 0..<1, using: &rng), v, f.time, Float.random(in: 0.7..<1.3, using: &rng))
            count += 1
            lastY = v
        }
        if let synth = f.sound, f.time - lastClick > 0.2 {
            lastClick = f.time
            // A pentatonic scale over two octaves, high at the top of the screen.
            let scale: [Float] = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24]
            let step = scale[min(Int((1 - lastY) * Float(scale.count)), scale.count - 1)]
            synth.play(.click, 440 * pow(2, step / 12), amp: 0.045, pan: 0.55, length: 0.06)
        }
    }

    private func poisson(_ mean: Float) -> Int {
        guard mean > 0 else { return 0 }
        let limit = exp(-min(mean, 30))
        var k = 0, p: Float = 1
        repeat { k += 1; p *= Float.random(in: 0..<1, using: &rng) } while p > limit && k < 60
        return k - 1
    }

    private static let source = """
    constant int slitW = \(nx), slitH = \(ny);

    // Signed distance from grid point c to the barrier: a wall at x = b.x, b.y thick on each side, with b.w slits.
    inline float slitSDF(float2 c, constant U &u) {
        float d = abs(c.x - u.b.x) - u.b.y;
        int n = int(u.b.w);
        for (int i = 0; i < n; i++) {
            float yc = u.c.z + (float(i) - 0.5 * float(n - 1)) * u.c.x;
            d = max(d, u.c.y - abs(c.y - yc));
        }
        return d;
    }

    // Absorbing layers on every edge, and behind the screen, so waves leave instead of reflecting.
    inline float slitDamp(int x, int y, constant U &u) {
        float s = max(1 - float(x) / 24, (float(x) - u.c.w) / (u.res.x - u.c.w));
        s = clamp(max(s, max(1 - float(y) / 24, (float(y) - (u.res.y - 25)) / 24)), 0.0, 1.0);
        return 1 - u.a.w * u.a.x * s * s;
    }

    // Visscher: Re advances by dt·H·Im, then Im by −dt·H·Re, with H = −½∇² + V on the grid.
    kernel void slitStepRe(constant U &u [[buffer(0)]], device float *re [[buffer(1)]], device const float *im [[buffer(2)]],
                           uint2 g [[thread_position_in_grid]]) {
        int w = int(u.res.x), h = int(u.res.y), x = g.x, y = g.y, i = y * w + x;
        float c = im[i];
        float l = x > 0 ? im[i - 1] : 0, r = x < w - 1 ? im[i + 1] : 0;
        float t = y > 0 ? im[i - w] : 0, b = y < h - 1 ? im[i + w] : 0;
        float v = slitSDF(float2(x, y), u) < 0 ? u.b.z : 0;
        float hpsi = -0.5 * (l + r + t + b - 4 * c) + v * c;
        re[i] = (re[i] + u.a.x * hpsi) * slitDamp(x, y, u);
    }

    kernel void slitStepIm(constant U &u [[buffer(0)]], device const float *re [[buffer(1)]], device float *im [[buffer(2)]],
                           uint2 g [[thread_position_in_grid]]) {
        int w = int(u.res.x), h = int(u.res.y), x = g.x, y = g.y, i = y * w + x;
        float c = re[i];
        float l = x > 0 ? re[i - 1] : 0, r = x < w - 1 ? re[i + 1] : 0;
        float t = y > 0 ? re[i - w] : 0, b = y < h - 1 ? re[i + w] : 0;
        float v = slitSDF(float2(x, y), u) < 0 ? u.b.z : 0;
        float hpsi = -0.5 * (l + r + t + b - 4 * c) + v * c;
        im[i] = (im[i] - u.a.x * hpsi) * slitDamp(x, y, u);
    }

    // Adds a Gaussian packet moving right with wavenumber a.y; Im is half a step ahead, as the leapfrog wants.
    kernel void slitLaunch(constant U &u [[buffer(0)]], device float *re [[buffer(1)]], device float *im [[buffer(2)]],
                           uint2 g [[thread_position_in_grid]]) {
        int i = g.y * int(u.res.x) + g.x;
        float2 d = (float2(g) - u.d.xy) / u.d.zw;
        float env = exp(-0.25 * dot(d, d));
        float ph = u.a.y * (float(g.x) - u.d.x);
        re[i] += env * cos(ph);
        im[i] += env * sin(ph - u.a.z * u.a.x * 0.5);
    }

    // Probability density at the screen, row by row, for the CPU to turn into detections.
    kernel void slitProbe(constant U &u [[buffer(0)]], device const float *re [[buffer(1)]], device const float *im [[buffer(2)]],
                          device float *col [[buffer(3)]], uint y [[thread_position_in_grid]]) {
        int w = int(u.res.x);
        float s = 0;
        for (int k = 1; k <= 4; k++) {
            int i = int(y) * w + int(u.c.w) - k;
            s += re[i] * re[i] + im[i] * im[i];
        }
        col[y] = s / 4;
    }

    inline float3 oklab(float3 c) {
        float l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
        float m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
        float s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
        l = l * l * l; m = m * m * m; s = s * s * s;
        return float3(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
    }

    // A hue wheel of equal perceived lightness, so brightness alone shows |ψ|².
    inline float3 phaseHue(float a) { return max(oklab(float3(0.72, 0.17 * cos(a), 0.17 * sin(a))), 0.0); }

    inline float2 psiAt(const device float *re, const device float *im, float2 q, int w, int h) {
        float2 f = q - 0.5;
        int2 i = int2(floor(f));
        float2 t = f - float2(i);
        int x0 = clamp(i.x, 0, w - 1), x1 = clamp(i.x + 1, 0, w - 1), y0 = clamp(i.y, 0, h - 1), y1 = clamp(i.y + 1, 0, h - 1);
        float2 a = float2(re[y0 * w + x0], im[y0 * w + x0]), b = float2(re[y0 * w + x1], im[y0 * w + x1]);
        float2 c = float2(re[y1 * w + x0], im[y1 * w + x0]), d = float2(re[y1 * w + x1], im[y1 * w + x1]);
        return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
    }

    // The wave tank, the barrier and the dark detector plate. pad.x is simulation time, d.x the first grid column shown.
    fragment float4 slitField(V in [[stage_in]], constant U &u [[buffer(0)]], const device float *re [[buffer(1)]],
                              const device float *im [[buffer(2)]], const device float *col [[buffer(3)]]) {
        int w = slitW, h = slitH;
        float cell = float(h) / u.res.y;
        float2 q = float2(in.uv.x * u.res.x * cell + u.d.x, in.uv.y * float(h));
        float3 c = float3(0.004, 0.005, 0.009) * (1.2 - 0.6 * abs(in.uv.y - 0.5));
        if (q.x < u.c.w) {
            float2 psi = psiAt(re, im, q, w, h);
            float rho = dot(psi, psi);
            // Hue is the phase relative to the incoming plane wave, so each diffracted direction and each fringe takes its
            // own colour; brightness ripples with the full phase, so the wavefronts still show as crests.
            float full = atan2(psi.y, psi.x) + u.a.z * u.pad.x;
            float a = full - u.a.y * q.x + u.time * 0.12;
            float b = 1 - exp(-pow(rho, 0.6) * 2.5);
            float crest = 0.6 + 0.4 * cos(full - u.a.z * u.pad.x * 0.2);
            c += phaseHue(a) * b * crest * 1.2 + float3(0.9, 0.95, 1.0) * max(rho - 1.5, 0.0) * 0.05;
            float sd = slitSDF(q, u);
            float edge = smoothstep(cell * 1.5, -cell * 0.5, sd);
            float rim = exp(-abs(sd) / (cell * 1.2));
            float3 wall = float3(0.03, 0.033, 0.042) * (0.8 + 0.4 * hash12(floor(q * 2))) + float3(0.16, 0.17, 0.2) * rim * 0.5
                        + phaseHue(a) * b * rim * 0.6;
            c = mix(c, wall, edge);
            // The plate glows where the wave lands.
            float land = col[clamp(int(q.y), 0, h - 1)];
            c += float3(0.6, 0.8, 1.0) * pow(land, 0.5) * 2.5 * exp(-(u.c.w - q.x) / 2.5);
        } else {
            float px = (q.x - u.c.w) / cell;
            float n = hash12(floor(in.pos.xy * 0.5)) * 0.004;
            c = float3(0.010, 0.012, 0.016) + n;
            float land = col[clamp(int(q.y), 0, h - 1)];
            c += float3(0.6, 0.8, 1.0) * (0.02 + pow(land, 0.5) * 2.0) * exp(-px / 3);
        }
        return float4(c, 1);
    }

    // Each detection is a dot on the plate that flashes as it lands and then stays. pad.y is time since the plate was wiped.
    vertex P slitHit(uint id [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *hits [[buffer(1)]]) {
        float4 hit = hits[id];
        float cell = float(slitH) / u.res.y;
        float x0 = (u.c.w - u.d.x) / cell + 10, x1 = u.res.x - 10;
        float2 px = float2(mix(x0, x1, hit.x), hit.y * u.res.y);
        float age = u.time - hit.z;
        float fresh = exp(-age * 2.5);
        float wipe = age > u.pad.y ? clamp(1 - u.pad.y / 2.5, 0.0, 1.0) : 1.0;
        P o;
        o.pos = float4(px / u.res * float2(2, -2) + float2(-1, 1), 0, 1);
        o.size = (4 + 9 * fresh) * u.res.y / 1000;
        float3 tint = mix(float3(1.0, 0.88, 0.7) * 1.1, float3(0.55, 0.85, 1.0) * 5, fresh) * hit.w;
        o.color = float4(tint * wipe * (age >= 0 ? 1.0 : 0.0), 0);
        return o;
    }

    fragment float4 slitTone(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                             texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = light.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.35;
        return float4(film(c), 1);
    }
    """
}
