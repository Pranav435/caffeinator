import Foundation
import AppKit
import Metal

/// Exact hydrogen wavefunctions ψ_nlm = R_nl(r) Y_lm(θ, φ) in atomic units (lengths in Bohr radii).
struct Orbital: Equatable {
    let n: Int, l: Int, m: Int

    static let letters = Array("spdfghik")
    /// Hydrogen's ionisation energy in eV (the Rydberg energy corrected for the proton's finite mass).
    static let rydberg = 13.598434
    var energy: Double { -Self.rydberg / Double(n * n) }
    var label: String { "\(n)\(Self.letters[l])" }

    /// Generalised Laguerre polynomial L_k^α(x) by its three-term recurrence.
    static func laguerre(_ k: Int, _ alpha: Double, _ x: Double) -> Double {
        if k == 0 { return 1 }
        var a = 1.0, b = 1 + alpha - x
        for i in stride(from: 1, to: k, by: 1) {
            let c = ((Double(2 * i + 1) + alpha - x) * b - (Double(i) + alpha) * a) / Double(i + 1)
            a = b
            b = c
        }
        return b
    }

    /// Associated Legendre function P_l^m(x) for m ≥ 0, with the Condon–Shortley phase.
    static func legendre(_ l: Int, _ m: Int, _ x: Double) -> Double {
        var pmm = 1.0
        let s = (1 - x * x).squareRoot()
        for i in 0..<m { pmm *= -Double(2 * i + 1) * s }
        if l == m { return pmm }
        var pm1 = x * Double(2 * m + 1) * pmm
        if l == m + 1 { return pm1 }
        for ll in (m + 2)...l {
            let p = (Double(2 * ll - 1) * x * pm1 - Double(ll + m - 1) * pmm) / Double(ll - m)
            pmm = pm1
            pm1 = p
        }
        return pm1
    }

    static func factorial(_ k: Int) -> Double { (0..<k).reduce(1.0) { $0 * Double($1 + 1) } }

    /// Normalisation of the radial part, so that ∫ R² r² dr = 1.
    var radialNorm: Double {
        let nn = Double(n)
        return (pow(2 / nn, 3) * Self.factorial(n - l - 1) / (2 * nn * Self.factorial(n + l))).squareRoot()
    }

    /// Normalisation of the angular part, so that ∫ |Y|² dΩ = 1.
    var angularNorm: Double {
        let am = abs(m)
        return (Double(2 * l + 1) / (4 * .pi) * Self.factorial(l - am) / Self.factorial(l + am)).squareRoot()
    }

    func radial(_ r: Double) -> Double {
        let rho = 2 * r / Double(n)
        return radialNorm * pow(rho, Double(l)) * exp(-rho / 2) * Self.laguerre(n - l - 1, Double(2 * l + 1), rho)
    }

    /// |Y_lm| as a function of cos θ (the phase e^{imφ} has modulus one).
    func angular(_ cosTheta: Double) -> Double { angularNorm * abs(Self.legendre(l, abs(m), cosTheta)) }

    /// ψ at a point in Bohr radii, as (real, imaginary).
    func value(_ x: Double, _ y: Double, _ z: Double) -> (Double, Double) {
        let r = (x * x + y * y + z * z).squareRoot()
        let mag = radial(r) * angularNorm * Self.legendre(l, abs(m), r > 0 ? z / r : 1)
        let phase = Double(m) * atan2(y, x)
        return (mag * cos(phase), mag * sin(phase))
    }

    /// The largest |ψ| anywhere, and the radius beyond which |ψ|² stays below `floor` times its peak.
    func extent(floor: Double = 2e-4) -> (peak: Double, radius: Double) {
        let ang = stride(from: -1.0, through: 1.0, by: 1.0 / 256).map { angular($0) }.max() ?? 1
        let top = Double(n * n) * 4 + 30
        var peak = 0.0
        let samples = stride(from: 0.0, through: top, by: top / 4000).map { abs(radial($0)) * ang }
        for v in samples { peak = max(peak, v) }
        var radius = top
        for (i, v) in samples.enumerated().reversed() where v * v > floor * peak * peak {
            radius = Double(i) * top / 4000
            break
        }
        return (peak, radius)
    }
}

/// Hydrogen orbitals as glowing nebulae: the exact ψ is baked into a 3D texture and ray-marched, with the phase of ψ
/// as colour, and 25,000 motes of electron dust drawn from |ψ|² drift along the probability current
/// (de Broglie–Bohm guidance), so they swirl round the axis when m ≠ 0 and follow the charge in superpositions,
/// which slosh at their energy difference slowed ten quadrillion times.
@MainActor final class Hydrogen: Scene {
    struct State {
        let a: Orbital
        let b: Orbital?

        init(_ n: Int, _ l: Int, _ m: Int) { a = Orbital(n: n, l: l, m: m); b = nil }
        init(_ a: (Int, Int, Int), _ b: (Int, Int, Int)) {
            self.a = Orbital(n: a.0, l: a.1, m: a.2)
            self.b = Orbital(n: b.0, l: b.1, m: b.2)
        }

        /// Everything the GPU needs to bake and draw this state, worked out off the main thread.
        struct Baked {
            /// Weights of the two components after scaling each to peak at one, so |ψ| peaks near one overall.
            let ampA: Double, ampB: Double
            /// Box radius in Bohr radii, the glow exponent, and each component's scale to peak at one.
            let radius: Double, gamma: Float, scaleA: Double, scaleB: Double
        }

        func baked() -> Baked {
            let ea = a.extent()
            let eb = b?.extent()
            let w: Double = b == nil ? 1 : 1 / 2.squareRoot()
            let norm = w * ea.peak + w * (eb?.peak ?? 0)
            let ampA = w * ea.peak / norm, ampB = w * (eb?.peak ?? 0) / norm
            let pa = ea.peak, pb = eb?.peak ?? 1
            let gamma = exposure(radius: max(ea.radius, eb?.radius ?? 0), ampA: ampA / pa, ampB: ampB / pb)
            // Size the box to where the glow fades below a few thousandths of its peak.
            let floor = pow(0.003, 1 / Double(gamma))
            var radius = a.extent(floor: min(floor / (ampA * ampA), 0.5)).radius
            if let b { radius = max(radius, b.extent(floor: min(floor / (ampB * ampB), 0.5)).radius) }
            return Baked(ampA: ampA, ampB: ampB, radius: radius * 1.1, gamma: gamma,
                         scaleA: a.radialNorm * a.angularNorm / pa, scaleB: (b ?? a).radialNorm * (b ?? a).angularNorm / (eb?.peak ?? pa))
        }

        /// The exponent that maps density to glow: chosen so the density enclosing half the probability glows at about
        /// a third of the peak, which keeps both a bright 1s core and a faint wide shell visible. `ampA` and `ampB`
        /// weight the raw wavefunctions so the combined |ψ|² peaks near one.
        func exposure(radius: Double, ampA: Double, ampB: Double) -> Float {
            var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
            func next() -> Double {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Double(seed >> 11) / Double(1 << 53) * 2 - 1
            }
            var rhos: [Double] = []
            rhos.reserveCapacity(12_000)
            while rhos.count < 12_000 {
                let (x, y, z) = (next(), next(), next())
                guard x * x + y * y + z * z <= 1 else { continue }
                let va = a.value(x * radius, y * radius, z * radius)
                let vb = b?.value(x * radius, y * radius, z * radius) ?? (0, 0)
                let re = ampA * va.0 + ampB * vb.0, im = ampA * va.1 + ampB * vb.1
                rhos.append(re * re + im * im)
            }
            rhos.sort(by: >)
            let total = rhos.reduce(0, +)
            var sum = 0.0
            for r in rhos {
                sum += r
                if sum >= total / 2 { return Float(max(0.3, min(1, log(0.3) / log(max(r, 1e-9))))) }
            }
            return 1
        }
    }

    static let states: [State] = [
        State(1, 0, 0), State(2, 1, 0), State((1, 0, 0), (2, 1, 0)), State(2, 1, 1), State(3, 2, 0),
        State((1, 0, 0), (2, 1, 1)), State(3, 0, 0), State(3, 2, 1), State((2, 1, 0), (3, 2, 0)), State(4, 3, 0),
        State(3, 2, 2), State(4, 3, 1), State((2, 1, 1), (3, 2, 2)), State(4, 3, 3), State(5, 4, 2),
        State((2, 1, 0), (4, 2, 0)), State(6, 5, 1), State(5, 2, 1), State(4, 1, 0), State((1, 0, 0), (3, 1, 0)),
    ]

    /// Real seconds are stretched 10^16 times, so a 2.5 petahertz Lyman alpha beat takes four seconds.
    static let slowdown = 1e16
    /// Atomic units of time (ħ / hartree, 2.4189 × 10⁻¹⁷ s) per second on screen.
    static let atomicTime = 1 / (slowdown * 2.4188843e-17)
    static let side = 128
    static let hold: Float = 40
    static let motes = 24_576

    private let fill: MTLComputePipelineState
    private let flow: MTLComputePipelineState
    private let march: MTLRenderPipelineState
    private let dots: MTLRenderPipelineState
    private let lens: Lens
    private let strip: MTLRenderPipelineState
    private var labelled = -2
    private var labelledSize = CGSize.zero
    private let dust: MTLBuffer
    private var volumes: [MTLTexture]
    private var light: MTLTexture?
    private var current = 0
    private var stateIndex: Int
    private var since: Float = 0
    private var filled = false
    private var switched = false
    /// Per slot: amplitudes of the two components (scaled so |ψ| peaks near one), the relative phase, the box radius.
    private var params: [SIMD4<Float>] = [.zero, .zero]
    private var rates: [Float] = [0, 0]
    private var gammas: [Float] = [1, 1]
    private var cache: [Int: State.Baked] = [:]
    private var lastBeat = 0

    var scale: Float { 0.5 }

    var facts: String? {
        let s = Self.states[stateIndex]
        guard let b = s.b else {
            let o = s.a
            return "Hydrogen \(o.label) orbital, n = \(o.n), l = \(o.l), m = \(o.m); energy \(Self.eV(o.energy))"
        }
        let line = Self.line(s.a.n, b.n)
        let thz = Int((Self.frequency(s.a, b) / 1e12).rounded()).formatted()
        let motion = b.m != s.a.m ? "circles the nucleus" : "sloshes back and forth"
        let nm = 1e9 * 299_792_458 / Self.frequency(s.a, b)
        return "Superposition of \(s.a.label) and \(b.label): the charge \(motion) at the \(line) frequency, \(thz) terahertz (\(Int(nm.rounded())) nanometres), shown 10 quadrillion times slower"
    }

    nonisolated static func eV(_ e: Double) -> String {
        "−" + String(format: abs(e) >= 10 ? "%.1f eV" : "%.2f eV", abs(e))
    }

    nonisolated static func frequency(_ a: Orbital, _ b: Orbital) -> Double {
        abs(a.energy - b.energy) * 2.417989e14 // hertz per electronvolt
    }

    nonisolated static func line(_ lo: Int, _ hi: Int) -> String {
        let series = ["Lyman", "Balmer", "Paschen", "Brackett", "Pfund"]
        let greek = ["alpha", "beta", "gamma", "delta", "epsilon"]
        let (a, b) = (min(lo, hi), max(lo, hi))
        guard a >= 1, a <= series.count, b - a - 1 < greek.count else { return "spectral line" }
        return "\(series[a - 1]) \(greek[b - a - 1])"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        fill = try gpu.kernel(library, "orbitalFill")
        flow = try gpu.kernel(library, "orbitalDust")
        march = try gpu.pipeline(library, "orbitalMarch")
        dots = try gpu.pipeline(library, "sprite", vertex: "orbitalMote", blend: .add)
        lens = try Lens(gpu: gpu)
        strip = try gpu.pipeline(library, "orbitalStrip", blend: .add)
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = .rgba16Float
        d.width = Self.side
        d.height = Self.side
        d.depth = Self.side
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        guard let v0 = gpu.device.makeTexture(descriptor: d), let v1 = gpu.device.makeTexture(descriptor: d) else {
            throw SceneError("No 3D textures")
        }
        volumes = [v0, v1]
        // Each mote: position in the unit ball and age, then lifetime, phase (cos, sin) and |ψ|. All start due a resample.
        dust = gpu.buffer([SIMD4<Float>](repeating: SIMD4(0, 0, 0, 1), count: Self.motes * 2))
        stateIndex = (Snapshot.variant ?? Int.random(in: 0..<Self.states.count)) % Self.states.count
    }

    /// Works out the next state's parameters in the background, so changing state never stalls a frame.
    private func prefetch(_ index: Int) {
        guard cache[index] == nil else { return }
        let state = Self.states[index]
        Task.detached(priority: .utility) { [weak self] in
            let baked = state.baked()
            await MainActor.run { self?.cache[index] = baked }
        }
    }

    /// Bakes a state into volume `slot`: ψ_a in rg and ψ_b in ba, each scaled to peak at one.
    private func bake(_ index: Int, into slot: Int, _ f: Frame) {
        let s = Self.states[index]
        let k = cache[index] ?? s.baked()
        cache[index] = k
        let b = s.b ?? s.a
        var u = f.uniforms([Float(s.a.n), Float(s.a.l), Float(s.a.m), Float(k.scaleA)],
                           [Float(b.n), Float(b.l), Float(b.m), Float(k.scaleB)],
                           [Float(k.radius), Float(Self.side), 0, 0])
        params[slot] = SIMD4(Float(k.ampA), Float(k.ampB), 0, Float(k.radius))
        gammas[slot] = k.gamma
        rates[slot] = s.b.map { Float(Self.frequency(s.a, $0) / Self.slowdown) * 2 * .pi } ?? 0
        guard let e = f.cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(fill)
        e.setBytes(&u, length: MemoryLayout<U>.stride, index: 0)
        e.setTexture(volumes[slot], index: 0)
        e.dispatchThreads(MTLSize(width: Self.side, height: Self.side, depth: Self.side),
                          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        e.endEncoding()
        prefetch((index + 1) % Self.states.count)
    }

    func draw(_ f: Frame) {
        if light?.width != f.target.width || light?.height != f.target.height { light = f.gpu.texture(f.target.width, f.target.height) }
        since += f.dt
        if !filled {
            bake(stateIndex, into: current, f)
            filled = true
            since = 4
        } else if since > Self.hold {
            let previous = Self.states[stateIndex]
            stateIndex = (stateIndex + 1) % Self.states.count
            current = 1 - current
            bake(stateIndex, into: current, f)
            since = 0
            switched = true
            if let synth = f.sound { announce(synth, from: previous, to: Self.states[stateIndex]) }
        }
        let s = Self.states[stateIndex]
        for i in 0..<2 { params[i].z = (params[i].z + rates[i] * f.dt).truncatingRemainder(dividingBy: 2 * .pi * 1000) }
        let fade = min(since / 4, 1)
        let eased = fade * fade * (3 - 2 * fade)

        let angle = f.time * 0.06 + f.seed
        let lift = 0.45 + 0.3 * sin(f.time * 0.021 + f.seed)
        let eye = SIMD3<Float>(cos(angle) * cos(lift), sin(lift), sin(angle) * cos(lift)) * 3.3
        let drift = SIMD2<Float>(0.08 * sin(f.time * 0.013), 0.06 * sin(f.time * 0.017 + 1))
        let u = f.uniforms(params[current], params[1 - current], SIMD4(eye, eased),
                           SIMD4(drift.x, drift.y, 3.3, gammas[1 - current] + (gammas[current] - gammas[1 - current]) * eased))

        var m = u
        m.b = SIMD4(f.dt * Float(Self.atomicTime), f.dt, switched ? 1 : 0, Float(f.index))
        f.cmd.compute(flow, Self.motes, uniforms: m, textures: [volumes[current]], buffers: [dust])
        switched = false

        f.cmd.draw(march, into: light!, uniforms: u, textures: [volumes[current], volumes[1 - current]])
        f.cmd.draw(dots, into: light!, clear: false, uniforms: u, buffers: [dust], vertices: Self.motes, primitive: .point)
        let line = Self.balmerLine(s)
        f.cmd.draw(strip, into: light!, clear: false, uniforms: f.uniforms(SIMD4(Float(line), 0, 0, 0)))
        labelSpectrum(f, line)
        lens.develop(f, light!, Lens.Look(exposure: 1.0, white: [0.97, 1.0, 0.99], bloom: 0.08, halation: 0.01, vignette: 0.5,
                                          aberration: 1.0, grain: 0.8))

        if let synth = f.sound { play(synth, s, params[current].z) }
    }

    static let balmer: [(nm: Double, name: String)] = [(656.3, "Hα"), (486.1, "Hβ"), (434.0, "Hγ"), (410.2, "Hδ")]

    /// Which Balmer line (0 to 3) the state's transition or decay goes through, or -1.
    nonisolated static func balmerLine(_ s: State) -> Int {
        let hi: Int
        if let b = s.b { guard min(s.a.n, b.n) == 2 else { return -1 }; hi = max(s.a.n, b.n) } else { hi = s.a.n }
        return (3...6).contains(hi) ? hi - 3 : -1
    }

    /// Small captions under the spectrum strip, redrawn only when the highlighted line or the display changes.
    private func labelSpectrum(_ f: Frame, _ line: Int) {
        let size = f.overlay.size
        guard line != labelled || size != labelledSize else { return }
        labelled = line
        labelledSize = size
        f.overlay.draw { ctx, size in
            for (i, l) in Self.balmer.enumerated() {
                let x = (0.27 + 0.46 * (l.nm - 400) / 300) * size.width
                let hot = i == line
                ctx.text(String(format: "%.1f", l.nm), at: CGPoint(x: x, y: 0.958 * size.height), size: 9, weight: hot ? .semibold : .regular,
                         mono: true, color: NSColor(white: 1, alpha: hot ? 0.75 : 0.4), anchor: CGPoint(x: 0.5, y: 0))
            }
            ctx.text("H I  BALMER SERIES  nm", at: CGPoint(x: 0.27 * size.width, y: 0.9 * size.height), size: 9, mono: true,
                     color: NSColor(white: 1, alpha: 0.35), anchor: CGPoint(x: 0, y: 1), tracking: 1)
        }
    }

    /// Folds a frequency by octaves into [lo, 2 lo).
    nonisolated static func fold(_ f: Double, _ lo: Double) -> Float {
        var f = f
        while f >= 2 * lo { f /= 2 }
        while f < lo { f *= 2 }
        return Float(f)
    }

    /// The drone sits at 1/n² of each level (the Rydberg ratios), folded into one octave; superpositions sound both
    /// levels and ring their spectral line once per slosh.
    private func play(_ synth: Synth, _ s: State, _ beat: Float) {
        let rootA = Self.fold(1760 / Double(s.a.n * s.a.n), 98)
        synth.drone(0, rootA, 0.09, pan: -0.25)
        synth.drone(1, rootA * 2, 0.03, pan: 0.25)
        synth.drone(2, rootA * 1.5, 0.025)
        if let b = s.b {
            let rootB = Self.fold(1760 / Double(b.n * b.n), 98)
            let swell = 0.5 + 0.5 * cos(beat)
            synth.drone(3, rootB * 2, 0.03 + 0.03 * swell, pan: 0.4)
            let count = Int(beat / (2 * .pi))
            if count != lastBeat {
                lastBeat = count
                synth.play(.bell, Self.lineTone(s.a.n, b.n), amp: 0.05, pan: -0.3, length: 3)
            }
        } else {
            synth.drone(3, rootA, 0)
            lastBeat = 0
        }
        synth.noise(0.015, cutoff: 220)
    }

    /// The photon's frequency from the Rydberg formula, scaled into hearing so Hα sounds G4 (392 Hz) and the other
    /// lines keep their true ratios, folded by octaves into 330 to 660 hertz.
    nonisolated static func lineTone(_ lo: Int, _ hi: Int) -> Float {
        let (a, b) = (Double(min(lo, hi)), Double(max(lo, hi)))
        return fold(392 * 36 / 5 * (1 / (a * a) - 1 / (b * b)), 330)
    }

    private func announce(_ synth: Synth, from: State, to: State) {
        let a = from.b?.n ?? from.a.n, b = to.a.n
        if a != b {
            synth.play(.bell, Self.lineTone(a, b), amp: 0.09, pan: 0, length: 4)
        } else {
            synth.play(.bell, Self.fold(1760 / Double(b * b), 196), amp: 0.05, pan: 0, length: 3)
        }
    }

    private static let source = """
    inline float sq(float x) { return x * x; }

    inline float laguerre(int k, float alpha, float x) {
        if (k == 0) return 1;
        float a = 1, b = 1 + alpha - x;
        for (int i = 1; i < k; i++) {
            float c = ((2 * i + 1 + alpha - x) * b - (i + alpha) * a) / float(i + 1);
            a = b; b = c;
        }
        return b;
    }

    inline float legendre(int l, int m, float x) {
        float pmm = 1, s = sqrt(max(1 - x * x, 0.0));
        for (int i = 0; i < m; i++) pmm *= -float(2 * i + 1) * s;
        if (l == m) return pmm;
        float pm1 = x * float(2 * m + 1) * pmm;
        if (l == m + 1) return pm1;
        for (int ll = m + 2; ll <= l; ll++) {
            float p = (float(2 * ll - 1) * x * pm1 - float(ll + m - 1) * pmm) / float(ll - m);
            pmm = pm1; pm1 = p;
        }
        return pm1;
    }

    // ψ_nlm at p (Bohr radii), times q.w, as a complex number. q = (n, l, m, scale).
    inline float2 psi(float3 p, float4 q) {
        int n = int(q.x), l = int(q.y), m = int(q.z);
        float r = length(p);
        float rho = 2 * r / float(n);
        float radial = pow(max(rho, 1e-6), float(l)) * exp(-rho / 2) * laguerre(n - l - 1, float(2 * l + 1), rho);
        float ct = r > 1e-6 ? p.z / r : 1;
        float phi = atan2(p.y, p.x) * float(m);
        return q.w * radial * legendre(l, abs(m), ct) * float2(cos(phi), sin(phi));
    }

    // a, b: the two orbitals. c.x: box radius in Bohr radii, c.y: texture side.
    kernel void orbitalFill(constant U &u [[buffer(0)]], texture3d<float, access::write> vol [[texture(0)]],
                            uint3 g [[thread_position_in_grid]]) {
        float3 p = ((float3(g) + 0.5) / u.c.y * 2 - 1) * u.c.x;
        vol.write(float4(psi(p, u.a), psi(p, u.b)), g);
    }

    // ψ at o in the unit ball for a baked state s = (amp a, amp b, relative phase, radius), scaled so |ψ| peaks near one.
    inline float2 field(texture3d<float> vol, float3 o, float4 s) {
        constexpr sampler lin(filter::linear, address::clamp_to_zero);
        float4 t = vol.sample(lin, o * 0.5 + 0.5);
        float2 e = float2(cos(s.z), -sin(s.z));
        return s.x * t.xy + s.y * float2(t.z * e.x - t.w * e.y, t.z * e.y + t.w * e.x);
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

    // Colour for a phase given as a unit vector (cos, sin), scaled by chroma: phase 0 is warm gold, π is blue.
    inline float3 phaseColor(float2 d, float chroma) {
        const float2x2 turn = float2x2(float2(0.4085, 0.9128), float2(-0.9128, 0.4085));
        float2 h = turn * d * chroma;
        return max(oklab(float3(0.76, h.x, h.y)), 0.0);
    }

    inline uint moteHash(uint i, uint f) { return pcg(i * 9781u + f * 6271u + 17u); }

    // Motes follow the de Broglie–Bohm velocity v = Im(∇ψ / ψ) in atomic units, which keeps them distributed as |ψ|².
    // Expired ones are redrawn from |ψ|² by rejection sampling. b = (atomic time step, seconds step, switched, frame).
    kernel void orbitalDust(constant U &u [[buffer(0)]], texture3d<float> vol [[texture(0)]], device float4 *dust [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
        float4 p = dust[2 * i], q = dust[2 * i + 1];
        uint s = moteHash(i, uint(u.b.w));
        if (u.b.z > 0) q.x = p.w + rnd(s) * 3.5;
        p.w += u.b.y;
        if (p.w > q.x) {
            q.w = 0;
            for (int k = 0; k < 40; k++) {
                float3 c = float3(rnd(s), rnd(s), rnd(s)) * 2 - 1;
                if (dot(c, c) > 1) continue;
                float2 w = field(vol, c, u.a);
                if (dot(w, w) > rnd(s)) {
                    p = float4(c, 0);
                    q = float4(3 + rnd(s) * 6, 0, 0, 1);
                    break;
                }
            }
            if (q.w == 0) { p.w = q.x; dust[2 * i] = p; dust[2 * i + 1] = q; return; }
        }
        float h = 1.0 / 128.0;
        float2 w = field(vol, p.xyz, u.a);
        float2 gx = field(vol, p.xyz + float3(h, 0, 0), u.a) - field(vol, p.xyz - float3(h, 0, 0), u.a);
        float2 gy = field(vol, p.xyz + float3(0, h, 0), u.a) - field(vol, p.xyz - float3(0, h, 0), u.a);
        float2 gz = field(vol, p.xyz + float3(0, 0, h), u.a) - field(vol, p.xyz - float3(0, 0, h), u.a);
        float rr = max(dot(w, w), 1e-6);
        // Im(conj(ψ) ∇ψ) / |ψ|², converted from texture units to Bohr radii, then back for the step.
        float3 v = float3(w.x * gx.y - w.y * gx.x, w.x * gy.y - w.y * gy.x, w.x * gz.y - w.y * gz.x) / (2 * h * rr);
        float3 step = v / (u.a.w * u.a.w) * u.b.x;
        float len = length(step);
        if (len > 0.02) step *= 0.02 / len;
        p.xyz += step;
        if (dot(p.xyz, p.xyz) > 1) p.w = q.x;
        float a = sqrt(rr);
        q.yz = w / a;
        q.w = a;
        dust[2 * i] = p;
        dust[2 * i + 1] = q;
    }

    inline float3 toWorld(float3 o) { return float3(o.x, o.z, -o.y); }

    // The reference spectrum of a hydrogen discharge tube: the four visible Balmer lines in their true colours, the
    // highlighted one (u.a.x, or -1) brighter. A thin strip along the bottom, drawn straight into the light.
    inline float3 wavelengthColor(float nm) {
        float3 c = float3(0);
        if (nm < 440) c = float3(-(nm - 440) / 60, 0, 1);
        else if (nm < 490) c = float3(0, (nm - 440) / 50, 1);
        else if (nm < 510) c = float3(0, 1, -(nm - 510) / 20);
        else if (nm < 580) c = float3((nm - 510) / 70, 1, 0);
        else if (nm < 645) c = float3(1, -(nm - 645) / 65, 0);
        else c = float3(1, 0, 0);
        float k = nm < 420 ? 0.3 + 0.7 * (nm - 380) / 40 : (nm > 700 ? 0.3 + 0.7 * (780 - nm) / 80 : 1.0);
        return pow(c * k, 2.2);
    }

    fragment float4 orbitalStrip(V in [[stage_in]], constant U &u [[buffer(0)]]) {
        float2 t = in.uv;
        float x = (t.x - 0.27) / 0.46;
        float y = (t.y - 0.915) / 0.03;
        if (x < -0.02 || x > 1.02 || y < -0.05 || y > 1.3) return float4(0);
        float3 c = 0;
        float nm = 400 + 300 * x;
        float pxnm = 300.0 / (0.46 * u.res.x);
        if (y >= 0 && y <= 1) {
            const float lines[4] = {656.3, 486.1, 434.0, 410.2};
            const float rel[4] = {1.0, 0.6, 0.45, 0.4};
            c = float3(0.004, 0.004, 0.005) + float3(0.012) * hash12(in.pos.xy);
            for (int i = 0; i < 4; i++) {
                float sigma = max(1.8, 1.4 * pxnm);
                float g = exp(-0.5 * sq((nm - lines[i]) / sigma));
                float boost = (int(u.a.x) == i) ? 2.6 : 1.0;
                c += wavelengthColor(lines[i]) * g * rel[i] * boost * 2.2;
            }
            float e = min(min(x, 1 - x) * 0.46 * u.res.x, min(y, 1 - y) * 0.03 * u.res.y);
            c += 0.03 * step(e, 1.0);
        } else if (y > 1 && y < 1.2) {
            // Wavelength ticks every 50 nm.
            float tick = abs(fract(nm / 50 + 0.5) - 0.5) * 50 / max(pxnm, 1e-3);
            c = float3(0.12) * step(tick, 0.8);
        }
        return float4(c, 1);
    }

    // Draws each mote as a soft point, fading in and out over its life. c: eye and fade, d: drift and zoom.
    vertex P orbitalMote(uint id [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *dust [[buffer(1)]]) {
        float4 p = dust[2 * id], q = dust[2 * id + 1];
        P o;
        o.pos = project(toWorld(p.xyz), u.c.xyz, float3(u.d.xy, 0), u.d.z, u.res.x / u.res.y);
        o.size = 2.6 * u.res.y / 600;
        float life = smoothstep(0.0, 0.8, p.w) * smoothstep(q.x, q.x - 1.2, p.w);
        float b = life * (q.w > 0 ? 1.0 : 0.0) * 0.6 * (0.3 + 0.7 * q.w);
        o.color = float4(float3(0.55, 1.0, 0.7) * b * 2.4, 0);
        float2 dp = o.pos.xy / o.pos.w * float2(u.res.x / u.res.y, 1) - float2(0, 0.06);
        if (length(dp) > 0.79) o.color = 0;
        return o;
    }

    // Emission and absorption through the cloud. Colour is carried as a chroma vector so phases blend smoothly.
    fragment float4 orbitalMarch(V in [[stage_in]], constant U &u [[buffer(0)]],
                                 texture3d<float> now [[texture(0)]], texture3d<float> before [[texture(1)]]) {
        float2 p = centered(in.uv, u.res);
        float3 eye = u.c.xyz;
        float3 rd = ray(p, eye, float3(u.d.xy, 0), u.d.z);
        float3 bg = float3(0.003, 0.005, 0.004) * (1 - 0.3 * dot(p, p));
        float2 hit = sphere(eye, rd, float3(0), 1);
        if (hit.y <= 0) return float4(bg, 1);
        const int steps = 96;
        float t0 = max(hit.x, 0.0), ds = (hit.y - t0) / float(steps);
        float jitter = fract(52.9829189 * fract(dot(in.pos.xy, float2(0.06711056, 0.00583715))));
        float glow = 0, hot = 0;
        float2 tint = 0;
        float trans = 1, fade = u.c.w;
        for (int i = 0; i < steps; i++) {
            float3 o = float3(1, -1, 1) * (eye + rd * (t0 + (float(i) + jitter) * ds)).xzy;
            float2 w = field(now, o, u.a) * fade;
            if (fade < 1) w += field(before, o, u.b) * (1 - fade);
            float len = length(w);
            float2 dir = w / max(len, 1e-6);
            float a = len * smoothstep(1.0, 0.82, length(o));
            float r = exp2(u.d.w * log2(max(a * a, 1e-12)));
            // Density glow, a faint halo for the outskirts, and a glassy sheen where |ψ|² crosses a third of its peak.
            float shell = exp(-sq((r - 0.3) * 9.0));
            float d = r * 1.6 + a * 0.12 + shell * 0.35;
            float e = trans * d * ds;
            glow += e;
            tint += e * dir;
            hot += trans * ds * (smoothstep(0.35, 1.0, r) + shell * 0.25);
            trans *= exp(-r * ds * 0.6);
            if (trans < 0.01) break;
        }
        // A detector image: the column density in phosphor green, with counting noise and a micro-channel-plate pore mesh,
        // inside a round active area with a machined bezel.
        float2 pg = in.pos.xy / max(u.res.y / 540, 1.0);
        float2 hex = float2(pg.x / 5.0, pg.y / 4.33 + 0.5 * floor(pg.x / 5.0));
        float2 cellp = fract(hex) - 0.5;
        float pore = 0.7 + 0.3 * smoothstep(0.5, 0.2, length(cellp * float2(1.0, 0.9)));
        float shot = 0.5 + 1.0 * hash12(floor(pg) + float2(u.frame % 7u, 0));
        float3 c = float3(0.55, 1.0, 0.7) * (glow * 0.9 + hot * 0.08) * pore * shot + float3(0.45, 1.0, 0.6) * 0.012 * pore;
        float2 dp = p - float2(0, 0.06);
        float dd = length(dp) - 0.8;
        if (dd > 0) {
            float ring = exp(-abs(dd - 0.012) / 0.006);
            float brushed = 0.6 + 0.4 * noise(float2(atan2(dp.y, dp.x) * 90.0, floor(dd * 40.0)));
            float side = 0.5 + 0.5 * dot(normalize(dp), normalize(float2(-0.6, 0.8)));
            c = float3(0.05, 0.052, 0.056) * brushed * (0.4 + side) + float3(0.5, 0.52, 0.55) * ring * 0.28 * side;
            trans = 0;
            bg = 0;
        }
        return float4(bg * trans + c, 1);
    }
    """
}
