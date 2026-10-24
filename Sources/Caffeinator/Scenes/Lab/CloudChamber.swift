import Foundation
import Metal
import simd

/// A diffusion cloud chamber seen from above, the view 12.5 cm tall. Tracks arrive at sea-level rates: cosmic-ray muons,
/// alphas from radon and its daughters, beta and Compton electrons, delta rays knocked out along the way, and now and
/// then an air shower. Each track is a line of alcohol droplets that condense within a fraction of a second, then
/// drift on the air, spread and evaporate.
@MainActor final class CloudChamber: Scene {
    enum Kind { case muon, alpha, electron, shower }

    /// Events per second for the whole field of view, given its area in square centimetres.
    struct Rates {
        let muons: Double, alphas: Double, electrons: Double, showers: Double
        init(area: Double) {
            muons = area / 60          // about one per square centimetre per minute at sea level
            alphas = 9.0 / 60          // radon-222, polonium-218 and polonium-214 in the chamber air
            electrons = 40.0 / 60      // beta decays and Compton electrons from background gamma rays
            showers = 0.3 / 60
        }
    }

    /// Radon chain alpha emitters: name, energy in MeV, range in air in centimetres, share of decays.
    static let alphaSources: [(String, Double, Double, Double)] = [
        ("radon 222", 5.49, 4.1, 0.4), ("polonium 218", 6.00, 4.7, 0.35), ("polonium 214", 7.69, 6.9, 0.25),
    ]

    /// Height of the view in millimetres, and the depth of the sensitive layer in which tracks form.
    static let height: Double = 125
    static let layer: Double = 40
    static let capacity = 1 << 18

    private let drops: MTLBuffer
    private var head = 0
    private let mist: MTLRenderPipelineState
    private let droplets: MTLRenderPipelineState
    private let rain: MTLRenderPipelineState
    private let tone: MTLRenderPipelineState
    private let bloom: Bloom
    private var light: MTLTexture?
    private var fog: MTLTexture?
    private var width: Double = 160
    private var started = false
    private var events: [(Kind, Float)] = []
    private var lastAlpha: (String, Double)?
    private var lastShower: Float = -1000
    private var now: Float = 0
    private var rng = SystemRandomNumberGenerator()
    private var clicks: [(Synth.Voice, Float, Float, Float, Float)] = []

    var scale: Float { 1 }

    var facts: String? {
        func count(_ k: Kind, _ name: String) -> String {
            let n = events.filter { $0.0 == k && now - $0.1 <= 60 }.count
            return "\(n.formatted()) \(name)\(n == 1 ? "" : "s")"
        }
        let window = now < 60 ? "So far" : "Last minute"
        var line = "\(window): \(count(.muon, "muon")), \(count(.alpha, "alpha")), \(count(.electron, "electron"))"
        if now - lastShower < 60 { line += ", and a cosmic-ray air shower" }
        if let (name, energy) = lastAlpha { line += "; last alpha " + String(format: "%.1f", energy) + " MeV from \(name)" }
        return line
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        mist = try gpu.pipeline(library, "chamberMist")
        droplets = try gpu.pipeline(library, "sprite", vertex: "chamberDrop", blend: .add)
        rain = try gpu.pipeline(library, "sprite", vertex: "chamberRain", blend: .add)
        tone = try gpu.pipeline(library, "chamberTone")
        bloom = try Bloom(gpu: gpu)
        drops = gpu.buffer([SIMD4<Float>](repeating: SIMD4(0, 0, -1000, 0), count: Self.capacity * 2))
    }

    private func random(_ r: ClosedRange<Double> = 0...1) -> Double { Double.random(in: r, using: &rng) }

    private func gaussian() -> Double {
        let u = max(random(), 1e-12), v = random()
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }

    /// How a kind of particle's trail looks: droplets per millimetre, sideways spread (σ, mm), how much thicker and
    /// brighter it gets toward the end (the Bragg peak), droplet brightness and glow size (mm), diffusion and life (s).
    struct Style {
        var density: Double, width: Double, bragg = 0.0, brightness: Double, size: Double, spread: Double, life: Double
    }

    static let muonTrail = Style(density: 34, width: 0.2, brightness: 1.1, size: 0.3, spread: 1, life: 3.6)
    static let alphaTrail = Style(density: 220, width: 0.32, bragg: 2.2, brightness: 0.75, size: 0.36, spread: 1.4, life: 5)
    static let electronTrail = Style(density: 18, width: 0.2, brightness: 1.0, size: 0.28, spread: 0.8, life: 3)

    /// Writes one droplet: position in millimetres, birth time, brightness, then diffusion, glow size and life.
    private func drop(_ x: Double, _ y: Double, _ t: Float, _ b: Double, _ style: Style) {
        let p = drops.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let a = random(0...(2 * .pi))
        let j = random(0.3...1) * style.spread
        p[2 * head] = SIMD4(Float(x), Float(y), t + Float(random(0...0.12)), Float(b))
        p[2 * head + 1] = SIMD4(Float(cos(a) * j), Float(sin(a) * j), Float(style.size), Float(style.life * random(0.75...1.15)))
        head = (head + 1) % Self.capacity
    }

    /// Clips a segment to the view plus a margin (Liang–Barsky), returning the kept range as fractions of it.
    private func clip(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> (Double, Double)? {
        let lo = SIMD2(-4.0, -4.0), hi = SIMD2(width + 4, Self.height + 4)
        var t0 = 0.0, t1 = 1.0
        let d = b - a
        for (p, q) in [(-d.x, a.x - lo.x), (d.x, hi.x - a.x), (-d.y, a.y - lo.y), (d.y, hi.y - a.y)] {
            if p == 0 { if q < 0 { return nil }; continue }
            let r = q / p
            if p < 0 { t0 = max(t0, r) } else { t1 = min(t1, r) }
            if t0 > t1 { return nil }
        }
        return (t0, t1)
    }

    /// Lays droplets along a polyline, with brightness that clumps like real ionisation.
    private func lay(_ path: [SIMD2<Double>], at t: Float, _ style: Style) {
        guard path.count > 1 else { return }
        var total = 0.0
        for i in 1..<path.count { total += simd_length(path[i] - path[i - 1]) }
        var walked = 0.0
        let clump = random(0...100)
        for i in 1..<path.count {
            let a = path[i - 1], b = path[i]
            let seg = simd_length(b - a)
            defer { walked += seg }
            guard seg > 0, let (c0, c1) = clip(a, b) else { continue }
            let dir = (b - a) / seg, side = SIMD2(-dir.y, dir.x)
            let n = Int(((c1 - c0) * seg * style.density + random()).rounded(.down))
            for _ in 0..<n {
                let s = random((c0 * seg)...(c1 * seg))
                let f = (walked + s) / total
                // Ionisation comes in clusters; a slow wobble of the density makes the beads.
                let bead = 0.7 + 0.3 * sin((walked + s) * 1.7 + clump) * sin((walked + s) * 0.63 + clump * 2)
                guard random() < bead else { continue }
                let peak = style.bragg * pow(f, 4)
                let w = style.width * (1 + peak * 0.8)
                let p = a + dir * s + side * gaussian() * w + dir * gaussian() * w * 0.3
                drop(p.x, p.y, t, style.brightness * (1 + peak), style)
            }
        }
    }

    /// A slowing electron: nearly straight at first, then curling more and more as multiple scattering takes over.
    private func curl(from start: SIMD2<Double>, angle: Double, length: Double, curliness: Double) -> [SIMD2<Double>] {
        var p = start, a = angle, path = [start]
        let step = 0.6
        var spin = random() < 0.5 ? -1.0 : 1.0
        var s = 0.0
        while s < length {
            let f = s / length
            if random() < 0.04 { spin = -spin }
            a += gaussian() * (0.04 + curliness * f * f) + spin * curliness * 0.12 * f
            p += SIMD2(cos(a), sin(a)) * step
            path.append(p)
            s += step
        }
        return path
    }

    private func spawn(_ kind: Kind, at t: Float, audible: Bool) {
        let w = width, h = Self.height
        switch kind {
        case .muon:
            // Zenith angle from the cos² law: the projected length through the layer is depth × tan θ.
            let c = pow(max(random(), 1e-6), 1.0 / 3)
            let length = min(Self.layer * (1 - c * c).squareRoot() / c, 400)
            let mid = SIMD2(random(-20...(w + 20)), random(-20...(h + 20)))
            let a = random(0...(2 * .pi))
            let d = SIMD2(cos(a), sin(a)) * length / 2
            lay([mid - d, mid + d], at: t, Self.muonTrail)
            if random() < 0.12 { delta(on: mid - d + (d * 2) * random(0.1...0.9), angle: a, at: t, energy: random(4...14)) }
            events.append((.muon, t))
            if audible { clicks.append((.click, Float(random(2800...3600)), 0.028, Float((mid.x / w) * 1.4 - 0.7), 0.008)) }
        case .alpha:
            var pick = random(), source = Self.alphaSources[0]
            for s in Self.alphaSources { if pick < s.3 { source = s; break }; pick -= s.3 }
            // Only alphas that run roughly along the layer leave their whole range in it.
            let length = source.2 * 10 * random(0.75...1)
            let start = SIMD2(random(5...(w - 5)), random(5...(h - 5)))
            let a = random(0...(2 * .pi))
            let bend = gaussian() * 0.15
            let end = start + SIMD2(cos(a), sin(a)) * length * 0.9
            let tip = end + SIMD2(cos(a + bend), sin(a + bend)) * length * 0.1
            // Ionisation rises toward the end of the range: the Bragg peak makes the track thicker and brighter there.
            lay([start, end, tip], at: t, Self.alphaTrail)
            if random() < 0.3 { delta(on: start + (end - start) * random(0.1...0.8), angle: a, at: t, energy: random(3...9)) }
            lastAlpha = (source.0, source.1)
            events.append((.alpha, t))
            if audible { clicks.append((.thump, Float(random(62...78)), 0.13, Float((start.x / w) * 1.4 - 0.7), 0.45)) }
        case .electron:
            let start = SIMD2(random(-5...(w + 5)), random(-5...(h + 5)))
            let length = min(-log(max(random(), 1e-6)) * 38 + 12, 160)
            let path = curl(from: start, angle: random(0...(2 * .pi)), length: length, curliness: random(0.5...1.1))
            lay(path, at: t, Self.electronTrail)
            events.append((.electron, t))
            if audible { clicks.append((.click, Float(random(5500...7500)), 0.018, Float((start.x / w) * 1.4 - 0.7), 0.004)) }
        case .shower:
            // Many nearly parallel muons and electrons from one cosmic ray high in the atmosphere.
            let a = random(0...(2 * .pi)), n = Int(random(7...16))
            let dir = SIMD2(cos(a), sin(a)), side = SIMD2(-dir.y, dir.x)
            let centre = SIMD2(w / 2, h / 2) + SIMD2(gaussian() * 20, gaussian() * 12)
            for _ in 0..<n {
                let offset = side * gaussian() * h * 0.35
                let d = SIMD2(cos(a + gaussian() * 0.03), sin(a + gaussian() * 0.03)) * 400
                let mid = centre + offset + dir * gaussian() * 10
                lay([mid - d, mid + d], at: t + Float(random(0...0.05)), Self.muonTrail)
                events.append((.muon, t))
            }
            lastShower = t
            if audible {
                for i in 0..<5 { clicks.append((.click, Float(random(2800...3800)), 0.03, Float(i) * 0.3 - 0.6, 0.008)) }
            }
        }
    }

    /// A short, curly electron knocked out of an atom by the passing particle.
    private func delta(on p: SIMD2<Double>, angle: Double, at t: Float, energy: Double) {
        let a = angle + (random() < 0.5 ? -1 : 1) * random(0.6...1.4)
        let path = curl(from: p, angle: a, length: energy, curliness: 1.6)
        lay(path, at: t + 0.02, Self.electronTrail)
    }

    func draw(_ f: Frame) {
        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            fog = f.gpu.texture(f.target.width / 4, f.target.height / 4)
        }
        width = Self.height * Double(f.aspect)
        let rates = Rates(area: width * Self.height / 100)
        if !started {
            started = true
            // Start mid-session: the last few seconds of tracks are already in the air, but not counted.
            var t: Float = -4
            while t < 0 {
                for (kind, rate) in [(Kind.muon, rates.muons), (.alpha, rates.alphas), (.electron, rates.electrons)]
                where random() < rate * 0.05 { spawn(kind, at: t, audible: false) }
                t += 0.05
            }
            events.removeAll()
            lastAlpha = nil
        }
        now = f.time
        let dt = Double(f.dt)
        for (kind, rate) in [(Kind.muon, rates.muons), (.alpha, rates.alphas), (.electron, rates.electrons), (.shower, rates.showers)] {
            var expected = rate * dt
            while expected > 0 {
                if random() < min(expected, 1) { spawn(kind, at: f.time, audible: f.sound != nil) }
                expected -= 1
            }
        }
        events.removeAll { f.time - $0.1 > 60 }

        let mmPerPixel = Float(Self.height) / f.size.y
        let u = f.uniforms(SIMD4(mmPerPixel, Float(width), Float(Self.height), 0))
        f.cmd.draw(mist, into: fog!, uniforms: u)
        f.cmd.draw(rain, into: light!, uniforms: u, vertices: 5000, primitive: .point)
        f.cmd.draw(droplets, into: light!, clear: false, uniforms: u, buffers: [drops], vertices: Self.capacity, primitive: .point)
        let glow = bloom.glow(f.cmd, light!)
        f.cmd.draw(tone, into: f.target, uniforms: u, textures: [light!, glow, fog!])

        if let synth = f.sound {
            // Geiger-style, at most three clicks a frame, so a shower is a short patter rather than a burst.
            for c in clicks.prefix(3) { synth.play(c.0, c.1, amp: c.2, pan: c.3, length: c.4) }
            synth.noise(0.025, cutoff: 260)
            synth.drone(0, 55, 0.025)
            synth.drone(1, 82.5, 0.012)
        }
        clicks.removeAll()
    }

    private static let source = """
    // a: millimetres per pixel, view width and height in millimetres.

    inline float sq(float x) { return x * x; }

    inline float2 toClip(float2 mm, constant U &u) {
        float2 px = mm / u.a.x;
        return px / u.res * float2(2, -2) + float2(-1, 1);
    }

    // Faint mist drifting over the black floor, lit from the left by the chamber's lamp. Rendered small; it is soft.
    fragment float4 chamberMist(V in [[stage_in]], constant U &u [[buffer(0)]]) {
        float2 mm = in.uv * float2(u.a.y, u.a.z);
        float t = u.time;
        float2 q = mm * 0.016 + float2(t * 0.010, t * 0.003);
        float2 warp = float2(fbm(q * 0.8 + float2(t * 0.007, 3.1), 4), fbm(q * 0.8 + float2(7.3, t * 0.006), 4));
        float m = fbm(q * 1.3 + warp * 1.6, 5);
        float wisp = smoothstep(0.42, 0.85, m);
        float lamp = exp(-mm.x / (u.a.y * 0.5)) * 0.85 + 0.15;
        float3 c = float3(0.5, 0.62, 0.8) * (wisp * wisp * 0.085 + 0.007) * lamp;
        return float4(c, 1);
    }

    // Each droplet condenses within a fraction of a second, then rides the air, spreads, and evaporates.
    // Droplet: (x, y, birth, brightness), (diffusion direction, growth, life).
    vertex P chamberDrop(uint id [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *d [[buffer(1)]]) {
        float4 a = d[2 * id], b = d[2 * id + 1];
        float age = u.time - a.z;
        P o;
        if (age < 0 || age > b.w) { o.pos = float4(2, 2, 0, 1); o.size = 0; o.color = 0; return o; }
        float2 p = a.xy;
        // Slow convection bends ageing trails; it changes over minutes, so each trail sags its own way.
        float2 flow = curl(p * 0.016 + float2(a.z * 0.004, 3.7)) * 0.75 + float2(0.25, 0.1);
        p += flow * age + b.xy * sqrt(age) * 0.45 * b.z;
        o.pos = float4(toClip(p, u), 0, 1);
        float form = smoothstep(0.0, 0.25, age);
        float fade = (1 - 0.45 * age / b.w) * smoothstep(b.w, b.w * 0.45, age);
        float mmSize = 0.16 + 0.22 * b.z * sqrt(age);
        o.size = max(mmSize / u.a.x, 1.6);
        float lamp = exp(-p.x / (u.a.y * 0.6)) * 0.55 + 0.45;
        float spread = 0.16 / mmSize;
        o.color = float4(float3(0.97, 0.98, 1.0) * a.w * form * fade * lamp * spread * 0.8, 0);
        return o;
    }

    // The steady drizzle of alcohol droplets falling through the light, each glinting for a moment.
    vertex P chamberRain(uint id [[vertex_id]], constant U &u [[buffer(0)]]) {
        float period = 1.5 + hash11(float(id) * 1.37) * 2.5;
        float t = u.time / period + hash11(float(id) * 7.13);
        float cycle = floor(t), ph = fract(t);
        float2 r = hash22(float2(float(id), cycle * 13.1 + 1.7));
        float2 p = r * float2(u.a.y, u.a.z);
        P o;
        o.pos = float4(toClip(p, u), 0, 1);
        o.size = max(0.12 / u.a.x, 1.0);
        float lamp = exp(-p.x / (u.a.y * 0.6)) * 0.6 + 0.4;
        float b = sin(ph * PI);
        o.color = float4(float3(0.8, 0.88, 1.0) * b * b * 0.05 * lamp * (0.3 + hash11(float(id) * 3.1)), 0);
        return o;
    }

    fragment float4 chamberTone(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                                texture2d<float> glow [[texture(1)]], texture2d<float> fog [[texture(2)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 e = abs(in.uv - 0.5) * 2;
        float vignette = 1 - 0.55 * pow(max(e.x, e.y), 5);
        float3 c = (light.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.45 + fog.sample(s, in.uv).rgb) * vignette;
        return float4(film(c), 1);
    }
    """
}
