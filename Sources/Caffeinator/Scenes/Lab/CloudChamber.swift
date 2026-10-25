import Foundation
import Metal
import simd

/// A diffusion cloud chamber filmed at an angle through its lid: black felt soaked in alcohol over dry ice, lit from the
/// left by an LED strip so only the supersaturated layer just above the felt glows. Tracks arrive at sea-level rates:
/// cosmic-ray muons, alphas from radon in the air, beta and Compton electrons, delta rays, now and then an air shower.
/// Each track is a line of alcohol droplets that condense within a fraction of a second, then grow, ride the convection,
/// sink out of the light and evaporate. Every few minutes a thoriated tungsten welding rod is laid on the felt, or taken
/// away, and its thorium chain adds a spray of alphas and betas.
@MainActor final class CloudChamber: Scene {
    enum Kind { case muon, alpha, electron, shower }

    /// Events per second over the chamber floor, given its area in square centimetres.
    struct Rates {
        let muons: Double, alphas: Double, electrons: Double, showers: Double
        init(area: Double) {
            muons = area / 60          // about one per square centimetre per minute at sea level
            alphas = 36.0 / 60         // radon-222 and its daughters in basement air
            electrons = 260.0 / 60     // beta decays and Compton electrons from background gamma rays
            showers = 0.4 / 60
        }
    }

    /// Alpha emitters: name, energy in MeV, range in air in centimetres, share of decays.
    typealias Emitter = (name: String, energy: Double, range: Double, share: Double)
    static let radon: [Emitter] = [("radon 222", 5.49, 4.1, 0.4), ("polonium 218", 6.00, 4.7, 0.35), ("polonium 214", 7.69, 6.9, 0.25)]
    /// The thorium-232 chain in a welding rod, in equilibrium: bismuth 212 alpha-decays 36 % of the time, the rest goes
    /// through polonium 212.
    static let thorium: [Emitter] = [
        ("thorium 232", 4.01, 2.8, 0.15), ("thorium 228", 5.42, 4.0, 0.15), ("radium 224", 5.69, 4.3, 0.15),
        ("radon 220", 6.29, 5.0, 0.15), ("polonium 216", 6.78, 5.6, 0.15), ("bismuth 212", 6.05, 4.7, 0.06),
        ("polonium 212", 8.78, 8.6, 0.19),
    ]

    /// The chamber floor in millimetres (x right, z away from the camera, the far glass wall at `far`), the depth of the
    /// sensitive layer, and the strip's height above the felt.
    static let half: Double = 200, near: Double = -170, far: Double = 98
    static let layer: Double = 12
    static let capacity = 1 << 19
    /// Seconds per look: the chamber on its own, then with the thorium rod.
    static let lookTime: Float = 240

    private let drops: MTLBuffer
    private var head = 0
    private let back: MTLRenderPipelineState
    private let mist: MTLRenderPipelineState
    private let droplets: MTLRenderPipelineState
    private let rain: MTLRenderPipelineState
    private let lens: Lens
    private var light: MTLTexture?
    private var fog: MTLTexture?
    private var started = false
    private var events: [(Kind, Float)] = []
    private var lastAlpha: (String, Double)?
    private var lastShower: Float = -1000
    private var now: Float = 0
    private var rng = LabRandom()
    private var clicks: [(Synth.Voice, Float, Float, Float, Float)] = []
    private var pendingDaughters: [(SIMD3<Double>, Float)] = []
    private var look: Int
    private var lookSince: Float = 0
    /// The rod: centre on the felt (x, z) and angle.
    private var rod = SIMD3<Double>(-38, -22, 1.9)

    var scale: Float { 1 }
    private var hasRod: Bool { look == 1 }

    var facts: String? {
        func count(_ k: Kind, _ name: String) -> String {
            let n = events.filter { $0.0 == k && now - $0.1 <= 60 }.count
            return "\(n.formatted()) \(name)\(n == 1 ? "" : "s")"
        }
        let window = now < 60 ? "So far" : "Last minute"
        var line = hasRod ? "A thoriated tungsten welding rod lies on the felt. " : ""
        line += "\(window): \(count(.muon, "muon")), \(count(.alpha, "alpha")), \(count(.electron, "electron"))"
        if now - lastShower < 60 { line += ", and a cosmic-ray air shower" }
        if let (name, energy) = lastAlpha { line += "; last alpha " + String(format: "%.1f", energy) + " MeV from \(name)" }
        return line
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(LabShaders.common + Self.source)
        back = try gpu.pipeline(library, "chamberBack")
        mist = try gpu.pipeline(library, "chamberFog")
        droplets = try gpu.pipeline(library, "sprite", vertex: "chamberDrop", blend: .add)
        rain = try gpu.pipeline(library, "sprite", vertex: "chamberRain", blend: .add)
        lens = try Lens(gpu: gpu)
        drops = gpu.buffer([SIMD4<Float>](repeating: SIMD4(0, 0, 0, -1000), count: Self.capacity * 3))
        look = (Snapshot.variant ?? Int.random(in: 0..<2)) % 2
        placeRod()
    }

    private func random(_ r: ClosedRange<Double> = 0...1) -> Double { rng.uniform(r) }
    private func gaussian() -> Double { rng.gaussian() }

    private func placeRod() {
        rod = SIMD3(random(-60...(-25)), random(-45...(-5)), .pi / 2 + random(-0.7...0.7))
    }

    /// How a kind of particle's trail looks: puffs of droplets per millimetre, the trail's width when it forms (σ, mm),
    /// how much thicker and brighter it gets toward the end (the Bragg peak), brightness, how fast it spreads
    /// (mm per √s) and how long it lasts (s).
    struct Style {
        var density: Double, width: Double, bragg = 0.0, brightness: Double, spread: Double, life: Double
    }

    static let muonTrail = Style(density: 30, width: 0.13, brightness: 0.9, spread: 0.5, life: 5.5)
    static let alphaTrail = Style(density: 120, width: 0.3, bragg: 1.6, brightness: 0.4, spread: 0.75, life: 6)
    static let electronTrail = Style(density: 22, width: 0.12, brightness: 0.75, spread: 0.45, life: 4.5)

    /// Writes one puff of droplets: position (mm) and birth time; drift direction, rise and life; brightness, width,
    /// fall speed and spread.
    private func drop(_ p: SIMD3<Double>, _ t: Float, _ b: Double, _ w: Double, _ style: Style) {
        let d = drops.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let a = random(0...(2 * .pi))
        let j = random(0.2...1)
        d[3 * head] = SIMD4(Float(p.x), Float(p.y), Float(p.z), t + Float(random(0...0.06)))
        d[3 * head + 1] = SIMD4(Float(cos(a) * j), Float(sin(a) * j), Float(random(-0.3...0.3)), Float(style.life * random(0.75...1.15)))
        d[3 * head + 2] = SIMD4(Float(b), Float(w), Float(random(1.0...2.6)), Float(style.spread * random(0.8...1.2)))
        head = (head + 1) % Self.capacity
    }

    /// Clips a segment to the sensitive layer of the chamber (Liang–Barsky), returning the kept range as fractions of it.
    private func clip(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> (Double, Double)? {
        let lo = SIMD3(-Self.half, 0.4, Self.near), hi = SIMD3(Self.half, Self.layer, Self.far - 0.5)
        var t0 = 0.0, t1 = 1.0
        let d = b - a
        for k in 0..<3 {
            for (p, q) in [(-d[k], a[k] - lo[k]), (d[k], hi[k] - a[k])] {
                if p == 0 { if q < 0 { return nil }; continue }
                let r = q / p
                if p < 0 { t0 = max(t0, r) } else { t1 = min(t1, r) }
                if t0 > t1 { return nil }
            }
        }
        return (t0, t1)
    }

    /// Lays droplets along a polyline, with brightness that clumps like real ionisation.
    private func lay(_ path: [SIMD3<Double>], at t: Float, _ style: Style, scale: Double = 1) {
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
            let dir = (b - a) / seg
            var side = simd_cross(dir, SIMD3(0, 1, 0))
            if simd_length(side) < 1e-3 { side = SIMD3(1, 0, 0) }
            side = simd_normalize(side)
            let up = simd_cross(side, dir)
            let n = Int(((c1 - c0) * seg * style.density + random()).rounded(.down))
            for _ in 0..<n {
                let s = random((c0 * seg)...(c1 * seg))
                let f = (walked + s) / total
                // Ionisation comes in clusters; a slow wobble of the density makes the beads.
                let bead = 0.65 + 0.35 * sin((walked + s) * 1.7 + clump) * sin((walked + s) * 0.63 + clump * 2)
                guard random() < bead else { continue }
                let peak = style.bragg * pow(f, 4)
                let w = style.width * (1 + peak * 0.9)
                let p = a + dir * s + side * gaussian() * w * 0.35 + up * gaussian() * w * 0.25
                drop(p, t, style.brightness * scale * (1 + peak * 0.6), w, style)
            }
        }
    }

    /// A slowing electron: nearly straight at first, then curling more and more as multiple scattering takes over.
    private func curl(from start: SIMD3<Double>, angle: Double, length: Double, curliness: Double) -> [SIMD3<Double>] {
        var p = start, a = angle, path = [start]
        var climb = gaussian() * 0.08
        let step = 0.6
        var spin = random() < 0.5 ? -1.0 : 1.0
        var s = 0.0
        while s < length {
            let f = s / length
            if random() < 0.04 { spin = -spin }
            a += gaussian() * (0.04 + curliness * f * f) + spin * curliness * 0.12 * f
            climb = max(-0.5, min(0.5, climb + gaussian() * 0.05 * curliness))
            p += SIMD3(cos(a), climb, sin(a)) * step
            path.append(p)
            s += step
        }
        return path
    }

    private func pick(_ table: [Emitter]) -> Emitter {
        var r = random()
        for e in table { if r < e.share { return e }; r -= e.share }
        return table[table.count - 1]
    }

    /// An alpha from `start` heading along `dir`, with its full range unless it leaves the layer.
    private func alpha(from start: SIMD3<Double>, dir: SIMD3<Double>, source: Emitter, at t: Float, audible: Bool) {
        // Cold air below the lid is denser than at room temperature, which shortens the range a little.
        let length = source.range * 10 * 0.9 * random(0.92...1)
        let bend = gaussian() * 0.12
        let end = start + dir * length * 0.88
        let turned = simd_normalize(dir + simd_cross(dir, SIMD3(0, 1, 0)) * bend)
        let tip = end + turned * length * 0.12
        // Ionisation rises toward the end of the range: the Bragg peak makes the track thicker and brighter there.
        lay([start, end, tip], at: t, Self.alphaTrail)
        if random() < 0.3 {
            let p = start + (end - start) * random(0.1...0.8)
            delta(on: p, angle: atan2(dir.z, dir.x), at: t, energy: random(3...9))
        }
        lastAlpha = (source.name, source.energy)
        events.append((.alpha, t))
        if audible { clicks.append((.thump, Float(random(62...78)), 0.13, Float(start.x / Self.half) * 0.7, 0.45)) }
    }

    /// A direction for an alpha that stays mostly in the layer: those that dive into the felt or climb out are cut short.
    private func flatDirection() -> SIMD3<Double> {
        let a = random(0...(2 * .pi)), dip = gaussian() * 0.12
        return simd_normalize(SIMD3(cos(a), dip, sin(a)))
    }

    private func spawn(_ kind: Kind, at t: Float, audible: Bool) {
        let w = Self.half, depth = Self.far - Self.near
        switch kind {
        case .muon:
            // Zenith angle from the cos² law: the track crosses the layer, so its length seen from above is depth × tan θ.
            let c = pow(max(random(), 1e-6), 1.0 / 3)
            let length = min(Self.layer * (1 - c * c).squareRoot() / c, 400)
            let mid = SIMD3(random(-w...w), Self.layer / 2, random(Self.near...Self.far))
            let a = random(0...(2 * .pi))
            let d = SIMD3(cos(a) * length / 2, -Self.layer / 2 - 0.5, sin(a) * length / 2)
            lay([mid - d, mid + d], at: t, Self.muonTrail)
            if random() < 0.12 { delta(on: mid - d + d * 2 * random(0.1...0.9), angle: a, at: t, energy: random(4...14)) }
            events.append((.muon, t))
            if audible { clicks.append((.click, Float(random(2800...3600)), 0.028, Float(mid.x / w) * 0.7, 0.008)) }
        case .alpha:
            let start = SIMD3(random(-w...w), random(1...(Self.layer - 1)), random(Self.near...Self.far))
            alpha(from: start, dir: flatDirection(), source: pick(Self.radon), at: t, audible: audible)
        case .electron:
            let start = SIMD3(random(-w...w), random(1...(Self.layer - 1)), random(Self.near...Self.far))
            let length = min(-log(max(random(), 1e-6)) * 38 + 12, 160)
            lay(curl(from: start, angle: random(0...(2 * .pi)), length: length, curliness: random(0.5...1.1)), at: t, Self.electronTrail)
            events.append((.electron, t))
            if audible { clicks.append((.click, Float(random(5500...7500)), 0.018, Float(start.x / w) * 0.7, 0.004)) }
        case .shower:
            // Many nearly parallel muons and electrons from one cosmic ray high in the atmosphere.
            let a = random(0...(2 * .pi)), n = Int(random(7...16))
            let dir = SIMD3(cos(a), 0, sin(a)), side = SIMD3(-dir.z, 0, dir.x)
            let centre = SIMD3(0, Self.layer / 2, (Self.near + Self.far) / 2) + SIMD3(gaussian() * 20, 0, gaussian() * 12)
            for _ in 0..<n {
                let offset = side * gaussian() * depth * 0.3
                let d = SIMD3(cos(a + gaussian() * 0.03) * 300, -Self.layer / 2 - 0.5, sin(a + gaussian() * 0.03) * 300)
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

    /// The thorium rod's own decays: alphas from its surface, thoron that drifts off and decays in the air followed
    /// 0.15 s later by its polonium daughter from the same spot (the forked pairs), and beta electrons.
    private func spawnRod(at t: Float, dt: Double, audible: Bool) {
        let axis = SIMD3(cos(rod.z), 0, sin(rod.z)), centre = SIMD3(rod.x, 1.2, rod.y)
        func onRod() -> SIMD3<Double> { centre + axis * random(-24...24) + SIMD3(0, random(0...1.2), 0) }
        var expected = 2.6 * dt
        while expected > 0 {
            if random() < min(expected, 1) {
                var dir = flatDirection()
                dir.y = abs(dir.y)
                alpha(from: onRod(), dir: dir, source: pick(Self.thorium), at: t, audible: audible)
            }
            expected -= 1
        }
        if random() < 0.35 * dt {
            let a = random(0...(2 * .pi)), r = random(6...28)
            let spot = onRod() + SIMD3(cos(a) * r, random(2...8), sin(a) * r)
            alpha(from: spot, dir: flatDirection(), source: Self.thorium[3], at: t, audible: audible)
            pendingDaughters.append((spot, t + Float(-log(max(random(), 1e-6)) * 0.21)))
        }
        if random() < 1.6 * dt {
            let start = onRod()
            let length = min(-log(max(random(), 1e-6)) * 45 + 15, 180)
            lay(curl(from: start, angle: random(0...(2 * .pi)), length: length, curliness: random(0.4...0.9)), at: t, Self.electronTrail)
            events.append((.electron, t))
            if audible { clicks.append((.click, Float(random(5500...7500)), 0.018, Float(start.x / Self.half) * 0.7, 0.004)) }
        }
        for (i, d) in pendingDaughters.enumerated().reversed() where d.1 <= t {
            alpha(from: d.0, dir: flatDirection(), source: Self.thorium[4], at: t, audible: audible)
            pendingDaughters.remove(at: i)
        }
    }

    /// A short, curly electron knocked out of an atom by the passing particle.
    private func delta(on p: SIMD3<Double>, angle: Double, at t: Float, energy: Double) {
        let a = angle + (random() < 0.5 ? -1 : 1) * random(0.6...1.4)
        lay(curl(from: p, angle: a, length: energy, curliness: 1.6), at: t + 0.02, Self.electronTrail)
    }

    func draw(_ f: Frame) {
        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            fog = f.gpu.texture(f.target.width / 4, f.target.height / 4)
        }
        let rates = Rates(area: 2 * Self.half * (Self.far - Self.near) / 100)
        let all: [(Kind, Double)] = [(.muon, rates.muons), (.alpha, rates.alphas), (.electron, rates.electrons), (.shower, rates.showers)]
        if !started {
            started = true
            // Start mid-session: the last few seconds of tracks are already in the air, but not counted.
            var t: Float = -5
            while t < 0 {
                for (kind, rate) in all where kind != .shower && random() < rate * 0.05 { spawn(kind, at: t, audible: false) }
                if hasRod { spawnRod(at: t, dt: 0.05, audible: false) }
                t += 0.05
            }
            events.removeAll()
            lastAlpha = nil
        }
        now = f.time
        lookSince += f.dt
        // The light goes off for a moment while the rod is laid on the felt or taken away.
        if lookSince > Self.lookTime {
            look = 1 - look
            lookSince = 0
            if hasRod { placeRod() }
            pendingDaughters.removeAll()
        }
        let lamp = min(lookSince / 2.5, 1, (Self.lookTime - lookSince) / 1.5)
        let dt = Double(f.dt)
        let audible = f.sound != nil
        for (kind, rate) in all {
            var expected = rate * dt
            while expected > 0 {
                if random() < min(expected, 1) { spawn(kind, at: f.time, audible: audible) }
                expected -= 1
            }
        }
        if hasRod { spawnRod(at: f.time, dt: dt, audible: audible) }
        events.removeAll { f.time - $0.1 > 60 }

        // A slow drift of the camera, as if on a slider, so nothing bright sits still on screen for long.
        let t = f.time + f.seed * 37
        let pan = Float(12 * sin(t * 0.0061 + 2))
        let eye = SIMD3<Float>(pan + 4 * sin(t * 0.0071), 252 + 6 * sin(t * 0.0053), -214 + 8 * sin(t * 0.0047 + 1))
        let target = SIMD3<Float>(pan, 0, -4 + 5 * sin(t * 0.0039))
        let focus = simd_length(target - eye)
        var u = f.uniforms(SIMD4(eye, 3.9), SIMD4(target, 22 * f.size.y / 1000),
                           SIMD4(Float(Self.half), Float(Self.near), Float(Self.far), focus),
                           SIMD4(Float(rod.x), Float(rod.y), Float(rod.z), hasRod ? 1 : 0))
        u.pad = SIMD2(lamp * lamp * (3 - 2 * lamp), Float(Self.layer))
        var m = u
        m.res = SIMD2(Float(fog!.width), Float(fog!.height))
        f.cmd.draw(mist, into: fog!, uniforms: m)
        f.cmd.draw(back, into: light!, uniforms: u, textures: [fog!])
        f.cmd.draw(rain, into: light!, clear: false, uniforms: u, vertices: 14000 * 3, primitive: .point)
        f.cmd.draw(droplets, into: light!, clear: false, uniforms: u, buffers: [drops], vertices: Self.capacity, instances: 2,
                   primitive: .point)
        lens.develop(f, light!, Lens.Look(exposure: 1.8, white: [0.97, 1.0, 1.04], bloom: 0.05, halation: 0.025, vignette: 0.45,
                                          aberration: 1.0, grain: 0.7))

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
    // a: eye, zoom; b: target, blur (circle of confusion in pixels per unit of relative defocus);
    // c: floor half width, near edge z, far glass wall z, focus distance; d: rod (x, z, angle, present);
    // pad: lamp (0 dark, 1 on), sensitive layer depth. Lengths in millimetres; y is up.

    constant float3 LED = float3(0.92, 0.97, 1.0);

    // The LED strip runs along the left wall a few millimetres above the felt; its light fans out and dims across
    // the chamber, so it lights a sheet of vapour rather than the whole box.
    inline float sheet(float3 p, constant U &u) {
        float d = max(p.x + u.c.x + 8, 0.0);
        float h = 6.5 + 0.05 * d;
        return exp(-sq((p.y - 6.0) / h)) * 40.0 / h / (1 + d / 110) * u.pad.x;
    }

    // Droplets a few microns across scatter mostly forward: light from the left reaches the camera best from the left.
    inline float mie(float3 p, float3 eye) {
        float3 v = normalize(eye - p);
        return 0.8 * henyey(v.x, 0.5) + 0.2 * henyey(v.x, -0.2);
    }

    // The 2.4 mm rod shades the felt and the lowest vapour just beyond it; the long strip blurs the shadow away
    // within a few centimetres.
    inline float rodShadow(float3 p, constant U &u) {
        if (u.d.w < 0.5) return 1;
        float2 axis = float2(cos(u.d.z), sin(u.d.z));
        float2 n = float2(-axis.y, axis.x);
        if (abs(n.x) < 1e-3) return 1;
        float k = dot(p.xz - u.d.xy, n) / n.x;
        if (k <= 0 || k > 60) return 1;
        float along = dot(p.xz + float2(-k, 0) - u.d.xy, axis);
        float d = max(p.x + u.c.x + 8, 1.0);
        float rayHeight = p.y + (6.0 - p.y) * k / d;
        float cover = smoothstep(25.0, 22.0, abs(along)) * smoothstep(3.2, 1.8, rayHeight) * smoothstep(60.0, 6.0, k);
        return 1 - 0.8 * cover;
    }

    // Black felt soaked in alcohol: almost no light comes back, just a soft mottling of the matted fibres. `blur`
    // is the circle of confusion in millimetres.
    inline float felt(float2 q, float blur) {
        float fine = 1 - smoothstep(0.1, 0.5, blur);
        float mottle = fbm(q * 0.3, 3);
        float fuzz = noise(q * 5.0) * 0.55 + noise(q * 11.0) * 0.45;
        return 0.006 * (0.5 + mottle) * mix(1.0, 0.6 + 0.8 * fuzz, fine);
    }

    // The supersaturated layer's faint fog, integrated through the lit sheet along the view ray. With `fold`, positions
    // beyond the far glass are folded back into the chamber, which is what the glass reflects.
    inline float3 vapour(float3 eye, float3 rd, float tFloor, constant U &u, float fold) {
        float3 c = 0;
        float top = u.pad.y + 12;
        float t = u.time;
        for (int i = 0; i < 5; i++) {
            float y = top * (float(i) + 0.5) / 5;
            float3 p = eye + rd * (tFloor - y / max(-rd.y, 1e-3));
            if (fold > 0 && p.z > u.c.z) p.z = 2 * u.c.z - p.z;
            float2 q = p.xz * 0.02;
            float2 warp = float2(fbm(float3(q * 0.7, t * 0.02 + 3.1), 3), fbm(float3(q * 0.7 + 5.2, t * 0.017), 3));
            float m = fbm(float3(q + warp * 1.6 + float2(t * 0.006, 0), y * 0.04 + t * 0.012), 4);
            float density = 0.15 + 1.5 * smoothstep(0.45, 0.9, m);
            c += LED * density * sheet(p, u) * mie(p, eye) * rodShadow(p, u);
        }
        return c * (top / 5) / max(-rd.y, 1e-3) * 0.0016;
    }

    inline float glassFresnel(float3 rd) { return 0.04 + 0.96 * pow(1 - abs(rd.z), 5.0); }

    // Rendered at a quarter of the resolution: the vapour on the floor side, its reflection on the glass side.
    fragment float4 chamberFog(V in [[stage_in]], constant U &u [[buffer(0)]]) {
        float2 q = centered(in.uv, u.res);
        float3 eye = u.a.xyz;
        float3 rd = ray(q, eye, u.b.xyz, u.a.w);
        float tf = rd.y < 0 ? -eye.y / rd.y : 1e6;
        float tw = rd.z > 0 ? (u.c.z - eye.z) / rd.z : 1e6;
        if (tf < tw) return float4(vapour(eye, rd, tf, u, 0), 1);
        float3 p = eye + rd * tw;
        float smudge = 0.6 + 0.8 * fbm(p.xy * 0.03, 3);
        return float4(vapour(eye, rd, tf, u, 1) * glassFresnel(rd) * smudge, 1);
    }

    inline float2 capsule(float3 ro, float3 rd, float3 a, float3 b, float r) {
        float3 ba = b - a, oa = ro - a;
        float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa), oaoa = dot(oa, oa);
        float qa = baba - bard * bard, qb = baba * rdoa - baoa * bard, qc = baba * oaoa - baoa * baoa - r * r * baba;
        float h = qb * qb - qa * qc;
        if (h < 0) return float2(-1, 0);
        float tt = (-qb - sqrt(h)) / qa;
        float y = baoa + tt * bard;
        if (y > 0 && y < baba) return float2(tt, y / baba);
        return float2(-1, 0);
    }

    fragment float4 chamberBack(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> fog [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 q = centered(in.uv, u.res);
        float3 eye = u.a.xyz;
        float3 rd = ray(q, eye, u.b.xyz, u.a.w);
        float pxPerUnit = 0.5 * u.res.y * u.a.w;
        float tf = rd.y < 0 ? -eye.y / rd.y : 1e6;
        float tw = rd.z > 0 ? (u.c.z - eye.z) / rd.z : 1e6;
        float3 c = fog.sample(s, in.uv).rgb;
        float lamp = u.pad.x;
        if (tf < tw) {
            float3 p = eye + rd * tf;
            float coc = u.b.w * abs(1 - u.c.w / tf);
            float lit = sheet(float3(p.x, 1.0, p.z), u) * rodShadow(float3(p.x, 0.5, p.z), u);
            c += LED * felt(p.xz, coc * tf / pxPerUnit) * lit;
            if (u.d.w > 0.5) {
                float2 axis = float2(cos(u.d.z), sin(u.d.z));
                float3 a = float3(u.d.x - axis.x * 25, 1.2, u.d.y - axis.y * 25), b = float3(u.d.x + axis.x * 25, 1.2, u.d.y + axis.y * 25);
                // Contact shadow on the felt around the rod.
                float3 pa = p - a, ba = b - a;
                float gap = length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
                c *= 0.3 + 0.7 * smoothstep(1.0, 5.0, gap);
                float2 hit = capsule(eye, rd, a, b, 1.2);
                if (hit.x > 0) {
                    float3 hp = eye + rd * hit.x;
                    float3 axisN = normalize(ba);
                    float3 n = normalize(hp - (a + ba * hit.y));
                    float3 toLight = normalize(float3(-1, 0.12, 0));
                    float3 h = normalize(toLight - rd);
                    float lit = sheet(hp, u);
                    bool tip = hit.y > 0.92;
                    float grime = 0.7 + 0.5 * noise(float2(dot(hp, axisN) * 0.8, atan2(n.y, dot(n, cross(axisN, float3(0, 1, 0)))) * 2));
                    float3 albedo = tip ? float3(0.36, 0.03, 0.028) : float3(0.12, 0.12, 0.125) * grime;
                    // Ground tungsten: grinding marks run around the rod, so its highlight is a soft broken streak.
                    float marks = noise(float2(dot(hp, axisN) * 9, 0.5)) * 0.5 + 0.75;
                    float spec = pow(max(dot(n, h), 0.0), tip ? 30.0 : 24.0) * (tip ? 0.2 : 0.9) * marks * grime;
                    c = LED * lit * (albedo * (0.04 + max(dot(n, toLight), 0.0) * 0.4) + spec * 0.35) + LED * 0.002 * lamp * albedo;
                }
            }
        } else {
            // The far glass wall: the dark room behind it, frost along its foot, the strip's light caught in the glass
            // edge, and beads of alcohol running down the inside.
            float3 p = eye + rd * tw;
            c += float3(0.0012, 0.0011, 0.001) * (0.7 + 0.6 * fbm(p.xy * 0.004 + 3, 3));
            float lit = sheet(float3(p.x, 3.0, p.z), u);
            float frostLine = 2.5 + 2.5 * fbm(float2(p.x * 0.08, 1.7), 3);
            float frost = smoothstep(frostLine, 0.0, p.y);
            float2 cell = floor(p.xy * 2.6);
            float2 h = hash22(cell);
            float crystal = step(0.75, h.x) * exp(-length(fract(p.xy * 2.6) - h) * 10);
            c += LED * lit * frost * (0.01 + 0.3 * crystal) * mie(p, eye);
            c += LED * lit * exp(-p.y / 0.3) * 0.05 * (0.3 + 1.4 * smoothstep(0.3, 0.8, fbm(float2(p.x * 0.05, 9.1), 4)));
            float2 bc = floor(p.xy * float2(0.5, 0.3));
            float2 bh = hash22(bc + 7.0);
            float2 bp = (fract(p.xy * float2(0.5, 0.3)) - bh) / float2(0.5, 0.3);
            float bead = step(0.8, bh.y) * exp(-dot(bp, bp) * 4.0) * exp(-p.y / 30);
            c += LED * bead * 0.006 * lamp * (0.5 + 0.5 * h.y) * mie(p, eye);
        }
        return float4(c, 1);
    }

    // Each puff of droplets condenses within a fraction of a second, grows and brightens, then spreads, rides the
    // convection, sinks out of the lit sheet and evaporates. Puff: (position, birth), (drift direction, rise, life),
    // (brightness, width, fall speed, spread). Instance 1 is its reflection in the far glass.
    vertex P chamberDrop(uint id [[vertex_id]], uint inst [[instance_id]], constant U &u [[buffer(0)]],
                         const device float4 *d [[buffer(1)]]) {
        float4 a = d[3 * id], b = d[3 * id + 1], k = d[3 * id + 2];
        float age = u.time - a.w;
        P o;
        o.pos = float4(2, 2, 0, 1); o.size = 0; o.color = 0;
        if (age < 0 || age > b.w) return o;
        float3 p = a.xyz;
        float root = sqrt(age);
        // Slow convection bends ageing trails; it changes over minutes, so each trail sags its own way.
        float2 flow = curl(p.xz * 0.011 + float2(a.w * 0.003, 3.7)) * 1.1 + float2(0.6, 0.25);
        p.xz += flow * age + b.xy * k.w * root * 0.7;
        p.y += b.z * root - k.z * age * age * 0.5;
        if (p.y < 0.2) return o;
        float3 world = p;
        if (inst == 1) {
            if (p.z < u.c.z - 40) return o;
            p.z = 2 * u.c.z - p.z;
        }
        float4 clip = project(p, u.a.xyz, u.b.xyz, u.a.w, u.res.x / u.res.y);
        o.pos = clip;
        float px = pixelsPer(clip.w, u.a.w, u.res.y);
        float coc = u.b.w * abs(1 - u.c.w / clip.w);
        float width = k.y + k.w * root * 0.8;
        float sharp = max(3.2 * width * px, 1.5);
        o.size = sqrt(sharp * sharp + coc * coc);
        float form = smoothstep(0.0, 0.15, age);
        float swell = 0.5 + 0.5 * smoothstep(0.05, 1.0, age);
        float fade = smoothstep(b.w, b.w * 0.5, age);
        float lit = sheet(world, u) * mie(world, u.a.xyz) * rodShadow(world, u);
        float glass = inst == 1 ? 0.035 : 1.0;
        o.color = float4(LED * k.x * form * swell * fade * lit * sq(1.5 / o.size) * glass * 2.2, 0);
        return o;
    }

    // The steady drizzle of alcohol droplets falling through the light sheet, each glinting as it crosses it.
    // Three points per droplet across a 1/50 s shutter smear each into a short streak.
    vertex P chamberRain(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {
        uint id = vid / 3;
        float speed = 5 + 14 * hash11(float(id) * 1.37);
        float span = 26;
        float period = span / speed;
        float t = (u.time - float(vid % 3) * 0.01) / period + hash11(float(id) * 7.13);
        float cycle = floor(t), ph = fract(t);
        float2 r = hash22(float2(float(id), cycle * 13.1 + 1.7));
        float3 p = float3(mix(-u.c.x, u.c.x, r.x), span * (1 - ph), mix(u.c.y, u.c.z, r.y));
        float4 clip = project(p, u.a.xyz, u.b.xyz, u.a.w, u.res.x / u.res.y);
        P o;
        o.pos = clip;
        float coc = u.b.w * abs(1 - u.c.w / clip.w);
        o.size = sqrt(1.5 * 1.5 + coc * coc);
        float glint = 0.3 + 0.7 * hash11(float(id) * 3.1 + cycle);
        o.color = float4(LED * sheet(p, u) * mie(p, u.a.xyz) * glint * 0.2 * sq(1.5 / o.size), 0);
        return o;
    }
    """
}
