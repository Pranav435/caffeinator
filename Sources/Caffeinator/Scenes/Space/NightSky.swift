import AppKit
import Metal
import simd

/// The real sky over the user right now: 5,000 catalogue stars coloured by temperature, the Milky Way from its
/// mapped outline, constellation figures, the planets and the Moon in its true phase, meteors from active showers,
/// above a dark horizon. The sky turns in real time while the view drifts slowly around the compass.
@MainActor final class NightSky: Scene {
    /// Matches `struct Sky` in the shader.
    private struct SkyU {
        var eq0, eq1, eq2: SIMD4<Float>
        var cam0, cam1, cam2: SIMD4<Float>
        var view: SIMD4<Float>
        var sun: SIMD4<Float>
        var moon: SIMD4<Float>
        var glow: SIMD4<Float>
    }

    private struct Body { var dir: SIMD4<Float>; var color: SIMD4<Float>; var extra: SIMD4<Float> }

    private struct Meteor {
        var start: SIMD3<Double>
        var tangent: SIMD3<Double>
        var speed: Double
        var age: Double = 0
        var life: Double
        var trail: Double
        var brightness: Float
        var color: SIMD3<Float>
    }

    private let back: MTLRenderPipelineState
    private let starPipe: MTLRenderPipelineState
    private let linePipe: MTLRenderPipelineState
    private let bodyPipe: MTLRenderPipelineState
    private let discPipe: MTLRenderPipelineState
    private let meteorPipe: MTLRenderPipelineState
    private let finish: MTLRenderPipelineState
    private let labelPipe: MTLRenderPipelineState
    private let groundLabelPipe: MTLRenderPipelineState
    private let bloom: Bloom
    private let milkyWay: MTLTexture
    private let stars: MTLBuffer
    private let dust: MTLBuffer
    private let segments: MTLBuffer
    private let starCount: Int
    private let dustCount: Int
    private let segmentCount: Int
    private let rings: [Ring]
    private var light: MTLTexture?
    /// The sky's glow and the Milky Way, smooth enough to render at half resolution.
    private var backdrop: MTLTexture?
    private var labels: Labels?

    private var meteors: [Meteor] = []
    private var meteorClock: Double = 0
    private let offset: TimeInterval
    private var heading: Double?
    private var aim: Double = 27
    private var summary = ""
    private var summaryAt: Float = -100
    private var lastBell: Float = -10

    var scale: Float { 1 }
    var facts: String? { summary.isEmpty ? nil : summary }

    init(gpu: GPU) throws {
        let library = try gpu.library(SpaceKit.metal + Self.source)
        back = try gpu.pipeline(library, "skyBack")
        starPipe = try gpu.pipeline(library, "glowFragment", vertex: "starVertex", blend: .add)
        linePipe = try gpu.pipeline(library, "lineFragment", vertex: "figureVertex", blend: .add)
        bodyPipe = try gpu.pipeline(library, "glowFragment", vertex: "bodyVertex", blend: .add)
        discPipe = try gpu.pipeline(library, "discFragment", vertex: "discVertex", blend: .add)
        meteorPipe = try gpu.pipeline(library, "meteorFragment", vertex: "meteorVertex", blend: .add)
        finish = try gpu.pipeline(library, "skyFinish")
        labelPipe = try gpu.pipeline(library, "skyLabelFragment", vertex: "labelVertex", blend: .alpha)
        groundLabelPipe = try gpu.pipeline(library, "labelFragment", vertex: "labelVertex", blend: .alpha)
        bloom = try Bloom(gpu: gpu)
        rings = (0..<6).map { _ in Ring(gpu: gpu, bytes: 4096) }
        // `--variant n` shows the sky n hours from now, for checking night and day in snapshots.
        offset = Double(Snapshot.variant ?? 0) * 3600

        var levels = SkyCatalog.milkyWay.map { $0 * 51 }
        milkyWay = levels.withUnsafeMutableBytes { gpu.texture(720, 360, .r8Unorm, bytes: $0.baseAddress!, bytesPerRow: 720) }

        // Stars: equatorial unit vector and magnitude, then colour and a twinkle seed.
        let catalogue = SkyCatalog.stars.flatMap { s -> [SIMD4<Float>] in
            let v = SkyCatalog.unit(ra: Double(s.ra), dec: Double(s.dec))
            let c = Self.tint(SkyCatalog.blackbody(SkyCatalog.temperature(bv: Double(s.bv))))
            return [SIMD4(Float(v.x), Float(v.y), Float(v.z), s.mag), SIMD4(c, .random(in: 0..<1))]
        }
        starCount = catalogue.count / 2
        stars = gpu.buffer(catalogue)

        // Unresolved stars too faint for the catalogue, scattered where the Milky Way outline says the sky is rich.
        var faint: [SIMD4<Float>] = []
        var tries = 0
        while faint.count < 2 * 24000, tries < 400_000 {
            tries += 1
            let z = Double.random(in: -1...1), phi = Double.random(in: 0..<2 * .pi)
            let g = SIMD3(sqrt(1 - z * z) * cos(phi), sqrt(1 - z * z) * sin(phi), z)
            let l = Astro.norm(Astro.deg(atan2(g.y, g.x))), b = Astro.deg(asin(z))
            let cell = min(Int((b + 90) * 2), 359) * 720 + min(Int(l * 2), 719)
            let rich = Double(SkyCatalog.milkyWay[cell]) / 5
            guard Double.random(in: 0..<1) < 0.05 + 0.95 * pow(rich, 0.7) else { continue }
            let rows = SkyCatalog.galactic
            let e = rows[0] * g.x + rows[1] * g.y + rows[2] * g.z
            let mag = Float(6.3 + 2.6 * pow(Double.random(in: 0..<1), 0.6))
            let c = Self.tint(SkyCatalog.blackbody(SkyCatalog.temperature(bv: .random(in: -0.1...1.4))))
            faint += [SIMD4(Float(e.x), Float(e.y), Float(e.z), mag), SIMD4(c, .random(in: 0..<1))]
        }
        dustCount = faint.count / 2
        dust = gpu.buffer(faint)

        // Constellation figures, cut into short pieces so they curve with the projection, with a gap at each star.
        var pieces: [SIMD4<Float>] = []
        for line in SkyCatalog.lines {
            for (a, b) in zip(line, line.dropFirst()) {
                let p = SkyCatalog.unit(ra: Double(a.x), dec: Double(a.y)), q = SkyCatalog.unit(ra: Double(b.x), dec: Double(b.y))
                let angle = acos(min(max((p * q).sum(), -1), 1))
                let gap = min(Astro.rad(0.9), angle * 0.3)
                guard angle > 2 * gap else { continue }
                let n = max(Int(Astro.deg(angle) / 2), 1)
                for k in 0..<n {
                    let t0 = (gap + (angle - 2 * gap) * Double(k) / Double(n)) / angle
                    let t1 = (gap + (angle - 2 * gap) * Double(k + 1) / Double(n)) / angle
                    let u = Self.slerp(p, q, t0, angle), v = Self.slerp(p, q, t1, angle)
                    pieces += [SIMD4(Float(u.x), Float(u.y), Float(u.z), 0), SIMD4(Float(v.x), Float(v.y), Float(v.z), 0)]
                }
            }
        }
        segmentCount = pieces.count / 2
        segments = gpu.buffer(pieces)
        Location.shared.request()
    }

    private static func slerp(_ p: SIMD3<Double>, _ q: SIMD3<Double>, _ t: Double, _ angle: Double) -> SIMD3<Double> {
        let s = sin(angle)
        return s < 1e-6 ? p : p * (sin((1 - t) * angle) / s) + q * (sin(t * angle) / s)
    }

    /// Star colours a touch richer than the eye sees them, but never far from white.
    private static func tint(_ c: SIMD3<Float>) -> SIMD3<Float> {
        let grey = (c * SIMD3(0.3, 0.55, 0.15)).sum()
        let rich = SIMD3<Float>(repeating: grey) + (c - grey) * 1.7
        return simd_clamp(rich / max(rich.max(), 0.01), .zero, SIMD3(repeating: 1))
    }

    // MARK: Sky geometry

    private struct View {
        var eq: [SIMD3<Double>]     // equatorial → horizon (east, north, up)
        var cam: [SIMD3<Double>]    // horizon → camera (right, up, forward)
        var scale: Double
        var aspect: Double
        var size: SIMD2<Double>

        func horizon(_ e: SIMD3<Double>) -> SIMD3<Double> { SIMD3((eq[0] * e).sum(), (eq[1] * e).sum(), (eq[2] * e).sum()) }

        /// Target pixels (top-left origin) of a horizon direction, or nil behind the viewer.
        func pixel(_ h: SIMD3<Double>) -> SIMD2<Float>? {
            let c = SIMD3((cam[0] * h).sum(), (cam[1] * h).sum(), (cam[2] * h).sum())
            guard c.z > -0.2 else { return nil }
            let k = 2 / (1 + c.z) * scale
            let x = (c.x * k / aspect * 0.5 + 0.5) * size.x, y = (0.5 - c.y * k * 0.5) * size.y
            guard x > -200, x < size.x + 200, y > -100, y < size.y + 100 else { return nil }
            return SIMD2(Float(x), Float(y))
        }
    }

    private static func enu(alt: Double, az: Double) -> SIMD3<Double> {
        let a = Astro.rad(alt), z = Astro.rad(az)
        return SIMD3(cos(a) * sin(z), cos(a) * cos(z), sin(a))
    }

    private static func altAz(_ h: SIMD3<Double>) -> (alt: Double, az: Double) {
        (Astro.deg(asin(min(max(h.z, -1), 1))), Astro.norm(Astro.deg(atan2(h.x, h.y))))
    }

    private func view(_ date: Date, _ f: Frame) -> View {
        let lat = Astro.rad(Location.shared.latitude)
        let theta = Astro.rad(Astro.lst(date, longitude: Location.shared.longitude))
        let eq: [SIMD3<Double>] = [SIMD3(-sin(theta), cos(theta), 0),
                                   SIMD3(-sin(lat) * cos(theta), -sin(lat) * sin(theta), cos(lat)),
                                   SIMD3(cos(lat) * cos(theta), cos(lat) * sin(theta), sin(lat))]
        // A slow drift round the compass, a full turn every half hour, with the gaze rising and falling a little.
        let t = Double(f.time)
        let az = (heading ?? 0) + t * 0.2
        let alt = aim + 4 * sin(t * 0.011 + Double(f.seed))
        let fwd = Self.enu(alt: alt, az: az)
        let right = simd_normalize(SIMD3(fwd.y, -fwd.x, 0))
        let up = simd_cross(right, fwd)
        let half = Astro.rad(36.0)
        return View(eq: eq, cam: [right, up, fwd], scale: 1 / (2 * tan(half / 2)), aspect: Double(f.aspect),
                    size: SIMD2(Double(f.size.x), Double(f.size.y)))
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        let date = f.now.addingTimeInterval(offset)
        if heading == nil { (heading, aim) = startHeading(date, f.seed) }
        let v = view(date, f)
        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            backdrop = f.gpu.texture(f.target.width / 2, f.target.height / 2)
            labels = makeLabels(f)
        }

        let sunEq = Astro.sun(date)
        let sun = v.horizon(SkyCatalog.unit(ra: sunEq.ra, dec: sunEq.dec))
        let moonEq = Astro.moon(date)
        var moon = v.horizon(SkyCatalog.unit(ra: moonEq.ra, dec: moonEq.dec))
        // Topocentric parallax: from the surface the Moon sits up to a degree lower than from Earth's centre.
        let moonAlt = Self.altAz(moon)
        moon = Self.enu(alt: moonAlt.alt - Astro.deg(asin(cos(Astro.rad(moonAlt.alt)) / max(moonEq.distance, 55))), az: moonAlt.az)
        let phase = Astro.moonPhase(date)
        let sunAlt = Self.altAz(sun).alt
        let moonUp = max(moon.z, 0) * phase.lit
        let density = Float(f.size.y / 1000)

        let f4 = { (d: SIMD3<Double>, w: Double) in SIMD4(Float(d.x), Float(d.y), Float(d.z), Float(w)) }
        let u = SkyU(eq0: f4(v.eq[0], 0), eq1: f4(v.eq[1], 0), eq2: f4(v.eq[2], 0),
                     cam0: f4(v.cam[0], 0), cam1: f4(v.cam[1], 0), cam2: f4(v.cam[2], 0),
                     view: SIMD4(Float(v.scale), f.aspect, f.size.y / 2, f.time.truncatingRemainder(dividingBy: 1000)),
                     sun: f4(sun, sunAlt), moon: f4(moon, phase.lit),
                     glow: SIMD4(Float(moonUp), Float(min(max((sunAlt + 18) / 18, 0), 1)), f.seed, density))
        let skyBuffer = rings[0].next(u)
        let light = light!
        let uni = f.uniforms()

        f.cmd.draw(back, into: backdrop!, uniforms: uni, textures: [milkyWay], buffers: [skyBuffer])
        f.cmd.draw(starPipe, into: light, uniforms: f.uniforms([0.55, 0, 0, 0]), buffers: [skyBuffer, dust],
                   vertices: 4, instances: dustCount, primitive: .triangleStrip)
        f.cmd.draw(linePipe, into: light, clear: false, uniforms: uni, buffers: [skyBuffer, segments],
                   vertices: 4, instances: segmentCount, primitive: .triangleStrip)
        f.cmd.draw(starPipe, into: light, clear: false, uniforms: f.uniforms([1, 0, 0, 0]), buffers: [skyBuffer, stars],
                   vertices: 4, instances: starCount, primitive: .triangleStrip)

        // Planets as steady points; the Moon and Sun as lit discs, drawn a few times their true size.
        var bodies: [Body] = []
        for body in Astro.Body.allCases {
            let q = Astro.planet(body, date)
            let h = v.horizon(SkyCatalog.unit(ra: q.ra, dec: q.dec))
            guard h.z > -0.05 else { continue }
            bodies.append(Body(dir: f4(h, 0), color: SIMD4(Self.planetColor(body), Float(Self.magnitude(body, date))), extra: .zero))
        }
        f.cmd.draw(bodyPipe, into: light, clear: false, uniforms: uni, buffers: [skyBuffer, rings[1].next(bodies)],
                   vertices: 4, instances: bodies.count, primitive: .triangleStrip)
        var discs: [Body] = []
        if moon.z > -0.05 {
            discs.append(Body(dir: f4(moon, 1), color: SIMD4(1.0, 0.96, 0.9, Float(phase.lit)), extra: .zero))
        }
        if sun.z > -0.05 {
            discs.append(Body(dir: f4(sun, 2), color: SIMD4(1.0, 0.8, 0.55, 1), extra: .zero))
        }
        f.cmd.draw(discPipe, into: light, clear: false, uniforms: uni, buffers: [skyBuffer, rings[2].next(discs)],
                   vertices: 4, instances: discs.count, primitive: .triangleStrip)

        updateMeteors(f, date, v)
        let streaks = meteors.map { m in
            [SIMD4(Float(m.start.x), Float(m.start.y), Float(m.start.z), Float(m.speed)),
             SIMD4(Float(m.tangent.x), Float(m.tangent.y), Float(m.tangent.z), Float(m.trail)),
             SIMD4(Float(m.age), Float(m.life), m.brightness, 0), SIMD4(m.color, 0)]
        }.flatMap { $0 }
        f.cmd.draw(meteorPipe, into: light, clear: false, uniforms: uni, buffers: [skyBuffer, rings[3].next(streaks)],
                   vertices: 4, instances: meteors.count, primitive: .triangleStrip)

        let glow = bloom.glow(f.cmd, light)
        f.cmd.draw(finish, into: f.target, uniforms: uni, textures: [light, glow, backdrop!], buffers: [skyBuffer])

        if let labels {
            let (sky, ground) = labelQuads(labels, v, date, density)
            let atlas = f.uniforms(SIMD4(labels.size.x, labels.size.y, 0, 0))
            f.cmd.draw(labelPipe, into: f.target, clear: false, uniforms: atlas, textures: [labels.texture],
                       buffers: [rings[4].next(sky), skyBuffer], vertices: 4, instances: sky.count, primitive: .triangleStrip)
            f.cmd.draw(groundLabelPipe, into: f.target, clear: false, uniforms: atlas, textures: [labels.texture],
                       buffers: [rings[5].next(ground)], vertices: 4, instances: ground.count, primitive: .triangleStrip)
        }

        if f.time - summaryAt > 5 || summary.isEmpty {
            summaryAt = f.time
            summary = describe(date, v, sun: sun, moon: moon, lit: phase.lit, waxing: phase.waxing)
        }
        if let synth = f.sound { play(synth, f, date, v, moonUp: moonUp) }
    }

    /// Faces the brightest thing up there to begin with: the Moon, a bright planet or a shower's radiant.
    private func startHeading(_ date: Date, _ seed: Float) -> (Double, Double) {
        let lat = Location.shared.latitude, lon = Location.shared.longitude
        var candidates: [(score: Double, alt: Double, az: Double)] = []
        let moon = Astro.horizontal(Astro.moon(date), date, latitude: lat, longitude: lon)
        candidates.append((3, moon.alt, moon.az))
        for body in [Astro.Body.venus, .jupiter, .mars, .saturn] {
            let h = Astro.horizontal(Astro.planet(body, date), date, latitude: lat, longitude: lon)
            candidates.append((2 - Self.magnitude(body, date) * 0.1, h.alt, h.az))
        }
        if let (shower, _) = Shower.active(on: date) {
            let r = shower.radiant(on: date)
            let h = Astro.horizontal(Astro.Equatorial(ra: r.ra, dec: r.dec, distance: 1), date, latitude: lat, longitude: lon)
            candidates.append((2.5, h.alt, h.az))
        }
        // Something that fits in the frame, starting a little to its left so the drift carries the view across it.
        guard let best = candidates.filter({ $0.alt > 8 && $0.alt < 52 }).max(by: { $0.score < $1.score }) else {
            return (Double(seed * 360).truncatingRemainder(dividingBy: 360), 28)
        }
        return (best.az - 25, min(max(best.alt - 4, 26), 31))
    }

    // MARK: Meteors

    private func updateMeteors(_ f: Frame, _ date: Date, _ v: View) {
        let dt = Double(f.dt)
        for i in meteors.indices { meteors[i].age += dt }
        meteors.removeAll { $0.age > $0.life }

        // A sporadic meteor every minute or so, plus the active shower's rate scaled up to suit a screen.
        let shower = Shower.active(on: date)
        var radiant: SIMD3<Double>?
        var showerRate = 0.0
        if let (s, zhr) = shower {
            let r = s.radiant(on: date)
            let h = v.horizon(SkyCatalog.unit(ra: r.ra, dec: r.dec))
            if h.z > 0 {
                radiant = h
                showerRate = zhr * h.z * 0.12 / 60
            }
        }
        let rate = 1.0 / 60 + showerRate
        meteorClock -= dt
        guard meteorClock <= 0 else { return }
        meteorClock = -log(Double.random(in: 0.0001..<1)) / rate
        guard meteors.count < 6 else { return }
        let fromShower = radiant != nil && Double.random(in: 0..<1) < showerRate / rate

        // Somewhere in view and well above the horizon.
        let px = Double.random(in: -0.85...0.85) * v.aspect, py = Double.random(in: -0.2...0.9)
        let q = SIMD2(px, py) / v.scale, r2 = (q * q).sum()
        let c = SIMD3(4 * q.x, 4 * q.y, 4 - r2) / (4 + r2)
        let p = simd_normalize(v.cam[0] * c.x + v.cam[1] * c.y + v.cam[2] * c.z)
        guard p.z > 0.12 else { return }
        let source = fromShower ? radiant! : simd_normalize(SIMD3(Double.random(in: -1...1), Double.random(in: -1...1), Double.random(in: -0.3...1)))
        let away = p * (p * source).sum() - source
        guard simd_length(away) > 0.05 else { return }
        let tangent = simd_normalize(away)
        // Meteors far from the radiant streak longer and faster; faster showers burn whiter and greener.
        let distance = acos(min(max((p * source).sum(), -1), 1))
        let speed = fromShower ? shower!.0.speed : Double.random(in: 25...65)
        let angular = Astro.rad(8 + 18 * sin(distance)) * speed / 50
        let mag = -1.5 + 5.5 * pow(Double.random(in: 0..<1), 0.6)
        let warm = SIMD3<Float>(1.0, 0.85, 0.6), fast = SIMD3<Float>(0.75, 1.0, 0.85)
        let color = simd_mix(warm, fast, SIMD3(repeating: Float(min(max((speed - 30) / 40, 0), 1))))
        meteors.append(Meteor(start: p, tangent: tangent, speed: angular, life: Double.random(in: 0.5...1.1),
                              trail: Astro.rad(Double.random(in: 4...9)), brightness: Float(pow(10, -0.4 * (mag - 2))), color: color))
        if let synth = f.sound, f.time - lastBell > 0.4 {
            lastBell = f.time
            let pitches: [Float] = [1174.7, 1318.5, 1568, 1760, 1975.5]
            let screenX = Float(px / v.aspect)
            synth.play(.bell, pitches.randomElement()! * (mag < 0 ? 0.5 : 1), amp: Float(min(0.03 + 0.03 * (2 - mag) / 3, 0.1)),
                       pan: screenX * 0.7, length: 3.5)
        }
    }

    // MARK: Labels

    private func makeLabels(_ f: Frame) -> Labels {
        var items: [(String, Labels.Style)] = []
        let small = Labels.Style(size: 10.5, weight: .regular, tracking: 1.6)
        for c in SkyCatalog.constellations { items.append((c.name.uppercased(), small)) }
        for s in SkyCatalog.starNames { items.append((s.name, Labels.Style(size: 10, weight: .regular))) }
        for b in Astro.Body.allCases { items.append((b.name, Labels.Style(size: 11.5, weight: .medium))) }
        items.append(("Moon", Labels.Style(size: 11.5, weight: .medium)))
        items.append(("Sun", Labels.Style(size: 11.5, weight: .medium)))
        for s in Shower.all { items.append((s.name, Labels.Style(size: 10.5, weight: .regular, tracking: 0.5))) }
        for c in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] { items.append((c, Labels.Style(size: 13, weight: .semibold))) }
        return Labels(gpu: f.gpu, density: f.overlay.scale * CGFloat(scale), items)
    }

    private func labelQuads(_ labels: Labels, _ v: View, _ date: Date, _ density: Float) -> ([LabelQuad], [LabelQuad]) {
        var sky: [LabelQuad] = []
        func add(_ text: String, _ h: SIMD3<Double>, dy: Float, _ color: SIMD4<Float>, minAlt: Double = 0.03) {
            guard h.z > minAlt, let p = v.pixel(h) else { return }
            let fade = Float(min((h.z - minAlt) / 0.08, 1))
            if let q = labels.quad(text, at: p + SIMD2(0, dy * density), anchor: [0.5, dy < 0 ? 1 : 0], color: color * fade) {
                sky.append(q)
            }
        }
        for c in SkyCatalog.constellations {
            let a: Float = c.rank == 1 ? 0.34 : c.rank == 2 ? 0.26 : 0.19
            add(c.name.uppercased(), v.horizon(SkyCatalog.unit(ra: Double(c.ra), dec: Double(c.dec))), dy: 0,
                .premultiplied(0.62, 0.74, 1.0, a))
        }
        for s in SkyCatalog.starNames {
            add(s.name, v.horizon(SkyCatalog.unit(ra: Double(s.ra), dec: Double(s.dec))), dy: 7, .premultiplied(1.0, 0.9, 0.75, 0.4))
        }
        for b in Astro.Body.allCases where b != .uranus && b != .neptune {
            let q = Astro.planet(b, date)
            add(b.name, v.horizon(SkyCatalog.unit(ra: q.ra, dec: q.dec)), dy: -9, .premultiplied(1.0, 0.88, 0.62, 0.75))
        }
        let m = Astro.moon(date)
        add("Moon", v.horizon(SkyCatalog.unit(ra: m.ra, dec: m.dec)), dy: -22, .premultiplied(1.0, 0.95, 0.85, 0.7))
        let sun = Astro.sun(date)
        add("Sun", v.horizon(SkyCatalog.unit(ra: sun.ra, dec: sun.dec)), dy: -22, .premultiplied(1.0, 0.9, 0.7, 0.7))
        if let (s, _) = Shower.active(on: date) {
            let r = s.radiant(on: date)
            add(s.name, v.horizon(SkyCatalog.unit(ra: r.ra, dec: r.dec)), dy: 0, .premultiplied(0.7, 1.0, 0.85, 0.42))
        }

        var ground: [LabelQuad] = []
        for (i, name) in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"].enumerated() {
            guard let p = v.pixel(Self.enu(alt: 0, az: Double(i) * 45)).map({ $0 + SIMD2(0, 24 * density) }) else { continue }
            let a: Float = i % 2 == 0 ? 0.55 : 0.32
            if let q = labels.quad(name, at: p, color: .premultiplied(0.75, 0.82, 1.0, a)) { ground.append(q) }
        }
        return (sky, ground)
    }

    // MARK: Facts

    private static func direction(_ az: Double) -> String {
        ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"][Int((az + 22.5) / 45) % 8]
    }

    private static func place(_ h: SIMD3<Double>) -> String {
        let p = altAz(h)
        if p.alt > 75 { return "overhead" }
        let d = direction(p.az)
        return p.alt > 50 ? "high in the \(d)" : p.alt < 12 ? "low in the \(d)" : "in the \(d)"
    }

    private func describe(_ date: Date, _ v: View, sun: SIMD3<Double>, moon: SIMD3<Double>, lit: Double, waxing: Bool) -> String {
        let time = date.formatted(date: .omitted, time: .shortened)
        let town = Location.shared.place.isEmpty ? "your location" : Location.shared.place
        let sunAlt = Self.altAz(sun).alt
        var head = "Sky over \(town) at \(time)"
        if sunAlt > 0 { head += ", shown without the daylight" } else if sunAlt > -12 { head += ", in twilight" }

        var seen: [String] = []
        let planets = Astro.Body.allCases.filter { $0 != .uranus && $0 != .neptune }
            .map { b -> (Astro.Body, Double, SIMD3<Double>) in
                let q = Astro.planet(b, date)
                return (b, Self.magnitude(b, date), v.horizon(SkyCatalog.unit(ra: q.ra, dec: q.dec)))
            }
            .filter { $0.2.z > 0.02 }
            .sorted { $0.1 < $1.1 }
        for (b, _, h) in planets.prefix(3) { seen.append("\(b.name) \(Self.place(h))") }
        if moon.z > 0 {
            seen.append("Moon \(Int((lit * 100).rounded())) percent lit, \(waxing ? "waxing" : "waning"), \(Self.place(moon))")
        }
        if sunAlt > 0 { seen.append("the Sun \(Self.place(sun))") }
        var text = head + (seen.isEmpty ? ". No planets or Moon above the horizon." : ": " + seen.joined(separator: ", ") + ".")
        if let (s, zhr) = Shower.active(on: date) {
            let r = s.radiant(on: date)
            let h = v.horizon(SkyCatalog.unit(ra: r.ra, dec: r.dec))
            text += " \(s.adjective) meteors active, up to \(Int(zhr.rounded())) an hour"
            text += h.z > 0 ? ", radiant \(Self.place(h))." : ", radiant below the horizon."
        }
        return text
    }

    // MARK: Planets

    private static func planetColor(_ b: Astro.Body) -> SIMD3<Float> {
        switch b {
        case .mercury: [0.9, 0.82, 0.72]
        case .venus: [1.0, 0.96, 0.86]
        case .mars: [1.0, 0.58, 0.34]
        case .jupiter: [1.0, 0.91, 0.78]
        case .saturn: [1.0, 0.88, 0.64]
        case .uranus: [0.72, 0.92, 1.0]
        case .neptune: [0.56, 0.68, 1.0]
        }
    }

    /// Apparent magnitude from distances and phase angle (Schlyter's formulas, rings ignored).
    static func magnitude(_ b: Astro.Body, _ date: Date) -> Double {
        let helio = Astro.heliocentric(b, date), sun = Astro.sunEcliptic(date)
        let geo = helio + sun
        let r = simd_length(helio), d = simd_length(geo), e = simd_length(sun)
        let i = Astro.deg(acos(min(max((r * r + d * d - e * e) / (2 * r * d), -1), 1)))
        let base = 5 * log10(r * d)
        switch b {
        case .mercury: return -0.36 + base + 0.027 * i + 2.2e-13 * pow(i, 6)
        case .venus: return -4.34 + base + 0.013 * i + 4.2e-7 * pow(i, 3)
        case .mars: return -1.51 + base + 0.016 * i
        case .jupiter: return -9.25 + base + 0.014 * i
        case .saturn: return -9.0 + base + 0.044 * i
        case .uranus: return -7.15 + base + 0.001 * i
        case .neptune: return -6.90 + base + 0.001 * i
        }
    }

    // MARK: Sound

    /// A quiet pad whose chord steps with the sidereal hour, a moonlit shimmer, and a breath of wind.
    private func play(_ synth: Synth, _ f: Frame, _ date: Date, _ v: View, moonUp: Double) {
        let lst = Astro.lst(date, longitude: Location.shared.longitude) / 15
        let roots: [Float] = [98, 110, 123.47, 146.83, 164.81, 146.83]
        let root = roots[Int(lst / 4) % roots.count]
        let breath = 0.8 + 0.2 * sin(f.time * 0.07)
        synth.drone(0, root / 2, 0.05 * breath)
        synth.drone(1, root, 0.04 * breath, pan: -0.3)
        synth.drone(2, root * 1.5, 0.03 * breath, pan: 0.3)
        synth.drone(3, root * 2.25, 0.016 * breath, pan: -0.15)
        synth.drone(4, root * 3, 0.01, pan: 0.2)
        let shimmer = Float(0.004 + 0.012 * moonUp)
        synth.drone(5, root * 6, shimmer, pan: -0.5)
        synth.drone(6, root * 6 + 0.35, shimmer, pan: 0.5)
        synth.noise(0.012, cutoff: 260)
    }

    // MARK: Shaders

    private static let source = """
    struct Sky {
        float4 eq[3];     // equatorial → horizon rows (east, north, up)
        float4 cam[3];    // horizon → camera rows (right, up, forward)
        float4 view;      // stereographic scale, aspect, pixels per unit, time
        float4 sun;       // horizon direction, altitude in degrees
        float4 moon;      // horizon direction, lit fraction
        float4 glow;      // moonlight, twilight 0...1, seed, pixel density
    };

    constant float3x3 GALACTIC = float3x3(float3(-0.0548755604, 0.4941094279, -0.8676661490),
                                          float3(-0.8734370902, -0.4448296300, -0.1980763734),
                                          float3(-0.4838350155, 0.7469822445, 0.4559837762));

    inline float3 rows(constant float4 *m, float3 v) { return float3(dot(m[0].xyz, v), dot(m[1].xyz, v), dot(m[2].xyz, v)); }
    inline float3 cols(constant float4 *m, float3 v) { return m[0].xyz * v.x + m[1].xyz * v.y + m[2].xyz * v.z; }

    // Stereographic projection to centered coordinates (y up), and back.
    inline float2 stereo(float3 c, float s) { return c.xy * (2 / max(1 + c.z, 1e-3)) * s; }
    inline float3 unstereo(float2 p, float s) {
        float2 q = p / s;
        float r2 = dot(q, q);
        return float3(4 * q, 4 - r2) / (4 + r2);
    }
    inline float2 toPixels(float2 p, constant Sky &k, float2 res) {
        return float2((p.x / k.view.y * 0.5 + 0.5) * res.x, (0.5 - p.y * 0.5) * res.y);
    }
    // The horizon direction under a pixel.
    inline float3 lookAt(float2 pixel, constant Sky &k, float2 res) {
        float2 p = float2((pixel.x / res.x - 0.5) * 2 * k.view.y, (0.5 - pixel.y / res.y) * 2);
        return cols(k.cam, unstereo(p, k.view.x));
    }

    // Altitude of the distant hills and treeline, in degrees, at an azimuth.
    inline float ridge(float az, float seed) {
        float2 c = float2(cos(az), sin(az));
        float hills = 0.4 + 2.3 * smoothstep(0.3, 0.75, fbm(c * 1.7 + seed, 4));
        float forest = smoothstep(0.45, 0.6, noise(c * 3.0 + seed * 1.7 + 9.0));
        float trees = forest * (0.45 + 0.9 * fbm(c * 70.0 + seed, 3)) * (0.6 + 0.6 * noise(c * 230.0));
        return hills + trees;
    }

    // How much of a pixel is ground, antialiased over about a pixel.
    inline float groundCover(float3 h, constant Sky &k, float2 res) {
        float alt = asin(clamp(h.z, -1.0, 1.0)) * 57.29578;
        if (alt > 4.6) return 0;    // above the highest treetop
        if (alt < 0) return 1;      // below the lowest hill
        float az = atan2(h.x, h.y);
        float perDegree = k.view.x * k.view.z * 0.01745 * 1.2;
        return clamp((ridge(az, k.glow.z) - alt) * perDegree + 0.5, 0.0, 1.0);
    }

    // Sky glow from altitude, twilight, moonlight and airglow, and the Milky Way.
    fragment float4 skyBack(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                            texture2d<float> mw [[texture(0)]]) {
        float3 h = lookAt(in.uv * u.res, k, u.res);
        float alt = h.z;
        float low = exp(-max(alt, 0.0) * 5.0);
        float3 sky = mix(float3(0.0022, 0.0032, 0.0085), float3(0.008, 0.011, 0.02), low);
        sky += float3(0.003, 0.0065, 0.004) * exp(-pow((alt - 0.17) / 0.12, 2.0));   // faint green airglow band

        // Twilight and daylight: a deep blue dome with a warm glow on the Sun's side of the horizon.
        float tw = k.glow.y;
        float toSun = dot(h, k.sun.xyz);
        float3 sunDir2 = normalize(float3(k.sun.xy, 0) + 1e-5);
        float side = max(dot(normalize(float3(h.xy, 0) + 1e-5), sunDir2), 0.0);
        sky += tw * tw * (float3(0.006, 0.013, 0.032) * (0.5 + 0.5 * low));
        sky += tw * tw * tw * float3(0.10, 0.045, 0.016) * pow(side, 3.0) * low * low;
        sky += smoothstep(-2.0, 6.0, k.sun.w) * float3(0.004, 0.012, 0.035) * (0.6 + 0.4 * low);
        sky += smoothstep(-6.0, 4.0, k.sun.w) * float3(0.3, 0.2, 0.1) * pow(max(toSun, 0.0), 40.0) * 0.4;

        // Moonlight scatters blue across the sky and pools around the Moon.
        float moon = k.glow.x;
        float toMoon = max(dot(h, k.moon.xyz), 0.0);
        sky += moon * (float3(0.004, 0.008, 0.018) + float3(0.04, 0.04, 0.045) * pow(toMoon, 60.0) + float3(0.02, 0.022, 0.03) * pow(toMoon, 8.0) * 0.2);

        // The Milky Way: the outline's brightness steps, softened, with mottled star clouds and dust lanes.
        float3 g = GALACTIC * cols(k.eq, h);
        constexpr sampler s(filter::linear, address::repeat);
        float2 uv = float2(fract(atan2(g.y, g.x) / TAU), asin(clamp(g.z, -1.0, 1.0)) / PI + 0.5);
        float2 o = float2(1.2 / 720, 1.2 / 360);
        float lev = mw.sample(s, uv).r * 0.36 + 0.16 * (mw.sample(s, uv + float2(o.x, 0)).r + mw.sample(s, uv - float2(o.x, 0)).r
                  + mw.sample(s, uv + float2(0, o.y)).r + mw.sample(s, uv - float2(0, o.y)).r);
        float3 galaxy = 0;
        if (lev > 0.003) {
            float3 q = g * 9.0;
            float clouds = fbm(q + 3.7, 5);
            float grain = fbm(q * 5.0 + 1.3, 3);
            float lanes = smoothstep(0.5, 0.72, fbm(q * 1.8 + 21.0, 4)) * exp(-abs(g.z) * 9.0);
            float b = lev * (0.35 + 1.3 * clouds * clouds) * (0.7 + 0.6 * grain) * (1 - 0.65 * lanes);
            float core = exp(-pow(acos(clamp(g.x, -1.0, 1.0)) / 0.75, 2.0));
            float3 tint = mix(float3(0.62, 0.70, 0.95), float3(1.0, 0.80, 0.58), core * 0.8 + lanes * 0.2);
            tint = mix(tint, float3(1.0, 0.55, 0.6), smoothstep(0.62, 0.8, fbm(q * 2.6 + 7.0, 3)) * 0.35);
            float clear = 1 - 0.6 * max(moon * 1.4, tw);
            galaxy = tint * b * 0.045 * clear * smoothstep(-0.02, 0.25, alt);
        }
        return float4(sky + galaxy, 1);
    }

    struct StarIn { float4 pos; float4 color; };

    // a.x: brightness scale. Atmospheric extinction dims and reddens stars low down; scintillation makes them
    // twinkle more the more air they shine through.
    vertex GlowOut starVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                              constant Sky &k [[buffer(1)]], const device StarIn *stars [[buffer(2)]]) {
        StarIn s = stars[iid];
        GlowOut o;
        o.pos = float4(0, 0, 2, 1);
        o.color = 0;
        o.uv = 0;
        o.extra = 0;
        float3 h = rows(k.eq, s.pos.xyz);
        float3 c = rows(k.cam, h);
        if (h.z < -0.03 || c.z < -0.2) return o;
        float airmass = 1 / max(h.z + 0.025 * exp(-11 * h.z), 0.02);
        float mag = s.pos.w + 0.2 * (airmass - 1);
        float wobble = 0.04 + 0.3 * clamp((airmass - 1) / 8, 0.0, 1.0);
        float flicker = noise(float2(k.view.w * 4.0, s.color.w * 731.0)) * 2 - 1;
        float flux = pow(10.0, -0.4 * (mag - 6.5)) * (1 + wobble * flicker);
        float d = k.glow.w;
        float sigma = d * (0.8 + 0.2 * max(6.5 - mag, 0.0));
        float peak = min(0.065 * pow(flux, 0.55), 4.0) * u.a.x;
        float3 color = s.color.rgb * exp(-float3(0.015, 0.04, 0.09) * (airmass - 1));
        float extent = sigma * 3.2 + (peak > 1 ? d * 4 * sqrt(peak) : 0);
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        float2 p = toPixels(stereo(c, k.view.x), k, u.res) + corner * extent;
        o.pos = pixelToClip(p, u.res);
        o.uv = corner;
        o.color = float4(color * peak, 0);
        float r = extent / sigma;
        o.extra = float4(r * r * 0.5, r * 0.9, peak > 1 ? 0.04 : 0.0, 0);
        return o;
    }

    vertex LineOut figureVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                constant Sky &k [[buffer(1)]], const device float4 *seg [[buffer(2)]]) {
        float3 ha = rows(k.eq, seg[2 * iid].xyz), hb = rows(k.eq, seg[2 * iid + 1].xyz);
        float3 ca = rows(k.cam, ha), cb = rows(k.cam, hb);
        float2 a = toPixels(stereo(ca, k.view.x), k, u.res), b = toPixels(stereo(cb, k.view.x), k, u.res);
        float fade = smoothstep(0.0, 0.12, min(ha.z, hb.z)) * step(-0.1, min(ca.z, cb.z));
        float clear = 1 - 0.5 * max(k.glow.x, k.glow.y);
        float4 color = float4(0.32, 0.5, 0.95, 0) * 0.075 * fade * clear;
        return lineQuad(a, b, 1.1 * k.glow.w, color, vid, u.res);
    }

    struct BodyIn { float4 dir; float4 color; float4 extra; };

    // Planets as steady points: colour and magnitude in `color`.
    vertex GlowOut bodyVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                              constant Sky &k [[buffer(1)]], const device BodyIn *bodies [[buffer(2)]]) {
        BodyIn b = bodies[iid];
        float2 center = toPixels(stereo(rows(k.cam, b.dir.xyz), k.view.x), k, u.res);
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        float d = k.glow.w;
        float mag = b.color.w + 0.2 * (1 / max(b.dir.z + 0.025, 0.02) - 1);
        float flux = pow(10.0, -0.4 * (mag - 6.5));
        float sigma = d * min(1.0 + 0.2 * max(6.5 - mag, 0.0), 3.0);
        float peak = min(0.065 * pow(flux, 0.55), 3.5);
        float extent = sigma * 3.2 + d * 5 * sqrt(peak);
        float r = extent / sigma;
        GlowOut o;
        o.pos = pixelToClip(center + corner * extent, u.res);
        o.uv = corner;
        o.extra = float4(r * r * 0.5, r * 0.7, 0.05, 0);
        o.color = float4(b.color.rgb * peak, 0);
        return o;
    }

    struct DiscOut { float4 pos [[position]]; float2 uv; float4 color; float4 light; float2 north; };

    // The Moon (dir.w 1) and Sun (dir.w 2), drawn three times their true size. The Moon's light direction is
    // in the quad's frame (x right, y up, z toward us), and `north` turns its face the way the sky does.
    vertex DiscOut discVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                              constant Sky &k [[buffer(1)]], const device BodyIn *bodies [[buffer(2)]]) {
        BodyIn b = bodies[iid];
        float3 c = rows(k.cam, b.dir.xyz);
        float2 center = toPixels(stereo(c, k.view.x), k, u.res);
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        float3 pole = float3(k.eq[0].z, k.eq[1].z, k.eq[2].z);
        float3 toward[2] = { k.sun.xyz, pole };
        float2 dirs[2];
        for (int i = 0; i < 2; i++) {
            float3 nudge = normalize(b.dir.xyz + 0.01 * (toward[i] - b.dir.xyz * dot(toward[i], b.dir.xyz)));
            float2 side = toPixels(stereo(rows(k.cam, nudge), k.view.x), k, u.res) - center;
            dirs[i] = length(side) > 1e-4 ? normalize(side) * float2(1, -1) : float2(0, 1);
        }
        float cosi = clamp(2 * b.color.w - 1, -1.0, 1.0);
        float sini = sqrt(1 - cosi * cosi);
        float radius = 0.26 * 0.01745 * k.view.x * k.view.z * 3.0;
        DiscOut o;
        o.pos = pixelToClip(center + corner * radius * 4.0, u.res);
        o.uv = corner;
        o.color = float4(b.color.rgb, smoothstep(-0.02, 0.03, b.dir.z));
        o.light = b.dir.w < 1.5 ? float4(dirs[0] * sini, cosi, 1) : float4(0, 0, 1, -1);
        o.north = dirs[1];
        return o;
    }

    // Dark maria where they lie on the Moon's near side, in disc coordinates with lunar north up.
    inline float maria(float2 q) {
        const float4 seas[8] = { float4(-0.3, 0.42, 0.27, 0.5), float4(0.17, 0.4, 0.17, 0.5), float4(0.3, 0.1, 0.21, 0.45),
                                 float4(0.68, 0.32, 0.11, 0.55), float4(-0.58, 0.02, 0.34, 0.4), float4(-0.15, -0.33, 0.18, 0.35),
                                 float4(0.55, -0.12, 0.14, 0.4), float4(0.0, 0.72, 0.14, 0.3) };
        float dark = 0;
        for (int i = 0; i < 8; i++) {
            float2 d = q - seas[i].xy;
            dark = max(dark, seas[i].w * smoothstep(seas[i].z, seas[i].z * 0.4, length(d)));
        }
        dark *= 0.8 + 0.4 * noise(q * 6.0);
        float tycho = 0.25 * smoothstep(0.08, 0.0, length(q - float2(-0.12, -0.7)));
        return 1 - dark + tycho;
    }

    // The Moon with its terminator and faint earthshine; the Sun as a limb-darkened disc. Both carry a halo, since
    // the eye never sees them as hard discs.
    fragment float4 discFragment(DiscOut in [[stage_in]]) {
        float2 q = in.uv * 4.0;
        float r2 = dot(q, q);
        float edge = smoothstep(1.06, 0.94, sqrt(r2));
        float3 c = 0;
        if (in.light.w > 0) {
            float3 n = float3(q, sqrt(max(1 - r2, 0.0)));
            float lit = smoothstep(-0.04, 0.08, dot(n, in.light.xyz));
            float2 m = float2(dot(q, float2(in.north.y, -in.north.x)), dot(q, in.north));
            c = in.color.rgb * (lit * maria(m) * 0.8 + 0.01) * edge;
            float lightness = (in.light.z + 1) * 0.5;
            c += in.color.rgb * lightness * (exp(-sqrt(r2) * 1.6) * 0.06 + exp(-r2 * 0.5) * 0.05) * (1 - edge);
        } else {
            float limb = sqrt(max(1 - r2, 0.0));
            c = in.color.rgb * (0.5 + 0.5 * limb) * 3.0 * edge;
            c += in.color.rgb * (exp(-sqrt(r2) * 1.2) * 0.35);
        }
        return float4(c * in.color.a * smoothstep(1.0, 0.85, length(in.uv)), 0);
    }

    struct MeteorOut { float4 pos [[position]]; float4 color; float along; float across; float width; };

    vertex MeteorOut meteorVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                  constant Sky &k [[buffer(1)]], const device float4 *ms [[buffer(2)]]) {
        float4 a = ms[4 * iid], t = ms[4 * iid + 1], info = ms[4 * iid + 2], col = ms[4 * iid + 3];
        float angle = a.w * info.x;
        float trail = min(t.w, angle);
        float3 head = a.xyz * cos(angle) + t.xyz * sin(angle);
        float3 tail = a.xyz * cos(angle - trail) + t.xyz * sin(angle - trail);
        float2 hp = toPixels(stereo(rows(k.cam, head), k.view.x), k, u.res);
        float2 tp = toPixels(stereo(rows(k.cam, tail), k.view.x), k, u.res);
        float life = info.x / info.y;
        float light = pow(sin(PI * clamp(life, 0.0, 1.0)), 0.7) * info.z;
        LineOut l = lineQuad(tp, hp, 2.4 * k.glow.w, float4(col.rgb * light, 0), vid, u.res);
        MeteorOut o;
        o.pos = l.pos;
        o.color = l.color;
        o.along = (vid & 1) ? 1.0 : 0.0;
        o.across = l.across;
        o.width = l.halfw;
        return o;
    }

    fragment float4 meteorFragment(MeteorOut in [[stage_in]]) {
        float w = clamp(in.width + 0.5 - abs(in.across), 0.0, 1.0) * exp(-in.across * in.across / max(in.width * in.width, 0.5));
        float body = in.along * in.along * in.along;
        return in.color * w * (body * 1.4 + smoothstep(0.92, 1.0, in.along) * 2.5);
    }

    // Puts the ground in front of everything, adds the glow and tone maps.
    fragment float4 skyFinish(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                              texture2d<float> light [[texture(0)]], texture2d<float> glow [[texture(1)]],
                              texture2d<float> back [[texture(2)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = back.sample(s, in.uv).rgb + light.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.55;
        float3 h = lookAt(in.uv * u.res, k, u.res);
        float cover = groundCover(h, k, u.res);
        float below = clamp(-h.z * 6, 0.0, 1.0);
        float3 ground = mix(float3(0.0045, 0.0055, 0.0085), float3(0.0018, 0.002, 0.003), below);
        ground *= 0.75 + 0.5 * fbm(in.uv * float2(u.res.x / u.res.y, 1) * 9.0 + k.glow.z, 4);
        ground += k.glow.y * k.glow.y * float3(0.006, 0.008, 0.014) + k.glow.x * float3(0.002, 0.0025, 0.004);
        ground += glow.sample(s, in.uv).rgb * 0.08;
        c = mix(c, ground, cover);
        return float4(film(c), 1);
    }

    // Labels in the sky hide behind the ground.
    fragment float4 skyLabelFragment(LabelOut in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(2)]],
                                     texture2d<float> atlas [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float a = atlas.sample(s, in.uv).r;
        if (a < 0.002) return 0;
        float cover = groundCover(lookAt(in.pos.xy, k, u.res), k, u.res);
        return in.color * a * (1 - cover);
    }
    """
}
