import AppKit
import Metal
import simd

/// The solar system from above with today's real planet positions, running forward three days a second for twelve
/// years and then starting again from today. Distances are compressed logarithmically so the inner planets show.
/// Each planet hums its orbital frequency raised by whole octaves, gliding as it speeds up near the Sun.
@MainActor final class Orrery: Scene {
    /// Days of simulated time per second, and how long a run lasts before starting over from today.
    static let pace = 3.0
    static let span = 12 * 365.25

    struct Planet {
        let name: String
        let color: SIMD3<Float>
        /// Radius in Earth radii.
        let radius: Double
        /// Orbital period in days.
        let period: Double
        /// Octaves to raise the orbital frequency by, chosen so the eight tones spread out rather than beat.
        let octaves: Int
        let loudness: Float
        let elements: (Double) -> Astro.Elements
    }

    static let planets: [Planet] = [
        Planet(name: "Mercury", color: [0.66, 0.6, 0.55], radius: 0.38, period: 87.969, octaves: 32, loudness: 0.5) { Astro.elements(.mercury, $0) },
        Planet(name: "Venus", color: [0.95, 0.86, 0.68], radius: 0.95, period: 224.701, octaves: 31, loudness: 1) { Astro.elements(.venus, $0) },
        Planet(name: "Earth", color: [0.3, 0.52, 0.95], radius: 1, period: 365.256, octaves: 33, loudness: 0.9) { d in
            Astro.Elements(N: 0, i: 0, w: 282.9404 + 4.70935e-5 * d + 180, a: 1, e: 0.016709 - 1.151e-9 * d, M: 356.0470 + 0.9856002585 * d)
        },
        Planet(name: "Mars", color: [0.9, 0.45, 0.27], radius: 0.53, period: 686.98, octaves: 35, loudness: 0.55) { Astro.elements(.mars, $0) },
        Planet(name: "Jupiter", color: [0.88, 0.74, 0.58], radius: 11.2, period: 4332.59, octaves: 36, loudness: 1) { Astro.elements(.jupiter, $0) },
        Planet(name: "Saturn", color: [0.92, 0.82, 0.6], radius: 9.45, period: 10759.22, octaves: 37, loudness: 1) { Astro.elements(.saturn, $0) },
        Planet(name: "Uranus", color: [0.62, 0.86, 0.92], radius: 4.0, period: 30685.4, octaves: 40, loudness: 0.6) { Astro.elements(.uranus, $0) },
        Planet(name: "Neptune", color: [0.38, 0.52, 0.98], radius: 3.88, period: 60189, octaves: 40, loudness: 0.8) { Astro.elements(.neptune, $0) },
    ]

    /// The planet's tone: its orbital frequency raised by whole octaves (Kepler's Harmonices Mundi).
    static func tone(period days: Double, octaves: Int) -> Double { 1 / (days * 86400) * pow(2, Double(octaves)) }

    /// Distance from the Sun on screen: logarithmic, so Mercury and Neptune both fit.
    static func compress(_ r: Double) -> Double { log(1 + r / 0.35) / log(1 + 31 / 0.35) }

    private struct Body { var pos: SIMD4<Float>; var light: SIMD4<Float>; var info: SIMD4<Float>; var axis: SIMD4<Float>; var meridian: SIMD4<Float> }

    /// North pole of each body as (right ascension, declination) in degrees (IAU), and a rotation offset in turns.
    private static let poles: [(ra: Double, dec: Double)] = [(281.01, 61.41), (272.76, 67.16), (0, 90), (317.68, 52.89),
                                                              (268.06, 64.50), (40.59, 83.54), (257.31, -15.18), (299.36, 43.46)]
    private static let textureNames = ["mercury", "venus_atmosphere", "earth_daymap", "mars", "jupiter", "saturn", "uranus", "neptune", "moon"]
    private static func textureURL(_ name: String, _ ext: String = "jpg") -> URL {
        URL(string: "https://www.solarsystemscope.com/textures/download/2k_\(name).\(ext)")!
    }
    private static let sunURL = URL(string: "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIIC.jpg")!

    private let orbitPipe: MTLRenderPipelineState
    private let dustPipe: MTLRenderPipelineState
    private let bodyPipe: MTLRenderPipelineState
    private let sunPipe: MTLRenderPipelineState
    private let sky: SkyBackdrop
    private let lens: Lens
    private let blank: MTLTexture
    private var maps: [MTLTexture?] = Array(repeating: nil, count: 9)
    private var clouds: MTLTexture?
    private var ringMap: MTLTexture?
    private var sunMap: MTLTexture?
    private var loading = true
    private let labelPipe: MTLRenderPipelineState
    private let dust: MTLBuffer
    private let dustCount: Int
    private let rings: [Ring]
    private var orbits: MTLBuffer?
    private let orbitTints: MTLBuffer
    private let orbitPoints = 512
    private var light: MTLTexture?
    private var labels: Labels?

    private var start = Date()
    private var runStart: Float = 0
    private var days = 0.0
    private var lastM: [Double] = []
    private var lastLon: [Double] = []
    private var lastPluck: Float = -10
    private var event = ""
    private var eventAt: Float = -100
    private var dateText = ""
    private var summary = ""
    private var summaryAt: Float = 0

    var scale: Float { 1 }
    var ready: Bool { !loading && sky.loaded }
    var facts: String? { summary }
    var credit: String? {
        (maps.contains { $0 != nil } ? "Planet maps: Solar System Scope (CC BY 4.0, NASA and USGS data)" : "")
            + (sunMap != nil ? "; Sun: NASA/SDO (HMI)" : "") + (sky.texture != nil ? "; Milky Way: ESO/S. Brunier" : "")
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(SpaceKit.metal + Self.source)
        orbitPipe = try gpu.pipeline(library, "lineFragment", vertex: "orbitVertex", blend: .add)
        dustPipe = try gpu.pipeline(library, "glowFragment", vertex: "dustVertex", blend: .add)
        bodyPipe = try gpu.pipeline(library, "planetFragment", vertex: "planetVertex", blend: .alpha)
        sunPipe = try gpu.pipeline(library, "sunFragment", vertex: "sunVertex", blend: .add)
        sky = try SkyBackdrop(gpu: gpu)
        lens = try Lens(gpu: gpu)
        var white: [UInt8] = [255, 255, 255, 255]
        blank = white.withUnsafeMutableBytes { gpu.texture(1, 1, .rgba8Unorm, bytes: $0.baseAddress!, bytesPerRow: 4) }
        labelPipe = try gpu.pipeline(library, "labelFragment", vertex: "labelVertex", blend: .alpha)
        rings = (0..<6).map { _ in Ring(gpu: gpu, bytes: 1024) }

        // Minor bodies, each a = semi-major axis, e, mean anomaly at the start, and kind:
        // 0 main belt (with the Kirkwood gaps Jupiter clears at its 3:1, 5:2, 7:3 and 2:1 resonances),
        // 1 Jupiter's Trojans, 60 degrees ahead and behind it, 2 the Kuiper belt. Plus a few far-off stars.
        var items: [SIMD4<Float>] = []
        let gaps = [2.502, 2.825, 2.958, 3.279]
        while items.count < 2 * 14000 {
            let a = Double.random(in: 2.1...3.4)
            if gaps.contains(where: { abs($0 - a) < 0.025 }), Double.random(in: 0..<1) < 0.9 { continue }
            if a > 3.3, Double.random(in: 0..<1) < 0.8 { continue }
            items += [SIMD4(Float(a), .random(in: 0...0.16), .random(in: 0..<360), 0), SIMD4(.random(in: 0..<360), .random(in: -4...4), .random(in: 0.3...1), 0)]
        }
        let jupiter = Astro.elements(.jupiter, Astro.dayNumber(Date()))
        for k in 0..<2600 {
            let side: Double = k % 2 == 0 ? 60 : -60
            let m = jupiter.M + side + Double.random(in: -18...18) + Double.random(in: -8...8)
            items += [SIMD4(Float(5.2 + .random(in: -0.12...0.12)), .random(in: 0...0.08), Float(m), 1),
                      SIMD4(Float(jupiter.w + jupiter.N), .random(in: -8...8), .random(in: 0.3...1), 0)]
        }
        for _ in 0..<5000 {
            items += [SIMD4(.random(in: 39...48), .random(in: 0...0.2), .random(in: 0..<360), 2),
                      SIMD4(.random(in: 0..<360), .random(in: -10...10), .random(in: 0.2...1), 0)]
        }
        dustCount = items.count / 2
        dust = gpu.buffer(items)
        orbitTints = gpu.buffer(Self.planets.map { SIMD4($0.color, 0) })
        Task { @MainActor [weak self] in
            // One map at a time, each fetched once and kept in the cache for months.
            for (i, name) in Self.textureNames.enumerated() {
                let t = await Imagery.texture(Self.textureURL(name), gpu, maxSize: 2048, maxAge: 365 * 86400)
                self?.maps[i] = t
            }
            self?.clouds = await Imagery.texture(Self.textureURL("earth_clouds"), gpu, maxSize: 2048, maxAge: 365 * 86400)
            self?.ringMap = await Imagery.texture(Self.textureURL("saturn_ring_alpha", "png"), gpu, maxSize: 2048, maxAge: 365 * 86400)
            self?.sunMap = await Imagery.texture(Self.sunURL, gpu, maxSize: 1024, maxAge: 6 * 3600)
            self?.loading = false
        }
    }

    // MARK: Camera

    private struct Camera {
        var eye: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>, forward: SIMD3<Double>
        var zoom: Double, aspect: Double, shift: SIMD2<Double>, size: SIMD2<Double>

        /// Target pixels of a scene point and its depth.
        func pixel(_ q: SIMD3<Double>) -> (SIMD2<Float>, Double) {
            let d = q - eye
            let z = max((d * forward).sum(), 0.01)
            let p = SIMD2((d * right).sum(), (d * up).sum()) * zoom / z + shift
            return (SIMD2(Float((p.x / aspect * 0.5 + 0.5) * size.x), Float((0.5 - p.y * 0.5) * size.y)), z)
        }
    }

    private func camera(_ f: Frame) -> Camera {
        let t = Double(f.time) + Double(f.seed) * 37
        // A slow turn, a gentle rise and fall of the viewing angle, and a long breath in toward the inner planets.
        let yaw = -Double.pi / 2 + t * 2 * .pi / 900
        let elevation = Astro.rad(58 + 10 * sin(t * 0.0041))
        let breath = 0.5 - 0.5 * cos(t * 2 * .pi / 240)
        let distance = 2.45 - 1.05 * breath * breath
        let eye = SIMD3(cos(elevation) * cos(yaw), cos(elevation) * sin(yaw), sin(elevation)) * distance
        let forward = -simd_normalize(eye)
        let right = simd_normalize(simd_cross(forward, SIMD3(0, 0, 1)))
        let up = simd_cross(right, forward)
        // The Sun wanders a little on screen so nothing bright stays put for hours.
        let shift = SIMD2(0.07 * sin(t * 0.0023), 0.05 * sin(t * 0.0031 + 1))
        return Camera(eye: eye, right: right, up: up, forward: forward, zoom: 2.0, aspect: Double(f.aspect), shift: shift,
                      size: SIMD2(Double(f.size.x), Double(f.size.y)))
    }

    private static func scenePoint(_ helio: SIMD3<Double>) -> SIMD3<Double> {
        let r = simd_length(helio)
        return r < 1e-9 ? .zero : helio * (compress(r) / r)
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        // Time runs from today; after twelve years it fades out and starts over.
        var runTime = Double(f.time - runStart)
        if runTime * Self.pace > Self.span {
            runStart = f.time
            runTime = 0
            start = f.now
            orbits = nil
            lastM = []
            lastLon = []
        }
        if f.index == 0 { start = f.now }
        days = runTime * Self.pace
        let date = start.addingTimeInterval(days * 86400)
        let d = Astro.dayNumber(date)
        let fade = Float(min(runTime / 2, 1, (Self.span / Self.pace - runTime) / 2))

        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            labels = Labels(gpu: f.gpu, density: f.overlay.scale * CGFloat(scale),
                            Self.planets.map { ($0.name, Labels.Style(size: 11, weight: .medium, tracking: 0.4)) })
        }
        if orbits == nil { orbits = makeOrbits(d, f.gpu) }
        let cam = camera(f)
        let density = f.size.y / 1000
        let f3 = { (v: SIMD3<Double>) in SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        let view = f.uniforms(SIMD4(f3(cam.eye), Float(cam.zoom)), SIMD4(f3(cam.right), Float(cam.shift.x)),
                              SIMD4(f3(cam.up), Float(cam.shift.y)), SIMD4(f3(cam.forward), density))

        // Planets now.
        var bodies: [Body] = []
        var helio: [SIMD3<Double>] = []
        var mean: [Float] = []
        let turn = Double(f.time)
        for (i, p) in Self.planets.enumerated() {
            let el = p.elements(d)
            let h = Astro.position(el)
            helio.append(h)
            mean.append(Float(Astro.norm(el.M) / 360))
            let q = Self.scenePoint(h)
            let size = Float(8 + 4.6 * log(1 + p.radius))
            let pole = Self.ecliptic(ra: Self.poles[i].ra, dec: Self.poles[i].dec)
            // Slow, calm spin: a real day passes in a blink at this pace, so the globes just turn gently.
            let meridian = Self.meridian(pole, angle: turn * 0.05 * (i % 2 == 0 ? 1 : 0.7) + Double(i) * 1.3)
            bodies.append(Self.body(q, size * density, h, i, pole, meridian, p.color, fade, cam))
        }
        // Earth's Moon, at its real angle round Earth, on a stretched-out orbit so it clears the planet.
        let moonLon = Astro.rad(Astro.ecliptic(Astro.moon(date), date))
        let earth = Self.scenePoint(helio[2])
        let toMoon = SIMD3(cos(moonLon), sin(moonLon), 0)
        let moonPos = earth + toMoon * 0.028
        bodies.append(Self.body(moonPos, 2.4 * density, helio[2], 8, SIMD3(0, 0, 1), -toMoon, SIMD3(0.75, 0.73, 0.7), fade, cam))

        let light = light!
        sky.draw(f.cmd, into: light, camera: Self.skyCamera(cam), exposure: 0.5 * fade, soften: 0.5, stars: 0.5 * fade)
        f.cmd.draw(dustPipe, into: light, clear: false, uniforms: view, buffers: [dust, rings[0].next(SIMD4(Float(days), fade, 0, 0))],
                   vertices: 4, instances: dustCount, primitive: .triangleStrip)
        let trails = Self.planets.map { Float($0.period / 150) }
        f.cmd.draw(orbitPipe, into: light, clear: false, uniforms: view, buffers: [orbits!, rings[1].next(mean + [fade] + trails), orbitTints],
                   vertices: 4, instances: Self.planets.count * orbitPoints, primitive: .triangleStrip)
        f.cmd.draw(sunPipe, into: light, clear: false, uniforms: view, textures: [sunMap ?? blank],
                   buffers: [rings[2].next(SIMD4(fade, sunMap == nil ? 0 : 1, 0, 0))], vertices: 4, primitive: .triangleStrip)
        // Far planets first, so near ones overlap them.
        let order = bodies.indices.sorted { cam.pixel(Self.point(bodies[$0])).1 > cam.pixel(Self.point(bodies[$1])).1 }
        let all = rings[3].next(bodies)
        for i in order {
            var u = view
            let map = maps[i], second = i == 5 ? ringMap : i == 2 ? clouds : nil
            u.pad = SIMD2(Float(i), Float((map == nil ? 0 : 1) + (second == nil ? 0 : 2)))
            f.cmd.draw(bodyPipe, into: light, clear: false, uniforms: u, textures: [map ?? blank, second ?? blank], buffers: [all],
                       vertices: 4, primitive: .triangleStrip)
        }

        lens.develop(f, light, Lens.Look(exposure: 1.1 * fade, white: [1, 0.99, 0.97], bloom: 0.05, halation: 0.025, vignette: 0.3,
                                         aberration: 1.0, grain: 0.35))

        if let labels {
            var quads: [LabelQuad] = []
            for (i, p) in Self.planets.enumerated() {
                let (px, _) = cam.pixel(Self.scenePoint(helio[i]))
                let offset = (bodies[i].pos.w + 7 * density)
                if let q = labels.quad(p.name, at: px + SIMD2(0, offset), anchor: [0.5, 0],
                                       color: .premultiplied(0.85, 0.88, 0.95, 0.55 * fade)) { quads.append(q) }
            }
            f.cmd.draw(labelPipe, into: f.target, clear: false, uniforms: f.uniforms(SIMD4(labels.size.x, labels.size.y, 0, 0)),
                       textures: [labels.texture], buffers: [rings[4].next(quads)], vertices: 4, instances: quads.count,
                       primitive: .triangleStrip)
        }

        let text = date.formatted(.dateTime.day().month(.wide).year())
        if text != dateText {
            dateText = text
            f.overlay.draw { ctx, size in
                ctx.text(text, at: CGPoint(x: 28, y: 24), size: 15, weight: .light, color: NSColor(white: 1, alpha: 0.55), tracking: 0.6)
                ctx.text("\(Int(Self.pace)) days a second", at: CGPoint(x: 28, y: 46), size: 10.5, weight: .regular,
                         color: NSColor(white: 1, alpha: 0.3), tracking: 0.4)
            }
        }
        listen(f, d, helio, cam)
        if f.time - summaryAt >= 1 || summary.isEmpty {
            summaryAt = f.time
            summarise(f, date, helio)
        }
    }

    /// A body ready for the GPU: scene position and size, light direction and pole in the camera's frame.
    private static func body(_ q: SIMD3<Double>, _ size: Float, _ helio: SIMD3<Double>, _ index: Int, _ pole: SIMD3<Double>,
                             _ meridian: SIMD3<Double>, _ color: SIMD3<Float>, _ fade: Float, _ cam: Camera) -> Body {
        func inCamera(_ v: SIMD3<Double>) -> SIMD3<Float> {
            SIMD3(Float((v * cam.right).sum()), Float((v * cam.up).sum()), Float(-(v * cam.forward).sum()))
        }
        let sun = simd_normalize(-q)
        return Body(pos: SIMD4(Float(q.x), Float(q.y), Float(q.z), size), light: SIMD4(inCamera(sun), Float(index)),
                    info: SIMD4(color, fade), axis: SIMD4(inCamera(pole), Float(simd_length(helio))),
                    meridian: SIMD4(inCamera(meridian), 0))
    }

    private static func point(_ b: Body) -> SIMD3<Double> { SIMD3(Double(b.pos.x), Double(b.pos.y), Double(b.pos.z)) }

    /// A pole given in equatorial coordinates, as an ecliptic unit vector.
    private static func ecliptic(ra: Double, dec: Double) -> SIMD3<Double> {
        let a = Astro.rad(ra), d = Astro.rad(dec), e = Astro.rad(23.4392911)
        let v = SIMD3(cos(d) * cos(a), cos(d) * sin(a), sin(d))
        return SIMD3(v.x, v.y * cos(e) + v.z * sin(e), -v.y * sin(e) + v.z * cos(e))
    }

    /// The direction of longitude zero: any vector across the pole, turned about it by `angle`.
    private static func meridian(_ pole: SIMD3<Double>, angle: Double) -> SIMD3<Double> {
        let across = simd_normalize(abs(pole.z) < 0.95 ? simd_cross(pole, SIMD3(0, 0, 1)) : SIMD3(1, 0, 0))
        return across * cos(angle) + simd_cross(pole, across) * sin(angle)
    }

    private static func skyCamera(_ c: Camera) -> SkyBackdrop.Camera {
        func eq(_ v: SIMD3<Double>) -> SIMD3<Float> { let e = SpaceSky.equatorial(ecliptic: v); return SIMD3(Float(e.x), Float(e.y), Float(e.z)) }
        return SkyBackdrop.Camera(right: eq(c.right), up: eq(c.up), forward: eq(c.forward), focal: Float(c.zoom))
    }

    /// Each orbit as a ring of points with its mean anomaly, so the shader can light a fading trail behind the planet.
    private func makeOrbits(_ d: Double, _ gpu: GPU) -> MTLBuffer {
        var points: [SIMD4<Float>] = []
        for (i, p) in Self.planets.enumerated() {
            var el = p.elements(d)
            let m0 = el.M
            for k in 0..<orbitPoints {
                el.M = m0 + 360 * Double(k) / Double(orbitPoints)
                let q = Self.scenePoint(Astro.position(el))
                points.append(SIMD4(Float(q.x), Float(q.y), Float(q.z), Float(i) + Float(Astro.norm(el.M) / 360) * 0.999))
            }
        }
        return gpu.buffer(points)
    }

    // MARK: Sound

    /// Each planet's tone glides with its angular speed round the Sun, so it rises near perihelion; a pluck marks each
    /// perihelion passage, and two planets lining up with the Sun pluck both their notes.
    private func listen(_ f: Frame, _ d: Double, _ helio: [SIMD3<Double>], _ cam: Camera) {
        let els = Self.planets.map { $0.elements(d) }
        let lons = helio.map { Astro.deg(atan2($0.y, $0.x)) }
        let ms = els.map { Astro.norm($0.M) }
        defer { lastM = ms; lastLon = lons }
        guard lastM.count == ms.count else { return }
        var notes: [(Float, Float, String)] = []  // frequency, pan, event
        var tones: [Float] = []
        for (i, p) in Self.planets.enumerated() {
            let el = els[i]
            let v = atan2(helio[i].y, helio[i].x) - Astro.rad(el.N + el.w)
            let speed = pow(1 + el.e * cos(v), 2) / pow(1 - el.e * el.e, 1.5)
            tones.append(Float(Self.tone(period: p.period, octaves: p.octaves) * speed))
            if ms[i] < lastM[i], lastM[i] - ms[i] > 180 { notes.append((tones[i] * 2, pan(cam, helio[i]), "\(p.name) passes perihelion")) }
        }
        for i in 0..<lons.count {
            for j in (i + 1)..<lons.count {
                var a = lons[i] - lons[j], b = lastLon[i] - lastLon[j]
                a = remainder(a, 360)
                b = remainder(b, 360)
                if (a > 0) != (b > 0), abs(a) < 20, abs(b) < 20 {
                    let names = (Self.planets[i].name, Self.planets[j].name)
                    let what = names.0 == "Earth" || names.1 == "Earth"
                        ? (i < 2 ? "\(names.0) passes between Earth and the Sun" : "\(names.0 == "Earth" ? names.1 : names.0) is at opposition, Earth passing between it and the Sun")
                        : "\(names.0) and \(names.1) line up with the Sun"
                    notes.append((tones[i], pan(cam, helio[i]), what))
                    notes.append((tones[j], pan(cam, helio[j]), ""))
                }
            }
        }
        // Mercury does something every few seconds, too often to announce unless Earth is involved.
        if let first = notes.first(where: { !$0.2.isEmpty && (!$0.2.hasPrefix("Mercury") || $0.2.contains("Earth")) }) {
            event = first.2
            eventAt = f.time
        }
        guard let synth = f.sound else { return }
        for (i, p) in Self.planets.enumerated() {
            synth.drone(i, tones[i], 0.024 * p.loudness, pan: pan(cam, helio[i]))
        }
        if !notes.isEmpty, f.time - lastPluck > 0.25 {
            lastPluck = f.time
            for n in notes.prefix(2) { synth.play(.pluck, n.0, amp: 0.07, pan: n.1, length: 2.5) }
        }
    }

    private func pan(_ cam: Camera, _ helio: SIMD3<Double>) -> Float {
        let q = Self.scenePoint(helio) - cam.eye
        return Float(max(-0.8, min(0.8, (q * cam.right).sum() * 0.9)))
    }

    private func summarise(_ f: Frame, _ date: Date, _ helio: [SIMD3<Double>]) {
        var text = "Showing \(date.formatted(.dateTime.day().month(.wide).year())), \(Int(Self.pace)) days a second; "
        if f.time - eventAt < 20, !event.isEmpty {
            text += event + "."
        } else {
            // Rotate through the planets' distances from Earth.
            let i = [3, 4, 1, 5, 0][Int(f.time / 20) % 5]
            let name = Self.planets[i].name
            let distance = simd_length(helio[i] - helio[2])
            text += "\(name) is \(SpaceKit.number(distance, decimals: 2)) AU from Earth."
        }
        summary = text
    }

    // MARK: Shaders

    private static let source = """
    // a: eye, zoom   b: right, shift x   c: up, shift y   d: forward, pixel density
    inline float3 viewOf(float3 q, constant U &u) {
        float3 d = q - u.a.xyz;
        float z = max(dot(d, u.d.xyz), 0.01);
        float2 p = float2(dot(d, u.b.xyz), dot(d, u.c.xyz)) * u.a.w / z + float2(u.b.w, u.c.w);
        return float3((p.x / (u.res.x / u.res.y) * 0.5 + 0.5) * u.res.x, (0.5 - p.y * 0.5) * u.res.y, z);
    }

    // Orbits: points[i].w is planet index + mean anomaly. Buffer 2 holds each planet's current mean anomaly, the fade,
    // then how sharply each trail falls off (its period over 150 days, so every trail shows the last few months).
    vertex LineOut orbitVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                               const device float4 *pts [[buffer(1)]], const device float *now [[buffer(2)]],
                               const device float4 *tints [[buffer(3)]]) {
        uint n = 512;
        uint planet = iid / n, k = iid % n;
        float4 a = pts[planet * n + k], b = pts[planet * n + (k + 1) % n];
        float3 pa = viewOf(a.xyz, u), pb = viewOf(b.xyz, u);
        float behind = fract(now[planet] - fract(a.w));
        float trail = exp(-behind * now[9 + planet]) * 0.7;
        float3 tint = mix(float3(0.5, 0.58, 0.8), tints[planet].rgb, 0.45 + 0.4 * trail);
        float4 color = float4(tint * (0.03 + 0.5 * trail), 0) * now[8];
        return lineQuad(pa.xy, pb.xy, 0.9 * u.d.w, color, vid, u.res);
    }

    // Minor bodies (a, e, mean anomaly, kind; perihelion longitude, height, brightness) and distant stars.
    vertex GlowOut dustVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                              const device float4 *items [[buffer(1)]], constant float4 &t [[buffer(2)]]) {
        float4 a = items[2 * iid], b = items[2 * iid + 1];
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        GlowOut o;
        o.uv = corner;
        float3 q;
        float3 color;
        float size = 1.1;
        if (a.w > 2.5) {
            // A star far away in every direction.
            q = a.xyz * 60.0 + u.a.xyz;
            color = mix(float3(0.7, 0.8, 1.0), float3(1.0, 0.85, 0.7), b.w) * (0.05 + 0.5 * b.z);
            size = 1.0 + b.z;
        } else {
            float n = 0.9856076686 / pow(a.x, 1.5);
            float M = (a.z + n * t.x) * 0.0174533;
            float e = a.y;
            float E = M + e * sin(M);
            E -= (E - e * sin(E) - M) / (1 - e * cos(E));
            float2 xy = float2(a.x * (cos(E) - e), a.x * sqrt(1 - e * e) * sin(E));
            float w = b.x * 0.0174533;
            xy = float2(xy.x * cos(w) - xy.y * sin(w), xy.x * sin(w) + xy.y * cos(w));
            float r = length(xy);
            float s = log(1 + r / 0.35) / log(1 + 31 / 0.35);
            q = float3(xy / r * s, b.y * 0.0174533 * s * 0.6);
            color = a.w < 0.5 ? float3(0.75, 0.72, 0.68) * 0.07 : a.w < 1.5 ? float3(0.95, 0.75, 0.5) * 0.09 : float3(0.6, 0.75, 1.0) * 0.06;
            size = 0.8;
            color *= b.z * t.y;
        }
        float3 p = viewOf(q, u);
        float px = size * u.d.w * 1.6;
        o.pos = pixelToClip(p.xy + corner * px, u.res);
        o.color = float4(color, 0);
        o.extra = float4(4.0, 0, 0, 0);
        return o;
    }

    vertex GlowOut sunVertex(uint vid [[vertex_id]], constant U &u [[buffer(0)]], constant float4 &t [[buffer(1)]]) {
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        float3 p = viewOf(float3(0), u);
        float radius = 28 * u.d.w * 0.9 * sqrt(2.2 / p.z);
        GlowOut o;
        o.pos = pixelToClip(p.xy + corner * radius * 6, u.res);
        o.uv = corner * 6;
        o.color = float4(1, 1, 1, t.x);
        o.extra = float4(u.time, radius, t.y, 0);
        return o;
    }

    // The Sun: NASA's Solar Dynamics Observatory image of today (sunspots, plage and the real limb darkening), tinted
    // from orange toward the white a 5,772 K surface really is, with a soft corona. Without it, a procedural disc.
    fragment float4 sunFragment(GlowOut in [[stage_in]], texture2d<float> sun [[texture(0)]]) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
        float r = length(in.uv);
        float3 c = 0;
        if (r < 1.02) {
            float mu = sqrt(max(1 - r * r, 0.0));
            float edge = smoothstep(1.02, 0.99, r);
            if (in.extra.z > 0.5) {
                float2 uv = float2(0.5 + in.uv.x * 0.4663, 0.5 - in.uv.y * 0.4663);
                float3 photo = sun.sample(s, uv, level(max(log2(1024 * 0.4663 * 2 / max(in.extra.y * 2, 1.0)), 0.0))).rgb;
                float i = photo.r;
                c = i * mix(float3(1.0, 0.72, 0.42), float3(1.0, 0.93, 0.82), mu) * 1.5 * edge;
            } else {
                float limb = 1 - 0.6 * (1 - mu) - 0.2 * (1 - mu) * (1 - mu);
                float g = fbm(float3(in.uv * 7.0, in.extra.x * 0.05), 4);
                c = float3(1.0, 0.86, 0.66) * limb * (0.85 + 0.3 * g) * 1.6 * edge;
            }
        }
        float corona = exp(-(r - 1) * 3.0) * 0.12 + exp(-(r - 1) * 0.7) * 0.025;
        float rays = 0.88 + 0.12 * noise(float2(atan2(in.uv.y, in.uv.x) * 5.0, in.extra.x * 0.03));
        if (r >= 1) c += float3(1.0, 0.8, 0.55) * corona * rays;
        return float4(c * in.color.a * smoothstep(6.0, 4.0, r), 0);
    }

    struct BodyIn { float4 pos; float4 light; float4 info; float4 axis; float4 meridian; };
    struct PlanetOut { float4 pos [[position]]; float2 uv; float4 light; float4 info; float4 axis; float3 meridian; float span; float2 flags; };

    vertex PlanetOut planetVertex(uint vid [[vertex_id]], constant U &u [[buffer(0)]], const device BodyIn *bodies [[buffer(1)]]) {
        BodyIn b = bodies[int(u.pad.x + 0.5)];
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        float3 p = viewOf(b.pos.xyz, u);
        float radius = b.pos.w * 0.9 * sqrt(2.2 / p.z);
        float span = abs(b.light.w - 5) < 0.5 ? 2.5 : 1.15;   // room for Saturn's rings
        PlanetOut o;
        o.pos = pixelToClip(p.xy + corner * radius * span, u.res);
        o.uv = corner * span;
        o.light = b.light;
        o.info = b.info;
        o.axis = b.axis;
        o.meridian = b.meridian.xyz;
        o.span = radius;
        o.flags = u.pad;
        return o;
    }

    constant float RING_IN = 1.11, RING_OUT = 2.33;

    // A lit globe from its real map: Lambert light with the Sun's direction, Lommel-Seeliger for airless rock, a thin
    // atmosphere at Earth's and Venus's limbs, ocean glint, and Saturn's rings that cast a shadow on the globe and
    // receive the globe's.
    fragment float4 planetFragment(PlanetOut in [[stage_in]], texture2d<float> map [[texture(0)]], texture2d<float> extra [[texture(1)]]) {
        constexpr sampler s(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge);
        constexpr sampler strip(filter::linear, mip_filter::linear, s_address::clamp_to_edge, t_address::clamp_to_edge);
        float2 q = in.uv;
        float r = length(q);
        int kind = int(in.light.w + 0.5);
        bool textured = fmod(in.flags.y, 2.0) > 0.5, second = in.flags.y > 1.5;
        float aa = 1.2 / max(in.span, 1.0);
        float disc = smoothstep(1 + aa, 1 - aa, r);
        float3 L = normalize(in.light.xyz);
        float3 P = normalize(in.axis.xyz);
        float3 n = float3(q, sqrt(max(1 - r * r, 0.0)));
        float mu = n.z;
        float3 e1 = normalize(in.meridian), e2 = cross(P, e1);
        float lat = asin(clamp(dot(n, P), -1.0, 1.0));
        float lon = atan2(dot(n, e2), dot(n, e1));
        float2 uv = float2(0.5 + lon / TAU, 0.5 - lat / PI);
        float lod = max(log2(float(map.get_width()) / (TAU * max(in.span, 1.0))) + 0.5, 0.0);
        float3 albedo = textured ? map.sample(s, uv, level(lod)).rgb : in.info.rgb * (0.85 + 0.25 * fbm(q * 3.0 + 2.0, 3));
        float sunlight = 1.6 / (1 + 0.2 * in.axis.w);

        float cosL = dot(n, L);
        float lit = max(cosL, 0.0);
        if (kind == 0 || kind == 8) lit = lit / (lit + mu + 0.02) * 2.0;
        if (kind == 1) lit = smoothstep(-0.3, 0.9, cosL) * 0.95;
        if (kind >= 4 && kind <= 7) lit = smoothstep(-0.12, 0.6, cosL);
        if (kind == 2 && second) {
            float cloud = extra.sample(s, uv + float2(0.004 * sin(in.axis.w * 40.0), 0), level(lod)).r;
            float ocean = smoothstep(0.0, 0.25, albedo.b - albedo.r - 0.02);
            albedo = mix(albedo, float3(0.92, 0.94, 0.96), cloud * 0.85);
            float3 H = normalize(L + float3(0, 0, 1));
            float glint = pow(max(dot(n, H), 0.0), 90.0) * ocean * (1 - cloud) * 0.9;
            albedo += glint / max(lit + 0.05, 0.05);
        }
        // Ring shadow on the globe: follow the light back to the ring plane.
        if (kind == 5 && second && cosL > 0) {
            float dl = dot(L, P);
            if (abs(dl) > 0.02) {
                float t = -dot(n, P) / dl;
                if (t > 0) {
                    float rho = length(n + L * t);
                    if (rho > RING_IN && rho < RING_OUT) lit *= 1 - 0.85 * extra.sample(strip, float2((rho - RING_IN) / (RING_OUT - RING_IN), 0.5), level(0)).a;
                }
            }
        }
        float3 c = albedo * (lit * sunlight + 0.004);
        if (kind == 2 || kind == 1) {
            float rim = pow(1 - mu, 3.0) * smoothstep(-0.25, 0.4, cosL);
            c += (kind == 2 ? float3(0.18, 0.4, 0.95) : float3(0.9, 0.78, 0.55)) * rim * (kind == 2 ? 0.7 : 0.35) * sunlight;
        }
        if (kind == 3) c += float3(0.9, 0.55, 0.35) * pow(1 - mu, 4.0) * 0.06 * smoothstep(-0.2, 0.5, cosL) * sunlight;
        c *= disc;
        float alpha = disc;

        if (kind == 5 && second) {
            float z = -(P.x * q.x + P.y * q.y) / (abs(P.z) > 0.04 ? P.z : 0.04);
            float3 R = float3(q, z);
            float rr = length(R);
            if (rr > RING_IN && rr < RING_OUT && !(z < 0 && r < 1)) {
                float4 t = extra.sample(strip, float2((rr - RING_IN) / (RING_OUT - RING_IN), 0.5), level(0));
                float b = dot(R, L), h = b * b - (dot(R, R) - 1);
                float shadow = b < 0 ? smoothstep(0.0, 0.08, h) : 0.0;
                bool sunSide = dot(L, P) * P.z > 0;
                float3 tint = float3(0.95, 0.84, 0.68);
                float3 rc = tint * dot(t.rgb, float3(0.333)) * 4.2 * sunlight * (sunSide ? 1.0 : 0.5) * (1 - 0.97 * shadow);
                float a = t.a * 0.96;
                c = rc * a + c * (1 - a);
                alpha = a + alpha * (1 - a);
            }
        }
        return float4(c, alpha) * in.info.w;
    }
    """
}
