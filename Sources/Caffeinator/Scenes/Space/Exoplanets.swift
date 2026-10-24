import AppKit
import Metal
import simd

/// Every confirmed exoplanet from the NASA Exoplanet Archive, its orbit drawn around one shared star on a logarithmic
/// scale of distance, each planet moving at a pace set by its real period, coloured by its temperature and sized by
/// its radius. Famous systems take turns in the spotlight, running at their true relative periods; between them,
/// the asteroids passing Earth this month from JPL.
@MainActor final class Exoplanets: Scene {
    struct Planet {
        var name: String
        var host: String
        /// Semi-major axis (AU), period (days), eccentricity.
        var a: Double, period: Double, e: Double
        /// Equilibrium temperature (K), measured or estimated from the star.
        var temperature: Double?
        /// Radius in Earth radii.
        var radius: Double?
        var year: Int
        var method: String
        var letter: String { String(name.split(separator: " ").last ?? "") }
    }

    struct Asteroid {
        var name: String
        var date: Date
        /// Closest approach in AU, speed relative to Earth in km/s, absolute magnitude.
        var distance: Double, speed: Double, h: Double
        var lunar: Double { distance / 0.00257 }
        /// Diameter in metres for a typical albedo of 0.14.
        var size: Double { 1329 / sqrt(0.14) * pow(10, -h / 5) * 1000 }
    }

    private struct Spotlight {
        let host: String
        let title: String
        let blurb: String
    }

    private static let spotlights = [
        Spotlight(host: "TRAPPIST-1", title: "TRAPPIST-1", blurb: "Earth-sized planets in a resonant chain around a cool red dwarf, 40 light-years away"),
        Spotlight(host: "KOI-351", title: "Kepler-90", blurb: "as many planets as the Solar System, all packed inside Earth's orbit"),
        Spotlight(host: "51 Peg", title: "51 Pegasi", blurb: "the first planet found orbiting a Sun-like star, in 1995: a hot Jupiter circling in 4.2 days"),
        Spotlight(host: "Proxima Cen", title: "Proxima Centauri", blurb: "the nearest star to the Sun, 4.2 light-years away"),
        Spotlight(host: "TOI-178", title: "TOI-178", blurb: "five of them locked in a chain of resonances"),
        Spotlight(host: "HR 8799", title: "HR 8799", blurb: "giant planets photographed directly, far out from their young star"),
        Spotlight(host: "55 Cnc", title: "55 Cancri", blurb: "including a lava world that circles in under 18 hours"),
        Spotlight(host: "Kepler-11", title: "Kepler-11", blurb: "six planets packed closer to their star than Venus is to the Sun"),
    ]

    private static let url = URL(string: "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,hostname,pl_orbsmax,pl_orbper,pl_orbeccen,pl_eqt,pl_rade,st_teff,disc_year,discoverymethod+from+pscomppars&format=csv")!
    private static let asteroidURL = URL(string: "https://ssd-api.jpl.nasa.gov/cad.api?date-min=now&date-max=%2B30&dist-max=0.05")!

    /// Screen radius for a semi-major axis: logarithmic from 0.006 AU to 600 AU.
    static func ring(_ a: Double) -> Double { 0.07 + 0.93 * min(max((log10(max(a, 1e-4)) + 2.2) / 5, 0), 1.04) }

    /// Mean motion on screen, radians per second: periods compressed as P^-0.4, so a day takes 12 seconds and a
    /// year two minutes, and Kepler's third law still orders them.
    static func pace(_ days: Double) -> Double { 0.5 * pow(max(days, 0.05), -0.4) }

    /// A pentatonic note from the period: the shortest periods ring highest.
    static func note(_ days: Double) -> Float {
        let steps = [0, 2, 4, 7, 9]
        let x = min(max((log10(max(days, 0.1)) + 0.5) / 4.5, 0), 1)
        let index = Int(((1 - x) * 14).rounded())
        return 196 * pow(2, Float(12 * (index / 5) + steps[index % 5]) / 12)
    }

    private enum Mode: Equatable { case field, spotlight(Int), asteroids }

    private let orbitPipe: MTLRenderPipelineState
    private let planetPipe: MTLRenderPipelineState
    private let starPipe: MTLRenderPipelineState
    private let flatPipe: MTLRenderPipelineState
    private let dotPipe: MTLRenderPipelineState
    private let backPipe: MTLRenderPipelineState
    private let tonePipe: MTLRenderPipelineState
    private let labelPipe: MTLRenderPipelineState
    private let bloom: Bloom
    private let rings: [Ring]
    private var light: MTLTexture?
    private var labels: Labels?

    private var planets: [Planet] = []
    private var asteroids: [Asteroid] = []
    private var orbitData: MTLBuffer?
    private var phase: [Float] = []
    private var rates: [Float] = []
    private var periapses: [Float] = []
    private var fresh = false
    private var summary = ""
    private var summaryAt: Float = -100
    private var loaded = false
    private var offline = false

    private var mode = Mode.field
    private var modeStart: Float = 0
    private var spotlightsShown = 0
    private var spotlight = 0
    private var members: [Int] = []
    private var focus: Float = 0
    private var interlude: Float = 0
    private var reach = 1.0
    private var plucks: [Float] = []
    private var captioned = false
    private var crossing: [Double] = []

    var scale: Float { 1 }
    var ready: Bool { loaded }
    var facts: String? { summary.isEmpty ? describe() : summary }

    init(gpu: GPU) throws {
        let library = try gpu.library(SpaceKit.metal + Self.source)
        orbitPipe = try gpu.pipeline(library, "solid", vertex: "exoOrbit", blend: .add)
        planetPipe = try gpu.pipeline(library, "glowFragment", vertex: "exoPlanet", blend: .add)
        starPipe = try gpu.pipeline(library, "glowFragment", vertex: "starVertex", blend: .add)
        flatPipe = try gpu.pipeline(library, "lineFragment", vertex: "flatLine", blend: .add)
        dotPipe = try gpu.pipeline(library, "glowFragment", vertex: "flatDot", blend: .add)
        backPipe = try gpu.pipeline(library, "exoBack")
        tonePipe = try gpu.pipeline(library, "exoTone")
        labelPipe = try gpu.pipeline(library, "labelFragment", vertex: "labelVertex", blend: .alpha)
        bloom = try Bloom(gpu: gpu)
        rings = (0..<6).map { _ in Ring(gpu: gpu, bytes: 65536) }
        spotlight = (Snapshot.variant ?? Int.random(in: 0..<Self.spotlights.count)) % Self.spotlights.count
        Task { @MainActor in
            async let table = Feed.data(Self.url, maxAge: 7 * 86400)
            async let flybys = Feed.object(Self.asteroidURL, maxAge: 12 * 3600)
            let parsed = (await table).map(Self.parse) ?? []
            asteroids = Self.parseFlybys(await flybys)
            if parsed.isEmpty { offline = true } else { setUp(parsed, gpu) }
            loaded = true
        }
    }

    // MARK: Data

    /// Splits one CSV line, honouring quoted fields.
    static func fields(_ line: Substring) -> [String] {
        var out: [String] = [], field = "", quoted = false
        for ch in line {
            if ch == "\"" { quoted.toggle() } else if ch == ",", !quoted { out.append(field); field = "" } else { field.append(ch) }
        }
        out.append(field)
        return out
    }

    static func parse(_ data: Data) -> [Planet] {
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        var lines = text.split(whereSeparator: \.isNewline)
        guard let header = lines.first else { return [] }
        let names = fields(header)
        func column(_ n: String) -> Int? { names.firstIndex(of: n) }
        guard let iName = column("pl_name"), let iHost = column("hostname"), let iA = column("pl_orbsmax"), let iP = column("pl_orbper")
        else { return [] }
        let iE = column("pl_orbeccen"), iT = column("pl_eqt"), iR = column("pl_rade"), iS = column("st_teff"),
            iY = column("disc_year"), iM = column("discoverymethod")
        lines.removeFirst()
        return lines.compactMap { line in
            let f = fields(line)
            guard f.count == names.count else { return nil }
            func num(_ i: Int?) -> Double? { i.flatMap { Double(f[$0]) } }
            var a = num(iA), p = num(iP)
            // Kepler's third law for a star like the Sun fills in whichever is missing.
            if a == nil, let p { a = pow(p / 365.25, 2.0 / 3) }
            if p == nil, let a { p = 365.25 * pow(a, 1.5) }
            guard let a, let p, a > 0, p > 0 else { return nil }
            var temperature = num(iT)
            if temperature == nil, let teff = num(iS) {
                let radius = min(max(pow(teff / 5772, 1.5), 0.1), 3)
                temperature = teff * sqrt(radius * 0.00465 / (2 * a))
            }
            return Planet(name: f[iName], host: f[iHost], a: a, period: p, e: min(max(num(iE) ?? 0, 0), 0.9),
                          temperature: temperature, radius: num(iR), year: Int(num(iY) ?? 0), method: iM.map { f[$0] } ?? "")
        }
    }

    static func parseFlybys(_ object: Any?) -> [Asteroid] {
        guard let root = object as? [String: Any], let fields = root["fields"] as? [String], let rows = root["data"] as? [[Any]] else { return [] }
        func index(_ n: String) -> Int? { fields.firstIndex(of: n) }
        guard let iD = index("des"), let iC = index("cd"), let iDist = index("dist") else { return [] }
        let iV = index("v_rel"), iH = index("h")
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MMM-dd HH:mm"
        func value(_ row: [Any], _ i: Int?) -> Double? {
            guard let i, i < row.count else { return nil }
            return (row[i] as? String).flatMap(Double.init) ?? (row[i] as? NSNumber)?.doubleValue
        }
        return rows.compactMap { row in
            guard row.count > max(iD, iC, iDist), let name = row[iD] as? String, let cd = row[iC] as? String,
                  let date = f.date(from: cd), let dist = value(row, iDist) else { return nil }
            return Asteroid(name: name, date: date, distance: dist, speed: value(row, iV) ?? 10, h: value(row, iH) ?? 25)
        }
        .filter { $0.date > Date().addingTimeInterval(-86400) }
        .sorted { $0.date < $1.date }
    }

    /// Static orbit data for the GPU: (ring radius, eccentricity, periapsis angle, tilt), then (colour, size).
    private func setUp(_ list: [Planet], _ gpu: GPU) {
        planets = list
        var data: [SIMD4<Float>] = []
        for p in list {
            var hash = UInt64(1469598103934665603)
            for b in p.name.utf8 { hash = (hash ^ UInt64(b)) &* 1099511628211 }
            let periapsis = Float(hash % 6283) / 1000
            let tilt = Float((hash >> 16) % 1000) / 1000 - 0.5
            data.append(SIMD4(Float(Self.ring(p.a)), Float(p.e), periapsis, tilt * 0.12))
            let size = Float(0.55 + 0.5 * sqrt(min(p.radius ?? 2, 25)))
            data.append(SIMD4(Self.color(p.temperature), size))
            phase.append(Float((hash >> 32) % 6283) / 1000)
            periapses.append(periapsis)
            rates.append(Float(Self.pace(p.period)))
        }
        orbitData = gpu.buffer(data)
        fresh = true
    }

    /// Cold planets blue, temperate ones green, hot ones gold to red; unknown ones grey.
    static func color(_ kelvin: Double?) -> SIMD3<Float> {
        guard let k = kelvin else { return [0.6, 0.62, 0.68] }
        let t = Float(min(max(log(k / 120) / log(3000.0 / 120), 0), 1))
        let stops: [SIMD3<Float>] = [[0.35, 0.5, 1.0], [0.3, 0.95, 0.85], [0.45, 1.0, 0.5], [1.0, 0.86, 0.35], [1.0, 0.5, 0.2], [1.0, 0.28, 0.42]]
        let x = t * Float(stops.count - 1)
        let i = min(Int(x), stops.count - 2)
        return simd_mix(stops[i], stops[i + 1], SIMD3(repeating: x - Float(i)))
    }

    // MARK: Timeline

    /// The field and a spotlight in turn, with the asteroid interlude after every third spotlight.
    private func advance(_ f: Frame) {
        let length: Float = switch mode { case .field: 22; case .spotlight: 42; case .asteroids: 34 }
        guard f.time - modeStart > length else { return }
        modeStart = f.time
        captioned = false
        switch mode {
        case .field:
            spotlight = (spotlight + 1) % Self.spotlights.count
            members = planets.indices.filter { planets[$0].host == Self.spotlights[spotlight].host }
            mode = members.isEmpty ? .field : .spotlight(spotlight)
        case .spotlight:
            spotlightsShown += 1
            mode = spotlightsShown % 3 == 2 && !asteroids.isEmpty ? .asteroids : .field
        case .asteroids:
            mode = .field
        }
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        if fresh {
            // Open on a spotlight so the first thing on screen is a named system.
            fresh = false
            members = planets.indices.filter { planets[$0].host == Self.spotlights[spotlight].host }
            if !members.isEmpty { mode = .spotlight(spotlight) }
            modeStart = f.time
            captioned = false
            light = nil
        }
        advance(f)
        let spot: Bool = if case .spotlight = mode { true } else { false }
        focus += ((spot ? 1 : 0) - focus) * min(f.dt * 1.2, 1)
        interlude += ((mode == .asteroids ? 1 : 0) - interlude) * min(f.dt * 1.0, 1)

        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            labels = makeLabels(f)
        }
        let light = light!
        let density = f.size.y / 1000

        // Camera: a slow turn over a disc seen from well above.
        let t = Double(f.time) + Double(f.seed) * 30
        let yaw = t * 2 * .pi / 600
        let elevation = Astro.rad(56 + 8 * sin(t * 0.009))
        // In a spotlight the camera comes in until the system's outermost orbit nearly fills the view.
        let outer = members.map { Self.ring(planets[$0].a * (1 + planets[$0].e)) }.max() ?? 1
        let goal = spot ? min(max(outer * 1.25, 0.32), 1) : (mode == .asteroids ? 1.0 : 1)
        reach += (goal - reach) * Double(min(f.dt * 0.5, 1))
        // The star wanders a little on screen so it never sits on the same pixels for hours.
        let aim = SIMD3(0.09 * sin(t * 0.0023), 0.07 * sin(t * 0.0031 + 1), 0) * reach
        let eye = aim + SIMD3(cos(elevation) * cos(yaw), cos(elevation) * sin(yaw), sin(elevation)) * 2.35 * reach
        let forward = simd_normalize(aim - eye)
        let right = simd_normalize(simd_cross(forward, SIMD3(0, 0, 1)))
        let up = simd_cross(right, forward)
        let f3 = { (v: SIMD3<Double>) in SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        let host = spot ? hostTemperature() : 5772
        let view = f.uniforms(SIMD4(f3(eye), 2.0), SIMD4(f3(right), Float(host)), SIMD4(f3(up), 1 - interlude),
                              SIMD4(f3(forward), density))

        // Advance every planet; the spotlit system runs at its true relative periods, innermost orbit in six seconds.
        let fastest = members.map { planets[$0].period }.min() ?? 1
        let dt = f.dt
        var state: [SIMD2<Float>] = []
        state.reserveCapacity(planets.count)
        let isMember = Set(spot ? members : [])
        let reference = Float(yaw)  // toward the viewer, where transits would be seen
        var crossed: [Int] = []
        for i in planets.indices {
            let member = isMember.contains(i)
            let rate = member ? Float(2 * .pi / 6 * fastest / planets[i].period) : rates[i]
            let before = phase[i]
            phase[i] = (phase[i] + rate * dt).truncatingRemainder(dividingBy: 2 * .pi)
            let angle = phase[i] + periapses[i] - reference
            let was = before + periapses[i] - reference
            if sin(angle) * sin(was) < 0, cos(angle) > 0 { crossed.append(i) }
            let dim: Float = spot ? (member ? 1 : 0.22) : 1
            state.append(SIMD2(phase[i], member ? 2 : dim))
        }

        f.cmd.draw(backPipe, into: light, uniforms: view)
        if interlude < 0.99 {
            let stateBuffer = rings[0].next(state)
            if let orbitData, !planets.isEmpty {
                f.cmd.draw(orbitPipe, into: light, clear: false, uniforms: view, buffers: [orbitData, stateBuffer],
                           vertices: planets.count * 48 * 2, primitive: .line)
            }
            var flat: [SIMD4<Float>] = []
            // The Solar System's orbits for scale, dashed.
            for a in [0.387, 0.723, 1.0, 1.524, 5.2, 9.58, 19.2, 30.1] {
                let r = Float(Self.ring(a))
                for k in stride(from: 0, to: 96, by: 2) {
                    let a0 = Float(k) / 96 * 2 * .pi, a1 = Float(k + 1) / 96 * 2 * .pi
                    flat += [SIMD4(r * cos(a0), r * sin(a0), r * cos(a1), r * sin(a1)), SIMD4(0.75, 0.8, 0.9, 0.07 * (1 - interlude))]
                }
            }
            // The spotlit system's orbits, crisp and in their own colours.
            for i in spot ? members : [] {
                let p = planets[i]
                let color = Self.color(p.temperature)
                var last: SIMD2<Float>?
                for k in 0...96 {
                    let E = Double(k) / 96 * 2 * .pi
                    let x = cos(E) - p.e, y = sqrt(1 - p.e * p.e) * sin(E)
                    let r = Self.ring(p.a * hypot(x, y)), v = atan2(y, x) + Double(periapses[i])
                    let q = SIMD2(Float(r * cos(v)), Float(r * sin(v)))
                    if let last { flat += [SIMD4(last.x, last.y, q.x, q.y), SIMD4(color, 0.35 * focus)] }
                    last = q
                }
            }
            f.cmd.draw(flatPipe, into: light, clear: false, uniforms: view, buffers: [rings[1].next(flat)],
                       vertices: 4, instances: flat.count / 2, primitive: .triangleStrip)
            if let orbitData, !planets.isEmpty {
                f.cmd.draw(planetPipe, into: light, clear: false, uniforms: view, buffers: [orbitData, stateBuffer],
                           vertices: 4, instances: planets.count, primitive: .triangleStrip)
            }
            f.cmd.draw(starPipe, into: light, clear: false, uniforms: view, vertices: 4, primitive: .triangleStrip)
        }
        var labelsNow: [LabelQuad] = []
        if interlude > 0.01 { drawAsteroids(f, light, view, &labelsNow, density: density, eye: eye, right: right, up: up, forward: forward) }

        let glow = bloom.glow(f.cmd, light)
        f.cmd.draw(tonePipe, into: f.target, uniforms: f.uniforms(), textures: [light, glow])

        if let labels {
            labelsNow += exoLabels(labels, f, eye: eye, right: right, up: up, forward: forward, density: density)
            f.cmd.draw(labelPipe, into: f.target, clear: false, uniforms: f.uniforms(SIMD4(labels.size.x, labels.size.y, 0, 0)),
                       textures: [labels.texture], buffers: [rings[2].next(labelsNow)], vertices: 4, instances: labelsNow.count,
                       primitive: .triangleStrip)
        }
        if !captioned {
            captioned = true
            caption(f)
        }
        if f.time - summaryAt > 2 {
            summaryAt = f.time
            summary = describe()
        }
        if let synth = f.sound { play(synth, f, crossed, isMember, eye: eye, right: right) }
    }

    private func hostTemperature() -> Double {
        let host = Self.spotlights[spotlight].host
        if host == "TRAPPIST-1" { return 2566 }
        if host == "Proxima Cen" { return 2900 }
        if host == "HR 8799" { return 7400 }
        return 5600
    }

    private func pixel(_ q: SIMD3<Double>, _ f: Frame, eye: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>,
                       forward: SIMD3<Double>) -> SIMD2<Float>? {
        let d = q - eye
        let z = (d * forward).sum()
        guard z > 0.05 else { return nil }
        let p = SIMD2((d * right).sum(), (d * up).sum()) * 2 / z
        return SIMD2(Float((p.x / Double(f.aspect) * 0.5 + 0.5) * Double(f.size.x)), Float((0.5 - p.y * 0.5) * Double(f.size.y)))
    }

    /// Where a planet is now, matching the shader: Kepler's equation on its log-scaled orbit.
    private func position(_ i: Int) -> SIMD3<Double> {
        let p = planets[i]
        let periapsis = Double(periapses[i])
        let M = Double(phase[i]), e = p.e
        var E = M
        for _ in 0..<6 { E -= (E - e * sin(E) - M) / (1 - e * cos(E)) }
        let x = cos(E) - e, y = sqrt(1 - e * e) * sin(E)
        let r = Self.ring(p.a * hypot(x, y)), v = atan2(y, x) + periapsis
        return SIMD3(r * cos(v), r * sin(v), 0)
    }

    private func makeLabels(_ f: Frame) -> Labels {
        var items: [(String, Labels.Style)] = []
        let small = Labels.Style(size: 9.5, weight: .medium, tracking: 1.2)
        for n in ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"] { items.append((n, small)) }
        for s in Self.spotlights {
            for i in planets.indices where planets[i].host == s.host {
                items.append((Self.display(planets[i].name), Labels.Style(size: 10.5, weight: .medium)))
            }
        }
        for a in asteroids {
            items.append((a.name, Labels.Style(size: 10.5, weight: .medium)))
            items.append((Self.flybyLine(a), Labels.Style(size: 9, weight: .regular)))
        }
        for n in ["MOON", "5 LD", "10 LD", "20 LD"] { items.append((n, small)) }
        return Labels(gpu: f.gpu, density: f.overlay.scale * CGFloat(scale), items)
    }

    /// Kepler-90's planets are catalogued under its older name, KOI-351.
    private static func display(_ name: String) -> String { name.replacingOccurrences(of: "KOI-351", with: "Kepler-90") }

    private static func flybyLine(_ a: Asteroid) -> String {
        "\(a.date.formatted(.dateTime.day().month(.abbreviated))) · \(SpaceKit.number(a.lunar, decimals: 1)) LD"
    }

    private func exoLabels(_ labels: Labels, _ f: Frame, eye: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>,
                           forward: SIMD3<Double>, density: Float) -> [LabelQuad] {
        var out: [LabelQuad] = []
        let field = 1 - interlude
        guard field > 0.01, !planets.isEmpty else { return out }
        let names = ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"]
        let solar = [0.387, 0.723, 1.0, 1.524, 5.2, 9.58, 19.2, 30.1]
        // Label the reference orbits at the point nearest the viewer, alternating sides so neighbours don't collide.
        let toward = atan2(eye.y, eye.x)
        for (i, a) in solar.enumerated() where i != 0 && i != 3 {
            let angle = toward + (i % 2 == 0 ? 0.35 : -0.35)
            let r = Self.ring(a)
            if let p = pixel(SIMD3(r * cos(angle), r * sin(angle), 0), f, eye: eye, right: right, up: up, forward: forward),
               let q = labels.quad(names[i], at: p + SIMD2(0, 3 * density), anchor: [0.5, 0], color: .premultiplied(0.8, 0.85, 0.95, 0.3 * field)) {
                out.append(q)
            }
        }
        if focus > 0.02 {
            for i in members {
                guard let p = pixel(position(i), f, eye: eye, right: right, up: up, forward: forward) else { continue }
                if let q = labels.quad(Self.display(planets[i].name), at: p + SIMD2(9 * density, -4 * density), anchor: [0, 1],
                                       color: .premultiplied(1, 0.95, 0.85, 0.8 * focus * field)) {
                    out.append(q)
                }
            }
        }
        return out
    }

    // MARK: Asteroids

    /// Distance from Earth on the flyby dial: square-root scaled so the closest passes don't crowd Earth.
    private static func dial(_ lunar: Double) -> Double { 0.9 * sqrt(min(lunar, 21) / 20) }

    private static func dialPoint(_ s: Asteroid, _ now: Date) -> SIMD2<Double> {
        let angle = -s.date.timeIntervalSince(now) / 86400 / 30 * 2 * .pi + .pi / 2
        return SIMD2(cos(angle), sin(angle)) * dial(s.lunar)
    }

    /// Earth at the centre with rings at 1 (the Moon's orbit), 5, 10 and 20 lunar distances, and each asteroid crossing
    /// on a short straight path through its real closest distance. Closest points sit round a 30-day dial, by date.
    private func drawAsteroids(_ f: Frame, _ light: MTLTexture, _ view: U, _ labelsNow: inout [LabelQuad], density: Float,
                               eye: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>, forward: SIMD3<Double>) {
        let a = interlude
        var lines: [SIMD4<Float>] = []
        var dots: [SIMD4<Float>] = []
        for (ld, bright) in [(1.0, 0.16), (5.0, 0.07), (10.0, 0.06), (20.0, 0.05)] {
            let r = Float(Self.dial(ld))
            for k in 0..<120 {
                let a0 = Float(k) / 120 * 2 * .pi, a1 = Float(k + 1) / 120 * 2 * .pi
                lines += [SIMD4(r * cos(a0), r * sin(a0), r * cos(a1), r * sin(a1)), SIMD4(0.6, 0.75, 1.0, Float(bright) * a)]
            }
        }
        let now = Date()
        let span = 0.32
        for (i, s) in asteroids.enumerated() {
            let c = Self.dialPoint(s, now)
            let n = simd_length(c) > 1e-6 ? simd_normalize(c) : SIMD2(0, 1)
            let along = SIMD2(-n.y, n.x)
            let p0 = c - along * span, p1 = c + along * span
            lines += [SIMD4(Float(p0.x), Float(p0.y), Float(c.x), Float(c.y)), SIMD4(0.85, 0.75, 0.6, 0.06 * a),
                      SIMD4(Float(c.x), Float(c.y), Float(p1.x), Float(p1.y)), SIMD4(0.85, 0.75, 0.6, 0.06 * a)]
            // Each crosses its path again and again, faster flybys faster, and rings a bell at its closest point.
            let travel = (Double(f.time) * s.speed * 0.004 + Double(i) * 0.37).truncatingRemainder(dividingBy: 2 * span) - span
            if i < crossing.count, crossing[i] < 0, travel >= 0, let synth = f.sound, a > 0.5 {
                synth.play(.bell, Float(220 * pow(2, (20 - min(s.lunar, 20)) / 10)), amp: 0.045, pan: Float(c.x), length: 3)
            }
            if i < crossing.count { crossing[i] = travel } else { crossing.append(travel) }
            let q = c + along * travel
            let fade = Float(1 - abs(travel) / span)
            let size = Float(min(max(0.5 + 0.3 * log10(s.size), 0.6), 1.4))
            dots += [SIMD4(Float(q.x), Float(q.y), 0, size), SIMD4(1.0, 0.85, 0.65, a * fade)]
            dots += [SIMD4(Float(c.x), Float(c.y), 0, 0.45), SIMD4(0.9, 0.8, 0.7, 0.3 * a)]
        }
        // Earth, and the Moon where it really is.
        dots += [SIMD4(0, 0, 0, 3.0), SIMD4(0.35, 0.6, 1.0, a)]
        let moon = Astro.rad(Astro.ecliptic(Astro.moon(now), now))
        let r1 = Self.dial(1)
        dots += [SIMD4(Float(r1 * cos(moon)), Float(r1 * sin(moon)), 0, 1.0), SIMD4(0.85, 0.85, 0.8, a)]
        f.cmd.draw(flatPipe, into: light, clear: false, uniforms: view, buffers: [rings[3].next(lines)],
                   vertices: 4, instances: lines.count / 2, primitive: .triangleStrip)
        f.cmd.draw(dotPipe, into: light, clear: false, uniforms: view, buffers: [rings[4].next(dots)],
                   vertices: 4, instances: dots.count / 2, primitive: .triangleStrip)

        guard let labels else { return }
        var taken: [SIMD4<Float>] = []
        func fits(_ q: LabelQuad) -> Bool {
            let x0 = q.pos.x - q.anchor.x * q.rect.z, y0 = q.pos.y - q.anchor.y * q.rect.w
            let box = SIMD4(x0 - 3, y0 - 1, x0 + q.rect.z + 3, y0 + q.rect.w + 1)
            return !taken.contains { $0.x < box.z && box.x < $0.z && $0.y < box.w && box.y < $0.w }
        }
        func take(_ q: LabelQuad) {
            let x0 = q.pos.x - q.anchor.x * q.rect.z, y0 = q.pos.y - q.anchor.y * q.rect.w
            taken.append(SIMD4(x0 - 3, y0 - 1, x0 + q.rect.z + 3, y0 + q.rect.w + 1))
        }
        for (text, ld) in [("MOON", 1.0), ("5 LD", 5.0), ("10 LD", 10.0), ("20 LD", 20.0)] {
            let r = Self.dial(ld)
            let angle = atan2(eye.y, eye.x)
            if let p = pixel(SIMD3(r * cos(angle), r * sin(angle), 0), f, eye: eye, right: right, up: up, forward: forward),
               let q = labels.quad(text, at: p + SIMD2(0, 3 * density), anchor: [0.5, 0], color: .premultiplied(0.7, 0.8, 1.0, 0.4 * a)) {
                labelsNow.append(q)
                take(q)
            }
        }
        for s in asteroids.sorted(by: { $0.distance < $1.distance }).prefix(12) {
            let c = Self.dialPoint(s, now)
            guard let p = pixel(SIMD3(c.x, c.y, 0), f, eye: eye, right: right, up: up, forward: forward),
                  let name = labels.quad(s.name, at: p + SIMD2(6 * density, -1 * density), anchor: [0, 1], color: .premultiplied(1, 0.9, 0.75, 0.75 * a)),
                  let line = labels.quad(Self.flybyLine(s), at: p + SIMD2(6 * density, 0), anchor: [0, 0], color: .premultiplied(0.8, 0.85, 0.95, 0.5 * a)),
                  fits(name), fits(line) else { continue }
            take(name)
            take(line)
            labelsNow += [name, line]
        }
    }

    // MARK: Words

    private func caption(_ f: Frame) {
        let count = SpaceKit.number(Double(planets.count))
        var title = offline ? "Exoplanets: offline" : "\(count) confirmed exoplanets"
        var line = "Every orbit around one star, distances on a log scale"
        switch mode {
        case .spotlight(let i):
            title = Self.spotlights[i].title
            line = "\(members.count) planets · \(Self.spotlights[i].blurb)"
        case .asteroids:
            title = "Asteroids passing Earth"
            line = "The next 30 days · within 0.05 AU, about 19 lunar distances"
        case .field: break
        }
        f.overlay.draw { ctx, _ in
            ctx.text(title, at: CGPoint(x: 30, y: 26), size: 22, weight: .ultraLight, color: NSColor(white: 1, alpha: 0.78), tracking: 1)
            ctx.text(line, at: CGPoint(x: 31, y: 58), size: 12, color: NSColor(white: 1, alpha: 0.45))
        }
    }

    private func describe() -> String {
        if offline || planets.isEmpty {
            return loaded ? "Offline: the exoplanet catalogue hasn't downloaded yet." : "Loading the exoplanet catalogue."
        }
        let count = SpaceKit.number(Double(planets.count))
        switch mode {
        case .spotlight(let i):
            let s = Self.spotlights[i]
            let periods = members.map { planets[$0].period }
            let range = periods.count < 2 ? periods.first.map { "; it orbits in \(Self.days($0))" } ?? ""
                : "; orbits from \(Self.days(periods.min()!)) to \(Self.days(periods.max()!))"
            return "\(count) confirmed exoplanets. Spotlight \(s.title): \(members.count == 1 ? "one planet" : "\(members.count) planets"), \(s.blurb)\(range)."
        case .asteroids:
            guard let closest = asteroids.min(by: { $0.distance < $1.distance }) else { return "\(count) confirmed exoplanets." }
            return "\(asteroids.count) asteroids pass within 19 lunar distances of Earth in the next 30 days. Closest: \(closest.name), "
                + "\(SpaceKit.number(closest.lunar, decimals: 1)) lunar distances \(Self.when(closest.date)), about \(Self.metres(closest.size)) across, "
                + "at \(SpaceKit.number(closest.speed, decimals: 1)) kilometres a second."
        case .field:
            let year = Calendar.current.component(.year, from: Date())
            let recent = planets.filter { $0.year >= year - 1 }.count
            let methods = Dictionary(grouping: planets, by: \.method).mapValues(\.count).max { $0.value < $1.value }
            var text = "\(count) confirmed exoplanets, \(SpaceKit.number(Double(recent))) of them found since \(year - 1)"
            if let methods { text += "; most, \(SpaceKit.number(Double(methods.value))), by the \(methods.key.lowercased()) method" }
            if let next = asteroids.first(where: { $0.date > Date() }) {
                text += ". Next close asteroid: \(next.name), \(SpaceKit.number(next.lunar, decimals: 1)) lunar distances \(Self.when(next.date))"
            }
            return text + "."
        }
    }

    private static func days(_ d: Double) -> String {
        d < 1 ? "\(Int((d * 24).rounded())) hours" : d < 1000 ? "\(SpaceKit.number(d, decimals: d < 10 ? 1 : 0)) days"
            : "\(SpaceKit.number(d / 365.25)) years"
    }

    private static func metres(_ m: Double) -> String {
        m < 1000 ? "\(Int((m / 5).rounded() * 5)) metres" : "\(SpaceKit.number(m / 1000, decimals: 1)) kilometres"
    }

    static func when(_ date: Date, now: Date = Date()) -> String {
        let cal = Calendar.current
        if cal.isDate(date, inSameDayAs: now) { return date < now ? "earlier today" : "today" }
        if let t = cal.date(byAdding: .day, value: 1, to: now), cal.isDate(date, inSameDayAs: t) { return "tomorrow" }
        return "on \(date.formatted(.dateTime.day().month(.wide)))"
    }

    // MARK: Sound

    /// Soft plucks as planets pass the line toward the viewer, where a transit would be seen: pitch from the period on
    /// a pentatonic scale, a few a second at most, like light rain on glass. Spotlight planets always sound.
    private func play(_ synth: Synth, _ f: Frame, _ crossed: [Int], _ members: Set<Int>, eye: SIMD3<Double>, right: SIMD3<Double>) {
        plucks = plucks.filter { f.time - $0 < 1 }
        for i in crossed.shuffled() {
            let member = members.contains(i)
            guard member || (plucks.count < 4 && focus < 0.5 && interlude < 0.5) else { continue }
            plucks.append(f.time)
            let q = position(i) - eye
            let pan = Float(max(-0.8, min(0.8, (q * right).sum())))
            synth.play(.pluck, Self.note(planets[i].period), amp: member ? 0.08 : 0.035, pan: pan, length: member ? 2.2 : 1.4)
        }
        synth.drone(0, 98, 0.035)
        synth.drone(1, 147, 0.02, pan: 0.3)
        synth.drone(2, 196, 0.012 * (1 - focus), pan: -0.3)
    }

    // MARK: Shaders

    private static let source = """
    // a: eye, zoom   b: right, host temperature   c: up, field visibility   d: forward, pixel density
    inline float3 viewOf(float3 q, constant U &u) {
        float3 d = q - u.a.xyz;
        float z = max(dot(d, u.d.xyz), 0.01);
        float2 p = float2(dot(d, u.b.xyz), dot(d, u.c.xyz)) * u.a.w / z;
        return float3((p.x / (u.res.x / u.res.y) * 0.5 + 0.5) * u.res.x, (0.5 - p.y * 0.5) * u.res.y, z);
    }

    fragment float4 exoBack(V in [[stage_in]], constant U &u [[buffer(0)]]) {
        float3 c0 = viewOf(float3(0), u);
        float r = length(in.uv * u.res - c0.xy) / u.res.y;
        float3 c = float3(0.0018, 0.002, 0.0038) + float3(0.006, 0.005, 0.008) * exp(-r * 2.5);
        return float4(c, 1);
    }

    // A point on an orbit at eccentric anomaly E: the real ellipse, its distance mapped to the log scale.
    inline float3 orbitPoint(float4 o, float E, float a) {
        float e = o.y;
        float2 xy = float2(cos(E) - e, sqrt(1 - e * e) * sin(E));
        float r = length(xy);
        float rr = 0.07 + 0.93 * clamp((log10(max(a * r, 1e-4)) + 2.2) / 5.0, 0.0, 1.04);
        float v = atan2(xy.y, xy.x) + o.z;
        return float3(rr * cos(v), rr * sin(v), rr * o.w * sin(v - o.z * 0.7));
    }

    // Recovers the semi-major axis in AU from the ring radius stored for the shader.
    inline float axisOf(float ring) { return pow(10.0, (ring - 0.07) / 0.93 * 5.0 - 2.2); }

    // buffer 1: (ring, e, periapsis, tilt), (colour, size) per planet. buffer 2: (mean anomaly, highlight) per planet.
    vertex P exoOrbit(uint vid [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *orbits [[buffer(1)]],
                      const device float2 *state [[buffer(2)]]) {
        uint segs = 48;
        uint planet = vid / (segs * 2);
        uint k = (vid / 2) % segs + (vid & 1);
        float4 o = orbits[2 * planet], c = orbits[2 * planet + 1];
        float2 s = state[planet];
        float E = float(k) / float(segs) * TAU;
        float3 q = orbitPoint(o, E, axisOf(o.x));
        float3 p = viewOf(q, u);
        P out;
        out.pos = pixelToClip(p.xy, u.res);
        out.size = 1;
        float hi = s.y > 1.5 ? 1.0 : 0.0;
        float level = hi > 0 ? 0.0 : 0.014 * s.y;
        out.color = float4(c.rgb * level * u.c.w, 0);
        return out;
    }

    vertex GlowOut exoPlanet(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                             const device float4 *orbits [[buffer(1)]], const device float2 *state [[buffer(2)]]) {
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        GlowOut o;
        o.uv = corner;
        float d = u.d.w;
        float4 orb = orbits[2 * iid], c = orbits[2 * iid + 1];
        float2 s = state[iid];
        float M = s.x, e = orb.y;
        float E = M + e * sin(M);
        for (int i = 0; i < 4; i++) E -= (E - e * sin(E) - M) / (1 - e * cos(E));
        float3 p = viewOf(orbitPoint(orb, E, axisOf(orb.x)), u);
        float hi = s.y > 1.5 ? 1.0 : 0.0;
        float size = c.w * d * (1 + hi * 1.6);
        float extent = size * 3.0;
        o.pos = pixelToClip(p.xy + corner * extent, u.res);
        o.color = float4(c.rgb * (hi > 0 ? 1.6 : 0.1 * s.y) * u.c.w, 0);
        o.extra = float4(4.5, 2.6, hi > 0 ? 0.06 : 0.0, 0);
        return o;
    }

    // The one star they all circle, coloured by the spotlit host's temperature.
    vertex GlowOut starVertex(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        float3 p = viewOf(float3(0), u);
        float r = 26 * u.d.w;
        GlowOut o;
        o.pos = pixelToClip(p.xy + corner * r, u.res);
        o.uv = corner;
        o.color = float4(blackbody(u.b.w) * 1.5 * u.c.w, 0);
        o.extra = float4(40, 6, 0.1, 0);
        return o;
    }

    // Flat segments in the plane: (x0, y0, x1, y1), (colour, alpha).
    vertex LineOut flatLine(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                            const device float4 *segs [[buffer(1)]]) {
        float4 s = segs[2 * iid], c = segs[2 * iid + 1];
        float3 a = viewOf(float3(s.xy, 0), u), b = viewOf(float3(s.zw, 0), u);
        return lineQuad(a.xy, b.xy, 1.0 * u.d.w, float4(c.rgb * c.a, 0), vid, u.res);
    }

    // Glowing dots in the plane: (x, y, z, size), (colour, brightness).
    vertex GlowOut flatDot(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                           const device float4 *dots [[buffer(1)]]) {
        float4 a = dots[2 * iid], c = dots[2 * iid + 1];
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        float3 p = viewOf(a.xyz, u);
        float size = a.w * 2.2 * u.d.w;
        GlowOut o;
        o.pos = pixelToClip(p.xy + corner * size * 3, u.res);
        o.uv = corner;
        o.color = float4(c.rgb * c.a * 1.2, 0);
        o.extra = float4(4.5, 3.0, 0.05, 0);
        return o;
    }

    fragment float4 exoTone(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                            texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = light.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.55;
        return float4(film(c), 1);
    }
    """
}
