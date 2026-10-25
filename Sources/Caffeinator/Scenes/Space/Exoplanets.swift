import AppKit
import Metal
import simd

/// Every confirmed exoplanet from the NASA Exoplanet Archive, circling one shared star on a logarithmic scale of
/// distance in front of the real Milky Way. Each planet is a point of its own star's reflected light, tinted by what it's
/// likely made of, the hottest glowing with their own heat, and each leaves a fading trail like a long exposure.
/// Famous systems take turns in the spotlight against the real sky behind them, two dissolving into the actual
/// photographs of their planets; between them, Earth and the asteroids passing it this month, from JPL.
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
        /// The host star's temperature (K) and radius (Suns), its place in the sky (degrees), its distance (parsecs).
        var starTemperature: Double? = nil
        var starRadius: Double? = nil
        var ra: Double? = nil, dec: Double? = nil
        var distance: Double? = nil
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
        let constellation: String
        var image: ExoImage? = nil
    }

    private static let spotlights = [
        Spotlight(host: "TRAPPIST-1", title: "TRAPPIST-1", blurb: "Earth-sized planets in a resonant chain around a cool red dwarf, 40 light-years away",
                  constellation: "Aquarius"),
        Spotlight(host: "KOI-351", title: "Kepler-90", blurb: "as many planets as the Solar System, all packed inside Earth's orbit", constellation: "Draco"),
        Spotlight(host: "51 Peg", title: "51 Pegasi", blurb: "the first planet found orbiting a Sun-like star, in 1995: a hot Jupiter circling in 4.2 days",
                  constellation: "Pegasus"),
        Spotlight(host: "Proxima Cen", title: "Proxima Centauri", blurb: "the nearest star to the Sun, 4.2 light-years away", constellation: "Centaurus"),
        Spotlight(host: "HR 8799", title: "HR 8799", blurb: "giant planets photographed directly, far out from their young star", constellation: "Pegasus",
                  image: .hr8799),
        Spotlight(host: "TOI-178", title: "TOI-178", blurb: "five of them locked in a chain of resonances", constellation: "Sculptor"),
        Spotlight(host: "55 Cnc", title: "55 Cancri", blurb: "including a lava world that circles in under 18 hours", constellation: "Cancer"),
        Spotlight(host: "TYC 8998-760-1", title: "YSES 1", blurb: "the first photograph of two giant planets around a young Sun-like star, 310 light-years away",
                  constellation: "Musca", image: .yses1),
        Spotlight(host: "Kepler-11", title: "Kepler-11", blurb: "six planets packed closer to their star than Venus is to the Sun", constellation: "Cygnus"),
    ]

    private static let url = URL(string: "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,hostname,pl_orbsmax,pl_orbper,pl_orbeccen,pl_eqt,pl_rade,st_teff,disc_year,discoverymethod,ra,dec,sy_dist,st_rad+from+pscomppars&format=csv")!
    private static let asteroidURL = URL(string: "https://ssd-api.jpl.nasa.gov/cad.api?date-min=now&date-max=%2B30&dist-max=0.05")!
    /// Shared with the orrery, so the map downloads once.
    private static let earthURL = URL(string: "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg")!
    /// Where the Kepler telescope stared for four years, in Cygnus and Lyra, and found most of these planets.
    private static let keplerField = SkyCatalog.unit(ra: 290.67, dec: 44.5)
    /// The photograph's own stars are drawn round only near the galactic plane; toward the galactic poles they streak.
    /// Sampled two mip levels softer they melt into a glow, with the catalogue's stars crisp on top.
    private static let soften: Float = 2.2

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

    private let trailPipe: MTLRenderPipelineState
    private let pointPipe: MTLRenderPipelineState
    private let spherePipe: MTLRenderPipelineState
    private let starPipe: MTLRenderPipelineState
    private let flatPipe: MTLRenderPipelineState
    private let dotPipe: MTLRenderPipelineState
    private let globePipe: MTLRenderPipelineState
    private let photoPipe: MTLRenderPipelineState
    private let addPipe: MTLRenderPipelineState
    private let pixelLinePipe: MTLRenderPipelineState
    private let labelPipe: MTLRenderPipelineState
    private let sky: SkyBackdrop
    private let lens: Lens
    private let rings: [Ring]
    private var light: MTLTexture?
    private var skyBefore: MTLTexture?
    private var labels: Labels?
    private let blank: MTLTexture
    private var earth: MTLTexture?
    private var photos: [String: MTLTexture] = [:]
    private var photosLoaded = false

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
    private var photo: Float = 0
    /// Turns the scene's frame into the sky's: the real sky behind the current view, eased from the previous one.
    private var orient = matrix_identity_double3x3
    private var before: simd_double3x3?
    private var orientAt: Float = -100

    var scale: Float { 0.7 }
    var ready: Bool { loaded && sky.loaded && photosLoaded }
    var facts: String? { summary.isEmpty ? describe() : summary }
    var credit: String? {
        ["Milky Way: ESO/S. Brunier", ExoImage.hr8799.credit, ExoImage.yses1.credit, "Earth map: Solar System Scope"].joined(separator: "; ")
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(SpaceKit.metal + SpaceSky.metal + Self.source)
        trailPipe = try gpu.pipeline(library, "trailFragment", vertex: "exoTrail", blend: .add)
        pointPipe = try gpu.pipeline(library, "exoPointFragment", vertex: "exoPoint", blend: .add)
        spherePipe = try gpu.pipeline(library, "exoSphereFragment", vertex: "exoSphere", blend: .alpha)
        starPipe = try gpu.pipeline(library, "exoStarFragment", vertex: "exoStar", blend: .alpha)
        flatPipe = try gpu.pipeline(library, "lineFragment", vertex: "flatLine", blend: .add)
        dotPipe = try gpu.pipeline(library, "exoPointFragment", vertex: "flatDot", blend: .add)
        globePipe = try gpu.pipeline(library, "exoGlobeFragment", vertex: "exoGlobe", blend: .alpha)
        photoPipe = try gpu.pipeline(library, "exoPhoto", blend: .alpha)
        addPipe = try gpu.pipeline(library, "exoAdd", blend: .add)
        pixelLinePipe = try gpu.pipeline(library, "lineFragment", vertex: "pixelLine", blend: .alpha)
        labelPipe = try gpu.pipeline(library, "labelFragment", vertex: "labelVertex", blend: .alpha)
        sky = try SkyBackdrop(gpu: gpu)
        lens = try Lens(gpu: gpu)
        rings = (0..<14).map { _ in Ring(gpu: gpu, bytes: 65536) }
        var zero: UInt32 = 0
        blank = gpu.texture(1, 1, .rgba8Unorm, bytes: &zero, bytesPerRow: 4)
        spotlight = (Snapshot.variant ?? Int.random(in: 0..<Self.spotlights.count)) % Self.spotlights.count
        Task { @MainActor [weak self] in
            async let table = Feed.data(Self.url, maxAge: 7 * 86400)
            async let flybys = Feed.object(Self.asteroidURL, maxAge: 12 * 3600)
            let parsed = (await table).map(Self.parse) ?? []
            let rocks = Self.parseFlybys(await flybys)
            guard let self else { return }
            asteroids = rocks
            if parsed.isEmpty { offline = true } else { setUp(parsed, gpu) }
            loaded = true
        }
        // The two photographs are small; fetch them one after the other, then Earth's map for the asteroid interlude.
        Task { @MainActor [weak self] in
            for image in [ExoImage.hr8799, ExoImage.yses1] {
                let t = await Imagery.texture(image.url, gpu, maxSize: 1400, maxAge: 365 * 86400)
                self?.photos[image.url.absoluteString] = t
            }
            self?.photosLoaded = true
            let map = await Imagery.texture(Self.earthURL, gpu, maxSize: 1024, maxAge: 365 * 86400)
            self?.earth = map
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
            iY = column("disc_year"), iM = column("discoverymethod"), iRA = column("ra"), iDec = column("dec"),
            iD = column("sy_dist"), iSR = column("st_rad")
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
                let radius = num(iSR) ?? min(max(pow(teff / 5772, 1.5), 0.1), 3)
                temperature = teff * sqrt(radius * 0.00465 / (2 * a))
            }
            return Planet(name: f[iName], host: f[iHost], a: a, period: p, e: min(max(num(iE) ?? 0, 0), 0.9),
                          temperature: temperature, radius: num(iR), year: Int(num(iY) ?? 0), method: iM.map { f[$0] } ?? "",
                          starTemperature: num(iS), starRadius: num(iSR), ra: num(iRA), dec: num(iDec), distance: num(iD))
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

    /// Static orbit data for the GPU: (ring radius, eccentricity, periapsis angle, tilt), (reflected colour, size),
    /// (thermal glow, 0).
    private func setUp(_ list: [Planet], _ gpu: GPU) {
        planets = list
        var data: [SIMD4<Float>] = []
        for p in list {
            var hash = UInt64(1469598103934665603)
            for b in p.name.utf8 { hash = (hash ^ UInt64(b)) &* 1099511628211 }
            let periapsis = Float(hash % 6283) / 1000
            let tilt = Float((hash >> 16) % 1000) / 1000 - 0.5
            data.append(SIMD4(Float(Self.ring(p.a)), Float(p.e), periapsis, tilt * 0.12))
            let look = Self.appearance(p)
            data.append(SIMD4(look.reflected, Float(Self.size(p))))
            data.append(SIMD4(look.glow, 0))
            phase.append(Float((hash >> 32) % 6283) / 1000)
            periapses.append(periapsis)
            rates.append(Float(Self.pace(p.period)))
        }
        orbitData = gpu.buffer(data)
        fresh = true
    }

    /// Radius in Earth radii, from the archive or guessed from nothing as a small Neptune.
    private static func size(_ p: Planet) -> Double { min(max(p.radius ?? 2.5, 0.4), 25) }

    /// How a planet would look: its star's light reflected with an albedo colour that suits its size and temperature
    /// (icy, rocky, hazy sub-Neptune, ice giant, banded cool giant, dark hot Jupiter), plus the glow of its own heat,
    /// a black body that turns dull red above about 800 K.
    static func appearance(_ p: Planet) -> (reflected: SIMD3<Float>, glow: SIMD3<Float>) {
        let r = size(p), t = p.temperature ?? 300
        let surface: SIMD3<Float>
        switch (r, t) {
        case (6..., ..<250): surface = [0.82, 0.74, 0.6]           // cool gas giant: ammonia clouds, Jupiter's cream
        case (6..., ..<900): surface = [0.42, 0.5, 0.64]            // warm giant: clear, deep blue air
        case (6..., _): surface = [0.16, 0.15, 0.16]                // hot Jupiter: darker than charcoal
        case (3..., ..<400): surface = [0.5, 0.72, 0.82]            // ice giant: methane blue
        case (3..., _), (1.6..., _): surface = t < 700 ? [0.56, 0.62, 0.7] : [0.4, 0.4, 0.43]   // hazy sub-Neptune
        case (_, ..<180): surface = [0.8, 0.83, 0.88]               // frozen rock
        case (_, ..<400): surface = [0.46, 0.5, 0.56]               // temperate rock, clouds and dark ground
        default: surface = [0.4, 0.34, 0.29]                        // bare hot rock
        }
        let star = SpaceSky.tint(p.starTemperature ?? 5772)
        let heat = Float(min(max((t - 800) / 1400, 0), 1.6))
        return (surface * star, SpaceSky.tint(t) * heat * heat)
    }

    // MARK: Timeline

    private static func length(_ m: Mode) -> Float { switch m { case .field: 24; case .spotlight: 44; case .asteroids: 34 } }

    /// The field and a spotlight in turn, with the asteroid interlude after every third spotlight.
    private func advance(_ f: Frame) {
        guard f.time - modeStart > Self.length(mode) else { return }
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
        reorient(f)
    }

    // MARK: Camera and sky

    private struct Camera { var eye, right, up, forward: SIMD3<Double> }

    /// A slow turn over the disc, seen from well above.
    private func camera(_ time: Float, _ seed: Float, reach: Double) -> Camera {
        let t = Double(time) + Double(seed) * 30
        let yaw = t * 2 * .pi / 600
        let elevation = Astro.rad(56 + 8 * sin(t * 0.009))
        // The star wanders a little on screen so it never sits on the same pixels for hours.
        let aim = SIMD3(0.09 * sin(t * 0.0023), 0.07 * sin(t * 0.0031 + 1), 0) * reach
        let eye = aim + SIMD3(cos(elevation) * cos(yaw), cos(elevation) * sin(yaw), sin(elevation)) * 2.35 * reach
        let forward = simd_normalize(aim - eye)
        let right = simd_normalize(simd_cross(forward, SIMD3(0, 0, 1)))
        return Camera(eye: eye, right: right, up: simd_cross(right, forward), forward: forward)
    }

    /// Points the real sky behind the view: the Kepler field for the whole catalogue, the spotlit star's own patch of
    /// sky as seen from Earth, and for the asteroids the true sky around Earth's orbit. It eases over from the last.
    private func reorient(_ f: Frame) {
        let goal: simd_double3x3
        if mode == .asteroids {
            goal = simd_double3x3(columns: (SpaceSky.equatorial(ecliptic: [1, 0, 0]), SpaceSky.equatorial(ecliptic: [0, 1, 0]),
                                            SpaceSky.equatorial(ecliptic: [0, 0, 1])))
        } else {
            var target = Self.keplerField
            if case .spotlight = mode, let p = members.first.map({ planets[$0] }), let ra = p.ra, let dec = p.dec {
                target = SkyCatalog.unit(ra: ra, dec: dec)
            }
            // Centred halfway through the mode, so the turning camera carries it across the middle.
            let c = camera(f.time + Self.length(mode) / 2, f.seed, reach: 1)
            let north = simd_normalize(SIMD3<Double>(0, 0, 1) - target * target.z)
            let east = simd_cross(target, north)
            let sceneAxes = simd_double3x3(columns: (c.right, c.up, c.forward))
            goal = simd_double3x3(columns: (east, north, target)) * sceneAxes.transpose
        }
        before = loaded ? orient : nil
        orient = goal
        orientAt = f.time
    }

    private func skyCamera(_ m: simd_double3x3, _ c: Camera) -> SkyBackdrop.Camera {
        let v = { (x: SIMD3<Double>) in SIMD3<Float>(simd_normalize(m * x)) }
        return SkyBackdrop.Camera(right: v(c.right), up: v(c.up), forward: v(c.forward), focal: 2)
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
            reorient(f)
            before = nil
        }
        if f.index == 0 { reorient(f) }
        advance(f)
        let spot: Bool = if case .spotlight = mode { true } else { false }
        focus += ((spot ? 1 : 0) - focus) * min(f.dt * 1.2, 1)
        interlude += ((mode == .asteroids ? 1 : 0) - interlude) * min(f.dt * 1.0, 1)
        let image = spot ? Self.spotlights[spotlight].image.flatMap { i in photos[i.url.absoluteString].map { (i, $0) } } : nil
        let inMode = f.time - modeStart
        let photoGoal: Float = image != nil && inMode > 12 && inMode < Self.length(mode) - 9 ? 1 : 0
        photo += (photoGoal - photo) * min(f.dt * 0.7, 1)

        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            skyBefore = f.gpu.texture(f.target.width, f.target.height)
            labels = makeLabels(f)
        }
        let light = light!
        let density = f.size.y / 1000

        // In a spotlight the camera comes in until the system's outermost orbit nearly fills the view.
        let outer = members.map { Self.ring(planets[$0].a * (1 + planets[$0].e)) }.max() ?? 1
        let goal = spot ? min(max(outer * 1.25, 0.32), 1) : 1
        reach += (goal - reach) * Double(min(f.dt * 0.5, 1))
        let cam = camera(f.time, f.seed, reach: reach)
        let (eye, right, up, forward) = (cam.eye, cam.right, cam.up, cam.forward)
        let f3 = { (v: SIMD3<Double>) in SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        let host = spot ? hostTemperature() : 5772
        let field = 1 - interlude
        let view = f.uniforms(SIMD4(f3(eye), 2.0), SIMD4(f3(right), Float(host)), SIMD4(f3(up), field), SIMD4(f3(forward), density))

        // Advance every planet; the spotlit system runs at its true relative periods, innermost orbit in six seconds.
        let fastest = members.map { planets[$0].period }.min() ?? 1
        let dt = f.dt
        var state: [SIMD2<Float>] = []
        state.reserveCapacity(planets.count)
        let isMember = Set(spot ? members : [])
        let reference = Float(atan2(eye.y, eye.x))  // toward the viewer, where transits would be seen
        var crossed: [Int] = []
        for i in planets.indices {
            let member = isMember.contains(i)
            let rate = member ? Float(2 * .pi / 6 * fastest / planets[i].period) : rates[i]
            let before = phase[i]
            phase[i] = (phase[i] + rate * dt).truncatingRemainder(dividingBy: 2 * .pi)
            let angle = phase[i] + periapses[i] - reference
            let was = before + periapses[i] - reference
            if sin(angle) * sin(was) < 0, cos(angle) > 0 { crossed.append(i) }
            let dim: Float = spot ? (member ? 1 : 0.07) : 1
            state.append(SIMD2(phase[i], member ? 2 : dim))
        }

        // The real sky, dissolving from the last view's patch of sky to this one's.
        let skyLevel = 0.6 * (1 - 0.85 * photo)
        let ease = min(max((f.time - orientAt) / 5, 0), 1)
        let mix = ease * ease * (3 - 2 * ease)
        if let before, mix < 1 {
            sky.draw(f.cmd, into: skyBefore!, camera: skyCamera(before, cam), exposure: skyLevel * (1 - mix), soften: Self.soften)
            sky.draw(f.cmd, into: light, camera: skyCamera(orient, cam), exposure: skyLevel * mix, soften: Self.soften)
            f.cmd.draw(addPipe, into: light, clear: false, uniforms: f.uniforms(), textures: [skyBefore!])
        } else {
            sky.draw(f.cmd, into: light, camera: skyCamera(orient, cam), exposure: skyLevel, soften: Self.soften)
            if mix >= 1 { self.before = nil }
        }

        if field > 0.01 {
            let stateBuffer = rings[0].next(state)
            if let orbitData, !planets.isEmpty {
                f.cmd.draw(trailPipe, into: light, clear: false, uniforms: view, buffers: [orbitData, stateBuffer],
                           vertices: 4, instances: planets.count * 48, primitive: .triangleStrip)
                f.cmd.draw(pointPipe, into: light, clear: false, uniforms: view, buffers: [orbitData, stateBuffer],
                           vertices: 4, instances: planets.count, primitive: .triangleStrip)
            }
            var flat: [SIMD4<Float>] = []
            // The Solar System's orbits for scale, dashed.
            for a in [0.387, 0.723, 1.0, 1.524, 5.2, 9.58, 19.2, 30.1] {
                let r = Float(Self.ring(a))
                for k in stride(from: 0, to: 120, by: 2) {
                    let a0 = Float(k) / 120 * 2 * .pi, a1 = Float(k + 1) / 120 * 2 * .pi
                    flat += [SIMD4(r * cos(a0), r * sin(a0), r * cos(a1), r * sin(a1)), SIMD4(0.85, 0.82, 0.76, 0.05 * field)]
                }
            }
            // The spotlit system's orbits, fine and in the colour of their planets.
            for i in spot ? members : [] {
                let p = planets[i]
                let look = Self.appearance(p)
                let color = simd_normalize(look.reflected + look.glow + 0.05) * 0.9
                var last: SIMD2<Float>?
                for k in 0...128 {
                    let E = Double(k) / 128 * 2 * .pi
                    let x = cos(E) - p.e, y = sqrt(1 - p.e * p.e) * sin(E)
                    let r = Self.ring(p.a * hypot(x, y)), v = atan2(y, x) + Double(periapses[i])
                    let q = SIMD2(Float(r * cos(v)), Float(r * sin(v)))
                    if let last { flat += [SIMD4(last.x, last.y, q.x, q.y), SIMD4(color, 0.22 * focus * field)] }
                    last = q
                }
            }
            f.cmd.draw(flatPipe, into: light, clear: false, uniforms: view, buffers: [rings[1].next(flat)],
                       vertices: 4, instances: flat.count / 2, primitive: .triangleStrip)
            // The star, then the spotlit planets in front of it or hidden behind it.
            let starRadius = spot ? hostRadius() : 0.045
            f.cmd.draw(starPipe, into: light, clear: false, uniforms: view, buffers: [rings[2].next(SIMD4(Float(starRadius), Float(host), field, 0))],
                       vertices: 4, primitive: .triangleStrip)
            if spot, focus > 0.01, let orbitData {
                f.cmd.draw(spherePipe, into: light, clear: false, uniforms: view,
                           buffers: [orbitData, stateBuffer, rings[3].next(members.map { UInt32($0) }),
                                     rings[4].next(SIMD4(Float(starRadius), focus * field, 0, 0))],
                           vertices: 4, instances: members.count, primitive: .triangleStrip)
            }
        }
        var labelsNow: [LabelQuad] = []
        var lines: [SIMD4<Float>] = []
        if interlude > 0.01 { drawAsteroids(f, light, view, &labelsNow, density: density, cam: cam) }
        if let (info, texture) = image, photo > 0.005 {
            let frame = photoFrame(info, f)
            f.cmd.draw(photoPipe, into: light, clear: false,
                       uniforms: f.uniforms(SIMD4(frame.origin.x, frame.origin.y, frame.scale, photo), SIMD4(info.size.x, info.size.y, 0, 0)),
                       textures: [texture], buffers: [rings[10].next(info.masks.isEmpty ? [SIMD4<Float>(-1, -1, -1, -1)] : info.masks),
                                                     rings[11].next(UInt32(info.masks.count))])
            photoMarks(info, frame, f, &labelsNow, &lines)
        }

        lens.develop(f, light, Lens.Look(exposure: 1, bloom: 0.035, halation: 0.018, vignette: 0.3, aberration: 0.8, grain: 0.35))

        if !lines.isEmpty {
            f.cmd.draw(pixelLinePipe, into: f.target, clear: false, uniforms: view, buffers: [rings[12].next(lines)],
                       vertices: 4, instances: lines.count / 2, primitive: .triangleStrip)
        }
        if let labels {
            labelsNow += exoLabels(labels, f, cam: cam, density: density)
            f.cmd.draw(labelPipe, into: f.target, clear: false, uniforms: f.uniforms(SIMD4(labels.size.x, labels.size.y, 0, 0)),
                       textures: [labels.texture], buffers: [rings[13].next(labelsNow)], vertices: 4, instances: labelsNow.count,
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
        if let t = members.compactMap({ planets[$0].starTemperature }).first { return t }
        return switch Self.spotlights[spotlight].host {
        case "TRAPPIST-1": 2566
        case "Proxima Cen": 2900
        case "HR 8799": 7400
        default: 5600
        }
    }

    /// The star's radius on the disc: a quarter-power of its true size, so a red dwarf is plainly smaller than the Sun.
    private func hostRadius() -> Double {
        let r = members.compactMap { planets[$0].starRadius }.first ?? 1
        return 0.045 * pow(min(max(r, 0.08), 3), 0.25)
    }

    private func pixel(_ q: SIMD3<Double>, _ f: Frame, _ c: Camera) -> SIMD2<Float>? {
        let d = q - c.eye
        let z = (d * c.forward).sum()
        guard z > 0.05 else { return nil }
        let p = SIMD2((d * c.right).sum(), (d * c.up).sum()) * 2 / z
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

    // MARK: Photographs

    /// The photograph fills most of the screen's height, centred: origin in target pixels and target pixels per file pixel.
    private func photoFrame(_ info: ExoImage, _ f: Frame) -> (origin: SIMD2<Float>, scale: Float) {
        let scale = f.size.y * 0.86 / info.size.y
        return ((f.size - info.size * scale) / 2, scale)
    }

    /// Thin figure marks over the photograph: a cross where the hidden star is, each planet's letter, a scale bar, the
    /// instrument and the credit.
    private func photoMarks(_ info: ExoImage, _ frame: (origin: SIMD2<Float>, scale: Float), _ f: Frame, _ out: inout [LabelQuad],
                            _ lines: inout [SIMD4<Float>]) {
        guard let labels else { return }
        let a = photo * photo
        let d = f.size.y / 1000
        let at = { (p: SIMD2<Float>) in frame.origin + p * frame.scale }
        let star = at(info.star)
        let ink = SIMD4<Float>(0.82, 0.86, 0.92, 0.5 * a)
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] as [(Float, Float)] {
            let dir = SIMD2(dx, dy)
            let p0 = star + dir * 9 * d, p1 = star + dir * 17 * d
            lines += [SIMD4(p0.x, p0.y, p1.x, p1.y), ink]
        }
        for (letter, p) in info.planets {
            if let q = labels.quad(letter, at: at(p) + SIMD2(19, -15) * d, anchor: [0, 0.5], color: .premultiplied(0.88, 0.9, 0.95, 0.62 * a)) {
                out.append(q)
            }
        }
        let left = frame.origin.x + 18 * d, bottom = frame.origin.y + info.size.y * frame.scale - 24 * d
        let length = Float(info.bar) * info.pixelsPerAU * frame.scale
        lines += [SIMD4(left, bottom, left + length, bottom), ink]
        lines += [SIMD4(left, bottom - 3 * d, left, bottom + 3 * d), ink, SIMD4(left + length, bottom - 3 * d, left + length, bottom + 3 * d), ink]
        if let q = labels.quad("\(info.bar) AU", at: SIMD2(left + length / 2, bottom - 6 * d), anchor: [0.5, 1], color: .premultiplied(0.82, 0.86, 0.92, 0.55 * a)) {
            out.append(q)
        }
        let below = frame.origin.y + info.size.y * frame.scale + 8 * d
        if let q = labels.quad(info.caption, at: SIMD2(frame.origin.x, below), anchor: [0, 0], color: .premultiplied(0.82, 0.86, 0.92, 0.5 * a)) {
            out.append(q)
        }
        if let q = labels.quad(info.credit, at: SIMD2(frame.origin.x + info.size.x * frame.scale, below), anchor: [1, 0],
                               color: .premultiplied(0.82, 0.86, 0.92, 0.36 * a)) {
            out.append(q)
        }
    }

    // MARK: Labels

    private func makeLabels(_ f: Frame) -> Labels {
        var items: [(String, Labels.Style)] = []
        let small = Labels.Style(size: 9, weight: .medium, tracking: 1.4)
        for n in ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"] { items.append((n, small)) }
        for s in Self.spotlights {
            for i in planets.indices where planets[i].host == s.host {
                items.append((Self.display(planets[i].name), Labels.Style(size: 10, weight: .regular, tracking: 0.3)))
            }
        }
        for a in asteroids {
            items.append((a.name, Labels.Style(size: 10, weight: .regular)))
            items.append((Self.flybyLine(a), Labels.Style(size: 8.5, weight: .regular, tracking: 0.3)))
        }
        for n in ["MOON", "5 LD", "10 LD", "20 LD"] { items.append((n, small)) }
        for image in [ExoImage.hr8799, ExoImage.yses1] {
            for p in image.planets { items.append((p.letter, Labels.Style(size: 13, weight: .light))) }
            items.append((image.caption, Labels.Style(size: 9, weight: .medium, tracking: 1.4)))
            items.append((image.credit, Labels.Style(size: 9, weight: .regular, tracking: 0.3)))
            items.append(("\(image.bar) AU", small))
        }
        return Labels(gpu: f.gpu, density: f.overlay.scale * CGFloat(scale), items)
    }

    /// Kepler-90's planets are catalogued under its older name, KOI-351.
    private static func display(_ name: String) -> String { name.replacingOccurrences(of: "KOI-351", with: "Kepler-90") }

    private static func flybyLine(_ a: Asteroid) -> String {
        "\(a.date.formatted(.dateTime.day().month(.abbreviated))) · \(SpaceKit.number(a.lunar, decimals: 1)) LD"
    }

    private func exoLabels(_ labels: Labels, _ f: Frame, cam: Camera, density: Float) -> [LabelQuad] {
        var out: [LabelQuad] = []
        let field = (1 - interlude) * (1 - photo)
        guard field > 0.01, !planets.isEmpty else { return out }
        let names = ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"]
        let solar = [0.387, 0.723, 1.0, 1.524, 5.2, 9.58, 19.2, 30.1]
        // Label the reference orbits at the point nearest the viewer, alternating sides so neighbours don't collide.
        let toward = atan2(cam.eye.y, cam.eye.x)
        for (i, a) in solar.enumerated() where i != 0 && i != 3 {
            let angle = toward + (i % 2 == 0 ? 0.35 : -0.35)
            let r = Self.ring(a)
            if let p = pixel(SIMD3(r * cos(angle), r * sin(angle), 0), f, cam),
               let q = labels.quad(names[i], at: p + SIMD2(0, 3 * density), anchor: [0.5, 0], color: .premultiplied(0.85, 0.85, 0.82, 0.24 * field)) {
                out.append(q)
            }
        }
        if focus > 0.02 {
            for i in members {
                guard let p = pixel(position(i), f, cam) else { continue }
                if let q = labels.quad(Self.display(planets[i].name), at: p + SIMD2(10 * density, -5 * density), anchor: [0, 1],
                                       color: .premultiplied(0.95, 0.93, 0.88, 0.6 * focus * field)) {
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

    /// Earth at the centre, lit by the Sun from where it really is and turned to the real time of day, the Moon in its
    /// true place and phase on the 1 lunar-distance ring, rings at 5, 10 and 20, and each asteroid crossing on a short
    /// straight path through its real closest distance. Closest points sit round a 30-day dial, by date.
    private func drawAsteroids(_ f: Frame, _ light: MTLTexture, _ view: U, _ labelsNow: inout [LabelQuad], density: Float, cam: Camera) {
        let a = interlude * (1 - photo)
        var lines: [SIMD4<Float>] = []
        var dots: [SIMD4<Float>] = []
        for (ld, bright) in [(1.0, 0.1), (5.0, 0.05), (10.0, 0.045), (20.0, 0.04)] {
            let r = Float(Self.dial(ld))
            for k in 0..<160 {
                let a0 = Float(k) / 160 * 2 * .pi, a1 = Float(k + 1) / 160 * 2 * .pi
                lines += [SIMD4(r * cos(a0), r * sin(a0), r * cos(a1), r * sin(a1)), SIMD4(0.8, 0.84, 0.9, Float(bright) * a)]
            }
        }
        let now = f.now
        let span = 0.32
        for (i, s) in asteroids.enumerated() {
            let c = Self.dialPoint(s, now)
            let n = simd_length(c) > 1e-6 ? simd_normalize(c) : SIMD2(0, 1)
            let along = SIMD2(-n.y, n.x)
            let p0 = c - along * span, p1 = c + along * span
            lines += [SIMD4(Float(p0.x), Float(p0.y), Float(c.x), Float(c.y)), SIMD4(0.85, 0.8, 0.72, 0.045 * a),
                      SIMD4(Float(c.x), Float(c.y), Float(p1.x), Float(p1.y)), SIMD4(0.85, 0.8, 0.72, 0.045 * a)]
            // Each crosses its path again and again, faster flybys faster, and rings a bell at its closest point.
            let travel = (Double(f.time) * s.speed * 0.004 + Double(i) * 0.37).truncatingRemainder(dividingBy: 2 * span) - span
            if i < crossing.count, crossing[i] < 0, travel >= 0, let synth = f.sound, a > 0.5 {
                synth.play(.bell, Float(220 * pow(2, (20 - min(s.lunar, 20)) / 10)), amp: 0.045, pan: Float(c.x), length: 3)
            }
            if i < crossing.count { crossing[i] = travel } else { crossing.append(travel) }
            let q = c + along * travel
            let fade = Float(1 - abs(travel) / span)
            let size = Float(min(max(0.45 + 0.25 * log10(s.size), 0.5), 1.2))
            dots += [SIMD4(Float(q.x), Float(q.y), 0, size), SIMD4(SIMD3<Float>(0.75, 0.68, 0.6) * 0.9, a * fade)]
            dots += [SIMD4(Float(c.x), Float(c.y), 0, 0.4), SIMD4(0.8, 0.78, 0.74, 0.18 * a)]
        }
        f.cmd.draw(flatPipe, into: light, clear: false, uniforms: view, buffers: [rings[5].next(lines)],
                   vertices: 4, instances: lines.count / 2, primitive: .triangleStrip)
        f.cmd.draw(dotPipe, into: light, clear: false, uniforms: view, buffers: [rings[6].next(dots)],
                   vertices: 4, instances: dots.count / 2, primitive: .triangleStrip)

        // Earth and the Moon as lit globes. The dial's plane is the ecliptic, x toward the March equinox.
        let sun = simd_normalize(Astro.sunEcliptic(now))
        let e = Astro.rad(23.4393)
        let pole = SIMD3(0, sin(e), cos(e))
        let theta = Astro.rad(Astro.gmst(now))
        let greenwich = SIMD3(cos(theta), sin(theta) * cos(e), -sin(theta) * sin(e))
        let moonLon = Astro.rad(Astro.ecliptic(Astro.moon(now), now))
        let moonAt = SIMD3(cos(moonLon), sin(moonLon), 0) * Self.dial(1)
        let toCamera = { (v: SIMD3<Double>) in SIMD4<Float>(Float((v * cam.right).sum()), Float((v * cam.up).sum()), Float(-(v * cam.forward).sum()), 0) }
        let globes: [[SIMD4<Float>]] = [
            [SIMD4(0, 0, 0, 0.034), toCamera(greenwich), toCamera(simd_cross(pole, greenwich)), toCamera(pole), toCamera(sun) + [0, 0, 0, earth == nil ? 0 : 1]],
            [SIMD4(Float(moonAt.x), Float(moonAt.y), 0, 0.011), toCamera([1, 0, 0]), toCamera([0, 1, 0]), toCamera([0, 0, 1]), toCamera(sun) + [0, 0, 0, 2]],
        ]
        let visibility = rings[9].next(SIMD4<Float>(a, 0, 0, 0))
        for (k, g) in globes.enumerated() {
            f.cmd.draw(globePipe, into: light, clear: false, uniforms: view, textures: [earth ?? blank],
                       buffers: [rings[7 + k].next(g), visibility], vertices: 4, primitive: .triangleStrip)
        }

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
            let angle = atan2(cam.eye.y, cam.eye.x)
            if let p = pixel(SIMD3(r * cos(angle), r * sin(angle), 0), f, cam),
               let q = labels.quad(text, at: p + SIMD2(0, 3 * density), anchor: [0.5, 0], color: .premultiplied(0.8, 0.84, 0.9, 0.32 * a)) {
                labelsNow.append(q)
                take(q)
            }
        }
        for s in asteroids.sorted(by: { $0.distance < $1.distance }).prefix(12) {
            let c = Self.dialPoint(s, now)
            guard let p = pixel(SIMD3(c.x, c.y, 0), f, cam),
                  let name = labels.quad(s.name, at: p + SIMD2(6 * density, -1 * density), anchor: [0, 1], color: .premultiplied(0.95, 0.92, 0.86, 0.6 * a)),
                  let line = labels.quad(Self.flybyLine(s), at: p + SIMD2(6 * density, 0), anchor: [0, 0], color: .premultiplied(0.82, 0.85, 0.9, 0.42 * a)),
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
        var line = "Every orbit around one star, distances on a log scale · the sky behind: Kepler's field in Cygnus"
        switch mode {
        case .spotlight(let i):
            let s = Self.spotlights[i]
            title = s.title
            var where_ = "in \(s.constellation)"
            if let d = members.compactMap({ planets[$0].distance }).first { where_ += " · \(Self.lightYears(d)) light-years" }
            line = "\(members.count == 1 ? "1 planet" : "\(members.count) planets") · \(where_) · the real sky behind it"
        case .asteroids:
            title = "Asteroids passing Earth"
            line = "The next 30 days · within 0.05 AU, about 19 lunar distances"
        case .field: break
        }
        f.overlay.draw { ctx, _ in
            ctx.text(title, at: CGPoint(x: 30, y: 26), size: 21, weight: .ultraLight, color: NSColor(white: 1, alpha: 0.72), tracking: 1)
            ctx.text(line, at: CGPoint(x: 31, y: 56), size: 11, color: NSColor(white: 1, alpha: 0.4), tracking: 0.3)
        }
    }

    private static func lightYears(_ parsecs: Double) -> String {
        let ly = parsecs * 3.26156
        return ly < 10 ? SpaceKit.number(ly, decimals: 1) : SpaceKit.number((ly / (ly < 100 ? 1 : 10)).rounded() * (ly < 100 ? 1 : 10))
    }

    private func describe() -> String {
        if offline || planets.isEmpty {
            return loaded ? "Offline: the exoplanet catalogue hasn't downloaded yet, so only the sky is showing."
                : "Loading the exoplanet catalogue."
        }
        let count = SpaceKit.number(Double(planets.count))
        switch mode {
        case .spotlight(let i):
            let s = Self.spotlights[i]
            let periods = members.map { planets[$0].period }
            let range = periods.count < 2 ? periods.first.map { "; it orbits in \(Self.days($0))" } ?? ""
                : "; orbits from \(Self.days(periods.min()!)) to \(Self.days(periods.max()!))"
            var text = "\(count) confirmed exoplanets. Spotlight \(s.title), in \(s.constellation): \(members.count == 1 ? "one planet" : "\(members.count) planets"), \(s.blurb)\(range)."
            if let image = s.image, photo > 0.5 { text += " \(image.spoken)." }
            return text
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
            let hot = planets.filter { ($0.temperature ?? 0) > 1500 }.count
            text += "; \(SpaceKit.number(Double(hot))) are hotter than 1,500 kelvin and glow with their own heat"
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
    // a: eye, focal   b: right, host temperature   c: up, field visibility   d: forward, pixel density
    inline float3 viewOf(float3 q, constant U &u) {
        float3 d = q - u.a.xyz;
        float z = max(dot(d, u.d.xyz), 0.01);
        float2 p = float2(dot(d, u.b.xyz), dot(d, u.c.xyz)) * u.a.w / z;
        return float3((p.x / (u.res.x / u.res.y) * 0.5 + 0.5) * u.res.x, (0.5 - p.y * 0.5) * u.res.y, z);
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

    inline float eccentricAnomaly(float M, float e) {
        float E = M + e * sin(M);
        for (int i = 0; i < 4; i++) E -= (E - e * sin(E) - M) / (1 - e * cos(E));
        return E;
    }

    struct TrailOut { float4 pos [[position]]; float4 color; float across; float halfw; };

    // One of 48 pieces of every orbit, lit like a long exposure: faint all the way round, brighter along the arc the
    // planet has just swept. buffer 1: (ring, e, periapsis, tilt), (reflected, size), (glow, 0) per planet;
    // buffer 2: (mean anomaly, weight) per planet, weight 1 for the field, less when dimmed, 2 in the spotlight.
    vertex TrailOut exoTrail(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                             const device float4 *orbits [[buffer(1)]], const device float2 *state [[buffer(2)]]) {
        uint planet = iid / 48, k = iid % 48;
        if (state[planet].y < 0.5) {   // dimmed out of a spotlight: too faint to show, so skip the work
            TrailOut skip;
            skip.pos = float4(0, 0, 2, 1);
            skip.color = 0;
            skip.across = 0;
            skip.halfw = 0;
            return skip;
        }
        float4 o = orbits[3 * planet];
        float3 tint = orbits[3 * planet + 1].rgb + orbits[3 * planet + 2].rgb * 1.4;
        float2 s = state[planet];
        float a = axisOf(o.x);
        float E0 = float(k) / 48.0 * TAU, E1 = float(k + 1) / 48.0 * TAU;
        float2 pa = viewOf(orbitPoint(o, E0, a), u).xy, pb = viewOf(orbitPoint(o, E1, a), u).xy;
        bool end = vid & 1;
        float E = end ? E1 : E0;
        float behind = fract((s.x - (E - o.y * sin(E))) / TAU);
        float trail = exp(-behind * 4.5) * smoothstep(1.0, 0.97, behind);
        float level = s.y > 1.5 ? 1.0 : s.y;
        float3 c = tint * (0.0009 + 0.04 * trail) * level * u.c.w;
        float width = 0.75 * u.d.w;
        float w = max(width, 1.0);
        float2 d = pb - pa;
        float len = length(d);
        float2 t = len > 1e-4 ? d / len : float2(1, 0);
        float2 n = float2(-t.y, t.x);
        float h = w * 0.5 + 1;
        float side = (vid & 2) ? 1.0 : -1.0;
        float2 p = (end ? pb : pa) + n * side * h;
        TrailOut out;
        out.pos = pixelToClip(p, u.res);
        out.color = float4(c * (width / w), 0);
        out.across = side * h;
        out.halfw = w * 0.5;
        return out;
    }

    fragment float4 trailFragment(TrailOut in [[stage_in]]) {
        return in.color * clamp(in.halfw + 0.5 - abs(in.across), 0.0, 1.0);
    }

    struct PointOut { float4 pos [[position]]; float2 offset; float4 color; float sigma; };

    inline PointOut gaussianPoint(float2 px, float sigma, float3 color, uint vid, float2 res) {
        float extent = sigma * 3.5;
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        PointOut o;
        o.pos = pixelToClip(px + corner * extent, res);
        o.offset = corner * extent;
        o.color = float4(color, 0);
        o.sigma = sigma;
        return o;
    }

    // Every planet as a point of light: its star's light reflected, plus its own heat.
    vertex PointOut exoPoint(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                             const device float4 *orbits [[buffer(1)]], const device float2 *state [[buffer(2)]]) {
        float4 orb = orbits[3 * iid], look = orbits[3 * iid + 1];
        float3 glow = orbits[3 * iid + 2].rgb;
        float2 s = state[iid];
        float3 p = viewOf(orbitPoint(orb, eccentricAnomaly(s.x, orb.y), axisOf(orb.x)), u);
        float level = s.y > 1.5 ? 0.0 : s.y;
        float sigma = (0.5 + 0.07 * sqrt(look.w)) * u.d.w;
        float3 c = (look.rgb * (0.5 + 0.06 * look.w) + glow * 1.6) * level * u.c.w * 0.7;
        return gaussianPoint(p.xy, sigma, c, vid, u.res);
    }

    fragment float4 exoPointFragment(PointOut in [[stage_in]]) {
        return in.color * exp(-dot(in.offset, in.offset) / (2 * in.sigma * in.sigma));
    }

    struct DiscOut { float4 pos [[position]]; float2 q; float4 a; float4 b; float4 c; float px; };

    // The star: a photosphere darkening toward its limb and reddening a little there, granulated when it's big enough
    // to show. buffer 1: radius, temperature, field visibility.
    vertex DiscOut exoStar(uint vid [[vertex_id]], constant U &u [[buffer(0)]], constant float4 &k [[buffer(1)]]) {
        float3 p = viewOf(float3(0), u);
        float radius = k.x / p.z * u.a.w * u.res.y * 0.5;
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        DiscOut o;
        o.pos = pixelToClip(p.xy + corner * radius * 3, u.res);
        o.q = corner * 3;
        o.a = k;
        o.b = float4(u.time, 0, 0, 0);
        o.c = 0;
        o.px = 1 / radius;
        return o;
    }

    fragment float4 exoStarFragment(DiscOut in [[stage_in]]) {
        float r = length(in.q);
        float cover = clamp((1 - r) / in.px + 0.5, 0.0, 1.0);
        float teff = in.a.y;
        // Light scattered in the lens around the disc.
        float3 halo = starTint(teff) * (exp(-max(r - 1, 0.0) * 3.0) * 0.06 + exp(-max(r - 1, 0.0) * 0.9) * 0.012) * in.a.z;
        if (cover <= 0) return float4(halo, 0);
        float mu = sqrt(max(1 - r * r, 0.0));
        float3 n = float3(in.q, mu);
        float3 spun = float3(n.x * cos(in.b.x * 0.01) - n.z * sin(in.b.x * 0.01), n.y, n.x * sin(in.b.x * 0.01) + n.z * cos(in.b.x * 0.01));
        float g = granulation(spun, 12.0, in.b.x * 0.05, in.px / max(mu, 0.2));
        // Exposed like a filtered photograph of the Sun, so the limb darkening shows: each star near the same
        // brightness, hotter ones a little brighter, its colour carried by its temperature.
        float bright = 0.62 * pow(teff / 5772.0, 0.6);
        float3 c = starTint(teff * (0.68 + 0.22 * mu)) * pow(limbDarkening(mu, teff), 1.9) * g * bright * in.a.z;
        return float4(c * cover + halo * (1 - cover), cover * in.a.z);
    }

    // Spotlit planets as small globes lit from the star: a crescent on the far side, full on the near side, and the
    // hottest glowing all over. buffer 3 lists the members, buffer 4 holds the star's radius and visibility.
    vertex DiscOut exoSphere(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                             const device float4 *orbits [[buffer(1)]], const device float2 *state [[buffer(2)]],
                             const device uint *list [[buffer(3)]], constant float4 &k [[buffer(4)]]) {
        uint i = list[iid];
        float4 orb = orbits[3 * i], look = orbits[3 * i + 1];
        float2 s = state[i];
        float3 q = orbitPoint(orb, eccentricAnomaly(s.x, orb.y), axisOf(orb.x));
        float3 p = viewOf(q, u);
        float3 star = viewOf(float3(0), u);
        float radius = (3.0 + 1.3 * sqrt(look.w)) * u.d.w;
        // Hidden behind the star's disc.
        float starPx = k.x / star.z * u.a.w * u.res.y * 0.5;
        float hidden = (p.z > star.z && length(p.xy - star.xy) < starPx + radius) ? 0.0 : 1.0;
        float3 toStar = normalize(-q);
        float3 L = float3(dot(toStar, u.b.xyz), dot(toStar, u.c.xyz), -dot(toStar, u.d.xyz));
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        DiscOut o;
        o.pos = pixelToClip(p.xy + corner * radius * 2.5, u.res);
        o.q = corner * 2.5;
        o.a = float4(L, k.y * hidden);
        o.b = look;
        o.c = orbits[3 * i + 2];
        o.px = 1 / radius;
        return o;
    }

    fragment float4 exoSphereFragment(DiscOut in [[stage_in]]) {
        float r = length(in.q);
        float cover = clamp((1 - r) / in.px + 0.5, 0.0, 1.0) * in.a.w;
        // A faint halo, the lens's glow around a small bright thing, so the planet reads at a glance.
        float3 halo = (in.b.rgb * (0.4 + 0.6 * max(in.a.z, 0.0)) + in.c.rgb) * exp(-max(r - 1, 0.0) * 3.0) * 0.08 * in.a.w;
        if (cover <= 0) return float4(halo, 0);
        float3 n = float3(in.q, sqrt(max(1 - r * r, 0.0)));
        float lit = max(dot(n, in.a.xyz), 0.0);
        float3 c = in.b.rgb * lit * 2.6 + in.c.rgb * (0.6 + 0.4 * n.z) * 1.4 + in.b.rgb * 0.02;
        return float4(c * cover + halo * (1 - cover), cover);
    }

    // Flat segments in the plane: (x0, y0, x1, y1), (colour, alpha).
    vertex LineOut flatLine(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                            const device float4 *segs [[buffer(1)]]) {
        float4 s = segs[2 * iid], c = segs[2 * iid + 1];
        float3 a = viewOf(float3(s.xy, 0), u), b = viewOf(float3(s.zw, 0), u);
        return lineQuad(a.xy, b.xy, 0.8 * u.d.w, float4(c.rgb * c.a, 0), vid, u.res);
    }

    // Segments in target pixels, drawn over the finished picture: (x0, y0, x1, y1), (colour, alpha).
    vertex LineOut pixelLine(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                             const device float4 *segs [[buffer(1)]]) {
        float4 s = segs[2 * iid], c = segs[2 * iid + 1];
        return lineQuad(s.xy, s.zw, 0.9 * u.d.w, float4(c.rgb * c.a, c.a), vid, u.res);
    }

    // Points of light in the plane: (x, y, z, size), (colour, brightness).
    vertex PointOut flatDot(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                            const device float4 *dots [[buffer(1)]]) {
        float4 a = dots[2 * iid], c = dots[2 * iid + 1];
        float3 p = viewOf(a.xyz, u);
        return gaussianPoint(p.xy, a.w * 0.9 * u.d.w, c.rgb * c.a * 1.6, vid, u.res);
    }

    // Earth (texture bound) or the Moon as a lit globe. buffer 1: centre and radius, body x, y, z axes and the Sun's
    // direction in camera space (w: 1 textured Earth, 0 Earth without its map, 2 the Moon); buffer 2: visibility.
    vertex DiscOut exoGlobe(uint vid [[vertex_id]], constant U &u [[buffer(0)]], const device float4 *g [[buffer(1)]],
                            constant float4 &k [[buffer(2)]]) {
        float3 p = viewOf(g[0].xyz, u);
        float radius = max(g[0].w / p.z * u.a.w * u.res.y * 0.5, 1.5);
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        DiscOut o;
        o.pos = pixelToClip(p.xy + corner * (radius * 1.12 + 1.5), u.res);
        o.q = corner * (radius * 1.12 + 1.5) / radius;
        o.a = g[4];
        o.b = float4(0, 0, 0, k.x);
        o.c = 0;
        o.px = 1 / radius;
        return o;
    }

    fragment float4 exoGlobeFragment(DiscOut in [[stage_in]], const device float4 *g [[buffer(1)]], texture2d<float> map [[texture(0)]]) {
        constexpr sampler s(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge);
        float r = length(in.q);
        float vis = in.b.w;
        float3 L = normalize(in.a.xyz);
        float kind = in.a.w;
        if (r > 1) {
            // A thin blue rim of air on Earth's day side.
            if (kind > 1.5) return 0;
            float rim = exp(-(r - 1) / 0.035) * smoothstep(1.12, 1.0, r);
            float day = smoothstep(-0.2, 0.4, dot(normalize(float3(in.q, 0)), L));
            float3 air = float3(0.25, 0.45, 1.0) * rim * day * 0.35 * vis;
            return float4(air, 0);
        }
        float cover = clamp((1 - r) / in.px + 0.5, 0.0, 1.0) * vis;
        float3 n = float3(in.q, sqrt(max(1 - r * r, 0.0)));
        float mu0 = dot(n, L);
        float3 c;
        if (kind > 1.5) {
            // The Moon: grey regolith, its terminator sharp, a little brighter toward full (opposition surge).
            c = float3(0.46, 0.44, 0.42) * max(mu0, 0.0) / (max(mu0, 0.0) + n.z + 0.05) * 1.8;
        } else {
            float3 X = g[1].xyz, Y = g[2].xyz, Z = g[3].xyz;
            float3 b = float3(dot(n, X), dot(n, Y), dot(n, Z));
            float lon = atan2(b.y, b.x), lat = asin(clamp(b.z, -1.0, 1.0));
            float2 uv = float2(0.5 + lon / TAU, 0.5 - lat / PI);
            float lod = log2(max(float(map.get_width()) / TAU * in.px / max(n.z, 0.2), 1e-3));
            float3 ground = kind > 0.5 ? map.sample(s, uv, level(lod)).rgb : float3(0.12, 0.2, 0.35);
            float day = smoothstep(-0.05, 0.12, mu0);
            c = ground * max(mu0, 0.0) * 1.6 + float3(0.25, 0.4, 0.8) * 0.06 * day * (1 - n.z);
        }
        return float4(c * cover, cover);
    }

    // A direct image of planets, fitted to the screen and laid over the scene as it fades in. a: origin in pixels,
    // scale, visibility; b: file size. Baked-in labels (buffer 1, count in buffer 2) are blanked to the dark sky.
    fragment float4 exoPhoto(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> photo [[texture(0)]],
                             const device float4 *masks [[buffer(1)]], constant uint &count [[buffer(2)]]) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
        float2 px = (in.uv * u.res - u.a.xy) / u.a.z;
        float2 uv = px / u.b.xy;
        float3 c = photo.sample(s, uv).rgb;
        for (uint i = 0; i < count; i++) {
            float4 m = masks[i];
            float2 inside = smoothstep(m.xy - 2, m.xy + 1, px) * smoothstep(m.zw + 2, m.zw - 1, px);
            c = mix(c, float3(0.0012), inside.x * inside.y);
        }
        float2 edge = min(uv, 1 - uv);
        float frame = smoothstep(0.0, 0.06, min(edge.x, edge.y));
        float a = u.a.w;
        return float4(c * 1.15 * frame * a, a);
    }

    fragment float4 exoAdd(V in [[stage_in]], texture2d<float> t [[texture(0)]]) {
        constexpr sampler s(filter::nearest, address::clamp_to_edge);
        return float4(t.sample(s, in.uv).rgb, 0);
    }
    """
}
