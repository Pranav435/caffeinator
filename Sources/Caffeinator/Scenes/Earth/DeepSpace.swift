import AppKit
import Metal
import simd

/// NASA's Deep Space Network as it is right now: the three complexes on a small Earth, a beam from each working dish
/// along its real pointing direction to the spacecraft it's talking to, placed at the logarithm of its distance, with
/// photons crawling along each link (inward on downlinks, outward on uplinks).
@MainActor final class DeepSpace: Scene {
    struct Signal: Sendable { var band: String; var rate: Double; var power: Double; var craft: String }
    struct Dish: Sendable {
        var name: String, site: String, azimuth: Double, elevation: Double, activity: String
        var down: [Signal] = [], up: [Signal] = []
        var targets: [(name: String, range: Double?, rtlt: Double?)] = []
    }
    struct Site: Sendable { var code: String, name: String, lat: Double, lon: Double }
    struct Config: Sendable {
        var sites: [String: Site] = [:]
        var dishTypes: [String: String] = [:]
        var names: [String: String] = [:]
    }

    /// One dish talking to one spacecraft.
    private struct Link {
        var site: Site, dish: String, type: String, craft: String
        var down: Signal?, up: Signal?
        var direction: SIMD3<Double>   // unit vector, inertial (TEME-like, z north, x toward the equinox)
    }
    /// A spacecraft, with every dish that's talking to it.
    private struct Craft {
        var code: String, name: String, km: Double, measured: Bool, rtlt: Double?
        var direction: SIMD3<Double>
        var links: [Int]
        var where_: SIMD3<Double> { direction * DeepSpace.layout(km) }
    }
    /// Matches `Beam` in the shader: station; spacecraft and band; activity (down, up), rate, phase.
    private struct Beam { var a: SIMD4<Float>; var b: SIMD4<Float>; var c: SIMD4<Float> }

    private static let dataURL = URL(string: "https://eyes.nasa.gov/dsn/data/dsn.xml")!
    private static let configURL = URL(string: "https://eyes.nasa.gov/dsn/config.xml")!
    private static let lightSpeed = 299_792.458
    private static let au = 149_597_870.7
    /// Linear colours by band; the shader draws the same colours in display space.
    static let bandColors: [String: SIMD3<Float>] = ["S": [1.0, 0.57, 0.26], "X": [0.32, 0.71, 1.0], "K": [0.67, 0.45, 1.0]]
    private static let defaultSites = [
        "gdscc": Site(code: "gdscc", name: "Goldstone", lat: 35.2443523, lon: -116.8895382),
        "mdscc": Site(code: "mdscc", name: "Madrid", lat: 40.2413554, lon: -4.2480085),
        "cdscc": Site(code: "cdscc", name: "Canberra", lat: -35.2209189, lon: 148.9812673),
    ]
    /// Where spacecraft are when the feed gives no range.
    private static let homes: [String: String] = {
        var h: [String: String] = [:]
        for c in ["LRO", "KPLO", "CAPS", "CH2O", "CH2", "CH3", "SLIM", "EM2", "ART2", "LTB", "THB", "THC", "LNSA", "ESCB", "ESCG"] { h[c] = "moon" }
        for c in ["ACE", "SOHO", "WIND", "DSCO", "IMAP", "SWFO", "CGO"] { h[c] = "l1" }
        for c in ["JWST", "GAIA", "RST", "EUCL", "PLAT"] { h[c] = "l2" }
        for c in ["CHDR", "XMM", "TESS", "MMS1", "MMS2", "MMS3", "MMS4", "GTL", "CLU1", "CLU2", "CLU3", "CLU4", "INTG", "HST"] { h[c] = "high" }
        for c in ["MRO", "MROS", "M01O", "M01S", "MVN", "MEX", "TGO", "EMM", "M20", "MSL", "NSYT"] { h[c] = "mars" }
        for c in ["JNO"] { h[c] = "jupiter" }
        return h
    }()

    private let globe: Globe
    private let canvas: Globe.Canvas
    private let beams: MTLRenderPipelineState
    private let photons: MTLRenderPipelineState
    private let dots: MTLRenderPipelineState
    private let rings: MTLRenderPipelineState
    private var config = Config()
    private var links: [Link] = []
    private var crafts: [Craft] = []
    private var dishesInUse = 0
    private var loaded = false
    private var fetching = false
    private var fetched = Date.distantPast
    private var camera: Globe.Camera?
    private var lastOverlay: Float = -1
    private var featured = 0
    private var featuredSince: Float = 0
    private var ringBuffer: MTLBuffer?
    private var heard = Set<String>()
    private var chimes: [(at: Float, freq: Float)] = []
    private var voices = [Float](repeating: 220, count: Synth.partials)

    var scale: Float { 1 }
    var ready: Bool { loaded && globe.ready }

    var facts: String? {
        guard loaded else { return "Loading the Deep Space Network's live status from NASA." }
        guard !crafts.isEmpty else { return "No Deep Space Network data right now; the feed couldn't be reached." }
        var s = "\(dishesInUse) dishes are working with \(crafts.count) spacecraft."
        let ordered = crafts.indices.sorted { crafts[$0].km > crafts[$1].km }
        let c = crafts[ordered[featured % ordered.count]]
        if let l = c.links.first.map({ links[$0] }) {
            let size = l.type.hasPrefix("70") ? "70-metre" : "34-metre"
            let verb = l.down != nil ? "listening to" : l.up != nil ? "sending commands to" : "pointed at"
            s += " \(l.site.name)'s \(size) dish is \(verb) \(c.name)"
            s += c.measured ? ", \(EarthText.kilometres(c.km)) away" : ", \(Self.whereabouts(c.code))"
            if c.measured || Self.homes[c.code] != nil { s += "; round-trip light time \(EarthText.duration(c.km / Self.lightSpeed * 2))" }
            if let d = l.down, d.rate > 0 { s += ", data at \(Self.rate(d.rate))" }
            s += "."
        }
        // The farthest one always gets a word about how old its signal is.
        if let far = ordered.first.map({ crafts[$0] }), far.measured, far.km / Self.lightSpeed > 6 * 3600 {
            s += " The signal arriving now from \(far.name) left it \(EarthText.duration(far.km / Self.lightSpeed)) ago."
        }
        return s
    }

    init(gpu: GPU) throws {
        globe = try Globe.shared(gpu)
        canvas = try Globe.Canvas(gpu: gpu)
        let library = try gpu.library(Globe.metal + Self.source)
        beams = try gpu.pipeline(library, "ribbonFill", vertex: "beam", blend: .alpha)
        photons = try gpu.pipeline(library, "photonFill", vertex: "photon", blend: .alpha)
        dots = try gpu.pipeline(library, "dotFill", vertex: "dot", blend: .alpha)
        rings = try gpu.pipeline(library, "ringFill", vertex: "scaleRing", blend: .alpha)
        config.sites = Self.defaultSites
        featured = Snapshot.variant ?? Int.random(in: 0..<20)
        fetch()
    }

    private func fetch() {
        guard !fetching else { return }
        fetching = true
        fetched = .now
        Task { @MainActor in
            async let cfg = Feed.data(Self.configURL, maxAge: 7 * 86400)
            async let dat = Feed.data(Self.dataURL, maxAge: 60)
            let (c, d) = await (cfg, dat)
            let parsed = await Task.detached { (c.map(Self.parseConfig), d.map(Self.parseData)) }.value
            fetching = false
            if let cfg = parsed.0, !cfg.sites.isEmpty { config = cfg }
            if let dishes = parsed.1 { build(dishes, at: .now) }
            loaded = true
        }
    }

    // MARK: Parsing

    private final class Reader: NSObject, XMLParserDelegate {
        var config = Config()
        var dishes: [Dish] = []
        var site = ""

        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?, qualifiedName: String?,
                    attributes a: [String: String] = [:]) {
            func num(_ k: String) -> Double? { a[k].flatMap(Double.init).flatMap { $0 < 0 ? nil : $0 } }
            switch name {
            case "site":
                if let code = a["name"], let lat = a["latitude"].flatMap(Double.init), let lon = a["longitude"].flatMap(Double.init) {
                    site = code
                    config.sites[code] = Site(code: code, name: a["friendlyName"] ?? code, lat: lat, lon: lon)
                }
            case "spacecraft":
                if let code = a["name"], let friendly = a["friendlyName"] { config.names[code.uppercased()] = friendly }
            case "station":
                site = a["name"] ?? site
            case "dish":
                if let dish = a["name"], let type = a["type"] { config.dishTypes[dish] = type }
                if let dish = a["name"], let az = a["azimuthAngle"].flatMap(Double.init), let el = a["elevationAngle"].flatMap(Double.init) {
                    dishes.append(Dish(name: dish, site: site, azimuth: az, elevation: el, activity: a["activity"] ?? ""))
                }
            case "downSignal", "upSignal":
                guard !dishes.isEmpty, a["active"] == "true", let craft = a["spacecraft"]?.uppercased() else { return }
                let s = Signal(band: a["band"] ?? "X", rate: num("dataRate") ?? 0, power: a["power"].flatMap(Double.init) ?? -150, craft: craft)
                if name == "downSignal" { dishes[dishes.count - 1].down.append(s) } else { dishes[dishes.count - 1].up.append(s) }
            case "target":
                guard !dishes.isEmpty, let craft = a["name"]?.uppercased() else { return }
                let range = [num("downlegRange"), num("uplegRange")].compactMap { $0 }.max()
                dishes[dishes.count - 1].targets.append((craft, range, num("rtlt")))
            default: break
            }
        }
    }

    nonisolated static func parseConfig(_ data: Data) -> Config {
        let r = Reader()
        let p = XMLParser(data: data)
        p.delegate = r
        p.parse()
        return r.config
    }

    nonisolated static func parseData(_ data: Data) -> [Dish] {
        let r = Reader()
        let p = XMLParser(data: data)
        p.delegate = r
        p.parse()
        return r.dishes
    }

    /// Turns dishes into links and spacecraft. Directions come from each dish's azimuth and elevation.
    private func build(_ dishes: [Dish], at date: Date) {
        var newLinks: [Link] = []
        var ranges: [String: (km: Double?, rtlt: Double?)] = [:]
        let gmst = Astro.rad(Astro.gmst(date))
        for d in dishes {
            guard let site = config.sites[d.site], d.elevation < 89.5 else { continue }
            for t in d.targets where !["DSN", "TEST", "TESTING"].contains(t.name) && !t.name.hasPrefix("DSS") {
                let down = d.down.filter { $0.craft == t.name }.max { $0.power < $1.power }
                let up = d.up.first { $0.craft == t.name }
                let lat = Astro.rad(site.lat), lon = Astro.rad(site.lon), az = Astro.rad(d.azimuth), el = Astro.rad(d.elevation)
                let east = SIMD3(-sin(lon), cos(lon), 0.0)
                let north = SIMD3(-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))
                let upward = SIMD3(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
                let fixed = cos(el) * sin(az) * east + cos(el) * cos(az) * north + sin(el) * upward
                let inertial = SIMD3(cos(gmst) * fixed.x - sin(gmst) * fixed.y, sin(gmst) * fixed.x + cos(gmst) * fixed.y, fixed.z)
                newLinks.append(Link(site: site, dish: d.name, type: config.dishTypes[d.name] ?? "34M", craft: t.name, down: down, up: up,
                                     direction: inertial))
                let old = ranges[t.name]
                ranges[t.name] = ((t.range ?? old?.km).map { max($0, old?.km ?? 0) }, t.rtlt ?? old?.rtlt)
            }
        }
        let fresh = Set(newLinks.compactMap { $0.down != nil ? $0.craft : nil })
        // A soft pluck for each spacecraft whose downlink just came on.
        for _ in fresh.subtracting(heard) where loaded && chimes.count < 4 { chimes.append((Float(chimes.count) * 0.4, 1567.98)) }
        heard = fresh
        links = newLinks
        dishesInUse = Set(newLinks.map(\.dish)).count
        crafts = Dictionary(grouping: newLinks.indices, by: { newLinks[$0].craft }).map { code, idx in
            let dir = simd_normalize(idx.map { newLinks[$0].direction }.reduce(.zero, +))
            let measured = ranges[code]?.km
            let km = measured ?? Self.fallback(code, date)
            return Craft(code: code, name: config.names[code] ?? code, km: km, measured: measured != nil, rtlt: ranges[code]?.rtlt,
                         direction: dir, links: idx)
        }.sorted { $0.km < $1.km }
    }

    private static func fallback(_ code: String, _ date: Date) -> Double {
        switch homes[code] {
        case "moon": Astro.moon(date).distance * 6378.14
        case "l1", "l2": 1.5e6
        case "high": 1.2e5
        case "mars": Astro.planet(.mars, date).distance * au
        case "jupiter": Astro.planet(.jupiter, date).distance * au
        default: au
        }
    }

    private static func whereabouts(_ code: String) -> String {
        switch homes[code] {
        case "moon": "at the Moon"
        case "l1": "about 1.5 million kilometres out at the L1 point, toward the Sun"
        case "l2": "about 1.5 million kilometres out at the L2 point, away from the Sun"
        case "high": "in high Earth orbit"
        case "mars": "at Mars"
        case "jupiter": "at Jupiter"
        default: "range not reported"
        }
    }

    private static func shortWhereabouts(_ code: String) -> String {
        switch homes[code] {
        case "moon": "at the Moon"
        case "l1": "Sun–Earth L1"
        case "l2": "Sun–Earth L2"
        case "high": "high Earth orbit"
        case "mars": "at Mars"
        case "jupiter": "at Jupiter"
        default: "range not reported"
        }
    }

    /// Distance from Earth's centre in the picture, Earth radii: a logarithmic scale from near-Earth to interstellar.
    nonisolated static func layout(_ km: Double) -> Double { 1.5 + 0.55 * log10(max(km, 20000) / 20000) }

    private static func rate(_ bps: Double) -> String {
        switch bps {
        case 1e6...: "\(EarthText.number(bps / 1e6)) megabits a second"
        case 1e3...: "\(EarthText.number(bps / 1e3)) kilobits a second"
        default: "\(EarthText.number(bps, digits: 0)) bits a second"
        }
    }

    private static func shortRate(_ bps: Double) -> String {
        switch bps {
        case 1e6...: "\(EarthText.number(bps / 1e6)) Mb/s"
        case 1e3...: "\(EarthText.number(bps / 1e3)) kb/s"
        default: "\(EarthText.number(bps, digits: 0)) b/s"
        }
    }

    private static func shortDistance(_ km: Double) -> String {
        switch km {
        case 1e9...: "\(EarthText.number(km / 1e9)) billion km"
        case 1e6...: "\(EarthText.number(km / 1e6)) million km"
        default: "\(EarthText.count(Int((km / 1000).rounded()) * 1000)) km"
        }
    }

    // MARK: Drawing

    /// Inertial to Earth-fixed at this moment.
    private static func fixed(_ v: SIMD3<Double>, _ gmst: Double) -> SIMD3<Float> {
        SIMD3(Float(cos(gmst) * v.x + sin(gmst) * v.y), Float(-sin(gmst) * v.x + cos(gmst) * v.y), Float(v.z))
    }

    func draw(_ f: Frame) {
        if f.now.timeIntervalSince(fetched) > 60 { fetch() }
        featuredSince += f.dt
        if featuredSince > 20 { featuredSince = 0; featured += 1 }
        let gmst = Astro.rad(Astro.gmst(f.now))
        let sun = Globe.sun(f.now)
        // Look down on the plane of the solar system from 30 degrees above it, turning slowly around, with the
        // ecliptic kept level so the planets' plane lies across the screen.
        let e = Astro.rad(23.44), lam = Double(f.seed) * 2 + Double(f.time) * 0.004, beta = Astro.rad(30 + 6 * sin(Double(f.time) * 0.005))
        let toCam = SIMD3(cos(beta) * cos(lam), cos(beta) * sin(lam) * cos(e) - sin(beta) * sin(e), cos(beta) * sin(lam) * sin(e) + sin(beta) * cos(e))
        let camFixed = Self.fixed(toCam, gmst)
        let spot = Globe.latLon(camFixed)
        var cam = Globe.Camera(lat: Float(spot.lat), lon: Float(spot.lon), distance: 20, zoom: 4.4, aspect: f.aspect, date: f.now)
        let pole = Self.fixed(SIMD3(0, -sin(e), cos(e)), gmst)
        let roll = atan2(-simd_dot(pole, cam.basis.columns.0), simd_dot(pole, cam.basis.columns.1))
        cam = Globe.Camera(lat: Float(spot.lat), lon: Float(spot.lon), distance: 20, zoom: 4.4, roll: roll, aspect: f.aspect, date: f.now)
        camera = cam
        let light = canvas.light(for: f)
        let view = canvas.camera(cam)
        var style = Globe.Style()
        style.stars = 0.3
        globe.refresh(f.now)
        globe.draw(f.cmd, into: light, camera: view, style: style, date: f.now, time: f.time)
        canvas.develop(f, Self.look)

        // The rest is the instrument display, over the picture. Reference rings in the ecliptic plane: the Moon's
        // distance, 1, 10 and 100 AU.
        let px = f.size.y / 1000
        if ringBuffer == nil {
            let e = Astro.rad(23.44)
            let ringList = [384_400.0, Self.au, 10 * Self.au, 100 * Self.au].map { km -> SIMD4<Float> in
                SIMD4(0, Float(-sin(e)), Float(cos(e)), Float(Self.layout(km)))
            }
            ringBuffer = f.gpu.buffer(ringList)
        }
        f.cmd.draw(rings, into: f.target, clear: false, uniforms: f.uniforms([px, Float(gmst), 0, 0]), buffers: [view, ringBuffer!],
                   vertices: 128 * 6, instances: 4)

        // Beams and photons, one beam per dish and spacecraft.
        var beamList: [Beam] = []
        for (i, c) in crafts.enumerated() {
            let p = Self.fixed(c.where_, gmst)
            for li in c.links {
                let l = links[li]
                let s = Globe.geo(l.site.lat, l.site.lon, 1.002)
                let band = l.down?.band ?? l.up?.band ?? "X"
                let code: Float = band.hasPrefix("S") ? 0 : band.hasPrefix("K") ? 2 : 1
                let rate = Float(log10(max(l.down?.rate ?? l.up?.rate ?? 1, 1)))
                beamList.append(Beam(a: SIMD4(s, code), b: SIMD4(p, Float(i)),
                                     c: [l.down != nil ? 1 : 0, l.up != nil ? 1 : 0, rate, Float(li) * 0.37]))
            }
        }
        // Spacecraft and reference bodies as points: xyz, kind (0 spacecraft, 1 planet, 2 the Sun).
        var dotList: [SIMD4<Float>] = crafts.map { SIMD4(Self.fixed($0.where_, gmst), 0) }
        for (q, kind) in referenceBodies(f.now) { dotList.append(SIMD4(Self.fixed(q, gmst), kind)) }
        for site in config.sites.values { dotList.append(SIMD4(Globe.geo(site.lat, site.lon, 1.002), 3)) }
        if !beamList.isEmpty {
            let b = canvas.stream(0, beamList)
            f.cmd.draw(beams, into: f.target, clear: false, uniforms: f.uniforms([px, f.time, 0, 0]), buffers: [view, b],
                       vertices: 32 * 6, instances: beamList.count)
            f.cmd.draw(photons, into: f.target, clear: false, uniforms: f.uniforms([px, f.time, 0, 0]), buffers: [view, b],
                       vertices: 64, instances: beamList.count, primitive: .point)
        }
        f.cmd.draw(dots, into: f.target, clear: false, uniforms: f.uniforms([px, f.time, 0, 0], SIMD4(sun, 0)), buffers: [view, canvas.stream(1, dotList)],
                   vertices: dotList.count, primitive: .point)

        if let synth = f.sound { play(synth, f) }
        if f.time - lastOverlay >= 0.1 || f.index == 0 {
            lastOverlay = f.time
            drawOverlay(f, gmst)
        }
    }

    /// The Moon, Sun, Mars, Jupiter and Saturn at their real directions and scaled distances, inertial.
    private func referenceBodies(_ date: Date) -> [(SIMD3<Double>, Float)] {
        func place(_ q: Astro.Equatorial, _ km: Double) -> SIMD3<Double> {
            let ra = Astro.rad(q.ra), dec = Astro.rad(q.dec)
            return SIMD3(cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)) * Self.layout(km)
        }
        let moon = Astro.moon(date), sun = Astro.sun(date)
        var out = [(place(moon, moon.distance * 6378.14), Float(1)), (place(sun, sun.distance * Self.au), 2)]
        for body in [Astro.Body.mars, .jupiter, .saturn] {
            let q = Astro.planet(body, date)
            out.append((place(q, q.distance * Self.au), 1))
        }
        return out
    }

    /// A quiet carrier for each downlink: S band low, X in the middle, Ka high, each a note of one chord, louder for
    /// a stronger signal. Voyager's carrier is faint and low with a slow tremolo.
    private func play(_ synth: Synth, _ f: Frame) {
        let chords: [String: [Float]] = ["S": [98, 146.8, 123.5], "X": [196, 246.9, 293.7, 392, 329.6], "K": [493.9, 587.3, 659.3]]
        var used: [String: Int] = [:]
        var tones: [(freq: Float, amp: Float, pan: Float)] = []
        let downs = links.compactMap { l in l.down.map { (l, $0) } }.sorted { $0.1.power > $1.1.power }
        for (l, d) in downs.prefix(Synth.partials) {
            let pan: Float = l.site.code.hasPrefix("g") ? -0.6 : l.site.code.hasPrefix("c") ? 0.6 : 0
            if l.craft.hasPrefix("VGR") {
                let two = l.craft == "VGR2"
                let tremolo = 0.6 + 0.4 * sin(f.time * 2 * .pi * 0.2 + (two ? 1.5 : 0))
                tones.append((two ? 55 : 73.4, 0.03 * tremolo, pan))
            } else {
                let key = d.band.hasPrefix("S") ? "S" : d.band.hasPrefix("K") ? "K" : "X"
                let notes = chords[key]!, n = used[key, default: 0]
                used[key] = n + 1
                tones.append((notes[n % notes.count] * (n >= notes.count ? 2 : 1), 0.01 + 0.035 * Float(min(max((d.power + 165) / 60, 0), 1)), pan))
            }
        }
        // Keep the whole chord gentle however many links there are.
        let total = tones.reduce(0) { $0 + $1.amp }
        let k: Float = total > 0.22 ? 0.22 / total : 1
        for i in 0..<Synth.partials {
            if i < tones.count {
                voices[i] = tones[i].freq
                synth.drone(i, tones[i].freq, tones[i].amp * k, pan: tones[i].pan)
            } else {
                synth.drone(i, voices[i], 0)
            }
        }
        synth.noise(0.012, cutoff: 900)
        var keep: [(at: Float, freq: Float)] = []
        for c in chimes {
            if c.at <= 0 { synth.play(.pluck, c.freq, amp: 0.04, pan: 0, length: 1.2) } else { keep.append((c.at - f.dt, c.freq)) }
        }
        chimes = keep
    }

    // MARK: Overlay

    private func drawOverlay(_ f: Frame, _ gmst: Double) {
        guard let cam = camera else { return }
        let drift = CGPoint(x: CGFloat(sin(f.time * 0.007)) * 6, y: CGFloat(cos(f.time * 0.009)) * 4)
        let bodies = referenceBodies(f.now)
        f.overlay.draw { ctx, size in
            let dim = NSColor(white: 1, alpha: 0.42), faint = NSColor(white: 1, alpha: 0.24)
            let left = 44 + drift.x, top = 38 + drift.y
            ctx.text("DEEP SPACE NETWORK", at: CGPoint(x: left, y: top), size: 11, weight: .semibold, color: dim, tracking: 2)
            ctx.text("\(self.crafts.count) spacecraft", at: CGPoint(x: left, y: top + 18), size: 22, weight: .light, color: NSColor(white: 1, alpha: 0.6))
            ctx.text("\(self.dishesInUse) dishes · \(EarthText.utc(f.now))", at: CGPoint(x: left, y: top + 48), size: 11, color: faint)
            var lx = left
            for (band, label) in [("S", "S BAND"), ("X", "X BAND"), ("K", "KA BAND")] {
                ctx.setFillColor(NSColor.earthLinear(Self.bandColors[band]!).cgColor)
                ctx.fillEllipse(in: CGRect(x: lx, y: top + 71, width: 7, height: 7))
                ctx.text(label, at: CGPoint(x: lx + 11, y: top + 68), size: 9.5, weight: .semibold, color: faint, tracking: 1)
                lx += 72
            }

            // Sites.
            for site in self.config.sites.values {
                let q = Globe.geo(site.lat, site.lon, 1.002)
                guard let p = cam.overlay(q, size), cam.facing(q) > 0.4 else { continue }
                ctx.text(site.name.uppercased(), at: CGPoint(x: p.point.x + 8, y: p.point.y + 4), size: 9.5, weight: .semibold,
                         color: NSColor(white: 1, alpha: 0.55), tracking: 1.2)
            }

            // Reference bodies and rings.
            let names = ["Moon", "Sun", "Mars", "Jupiter", "Saturn"]
            for ((q, _), name) in zip(bodies, names) {
                guard let p = cam.overlay(Self.fixed(q, gmst), size) else { continue }
                ctx.text(name, at: CGPoint(x: p.point.x, y: p.point.y + 7), size: 9.5, color: NSColor(white: 1, alpha: 0.28),
                         anchor: CGPoint(x: 0.5, y: 0))
            }
            let e = Astro.rad(23.44)
            let axisA = SIMD3(1.0, 0, 0), axisB = SIMD3(0, cos(e), sin(e))
            for (km, label) in [(Self.au, "1 AU"), (10 * Self.au, "10 AU"), (100 * Self.au, "100 AU")] {
                // Label each ring where it's lowest on screen.
                var best: CGPoint?
                for k in 0..<48 {
                    let a = Double(k) / 48 * 2 * .pi
                    let q = (axisA * cos(a) + axisB * sin(a)) * Self.layout(km)
                    if let p = cam.overlay(Self.fixed(q, gmst), size), best == nil || p.point.y > best!.y { best = p.point }
                }
                if let b = best { ctx.text(label, at: CGPoint(x: b.x, y: b.y + 4), size: 9, color: faint, anchor: CGPoint(x: 0.5, y: 0)) }
            }

            // Spacecraft, grouped when they sit together (the Mars orbiters, say).
            var groups: [(point: CGPoint, crafts: [Craft])] = []
            for c in self.crafts {
                guard let p = cam.overlay(Self.fixed(c.where_, gmst), size), p.shown > 0.3 else { continue }
                if let g = groups.firstIndex(where: { hypot($0.point.x - p.point.x, $0.point.y - p.point.y) < 26 }) {
                    groups[g].crafts.append(c)
                } else {
                    groups.append((p.point, [c]))
                }
            }
            // Labels sit on the side away from the Earth, nudged down until they don't collide.
            let centre = cam.overlay(.zero, size)?.point ?? CGPoint(x: size.width / 2, y: size.height / 2)
            var placed: [CGRect] = self.config.sites.values.compactMap { site in
                let q = Globe.geo(site.lat, site.lon, 1.002)
                guard let p = cam.overlay(q, size), cam.facing(q) > 0.4 else { return nil }
                return CGRect(x: p.point.x + 6, y: p.point.y, width: CGFloat(site.name.count) * 8 + 10, height: 16)
            }
            let earth = (cam.overlay(Globe.geo(0, 0, 1), size).map { hypot($0.point.x - centre.x, $0.point.y - centre.y) } ?? 60) + 8
            placed.append(CGRect(x: centre.x - earth, y: centre.y - earth, width: earth * 2, height: earth * 2))
            placed.append(CGRect(x: 0, y: 0, width: left + 300, height: top + 90))
            for g in groups.sorted(by: { abs($0.point.y - centre.y) < abs($1.point.y - centre.y) }) {
                let first = g.crafts.max { $0.km < $1.km }!
                let names = g.crafts.map(\.name).joined(separator: " · ")
                let title = names.count > 46 ? g.crafts.map(\.code).joined(separator: " · ") : names
                var detail = first.measured ? Self.shortDistance(first.km) : Self.shortWhereabouts(first.code)
                let rates = g.crafts.flatMap { $0.links.compactMap { self.links[$0].down?.rate } }.filter { $0 > 0 }
                if g.crafts.count == 1, let r = rates.max() { detail += " · \(Self.shortRate(r))" }
                let oneWay = first.km / Self.lightSpeed
                if oneWay > 3600, first.measured { detail += " · signal left \(Self.shortDuration(oneWay)) ago" }
                let width = CGFloat(max(title.count, detail.count)) * 6.1
                let right = g.point.x >= centre.x ? g.point.x + 12 + width < size.width - 20 : g.point.x - 12 - width < 20
                let x = right ? g.point.x + 12 : g.point.x - 12
                var y = g.point.y - 10
                var rect = CGRect(x: right ? x : x - width, y: y, width: width, height: 28)
                var tries = 0
                while tries < 12, placed.contains(where: { $0.intersects(rect) }) {
                    tries += 1
                    y = g.point.y - 10 + CGFloat((tries + 1) / 2) * 16 * (tries % 2 == 0 ? -1 : 1)
                    rect.origin.y = y
                }
                placed.append(rect)
                if abs(y - (g.point.y - 10)) > 4 {
                    ctx.earthLine(CGPoint(x: g.point.x + (right ? 5 : -5), y: g.point.y), CGPoint(x: x + (right ? -2 : 2), y: y + 8),
                                  color: NSColor(white: 1, alpha: 0.2))
                }
                let anchor = CGPoint(x: right ? 0 : 1, y: 0)
                ctx.text(title, at: CGPoint(x: x, y: y), size: 11, weight: .medium, color: NSColor(white: 1, alpha: 0.82), anchor: anchor)
                ctx.text(detail, at: CGPoint(x: x, y: y + 14), size: 9.5, color: NSColor(white: 1, alpha: 0.45), anchor: anchor)
            }
        }
    }

    private static func shortDuration(_ s: Double) -> String {
        s >= 7200 ? "\(EarthText.number(s / 3600)) h" : "\(Int((s / 60).rounded())) min"
    }

    /// The Earth is small here, so a slightly longer exposure lets the Milky Way show.
    private static let look = Lens.Look(exposure: 1.4, bloom: 0.03, halation: 0.012, vignette: 0.3, aberration: 0.8, grain: 0.4)

    private static let source = """
    struct Beam { float4 a; float4 b; float4 c; };

    // Display colours by band: S warm, X cool, Ka violet.
    constant float3 bandColors[3] = { float3(1.0, 0.78, 0.55), float3(0.6, 0.86, 1.0), float3(0.84, 0.7, 1.0) };

    // Each beam is a hairline in 32 short pieces so the Earth can hide the part behind it. u.a: pixel scale, time
    vertex Ribbon beam(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                       const device Beam *beams [[buffer(2)]]) {
        Beam b = beams[iid];
        uint seg = vid / 6, corner = vid % 6;
        bool atB = corner == 2 || corner == 3 || corner == 5;
        float s0 = float(seg) / 32, s1 = float(seg + 1) / 32;
        float3 p0 = mix(b.a.xyz, b.b.xyz, s0), p1 = mix(b.a.xyz, b.b.xyz, s1);
        float3 q = atB ? p1 : p0;
        float s = atB ? s1 : s0;
        float active = max(b.c.x, b.c.y);
        float alpha = (0.18 + 0.32 * active) * (0.45 + 0.55 * pow(1 - s, 0.7)) * globeShown(q, c);
        return ribbon(corner, globeClip(p0, c), globeClip(p1, c), 1.0 * u.a.x, float4(bandColors[int(b.a.w)], 1) * alpha, u.res);
    }

    struct Photon { float4 pos [[position]]; float size [[point_size]]; float4 color; };

    // Photons crawl along each beam at a visual light speed: inward on downlinks, outward on uplinks, more of them for
    // a faster data rate.
    vertex Photon photon(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                         const device Beam *beams [[buffer(2)]]) {
        Beam b = beams[iid];
        Photon o;
        bool down = vid % 2 == 0;
        float on = down ? b.c.x : b.c.y;
        float len = length(b.b.xyz - b.a.xyz);
        float spacing = mix(0.5, 0.18, saturate(b.c.z / 6));
        float k = float(vid / 2);
        float s = (k * spacing + fract(u.a.y * 0.28 / spacing + b.c.w) * spacing) / len;
        if (on <= 0 || s > 1) { o.pos = float4(0, 0, -1, 1); o.size = 0; return o; }
        if (down) s = 1 - s;
        float3 q = mix(b.a.xyz, b.b.xyz, s);
        o.pos = globeClip(q, c);
        o.size = 4 * u.a.x;
        float edge = smoothstep(0.0, 0.04, s) * smoothstep(1.0, 0.96, s);
        o.color = float4(mix(bandColors[int(b.a.w)], float3(1), 0.4), 1) * 0.9 * edge * globeShown(q, c);
        return o;
    }

    fragment float4 photonFill(Photon in [[stage_in]], float2 pc [[point_coord]]) {
        float d = length(pc - 0.5) * 2;
        float px = max(fwidth(d), 1e-3);
        return in.color * (1 - smoothstep(0.42 - px, 0.42 + px, d));
    }

    struct Dot { float4 pos [[position]]; float size [[point_size]]; float4 color; float kind; };

    // Spacecraft (0), planets (1), the Sun (2) and the network's sites (3). u.b: sun
    vertex Dot dot(uint i [[vertex_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                   const device float4 *pts [[buffer(2)]]) {
        float4 p = pts[i];
        Dot o;
        o.pos = globeClip(p.xyz, c);
        o.kind = p.w;
        float shown = p.w > 2.5 ? globeFacing(p.xyz, c) : globeShown(p.xyz, c);
        float sizes[4] = { 14, 8, 12, 12 };
        o.size = sizes[int(p.w)] * u.a.x;
        float4 colors[4] = { float4(0.95, 0.96, 0.97, 1) * 0.9, float4(0.75, 0.78, 0.82, 1) * 0.55, float4(1.0, 0.86, 0.6, 1) * 0.9,
                             float4(1.0, 0.86, 0.66, 1) * 0.9 };
        o.color = colors[int(p.w)] * shown;
        if (shown <= 0) o.pos = float4(0, 0, -1, 1);
        return o;
    }

    // Spacecraft get a small open square, like a tracking box; planets and the Sun a dot; sites a tick mark.
    fragment float4 dotFill(Dot in [[stage_in]], float2 pc [[point_coord]]) {
        float2 p = (pc - 0.5) * 2, a = abs(p);
        float px = max(fwidth(p.x), 1e-3) * 1.1;
        float ink;
        if (in.kind < 0.5) {
            float box = max(a.x, a.y);
            ink = (1 - smoothstep(0.0, px, abs(box - 0.62))) + (1 - smoothstep(0.12, 0.12 + px, length(p)));
        } else if (in.kind < 2.5) {
            ink = 1 - smoothstep((in.kind > 1.5 ? 0.6 : 0.42) - px, (in.kind > 1.5 ? 0.6 : 0.42) + px, length(p));
        } else {
            ink = (1 - smoothstep(0.0, px, a.x)) * step(a.y, 0.9) + (1 - smoothstep(0.0, px, a.y)) * step(a.x, 0.9);
            ink *= step(0.3, max(a.x, a.y));
        }
        return in.color * saturate(ink);
    }

    // Faint dashed circles in the ecliptic plane at the scaled distances of the Moon, 1, 10 and 100 AU, turned with
    // the sky. Each ring: plane normal (inertial) and radius. u.a: pixel scale, GMST
    vertex Ribbon scaleRing(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                            constant GlobeCamera &c [[buffer(1)]], const device float4 *ringsIn [[buffer(2)]]) {
        float4 r = ringsIn[iid];
        uint seg = vid / 6, corner = vid % 6;
        bool atB = corner == 2 || corner == 3 || corner == 5;
        float3 n = r.xyz;
        float3 e1 = float3(1, 0, 0), e2 = normalize(cross(n, e1));
        float g = u.a.y;
        float a0 = float(seg) / 128 * TAU, a1 = float(seg + 1) / 128 * TAU;
        float3 i0 = (e1 * cos(a0) + e2 * sin(a0)) * r.w, i1 = (e1 * cos(a1) + e2 * sin(a1)) * r.w;
        float cg = cos(g), sg = sin(g);
        float3 p0 = float3(cg * i0.x + sg * i0.y, -sg * i0.x + cg * i0.y, i0.z), p1 = float3(cg * i1.x + sg * i1.y, -sg * i1.x + cg * i1.y, i1.z);
        float3 q = atB ? p1 : p0;
        float4 color = float4(0.8, 0.84, 0.9, 1) * 0.22 * globeShown(q, c);
        Ribbon o = ribbon(corner, globeClip(p0, c), globeClip(p1, c), 1.0 * u.a.x, color, u.res);
        o.along = float(seg) + (atB ? 1 : 0);
        return o;
    }

    fragment float4 ringFill(Ribbon in [[stage_in]]) {
        float dash = step(0.5, fract(in.along * 0.5));
        return in.color * saturate(in.half_width + 0.5 - abs(in.across)) * dash;
    }
    """
}
