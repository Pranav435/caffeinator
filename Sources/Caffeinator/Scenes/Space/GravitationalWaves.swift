import AppKit
import Metal
import simd

/// Real binary black hole mergers heard by LIGO. The two black holes spiral together in front of ESO's photograph of
/// the Milky Way and bend its light as gravitational lenses, with the exact Schwarzschild deflection: black shadows
/// ringed by thin photon rings, the star clouds smeared into Einstein rings and doubled images, the two shadows merging
/// into one. The orbit follows leading-order post-Newtonian timing from the measured masses, slowed down to watch.
/// In the corner, the real Hanford and Livingston strain from GWOSC, whitened and band-passed as in the discovery
/// paper, and its time–frequency chirp, with a playhead that keeps time with the orbit; the sound is that data too.
@MainActor final class GravitationalWaves: Scene {
    struct Merger {
        var name: String
        var m1: Double, m2: Double
        var final: Double?
        /// Luminosity distance in megaparsecs.
        var distance: Double
        var gps: Double = 0
        var redshift: Double = 0
    }

    /// The inspiral of one merger in "visual" time: the true chirp, slowed so the orbit at the innermost stable circular
    /// orbit takes `visualOrbit` seconds. Leading-order (Newtonian quadrupole) timing: time to coalescence
    /// τ = (5/256)(G𝓜/c³)^(−5/3)(πf)^(−8/3), so separation ∝ τ^(1/4), orbital frequency ∝ τ^(−3/8), amplitude ∝ f^(2/3).
    struct Inspiral {
        static let solarTime = 4.925491e-6        // G M☉ / c³ in seconds
        static let visualOrbit = 0.75             // orbits per second at the innermost stable orbit, on screen
        static let inspiralTime = 26.0            // seconds of inspiral shown, unless set
        static let plunge = 0.45
        static let ringdown = 0.9

        let m1: Double, m2: Double
        /// Seconds of inspiral shown before the plunge.
        var duration = Self.inspiralTime
        var total: Double { m1 + m2 }
        var eta: Double { m1 * m2 / (total * total) }
        var chirpMass: Double { total * pow(eta, 0.6) }
        /// Gravitational-wave frequency at the innermost stable circular orbit, about 4,400 Hz × M☉ / M.
        var iscoFrequency: Double { 1 / (pow(6, 1.5) * .pi * total * Self.solarTime) }
        /// How many times slower than real life the scene runs.
        var stretch: Double { iscoFrequency / 2 / Self.visualOrbit }
        /// Real time from the innermost stable orbit to (Newtonian) coalescence, slowed by `stretch`.
        var tauISCO: Double {
            5.0 / 256 * pow(chirpMass * Self.solarTime, -5.0 / 3) * pow(.pi * iscoFrequency, -8.0 / 3) * stretch
        }
        var tauStart: Double { tauISCO + duration }
        var merger: Double { duration + Self.plunge }

        /// Orbital phase (radians), separation (in units of the innermost stable orbit), wave amplitude (1 at that orbit)
        /// and orbital angular speed, `s` seconds after the shown inspiral begins.
        func state(_ s: Double) -> (phase: Double, separation: Double, amplitude: Double, omega: Double) {
            let wI = 2 * .pi * Self.visualOrbit
            let k = 8.0 / 5 * wI * tauISCO
            let startX = tauStart / tauISCO
            if s < duration {
                let x = (tauStart - s) / tauISCO
                return (k * (pow(startX, 0.625) - pow(x, 0.625)), pow(x, 0.25), pow(x, -0.25), wI * pow(x, -0.375))
            }
            let phaseISCO = k * (pow(startX, 0.625) - 1)
            if s < merger {
                let u = (s - duration) / Self.plunge
                return (phaseISCO + wI * Self.plunge * (u + 0.3 * u * u), pow(1 - u, 1.5), 1 + 0.4 * u, wI * (1 + 0.6 * u))
            }
            let v = s - merger
            let phaseMerger = phaseISCO + wI * Self.plunge * 1.3
            return (phaseMerger + 1.6 * wI * v, 0, 1.4 * exp(-v / Self.ringdown), 1.6 * wI)
        }
    }

    /// Matches `struct GWLens` in the shader.
    private struct LensParams {
        var hole1: SIMD4<Float>     // position (in total masses), mass
        var hole2: SIMD4<Float>
        var hole3: SIMD4<Float>     // the remnant
        var eye: SIMD4<Float>       // position, focal length
        var right: SIMD4<Float>     // axis, fade
        var up: SIMD4<Float>        // axis, exposure
        var forward: SIMD4<Float>   // axis, 1 when the photograph is missing
        var sky0: SIMD4<Float>      // rows of the turn from the scene into equatorial coordinates
        var sky1: SIMD4<Float>
        var sky2: SIMD4<Float>
        var scrim: SIMD4<Float>     // the figure's box in uv, darkened a little so the traces read
    }

    private static let url = URL(string: "https://gwosc.org/eventapi/json/allevents/")!
    private static let alertURL = URL(string: "https://gracedb.ligo.org/api/superevents/?query=category%3A+Production&format=json&count=1")!
    /// The shown events' parameters (GWTC-1, 2.1 and 3, source frame), used until the catalogue arrives or offline.
    private static let remembered = [
        Merger(name: "GW150914", m1: 34.6, m2: 30.0, final: 61.5, distance: 470, gps: 1_126_259_462.4, redshift: 0.10),
        Merger(name: "GW200129", m1: 34.5, m2: 29.0, final: 60.2, distance: 890, gps: 1_264_316_116.4, redshift: 0.18),
        Merger(name: "GW170814", m1: 30.9, m2: 24.9, final: 53.2, distance: 610, gps: 1_186_741_861.5, redshift: 0.13),
        Merger(name: "GW190521", m1: 98.4, m2: 57.2, final: 147.4, distance: 3310, gps: 1_242_442_967.4, redshift: 0.56),
        Merger(name: "GW190412", m1: 27.7, m2: 9.0, final: 35.6, distance: 720, gps: 1_239_082_262.1, redshift: 0.15),
        Merger(name: "GW190521_074359", m1: 43.4, m2: 33.4, final: 72.6, distance: 1080, gps: 1_242_459_857.4, redshift: 0.21),
        Merger(name: "GW170104", m1: 28.7, m2: 20.8, final: 47.5, distance: 1110, gps: 1_167_559_936.6, redshift: 0.22),
        Merger(name: "GW200224", m1: 40.0, m2: 32.7, final: 68.7, distance: 1710, gps: 1_266_618_172.4, redshift: 0.32),
        Merger(name: "GW191109", m1: 65.0, m2: 47.0, final: 107.0, distance: 1290, gps: 1_257_296_855.2, redshift: 0.25),
        Merger(name: "GW190828_063405", m1: 31.9, m2: 25.8, final: 54.3, distance: 2070, gps: 1_251_009_263.7, redshift: 0.38),
        Merger(name: "GW200311", m1: 34.2, m2: 27.7, final: 59.0, distance: 1170, gps: 1_267_963_151.3, redshift: 0.23),
    ]
    /// Rich stretches of the Milky Way to put behind the binary, as galactic longitude and latitude in degrees.
    private static let backdrops: [SIMD2<Double>] = [[2, -3], [287.5, -0.8], [27, -2], [78, 1.5], [302, -1], [335, -1], [9, -5], [263, -3]]
    /// The chord the eight frequency bands sound as, lowest band first.
    private static let chord: [Float] = [110, 146.83, 164.81, 220, 246.94, 293.66, 329.63, 440]

    private let lensPipe: MTLRenderPipelineState
    private let tracePipe: MTLRenderPipelineState
    private let mapPipe: MTLRenderPipelineState
    private let lens: Lens
    private let rings: [Ring]
    private let deflection: MTLBuffer
    private let outline: MTLTexture
    private var photo: MTLTexture?
    private var photoDone = false
    private var light: MTLTexture?

    private var catalogue: [Merger] = []
    private var detections = 0
    private var alert: String?
    private var order: [Int]
    private var shown = 0
    private var current: Merger
    private var inspiral: Inspiral
    private var eventStart: Float = 0
    private var signal: GWSignal?
    private var signalFor: Int?
    private var loading: Int?
    private var failed: Set<Int> = []
    private var firstDone = false
    private var map: MTLTexture?
    private var traces: MTLBuffer?
    private var traceCount = 0
    private var window = (start: -0.6, end: 0.12)
    private var range: Float = 6
    private var started = false
    private var chirped = false
    private var merged = false
    private var captioned = false

    var scale: Float { 1 }
    var ready: Bool { photoDone && firstDone }
    var facts: String? { describe() }
    var credit: String? { "Milky Way: ESO/S. Brunier. Strain: LIGO Scientific Collaboration, Virgo and KAGRA, via GWOSC" }

    init(gpu: GPU) throws {
        let library = try gpu.library(SpaceKit.metal + SpaceSky.metal + Self.source)
        lensPipe = try gpu.pipeline(library, "gwLensing")
        tracePipe = try gpu.pipeline(library, "lineFragment", vertex: "gwTraceVertex", blend: .alpha)
        mapPipe = try gpu.pipeline(library, "gwMapFragment", vertex: "gwMapVertex", blend: .alpha)
        lens = try Lens(gpu: gpu)
        rings = (0..<3).map { _ in Ring(gpu: gpu, bytes: 4096) }
        deflection = gpu.buffer(GWData.deflectionTable())
        var levels = SkyCatalog.milkyWay.map { $0 * 51 }
        outline = levels.withUnsafeMutableBytes { gpu.texture(720, 360, .r8Unorm, bytes: $0.baseAddress!, bytesPerRow: 720) }
        let n = GWData.sources.count
        let first = (Snapshot.variant ?? 0) % n
        order = [first] + Array(0..<n).filter { $0 != first }.shuffled()
        current = Self.remembered[first]
        inspiral = Self.inspiral(for: current)
        Task { @MainActor [weak self] in
            let texture = await MilkyWay.texture(gpu)
            self?.photo = texture
            self?.photoDone = true
        }
        fetch(first)
        Task { @MainActor [weak self] in
            let object = await Feed.object(Self.url, maxAge: 30 * 86400)
            let parsed = Self.parse(object)
            self?.catalogue = parsed
            self?.detections = parsed.count
            if let object = await Feed.object(Self.alertURL, maxAge: 6 * 3600) { self?.alert = Self.parseAlert(object) }
        }
    }

    // MARK: Data

    /// Every event with both component masses, one entry per event (the most complete version), named by its short
    /// name when that is unambiguous.
    static func parse(_ object: Any?) -> [Merger] {
        guard let root = object as? [String: Any], let all = root["events"] as? [String: Any] else { return [] }
        func number(_ v: Any?) -> Double? { (v as? NSNumber)?.doubleValue }
        var best: [Int: (score: Int, merger: Merger)] = [:]
        for (_, value) in all {
            guard let e = value as? [String: Any], let name = e["commonName"] as? String,
                  let m1 = number(e["mass_1_source"]), let m2 = number(e["mass_2_source"]), m1 > 0, m2 > 0,
                  let gps = number(e["GPS"]) else { continue }
            let distance = number(e["luminosity_distance"]) ?? 0
            let final = number(e["final_mass_source"])
            let version = (e["version"] as? Int) ?? 0
            let score = (final != nil ? 100 : 0) + (distance > 0 ? 50 : 0) + version
            let key = Int(gps.rounded())
            if let old = best[key], old.score >= score { continue }
            best[key] = (score, Merger(name: name, m1: max(m1, m2), m2: min(m1, m2), final: final, distance: distance, gps: gps,
                                       redshift: number(e["redshift"]) ?? 0))
        }
        var list = best.values.map(\.merger).filter { $0.distance > 0 }
        // "GW250114_082203" reads as "GW250114" unless two events share a day.
        let days = Dictionary(grouping: list) { String($0.name.prefix(8)) }
        for i in list.indices where list[i].name.count > 8 {
            let day = String(list[i].name.prefix(8))
            if days[day]?.count == 1 { list[i].name = day }
        }
        return list.sorted { $0.name < $1.name }
    }

    static func parseAlert(_ object: Any) -> String? {
        guard let root = object as? [String: Any], let list = root["superevents"] as? [[String: Any]], let e = list.first,
              let id = e["superevent_id"] as? String else { return nil }
        var text = "The most recent public alert is \(id)"
        if let created = e["created"] as? String {
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "yyyy-MM-dd HH:mm:ss 'UTC'"
            f.timeZone = TimeZone(identifier: "UTC")
            if let date = f.date(from: created) {
                text += ", from \(date.formatted(.dateTime.day().month(.wide).year()))"
            }
        }
        let labels = e["labels"] as? [String] ?? []
        if labels.contains(where: { $0.hasPrefix("LOW_SIGNIF") }) { text += ", a low-significance candidate" }
        if let instruments = (e["preferred_event_data"] as? [String: Any])?["instruments"] as? String {
            let names = instruments.split(separator: ",").compactMap { ["H1": "Hanford", "L1": "Livingston", "V1": "Virgo", "K1": "KAGRA"][String($0)] }
            if !names.isEmpty { text += " seen by " + ListFormatter.localizedString(byJoining: names) }
        }
        return text + "."
    }

    /// The orbit in the detectors' frame: masses redshifted by (1 + z), so the slowed chirp keeps time with the data.
    /// Heavier binaries chirp for less time, so the shown inspiral covers about the last 0.6 s of the recording.
    private static func inspiral(for m: Merger) -> Inspiral {
        let z = 1 + m.redshift
        var i = Inspiral(m1: m.m1 * z, m2: m.m2 * z)
        i.duration = min(max(i.stretch * 0.6, 14), 30)
        return i
    }

    /// Downloads and processes one event's strain, one event at a time.
    private func fetch(_ index: Int) {
        guard loading == nil, signalFor != index, !failed.contains(index) else { return }
        loading = index
        Task { @MainActor [weak self] in
            let result = await GWData.load(GWData.sources[index])
            guard let self else { return }
            loading = nil
            if let result {
                pending = (index, result)
            } else {
                failed.insert(index)
            }
            if index == order[0] { firstDone = true }
        }
    }

    /// The most recently downloaded event, waiting for its turn.
    private var pending: (Int, GWSignal)?

    /// Starts the next event in the rotation.
    private func next(at time: Float, _ f: Frame) {
        if started { shown = (shown + 1) % order.count }
        started = true
        let index = order[shown]
        let remembered = Self.remembered[index]
        current = catalogue.first { abs($0.gps - remembered.gps) < 1 }.map { var m = $0; m.name = remembered.name; return m } ?? remembered
        if current.redshift == 0 { current.redshift = remembered.redshift }
        inspiral = Self.inspiral(for: current)
        eventStart = time
        chirped = false
        merged = false
        captioned = false
        if let (i, s) = pending, i == index {
            signal = s
            signalFor = i
            pending = nil
        } else if signalFor != index {
            signal = nil
            signalFor = nil
        }
        let w = min(max(inspiral.duration / inspiral.stretch, 0.3), 0.6)
        window = (-w, 0.12)
        prepareFigure(f)
        fetch(order[(shown + 1) % order.count])
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        if !started { next(at: f.time, f) }
        let total = Float(inspiral.merger + 12) + 1.5
        if f.time - eventStart > total { next(at: f.time, f) }
        // A late download for the event on screen joins it while the inspiral is still young.
        if signal == nil, let (i, s) = pending, i == order[shown], f.time - eventStart < 6 {
            signal = s
            signalFor = i
            pending = nil
            prepareFigure(f)
            captioned = false
        }
        let e = Double(f.time - eventStart)
        let s = e - 1.5
        let fade = Float(min(e / 1.5, 1, (Double(total) - e) / 1.5))
        let st = inspiral.state(s)

        // Positions in units of the total mass, about the centre of mass; the innermost stable orbit is 6 M across.
        let M = current.m1 + current.m2
        let q1 = current.m1 / M, q2 = current.m2 / M
        let r = 6 * st.separation
        let dir = SIMD2(cos(st.phase), sin(st.phase))
        let p1 = dir * r * q2, p2 = -dir * r * q1
        // The common horizon forms over the last moments of the plunge.
        let formed = Float(smoothstep(inspiral.merger - 0.22, inspiral.merger + 0.12, s))
        let remnant = Float((current.final ?? 0.95 * M) / M)

        // A slow drift of the camera around the binary, and a different stretch of the Milky Way behind each event.
        let t = Double(f.time) + Double(f.seed) * 40
        let yaw = t * 2 * .pi / 420
        let tilt = Astro.rad(64 + 9 * sin(t * 0.013))
        let distance = 26.0
        let eye = SIMD3(sin(tilt) * cos(yaw), sin(tilt) * sin(yaw), cos(tilt)) * distance
        let up0 = SIMD3<Double>(0, 0, 1)
        let forward0 = simd_normalize(-eye)
        let right0 = simd_normalize(simd_cross(forward0, up0))
        let upAxis = simd_cross(right0, forward0)
        // Aim a little below the binary so it sits above the middle of the screen.
        let target = -upAxis * 4.6
        let forward = simd_normalize(target - eye)
        let right = simd_normalize(simd_cross(forward, up0))
        let up = simd_cross(right, forward)
        let sky = skyTurn(e, right: right, up: up, forward: forward)

        let figure = figureBox()
        let f3 = { (v: SIMD3<Double>) in SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        let params = LensParams(
            hole1: SIMD4(Float(p1.x), Float(p1.y), 0, Float(q1) * (1 - formed)),
            hole2: SIMD4(Float(p2.x), Float(p2.y), 0, Float(q2) * (1 - formed)),
            hole3: SIMD4(0, 0, 0, remnant * formed),
            eye: SIMD4(f3(eye), 1.35), right: SIMD4(f3(right), fade), up: SIMD4(f3(up), 1.5),
            forward: SIMD4(f3(forward), photo == nil ? 1 : 0),
            sky0: SIMD4(f3(sky[0]), 0), sky1: SIMD4(f3(sky[1]), 0), sky2: SIMD4(f3(sky[2]), 0),
            scrim: SIMD4(Float(figure.minX) - 0.03, Float(figure.minY) - 0.05, Float(figure.maxX) + 0.03, Float(figure.maxY) + 0.03))

        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
        }
        let light = light!
        f.cmd.draw(lensPipe, into: light, uniforms: f.uniforms(), textures: [photo ?? outline, outline],
                   buffers: [rings[0].next(params), deflection])
        lens.develop(f, light, Lens.Look(exposure: 1, bloom: 0.035, halation: 0.02, vignette: 0.3, aberration: 1.0, grain: 0.45))
        drawFigure(f, s: s, fade: fade, box: figure)

        if !captioned {
            captioned = true
            caption(f, box: figure)
        }
        if let synth = f.sound { play(synth, s) }
    }

    /// Rows of the rotation taking scene directions to equatorial ones: the camera's view lands on this event's
    /// stretch of the Milky Way, wandering a few degrees along it, the band tilted across the frame.
    private func skyTurn(_ e: Double, right: SIMD3<Double>, up: SIMD3<Double>, forward: SIMD3<Double>) -> [SIMD3<Double>] {
        let base = Self.backdrops[shown % Self.backdrops.count]
        let l = Astro.rad(base.x + 5 * sin(e * 0.02)), b = Astro.rad(base.y + 1.5 * sin(e * 0.031))
        let target = SpaceSky.equatorial(galactic: SIMD3(cos(b) * cos(l), cos(b) * sin(l), sin(b)))
        let pole = SpaceSky.equatorial(galactic: SIMD3(0, 0, 1))
        let roll = Astro.rad(shown % 2 == 0 ? 24 : -18)
        let r0 = simd_normalize(simd_cross(target, pole))
        let u0 = simd_cross(r0, target)
        let rightE = r0 * cos(roll) + u0 * sin(roll), upE = u0 * cos(roll) - r0 * sin(roll)
        // R = [rightE upE target] · [right up forward]ᵀ, stored as rows.
        func column(_ v: SIMD3<Double>) -> SIMD3<Double> { rightE * v.x + upE * v.y + target * v.z }
        let cx = column(SIMD3(right.x, up.x, forward.x)), cy = column(SIMD3(right.y, up.y, forward.y))
        let cz = column(SIMD3(right.z, up.z, forward.z))
        return [SIMD3(cx.x, cy.x, cz.x), SIMD3(cx.y, cy.y, cz.y), SIMD3(cx.z, cy.z, cz.z)]
    }

    // MARK: The figure

    /// The figure's box in uv (0...1, top left origin): the strain above, the time–frequency map below. It changes
    /// sides from one event to the next so nothing sits on the same pixels for hours.
    private func figureBox() -> CGRect {
        let x = shown % 2 == 0 ? 0.05 : 0.6
        return CGRect(x: x, y: 0.69, width: 0.35, height: 0.235)
    }

    private func prepareFigure(_ f: Frame) {
        guard let signal else { map = nil; traces = nil; return }
        // The time–frequency map over the window, 480 columns.
        let columns = 480
        var pixels = [Float](repeating: 0, count: columns * GWSignal.rows)
        for row in 0..<GWSignal.rows {
            for c in 0..<columns {
                let t = window.start + (window.end - window.start) * (Double(c) + 0.5) / Double(columns)
                pixels[(GWSignal.rows - 1 - row) * columns + c] = signal.energy(row: row, at: t)
            }
        }
        map = pixels.withUnsafeBytes { f.gpu.texture(columns, GWSignal.rows, .r32Float, bytes: $0.baseAddress!, bytesPerRow: columns * 4) }
        // Both traces over the window, as (time, Hanford, Livingston, 0).
        var points: [SIMD4<Float>] = []
        var peak: Float = 0
        let i0 = Int((window.start + GWSignal.before) * GWSignal.rate), i1 = Int((window.end + GWSignal.before) * GWSignal.rate)
        for i in max(i0, 0)..<min(i1, signal.hanford.count) {
            let t = Double(i) / GWSignal.rate - GWSignal.before
            points.append(SIMD4(Float(t), signal.hanford[i], signal.livingston[i], 0))
            peak = max(peak, abs(signal.hanford[i]), abs(signal.livingston[i]))
        }
        range = max(peak * 1.12, 5)
        traceCount = points.count
        traces = f.gpu.buffer(points)
    }

    private func drawFigure(_ f: Frame, s: Double, fade: Float, box: CGRect) {
        guard let traces, let map, traceCount > 1 else { return }
        let now = Float((s - inspiral.merger) / inspiral.stretch)
        let u = f.uniforms(SIMD4(Float(box.minX), Float(box.minY), Float(box.width), Float(box.height)),
                           SIMD4(Float(window.start), Float(window.end), now, fade),
                           SIMD4(range, f.size.y / 1000, 0, 0), SIMD4(Float(traceCount - 1), 0, 0, 0))
        f.cmd.draw(mapPipe, into: f.target, clear: false, uniforms: u, textures: [map], vertices: 4, primitive: .triangleStrip)
        // Livingston under Hanford, then the zero line and the playhead.
        f.cmd.draw(tracePipe, into: f.target, clear: false, uniforms: u, buffers: [traces],
                   vertices: 4, instances: 2 * (traceCount - 1) + 2, primitive: .triangleStrip)
    }

    private func caption(_ f: Frame, box: CGRect) {
        let c = current
        let both = "\(Self.mass(c.m1)) + \(Self.mass(c.m2)) M☉" + (c.final.map { " → \(SpaceKit.number($0, decimals: 1)) M☉" } ?? "")
        let ly = c.distance * 3.2616e6
        let far = ly >= 1e9 ? "\(SpaceKit.number(ly / 1e9, decimals: 1)) billion light-years" : "\(Int((ly / 1e6).rounded())) million light-years"
        let signal = signal
        let window = window
        let stretch = SpaceKit.number(inspiral.stretch)
        let date = Self.date(gps: signal?.gps ?? c.gps)
        f.overlay.draw { ctx, size in
            let x = box.minX * size.width, w = box.width * size.width
            let y = box.minY * size.height, h = box.height * size.height
            let strainBottom = y + h * 0.47, mapTop = y + h * 0.53
            let dim = NSColor(white: 1, alpha: 0.38), faint = NSColor(white: 1, alpha: 0.24)
            ctx.text(c.name, at: CGPoint(x: x - 1, y: y - 74), size: 21, weight: .ultraLight, color: NSColor(white: 1, alpha: 0.78), tracking: 1.2)
            ctx.text("\(both)  ·  \(SpaceKit.number(c.distance)) Mpc, \(far)", at: CGPoint(x: x, y: y - 46), size: 10.5,
                     color: NSColor(white: 1, alpha: 0.5))
            ctx.text(date + "  ·  orbit slowed \(stretch)×", at: CGPoint(x: x, y: y - 31), size: 10.5, color: dim)
            guard let signal else {
                ctx.text("LIGO strain not downloaded  ·  model chirp", at: CGPoint(x: x, y: y + 4), size: 9.5, weight: .medium,
                         color: faint, tracking: 1.0)
                return
            }
            let hanford = NSColor(red: 1.0, green: 0.74, blue: 0.5, alpha: 0.62)
            let livingston = NSColor(red: 0.62, green: 0.78, blue: 1.0, alpha: 0.62)
            ctx.text("HANFORD H1", at: CGPoint(x: x, y: y - 14), size: 8.5, weight: .medium, color: hanford, tracking: 1.2)
            let flip = signal.flipped ? ", inverted" : ""
            ctx.text("LIVINGSTON L1  (shifted \(SpaceKit.number(abs(signal.lag), decimals: 1)) ms\(flip))", at: CGPoint(x: x + 82, y: y - 14),
                     size: 8.5, weight: .medium, color: livingston, tracking: 1.2)
            ctx.text("WHITENED STRAIN (σ)  ·  30–400 Hz", at: CGPoint(x: x + w, y: y - 14), size: 8, weight: .regular, color: faint,
                     anchor: CGPoint(x: 1, y: 0), tracking: 0.8)
            // Hairlines: the strain panel's frame ticks and the time axis under the map.
            ctx.setLineWidth(0.6)
            ctx.setStrokeColor(NSColor(white: 1, alpha: 0.16).cgColor)
            ctx.stroke(CGRect(x: x, y: y, width: w, height: strainBottom - y))
            ctx.stroke(CGRect(x: x, y: mapTop, width: w, height: y + h - mapTop))
            let span = window.end - window.start
            var tick = (window.start / 0.1).rounded(.up) * 0.1
            while tick <= window.end + 1e-6 {
                let tx = x + w * (tick - window.start) / span
                ctx.move(to: CGPoint(x: tx, y: y + h))
                ctx.addLine(to: CGPoint(x: tx, y: y + h + 4))
                let label = abs(tick) < 1e-6 ? "0 s" : String(format: "%+.1f", tick).replacingOccurrences(of: "-", with: "−")
                ctx.text(label, at: CGPoint(x: tx, y: y + h + 6), size: 8.5, color: faint, anchor: CGPoint(x: 0.5, y: 0))
                tick += 0.1
            }
            ctx.strokePath()
            ctx.text("TIME FROM MERGER", at: CGPoint(x: x + w, y: y + h + 18), size: 8, color: faint, anchor: CGPoint(x: 1, y: 0), tracking: 0.8)
            // Frequency ticks on the map, log scale.
            for hz in [32.0, 64, 128, 256] {
                let row = log(hz / GWSignal.low) / log(GWSignal.high / GWSignal.low)
                let ty = y + h - (y + h - mapTop) * row
                ctx.move(to: CGPoint(x: x - 4, y: ty))
                ctx.addLine(to: CGPoint(x: x, y: ty))
                ctx.text("\(Int(hz))", at: CGPoint(x: x - 6, y: ty), size: 8, color: faint, anchor: CGPoint(x: 1, y: 0.5))
            }
            ctx.strokePath()
            ctx.text("Hz", at: CGPoint(x: x - 6, y: mapTop - 2), size: 8, color: faint, anchor: CGPoint(x: 1, y: 0))
        }
    }

    /// The UTC date of a GPS time (18 leap seconds since 1980 by 2017; close enough for a date).
    private static func date(gps: Double) -> String {
        let unix = gps + 315_964_800 - 18
        let d = Date(timeIntervalSince1970: unix)
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "d MMMM yyyy, HH:mm:ss 'UTC'"
        return f.string(from: d)
    }

    private var radiated: Double? {
        guard let f = current.final else { return nil }
        let r = current.m1 + current.m2 - f
        return r >= 0.1 ? r : nil
    }

    private static func mass(_ m: Double) -> String { SpaceKit.number(m, decimals: m < 3 ? 2 : 1) }

    private func describe() -> String {
        let c = current
        var text = "\(c.name): black holes of \(Self.mass(c.m1)) and \(Self.mass(c.m2)) solar masses"
        let ly = c.distance * 3.2616e6
        let far = ly >= 1e9 ? "\(SpaceKit.number(ly / 1e9, decimals: 1)) billion" : "\(Int((ly / 1e6).rounded())) million"
        text += ", about \(far) light-years away"
        if let r = radiated { text += "; about \(SpaceKit.number(r, decimals: 1)) solar masses became gravitational waves" }
        text += ". "
        if let signal {
            text += "Real strain from LIGO Hanford and Livingston, \(SpaceKit.number(abs(signal.lag), decimals: 1)) milliseconds apart; "
            text += "the orbit and the sound are slowed \(SpaceKit.number(inspiral.stretch)) times."
        } else if loading == order[shown] {
            text += "Downloading the LIGO strain; meanwhile a model chirp from the masses."
        } else {
            text += "The LIGO strain hasn't downloaded, so the sound is a model chirp from the masses."
        }
        if detections > 0, shown % 3 == 1 { text += " One of \(SpaceKit.number(Double(detections))) mergers in the open catalogue." }
        if let alert, shown % 3 == 2 { text += " " + alert }
        return text
    }

    // MARK: Sound

    /// The real data as a chord: each of eight frequency bands from 24 to 480 Hz is one note, as loud as the strain's
    /// energy in that band at the playhead, panned toward whichever detector heard more. The noise is a soft shimmer;
    /// the chirp climbs through the chord as it rises in frequency, then the remnant rings.
    private func play(_ synth: Synth, _ s: Double) {
        let now = (s - inspiral.merger) / inspiral.stretch
        if let signal {
            for band in 0..<8 {
                var h: Float = 0, l: Float = 0
                for row in (band * 8)..<(band * 8 + 8) {
                    h += signal.energy(row: row, at: now, detector: 0)
                    l += signal.energy(row: row, at: now, detector: 1)
                }
                h /= 8
                l /= 8
                let e = 0.5 * (h + l)
                let shimmer = 0.009 * min(e / 3, 1)
                let rise = 0.075 * pow(min(max((e - 2) / 18, 0), 1), 0.8)
                synth.drone(band, Self.chord[band], s < -1 ? 0 : shimmer + rise, pan: (l - h) / (l + h + 0.5) * 0.7)
            }
        } else {
            // A model chirp from the masses while the data is missing.
            if !chirped, s >= 0 {
                chirped = true
                let end: Float = 330
                let start = end * Float(pow(inspiral.tauISCO / inspiral.tauStart, 0.375))
                synth.play(.chirp, start, amp: 0.12, length: Float(inspiral.tauStart), to: end)
            }
            for band in 0..<8 { synth.drone(band, Self.chord[band], band % 3 == 0 ? 0.008 : 0) }
        }
        if !merged, s >= inspiral.merger {
            merged = true
            synth.play(.thump, 46, amp: 0.12, length: 2.5)
            synth.play(.bell, 440, amp: 0.035, length: 5)
        }
        synth.noise(0.01, cutoff: 150)
    }

    // MARK: Shaders

    private static let source = """
    struct GWLens {
        float4 hole1, hole2, hole3;   // position in total masses, mass
        float4 eye;                   // position, focal length
        float4 right;                 // axis, fade
        float4 up;                    // axis, exposure
        float4 fwd;                   // axis, no photograph
        float4 sky0, sky1, sky2;      // scene → equatorial rows
        float4 scrim;                 // figure box in uv
    };

    // Exact Schwarzschild bending for impact parameter b past mass m: the table covers b = √27 m (1 + s) for
    // s = 10⁻⁶...10²; outside it, the strong-deflection limit and the weak-field series.
    inline float bendAngle(float b, float m, const device float *table) {
        float s = b / (5.1961524 * m) - 1;
        if (s >= 100) { float x = m / b; return 4 * x + 11.780972 * x * x; }
        if (s <= 1e-6) return -log(max(s, 1e-30)) - 0.40023;
        float i = (log10(s) + 6) / 8 * 255;
        int k = min(int(i), 254);
        return mix(table[k], table[k + 1], i - float(k));
    }

    // ESO's photograph toward equatorial direction d, filtered along the pixel's footprint on the sky: lensing stretches
    // that footprint into long thin arcs, and anisotropic filtering keeps them sharp across and smooth along. Jupiter,
    // caught in the photograph in 2009, is painted over as in `milkyWay`.
    inline float3 lensedSky(texture2d<float> t, float3 d) {
        constexpr sampler s(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge, max_anisotropy(16));
        float3 g = EQ_TO_PHOTO * d;
        float2 uv = float2(0.5 - atan2(g.y, g.x) / TAU, 0.5 - asin(clamp(g.z, -1.0, 1.0)) / PI);
        float2 dx = dfdx(uv), dy = dfdy(uv);
        dx.x -= round(dx.x);
        dy.x -= round(dy.x);
        float3 c = t.sample(s, uv, gradient2d(dx, dy)).rgb;
        float jupiter = smoothstep(0.99985, 0.99992, dot(g, float3(0.19237, 0.48919, -0.85068)));
        if (jupiter > 0) c = mix(c, t.sample(s, uv - float2(0, 0.012), gradient2d(dx, dy)).rgb, jupiter);
        return c;
    }

    fragment float4 gwLensing(V in [[stage_in]], constant U &u [[buffer(0)]], constant GWLens &g [[buffer(1)]],
                              const device float *table [[buffer(2)]], texture2d<float> photo [[texture(0)]],
                              texture2d<float> outline [[texture(1)]]) {
        float2 p = centered(in.uv, u.res);
        float3 rd = normalize(g.fwd.xyz * g.eye.w + p.x * g.right.xyz + p.y * g.up.xyz);
        float3 ro = g.eye.xyz;
        float pixel = 2 / (u.res.y * g.eye.w);
        float4 holes[3] = { g.hole1, g.hole2, g.hole3 };
        // The holes sit within a few masses of each other, far closer than the camera, so their bending adds as in a
        // thin lens: each turns the ray toward itself by its exact Schwarzschild angle, summed as rotations.
        float3 turn = 0;
        float shadow = 0;
        for (int k = 0; k < 3; k++) {
            float4 h = holes[k];
            float m = h.w;
            float3 oc = h.xyz - ro;
            float along = dot(oc, rd);
            if (m <= 1e-4 || along <= 0) continue;
            float3 perp = oc - rd * along;
            float b = max(length(perp), 1e-6);
            float bc = 5.1961524 * m;
            float px = along * pixel;
            shadow = max(shadow, smoothstep(bc + 0.8 * px, bc - 0.8 * px, b));
            turn += cross(rd, perp / b) * bendAngle(max(b, bc * 1.000001), m, table);
        }
        float angle = length(turn);
        if (angle > 1e-6) rd = normalize(rd * cos(angle) + cross(turn / angle, rd) * sin(angle));
        float3 d = float3(dot(g.sky0.xyz, rd), dot(g.sky1.xyz, rd), dot(g.sky2.xyz, rd));
        float3 col = g.fwd.w > 0.5 ? milkyWayOutline(outline, d) : lensedSky(photo, d);
        col *= g.up.w * (1 - shadow);
        // A soft darkening behind the figure, like a graduated filter, so its hairlines read over the star clouds.
        float2 q = in.uv;
        float inside = smoothstep(g.scrim.x - 0.07, g.scrim.x + 0.03, q.x) * smoothstep(g.scrim.z + 0.07, g.scrim.z - 0.03, q.x)
                     * smoothstep(g.scrim.y - 0.1, g.scrim.y + 0.04, q.y);
        col *= 1 - 0.62 * inside;
        return float4(col * g.right.w, 1);
    }

    // a: figure box (x, y, w, h in uv)   b: window start, end, playhead (s), fade   c: strain range (σ), pixel density.
    // The strain panel is the box's top 47 %, the map its bottom 47 %.
    inline float2 strainPoint(float t, float v, constant U &u) {
        float x = u.a.x + u.a.z * (t - u.b.x) / (u.b.y - u.b.x);
        float y = u.a.y + u.a.w * 0.235 * (1 - v / u.c.x);
        return float2(x, y) * u.res;
    }

    // Instances: Livingston's segments, Hanford's, then the zero line and the playhead.
    vertex LineOut gwTraceVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                 const device float4 *pts [[buffer(1)]]) {
        uint segs = (uint(u.d.x) > 0 ? uint(u.d.x) : 0);
        float d = u.c.y;
        float fade = u.b.w;
        uint count = segs;
        LineOut o;
        if (iid < 2 * count) {
            bool hanford = iid >= count;
            uint i = iid % count;
            float4 a = pts[i], b = pts[i + 1];
            float va = hanford ? a.y : a.z, vb = hanford ? b.y : b.z;
            float past = a.x <= u.b.z ? 1.0 : 0.22;
            float3 tint = hanford ? float3(1.0, 0.74, 0.5) : float3(0.62, 0.78, 1.0);
            float alpha = (hanford ? 0.72 : 0.58) * past * fade;
            return lineQuad(strainPoint(a.x, va, u), strainPoint(b.x, vb, u), 1.0 * d, float4(tint * alpha, alpha), vid, u.res);
        }
        if (iid == 2 * count) {
            return lineQuad(strainPoint(u.b.x, 0, u), strainPoint(u.b.y, 0, u), 0.6 * d, float4(0.12, 0.12, 0.12, 0.12) * fade, vid, u.res);
        }
        float shown = (u.b.z > u.b.x && u.b.z < u.b.y) ? 1.0 : 0.0;
        float x = strainPoint(clamp(u.b.z, u.b.x, u.b.y), 0, u).x;
        float a = 0.32 * shown * fade;
        return lineQuad(float2(x, u.a.y * u.res.y), float2(x, (u.a.y + u.a.w) * u.res.y), 0.8 * d, float4(a, a, a, a), vid, u.res);
    }

    struct MapOut { float4 pos [[position]]; float2 uv; };

    vertex MapOut gwMapVertex(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {
        float2 corner = float2(vid & 1, vid >> 1);
        float2 p = float2(u.a.x + u.a.z * corner.x, u.a.y + u.a.w * (0.53 + 0.47 * corner.y));
        MapOut o;
        o.pos = float4(p.x * 2 - 1, 1 - p.y * 2, 0, 1);
        o.uv = corner;
        return o;
    }

    // Energy over noise as light: faint speckle for the noise, the chirp burning amber to near white. Past the playhead
    // the map is dimmed, still to come.
    fragment float4 gwMapFragment(MapOut in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> map [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float e = map.sample(s, in.uv).r;
        float v = clamp(log(1 + max(e - 0.5, 0.0)) / log(31.0), 0.0, 1.0);
        float3 c = mix(float3(0.22, 0.3, 0.42), float3(1.0, 0.66, 0.32), smoothstep(0.25, 0.7, v));
        c = mix(c, float3(1.0, 0.95, 0.85), smoothstep(0.7, 1.0, v));
        float t = u.b.x + (u.b.y - u.b.x) * in.uv.x;
        float past = t <= u.b.z ? 1.0 : 0.3;
        float alpha = smoothstep(0.08, 0.6, v) * 0.85 * past * u.b.w;
        return float4(c * alpha, alpha);
    }
    """
}

private func smoothstep(_ a: Double, _ b: Double, _ x: Double) -> Double {
    let t = min(max((x - a) / (b - a), 0), 1)
    return t * t * (3 - 2 * t)
}
