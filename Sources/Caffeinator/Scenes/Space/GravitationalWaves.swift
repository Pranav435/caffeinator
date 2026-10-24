import AppKit
import Metal
import simd

/// Real black hole and neutron star mergers from the LIGO, Virgo and KAGRA catalogue (GWOSC). Two bodies spiral in
/// with leading-order post-Newtonian timing from their measured masses, slowed down to watch, while spacetime ripples
/// outward in the two-armed quadrupole pattern; then the merger, the ringdown, and the strain trace along the bottom.
@MainActor final class GravitationalWaves: Scene {
    struct Merger {
        var name: String
        var m1: Double, m2: Double
        var final: Double?
        /// Luminosity distance in megaparsecs.
        var distance: Double
    }

    /// The inspiral of one merger in "visual" time: the true chirp, slowed so the orbit at the innermost stable circular
    /// orbit takes `visualOrbit` seconds. Leading-order (Newtonian quadrupole) timing: time to coalescence
    /// τ = (5/256)(G𝓜/c³)^(−5/3)(πf)^(−8/3), so separation ∝ τ^(1/4), orbital frequency ∝ τ^(−3/8), amplitude ∝ f^(2/3).
    struct Inspiral {
        static let solarTime = 4.925491e-6        // G M☉ / c³ in seconds
        static let visualOrbit = 0.75             // orbits per second at the innermost stable orbit, on screen
        static let inspiralTime = 26.0            // seconds of inspiral shown
        static let plunge = 0.45
        static let ringdown = 0.9

        let m1: Double, m2: Double
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
        var tauStart: Double { tauISCO + Self.inspiralTime }
        var merger: Double { Self.inspiralTime + Self.plunge }

        /// Orbital phase (radians), separation (in units of the innermost stable orbit), wave amplitude (1 at that orbit)
        /// and orbital angular speed, `s` seconds after the shown inspiral begins.
        func state(_ s: Double) -> (phase: Double, separation: Double, amplitude: Double, omega: Double) {
            let wI = 2 * .pi * Self.visualOrbit
            let k = 8.0 / 5 * wI * tauISCO
            let startX = tauStart / tauISCO
            if s < Self.inspiralTime {
                let x = (tauStart - s) / tauISCO
                return (k * (pow(startX, 0.625) - pow(x, 0.625)), pow(x, 0.25), pow(x, -0.25), wI * pow(x, -0.375))
            }
            let phaseISCO = k * (pow(startX, 0.625) - 1)
            if s < merger {
                let u = (s - Self.inspiralTime) / Self.plunge
                return (phaseISCO + wI * Self.plunge * (u + 0.3 * u * u), pow(1 - u, 1.5), 1 + 0.4 * u, wI * (1 + 0.6 * u))
            }
            let v = s - merger
            let phaseMerger = phaseISCO + wI * Self.plunge * 1.3
            return (phaseMerger + 1.6 * wI * v, 0, 1.4 * exp(-v / Self.ringdown), 1.6 * wI)
        }
    }

    /// Matches `struct GW` in the shader.
    private struct Params {
        var body1: SIMD4<Float>     // position xy, radius, kind (0 black hole, 1 neutron star, -1 hidden)
        var body2: SIMD4<Float>
        var remnant: SIMD4<Float>   // radius, visibility, kind, whether its well has formed
        var wave: SIMD4<Float>      // speed, window, samples, amplitude
        var flash: SIMD4<Float>     // merger flash, kilonova radius, kilonova brightness, well depth
        var masses: SIMD4<Float>    // mass shares of the two bodies, fade, time
    }

    private static let url = URL(string: "https://gwosc.org/eventapi/json/allevents/")!
    private static let alertURL = URL(string: "https://gracedb.ligo.org/api/superevents/?query=category%3A+Production&format=json&count=1")!
    private static let famous = ["GW150914", "GW170817", "GW190521", "GW250114", "GW190814", "GW231123", "GW200115", "GW230529",
                                 "GW151226", "GW170814", "GW190425"]
    /// Used until the catalogue arrives, or offline with nothing cached (GWTC-1 and GWTC-2.1 values).
    private static let remembered = [
        Merger(name: "GW150914", m1: 35.6, m2: 30.6, final: 63.1, distance: 440),
        Merger(name: "GW170817", m1: 1.46, m2: 1.27, final: 2.8, distance: 40),
        Merger(name: "GW190521", m1: 98.4, m2: 57.2, final: 147.4, distance: 3310),
    ]

    private let surface: MTLRenderPipelineState
    private let tracePipe: MTLRenderPipelineState
    private let tonePipe: MTLRenderPipelineState
    private let bloom: Bloom
    private let rings: [Ring]
    private var light: MTLTexture?

    private var events: [Merger] = remembered
    private var loaded = false
    private var offline = false
    private var alert: String?
    private var order: [Int] = []
    private var shown = 0
    private var current: Merger
    private var inspiral: Inspiral
    private var eventStart: Float = 0
    private var chirped = false
    private var merged = false
    private var captioned = false
    private let samples = 512
    private let window = 8.0

    var scale: Float { 0.75 }
    var ready: Bool { loaded }
    var facts: String? { describe() }

    init(gpu: GPU) throws {
        let library = try gpu.library(SpaceKit.metal + Self.source)
        surface = try gpu.pipeline(library, "spacetime")
        tracePipe = try gpu.pipeline(library, "lineFragment", vertex: "strainVertex", blend: .add)
        tonePipe = try gpu.pipeline(library, "gwTone")
        bloom = try Bloom(gpu: gpu)
        rings = (0..<3).map { _ in Ring(gpu: gpu, bytes: 8192) }
        current = Self.remembered[0]
        inspiral = Inspiral(m1: current.m1, m2: current.m2)
        Task { @MainActor in
            let catalogue = await Feed.object(Self.url, maxAge: 7 * 86400)
            let parsed = Self.parse(catalogue)
            if parsed.isEmpty { offline = true } else { events = parsed }
            order = []
            loaded = true
            next(at: eventStart)
            if let object = await Feed.object(Self.alertURL, maxAge: 6 * 3600) { alert = Self.parseAlert(object) }
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
            best[key] = (score, Merger(name: name, m1: max(m1, m2), m2: min(m1, m2), final: final, distance: distance))
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

    /// Famous events in turn, each followed by one picked at random from the rest of the catalogue.
    private func next(at time: Float) {
        if order.isEmpty {
            let famous = Self.famous.compactMap { name in events.firstIndex { $0.name == name } }
            var rest = Array(events.indices).filter { !famous.contains($0) }.shuffled()
            order = famous.flatMap { f -> [Int] in rest.isEmpty ? [f] : [f, rest.removeLast()] }
            if order.isEmpty { order = Array(events.indices) }
            shown = (Snapshot.variant ?? 0) % max(order.count, 1)
        } else {
            shown = (shown + 1) % order.count
        }
        current = events[order[shown]]
        inspiral = Inspiral(m1: current.m1, m2: current.m2)
        eventStart = time
        chirped = false
        merged = false
        captioned = false
    }

    private var radiated: Double? {
        guard let f = current.final else { return nil }
        let r = current.m1 + current.m2 - f
        return r >= 0.1 && current.m2 > 3 ? r : nil
    }

    private static func kind(_ m: Double) -> String { m < 3 ? "neutron star" : "black hole" }

    private static func mass(_ m: Double) -> String { SpaceKit.number(m, decimals: m < 3 ? 2 : 1) }

    private func describe() -> String {
        let c = current
        var text = "\(c.name): "
        if Self.kind(c.m1) == Self.kind(c.m2) {
            text += "\(Self.kind(c.m1))s of \(Self.mass(c.m1)) and \(Self.mass(c.m2)) solar masses"
        } else {
            text += "a \(Self.kind(c.m1)) of \(Self.mass(c.m1)) and a \(Self.kind(c.m2)) of \(Self.mass(c.m2)) solar masses"
        }
        let ly = c.distance * 3.2616e6
        let far = ly >= 1e9 ? "\(SpaceKit.number(ly / 1e9, decimals: 1)) billion" : "\(Int((ly / 1e6).rounded())) million"
        text += ", \(SpaceKit.number(c.distance)) megaparsecs, about \(far) light-years away"
        if let r = radiated { text += "; about \(SpaceKit.number(r, decimals: 1)) solar masses became gravitational waves" }
        text += ". Slowed \(SpaceKit.number(inspiral.stretch)) times."
        if offline { text += " Offline, so showing three famous events from memory." }
        if let alert, shown % 3 == 2 { text += " " + alert }
        return text
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        let total = Float(inspiral.merger + 12) + 1.5
        if f.time - eventStart > total { next(at: f.time) }
        let e = Double(f.time - eventStart)
        let s = e - 1.5
        let fade = Float(min(e / 1.5, 1, (Double(total) - e) / 1.5))
        let st = inspiral.state(s)

        // Bodies: positions about the centre of mass, sizes from mass (neutron stars are bigger than their mass suggests).
        let M = current.m1 + current.m2
        let rISCO = 0.16
        let r = rISCO * st.separation
        let dir = SIMD2(cos(st.phase), sin(st.phase))
        let p1 = dir * r * current.m2 / M, p2 = -dir * r * current.m1 / M
        func radius(_ m: Double) -> Float { m < 3 ? 0.034 : Float(0.068 * pow(m / M, 0.6)) }
        let gone = s >= inspiral.merger
        let neutron = current.m2 < 3
        let sinceMerger = Float(s - inspiral.merger)
        let flash = sinceMerger > -0.3 ? exp(-pow(sinceMerger / 0.45, 2)) : 0
        let remnant = Float(min(max((s - inspiral.merger + 0.15) / 0.3, 0), 1))
        let kilonova: SIMD2<Float> = neutron && sinceMerger > 0 ? SIMD2(min(0.12 + 0.11 * sinceMerger, 0.6), 1.4 * exp(-sinceMerger / 3.5)) : .zero
        let finalKind: Float = current.m1 < 3 && (current.final ?? M) < 3 ? 1 : 0
        let p = Params(
            body1: SIMD4(Float(p1.x), Float(p1.y), radius(current.m1), gone ? -1 : (current.m1 < 3 ? 1 : 0)),
            body2: SIMD4(Float(p2.x), Float(p2.y), radius(current.m2), gone ? -1 : (current.m2 < 3 ? 1 : 0)),
            remnant: SIMD4(current.m1 < 3 ? 0.042 : 0.075, remnant, finalKind, gone ? 1 : 0),
            wave: SIMD4(Float(3.6 / window), Float(window), Float(samples), 0.032),
            flash: SIMD4(flash, kilonova.x, kilonova.y, 0.2),
            masses: SIMD4(gone ? 0 : Float(current.m1 / M), gone ? 0 : Float(current.m2 / M), fade, f.time.truncatingRemainder(dividingBy: 1000)))

        // The wave's phase and amplitude over the last few seconds, read by the surface at each retarded time.
        var history: [SIMD2<Float>] = []
        history.reserveCapacity(samples)
        for i in 0..<samples {
            let h = inspiral.state(s - Double(i) * window / Double(samples - 1))
            history.append(SIMD2(Float(h.phase), Float(h.amplitude)))
        }

        // A slow orbit of the camera, leaning in a little at the merger.
        let t = Double(f.time) + Double(f.seed) * 40
        let yaw = t * 2 * .pi / 150
        let elevation = Astro.rad(30 + 8 * sin(t * 0.021))
        let distance = 2.0 - 0.2 * Double(flash)
        // The view's centre wanders a little, so the merger never lands on the same pixels for hours.
        let aim = SIMD3(0.12 * sin(t * 0.0029), 0.1 * sin(t * 0.0037 + 2), 0)
        let eye = aim + SIMD3(cos(elevation) * cos(yaw), cos(elevation) * sin(yaw), sin(elevation)) * distance
        let forward = simd_normalize(aim - eye)
        let right = simd_normalize(simd_cross(forward, SIMD3(0, 0, 1)))
        let up = simd_cross(right, forward)
        let f3 = { (v: SIMD3<Double>) in SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        let view = f.uniforms(SIMD4(f3(eye), 1.7), SIMD4(f3(right), 0), SIMD4(f3(up), 0), SIMD4(f3(forward), f.size.y / 750))

        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
        }
        let light = light!
        let params = rings[0].next(p)
        let hist = rings[1].next(history)
        f.cmd.draw(surface, into: light, uniforms: view, buffers: [params, hist])
        // The strain trace, newest on the right; it drifts a few pixels over the hours so it never burns in.
        let drift = Float(sin(Double(f.time) * 0.0013)) * 0.008
        f.cmd.draw(tracePipe, into: light, clear: false, uniforms: f.uniforms(SIMD4(fade, drift, 0, 0), view.b, view.c, view.d),
                   buffers: [hist], vertices: 4, instances: samples, primitive: .triangleStrip)
        let glow = bloom.glow(f.cmd, light)
        f.cmd.draw(tonePipe, into: f.target, uniforms: f.uniforms(), textures: [light, glow])

        if !captioned {
            captioned = true
            caption(f)
        }
        if let synth = f.sound { play(synth, s, st.amplitude) }
    }

    private func caption(_ f: Frame) {
        let c = current
        let both = "\(Self.mass(c.m1)) + \(Self.mass(c.m2)) solar masses"
        let result = c.final.map { " → \(SpaceKit.number($0, decimals: 1))" } ?? ""
        let ly = c.distance * 3.2616e6
        let far = ly >= 1e9 ? "\(SpaceKit.number(ly / 1e9, decimals: 1)) billion light-years" : "\(Int((ly / 1e6).rounded())) million light-years"
        let radiated = radiated
        let stretch = SpaceKit.number(inspiral.stretch)
        f.overlay.draw { ctx, size in
            ctx.text(c.name, at: CGPoint(x: 30, y: 26), size: 24, weight: .ultraLight, color: NSColor(white: 1, alpha: 0.8), tracking: 1.5)
            ctx.text(both + result, at: CGPoint(x: 31, y: 60), size: 12.5, weight: .regular, color: NSColor(white: 1, alpha: 0.55))
            ctx.text("\(SpaceKit.number(c.distance)) megaparsecs · \(far)", at: CGPoint(x: 31, y: 79), size: 12.5,
                     color: NSColor(white: 1, alpha: 0.42))
            if let radiated {
                ctx.text("\(SpaceKit.number(radiated, decimals: 1)) solar masses radiated as gravitational waves", at: CGPoint(x: 31, y: 98),
                         size: 12.5, color: NSColor(white: 1, alpha: 0.42))
            }
            ctx.text("STRAIN  ·  SLOWED \(stretch)×", at: CGPoint(x: size.width * 0.06, y: size.height * 0.815), size: 9.5,
                     weight: .medium, color: NSColor(red: 1, green: 0.7, blue: 0.4, alpha: 0.55), tracking: 1.6)
        }
    }

    // MARK: Sound

    /// The chirp from the real masses: its final pitch follows the frequency at the innermost stable orbit
    /// (about 4,400 Hz × M☉ / M), squeezed into a pleasant range so heavier mergers still sound lower.
    private func play(_ synth: Synth, _ s: Double, _ amplitude: Double) {
        let M = current.m1 + current.m2
        let end = Float(min(max(330 * pow(66 / M, 0.35), 160), 1100))
        if !chirped, s >= 0 {
            chirped = true
            let start = end * Float(pow(inspiral.tauISCO / inspiral.tauStart, 0.375))
            synth.play(.chirp, start, amp: 0.16 * min(1, sqrt(400 / end)), length: Float(inspiral.tauStart), to: end)
        }
        if !merged, s >= inspiral.merger {
            merged = true
            synth.play(.thump, 48, amp: 0.16, length: 2.5)
            synth.play(.bell, end * 1.5, amp: 0.05, length: 4)
        }
        let swell = Float(min(amplitude, 1.5)) * 0.02
        synth.drone(0, 55, 0.05 + swell)
        synth.drone(1, 82.4, 0.03 + swell)
        synth.drone(2, 110, 0.015)
        synth.noise(0.015, cutoff: 140)
    }

    // MARK: Shaders

    private static let source = """
    struct GW {
        float4 body1;    // xy, radius, kind (0 black hole, 1 neutron star, -1 gone)
        float4 body2;
        float4 remnant;  // radius, visibility, kind, whether its well has formed
        float4 wave;     // speed, window seconds, samples, amplitude
        float4 flash;    // merger flash, kilonova radius, kilonova brightness, well depth
        float4 masses;   // mass shares, fade, time
    };

    inline float2 waveAt(const device float2 *h, float r, constant GW &g) {
        float i = clamp(r / g.wave.x / g.wave.y * (g.wave.z - 1), 0.0, g.wave.z - 1.001);
        uint k = uint(i);
        return mix(h[k], h[k + 1], i - float(k));
    }

    // Height of the rubber sheet: a well under each mass, plus the outgoing wave h+ = A(t - r/c) cos(2φ - 2Φ(t - r/c)).
    inline float sheet(float2 q, constant GW &g, const device float2 *h, thread float &strain) {
        float depth = g.flash.w;
        float well = 0;
        well -= depth * g.masses.x / sqrt(1 + dot(q - g.body1.xy, q - g.body1.xy) / 0.03);
        well -= depth * g.masses.y / sqrt(1 + dot(q - g.body2.xy, q - g.body2.xy) / 0.03);
        well -= depth * g.remnant.w / sqrt(1 + dot(q, q) / 0.03);
        float r = length(q);
        float2 s = waveAt(h, r, g);
        float phi = atan2(q.y, q.x);
        strain = s.y * cos(2 * phi - 2 * s.x) * smoothstep(0.08, 0.35, r);
        return well + strain * g.wave.w * smoothstep(0.12, 0.7, r) / (0.4 + r) * smoothstep(4.2, 2.8, r);
    }

    // Bends a ray passing a black hole like a point lens, by θE²/θ toward it, so the grid behind is lensed into a ring.
    inline float3 bend(float3 ro, float3 rd, float3 c, float radius) {
        float3 oc = c - ro;
        float along = dot(oc, rd);
        if (along <= 0 || radius <= 0) return rd;
        float3 perp = oc - rd * along;
        float b = length(perp);
        float theta = b / along, edge = radius / along;
        if (theta < edge) return rd;   // into the shadow: nothing comes back
        float alpha = (1.4 * edge) * (1.4 * edge) / max(theta, edge * 0.7);
        return normalize(rd + perp / max(b, 1e-6) * alpha);
    }

    inline float3 bodyCenter(float4 b, constant GW &g, const device float2 *h) {
        float s;
        return float3(b.xy, sheet(b.xy, g, h, s) + b.z * 1.1);
    }

    // Lays a black hole (a shadow ringed by a thin photon ring) or a neutron star (hot and bright) over `col`,
    // unless the sheet is in front of it. `hitT` is how far along the ray the sheet is.
    inline float3 shadeBody(float3 col, float3 ro, float3 rd, float3 c, float radius, float kind, float hitT) {
        float3 oc = c - ro;
        float along = dot(oc, rd);
        if (along <= 0 || radius <= 0) return col;
        float b = length(oc - rd * along);
        float entry = along - sqrt(max(radius * radius - b * b, 0.0));
        bool visible = entry < hitT + radius;
        float px = 0.004 * length(oc);
        if (kind < 0.5) {
            float shadow = visible ? smoothstep(radius + px, radius - px, b) : 0.0;
            float ring = exp(-pow((b - radius * 1.08) / (radius * 0.04 + px), 2.0)) * 0.9 + exp(-max(b - radius, 0.0) / (radius * 0.5)) * 0.08;
            float shown = along - radius < hitT ? 1.0 : 0.25;
            return col * (1 - shadow) + float3(0.6, 0.75, 1.0) * ring * (1 - shadow) * shown;
        }
        float core = visible ? smoothstep(radius + px, radius - px, b) : 0.0;
        float mu = sqrt(max(1 - b * b / (radius * radius), 0.0));
        float3 hot = float3(0.62, 0.8, 1.0) * (0.9 + 0.6 * (1 - mu));
        float halo = exp(-max(b - radius, 0.0) / (radius * 0.7)) * 0.3 * (along - radius < hitT ? 1.0 : 0.3);
        return mix(col, hot, core) + float3(0.5, 0.7, 1.0) * halo * (1 - core);
    }

    fragment float4 spacetime(V in [[stage_in]], constant U &u [[buffer(0)]], constant GW &g [[buffer(1)]],
                              const device float2 *h [[buffer(2)]]) {
        float2 p = (in.uv * 2 - 1) * float2(u.res.x / u.res.y, -1);
        float3 ro = u.a.xyz;
        float3 rd = normalize(u.d.xyz + (p.x * u.b.xyz + p.y * u.c.xyz) / u.a.w);
        float3 seen = rd;

        float3 c1 = bodyCenter(g.body1, g, h), c2 = bodyCenter(g.body2, g, h);
        float3 cr = float3(0, 0, -g.flash.w + g.remnant.x * 1.1);
        if (g.body1.w == 0) rd = bend(ro, rd, c1, g.body1.z);
        if (g.body2.w == 0) rd = bend(ro, rd, c2, g.body2.z);
        if (g.remnant.z < 0.5) rd = bend(ro, rd, cr, g.remnant.x * g.remnant.y);

        // Background: a few faint stars.
        float3 col = float3(0.0015, 0.0018, 0.004);
        float2 cell = floor(float2(atan2(rd.y, rd.x), asin(rd.z)) * 260.0);
        float star = hash12(cell);
        col += step(0.9975, star) * float3(0.5, 0.6, 0.8) * hash12(cell + 7.1) * 0.25;

        // March to the displaced sheet: intersect the plane, then refine against the height a few times.
        float3 surf = 0;
        float hitT = 1e9;
        if (rd.z < -0.02) {
            float t = -ro.z / rd.z;
            float strain = 0;
            for (int i = 0; i < 6; i++) {
                float2 q = (ro + rd * t).xy;
                float z = sheet(q, g, h, strain);
                t = mix(t, (z - ro.z) / rd.z, 0.75);
            }
            float3 pos = ro + rd * t;
            float z = sheet(pos.xy, g, h, strain);
            hitT = t;
            float r = length(pos.xy);
            // Fine grid lines, antialiased, every fifth one stronger.
            float2 gq = pos.xy / 0.075;
            float2 fw = fwidth(gq);
            float2 lines = abs(fract(gq - 0.5) - 0.5) / max(fw, 1e-4);
            // Lines fade out where they crowd closer than a few pixels, so the far sheet never shimmers.
            float line = (1 - clamp(min(lines.x, lines.y) - 0.15, 0.0, 1.0)) * clamp(1.6 - max(fw.x, fw.y) * 4.0, 0.0, 1.0);
            float2 gq5 = pos.xy / 0.375;
            float2 fw5 = fwidth(gq5);
            float2 lines5 = abs(fract(gq5 - 0.5) - 0.5) / max(fw5, 1e-4);
            float major = (1 - clamp(min(lines5.x, lines5.y) - 0.3, 0.0, 1.0)) * clamp(1.6 - max(fw5.x, fw5.y) * 4.0, 0.0, 1.0);
            float fade = exp(-max(t - 1.5, 0.0) / 2.2) * smoothstep(4.4, 2.6, r);
            float lift = clamp(strain, -1.5, 1.5);
            float3 crest = mix(float3(0.22, 0.32, 1.0), float3(0.55, 0.9, 1.0), clamp(lift * 0.6 + 0.5, 0.0, 1.0));
            float3 lineColor = crest * (0.12 + 0.55 * abs(lift));
            float well = clamp(-z / g.flash.w, 0.0, 1.0);
            lineColor += float3(0.6, 0.4, 1.0) * well * 0.4;
            surf = lineColor * (line * 0.5 + major * 0.45);
            // The wave itself glows on the sheet: crests cyan, troughs violet, so the two-armed spiral shows.
            float near = smoothstep(4.5, 2.2, t);
            surf += (float3(0.05, 0.22, 0.4) * pow(max(lift, 0.0), 2.0) * 0.35 + float3(0.16, 0.05, 0.3) * pow(max(-lift, 0.0), 2.0) * 0.25) * near;
            surf += float3(0.006, 0.008, 0.02);
            // The merger flash lights the sheet near the centre.
            surf += float3(0.9, 0.85, 1.0) * g.flash.x * exp(-r * r / 0.18) * 0.6;
            surf *= fade;
            col = mix(col, surf, fade);
        }

        // The bodies, nearest last.
        bool firstNear = dot(c1 - ro, rd) < dot(c2 - ro, rd);
        float4 far = firstNear ? g.body2 : g.body1, near = firstNear ? g.body1 : g.body2;
        float3 cf = firstNear ? c2 : c1, cn = firstNear ? c1 : c2;
        if (far.w >= 0) col = shadeBody(col, ro, seen, cf, far.z, far.w, hitT);
        if (near.w >= 0) col = shadeBody(col, ro, seen, cn, near.z, near.w, hitT);
        if (g.remnant.y > 0) col = shadeBody(col, ro, seen, cr, g.remnant.x * g.remnant.y, g.remnant.z, hitT);

        // Merger flash and, for neutron stars, the kilonova's glowing red cloud.
        float3 oc = float3(0, 0, -g.flash.w * 0.5) - ro;
        float bc = length(oc - rd * dot(oc, rd));
        col += float3(0.85, 0.9, 1.0) * g.flash.x * (exp(-bc * bc / 0.003) * 1.6 + exp(-bc / 0.22) * 0.25);
        if (g.flash.z > 0) {
            float k = g.flash.y;
            float cloud = exp(-bc * bc / (k * k)) * (0.7 + 0.3 * fbm(rd.xy * 30.0 + g.masses.w * 0.05, 3));
            col += float3(1.0, 0.42, 0.18) * cloud * g.flash.z * 0.5;
        }
        return float4(col * g.masses.z, 1);
    }

    // The strain trace along the bottom, from the same history: buffer 1 holds (phase, amplitude) newest first.
    // Instance 511 draws the faint zero line.
    vertex LineOut strainVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                const device float2 *h [[buffer(1)]]) {
        float x0 = 0.06 * u.res.x, x1 = 0.94 * u.res.x;
        float y = (0.9 + u.a.y) * u.res.y, amp = 0.045 * u.res.y;
        float d = u.d.w;
        if (iid >= 511) {
            return lineQuad(float2(x0, y), float2(x1, y), 0.8 * d, float4(1.0, 0.6, 0.3, 0) * 0.05 * u.a.x, vid, u.res);
        }
        float2 a = h[iid], b = h[iid + 1];
        float ya = y - amp * a.y * cos(2 * a.x) / 1.4, yb = y - amp * b.y * cos(2 * b.x) / 1.4;
        float xa = mix(x1, x0, float(iid) / 511.0), xb = mix(x1, x0, float(iid + 1) / 511.0);
        float age = float(iid) / 511.0;
        float4 color = float4(1.0, 0.62, 0.28, 0) * (0.5 * (1 - age * 0.6)) * u.a.x;
        return lineQuad(float2(xa, ya), float2(xb, yb), 1.3 * d, color, vid, u.res);
    }

    fragment float4 gwTone(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                           texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = light.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.6;
        return float4(film(c), 1);
    }
    """
}
