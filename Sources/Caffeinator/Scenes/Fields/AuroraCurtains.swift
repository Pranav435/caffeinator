import Foundation
import Metal
import simd

/// The aurora from the shore of a still Arctic fjord, filmed like a ten-second exposure on a tripod, under the real sky.
/// The place is a real aurora spot where it's dark right now (Ersfjordbotn near Tromsø whenever it's night there), and
/// the stars are ESO's photograph of the Milky Way turned to that place's sidereal time, dimmed and reddened by the
/// air toward the horizon, with the Moon where it really is. The aurora follows NOAA's live planetary Kp index: the
/// auroral oval's edge moves toward the equator as Kp rises, so the curtains stand where the place's geomagnetic
/// latitude puts them. Each curtain is a thin sheet of light hanging along a folded east–west line; a pixel's ray is
/// marched to find where it crosses each sheet, coloured by altitude: pink nitrogen at the sharp lower edge near
/// 100 km, oxygen's 557.7 nm green above it, oxygen's 630 nm red high up. The mountains are ray-marched once per shot
/// into a G-buffer, then lit every frame by starlight, airglow, the aurora and the Moon, and mirrored in the water.
@MainActor final class AuroraCurtains: Scene {
    static let feed = URL(string: "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")!

    /// The latest planetary Kp and the start of its three-hour window.
    struct Reading: Equatable {
        let kp: Double
        let time: Date
    }

    /// A real place where aurora photographers stand, with the shape of the water and mountains in front of them.
    struct Site {
        let name: String
        let latitude: Double, longitude: Double
        let zone: String
        let water: String
        /// Half-width of the water and height of the mountains, and the distance to the far shore, in metres.
        let width: Float, peaks: Float, far: Float
        let seed: Float
    }

    /// Between them it's always properly dark (the Sun 12 degrees down or more) at one of these, all year round.
    static let sites = [
        Site(name: "Ersfjordbotn, near Tromsø in Norway", latitude: 69.68, longitude: 18.62, zone: "Europe/Oslo",
             water: "fjord", width: 900, peaks: 950, far: 13000, seed: 1.3),
        Site(name: "Resurrection Bay, at Seward in Alaska", latitude: 60.10, longitude: -149.44, zone: "America/Anchorage",
             water: "bay", width: 1700, peaks: 1250, far: 19000, seed: 4.1),
        Site(name: "the Gulf of Morbihan, Kerguelen Islands", latitude: -49.35, longitude: 70.22, zone: "Indian/Kerguelen",
             water: "gulf", width: 2600, peaks: 650, far: 24000, seed: 7.7),
        Site(name: "Lake Tekapo, New Zealand", latitude: -44.0, longitude: 170.48, zone: "Pacific/Auckland",
             water: "lake", width: 2400, peaks: 900, far: 26000, seed: 2.9),
        Site(name: "the Beagle Channel, at Ushuaia in Argentina", latitude: -54.80, longitude: -68.30, zone: "America/Argentina/Ushuaia",
             water: "channel", width: 2300, peaks: 1100, far: 21000, seed: 5.5),
    ]

    /// The camera, turned a little from the pole for each shot.
    static let shots: [Float] = [0, -14, 10, -6, 16, 4]
    static let shotLength: Float = 660

    /// Matches `struct Params` in the shader.
    struct Params {
        var camera = SIMD4<Float>()   // pitch, zoom, curtain turn, unused
        var east = SIMD4<Float>()     // the camera's right, forward and up, as equatorial directions (J2000)
        var ahead = SIMD4<Float>()
        var zenith = SIMD4<Float>()
        var sky = SIMD4<Float>()      // photo ready, unused, valley offset (metres), unused
        var sun = SIMD4<Float>()      // direction, altitude in degrees
        var moon = SIMD4<Float>()     // direction, brightness (0 once set)
        var profile = SIMD4<Float>()  // lower border (km), red, pink, green scale height (km)
        var aurora = SIMD4<Float>()   // activity, curtains, brightness, clock
        var ambient = SIMD4<Float>()  // light from the whole sky on the land
        var glow = SIMD4<Float>()     // aurora light on the land
        var glowDir = SIMD4<Float>()
        var moonLight = SIMD4<Float>()
        var land = SIMD4<Float>()     // half-width, peaks, far shore (metres), seed
        var temporal = SIMD4<Float>() // jitter xy, history blend, row offset
        var distances = SIMD4<Float>()
        var strengths = SIMD4<Float>()
    }

    private let table: MTLComputePipelineState
    private let landPass: MTLComputePipelineState
    private let skyPass: MTLRenderPipelineState
    private let compose: MTLRenderPipelineState
    private let lens: Lens
    private let curtains: MTLTexture
    private let placeholder: MTLTexture
    private var photo: MTLTexture?
    private var photoDone = false
    private var frames: [MTLTexture] = []
    private var light: MTLTexture?
    /// Two sets of land G-buffers: one on screen, one being built for the next shot.
    private var gbuffers: [[MTLTexture]] = []
    private var shown = 0
    private var built = 0
    private var building = -1
    private var buildRow = 0
    private var stale = true
    private let params: [MTLBuffer]

    private var reading: Reading?
    private var loaded = false
    private var fetched = Date.distantPast
    private var override: Double?
    private var level: Float = -1
    private var settled = false
    private var clock: Float = 0
    private var surgeStart: Float = -1000
    private var nextSurge: Float = 45
    private var brightness: Float = 0
    private var gust: Float = 0

    private var site: Int
    private var pendingSite: Int
    private var shot: Int
    private var shotStart: Float = 0
    private var dip: Float = 1
    private var checked = Date.distantPast
    private var moonUp: (alt: Double, az: Double, lit: Double) = (-90, 0, 0)
    private var localTime = ""
    private var described = Date.distantPast

    var scale: Float { 0.5 }
    var ready: Bool { loaded && photoDone }
    var credit: String? { photo == nil ? nil : MilkyWay.credit }
    var facts: String? {
        let s = Self.sites[site]
        let where_ = "\(s.name), \(localTime) local time"
        var moon = ""
        if moonUp.alt > 0 {
            moon = " The Moon, \(Int((moonUp.lit * 100).rounded())) percent lit, is up in the \(Self.compass(moonUp.az))."
        }
        let science = " Oxygen glows green at 557.7 nanometres 100 to 150 kilometres up and red at 630 nanometres above that; "
            + "nitrogen glows pink along the lower edge."
        if let override { return "Showing a Kp index of \(Self.format(override)), \(Self.describe(override)), over \(where_)." + moon + science }
        guard let reading else {
            return loaded ? "The Kp index is unavailable offline, so this is a quiet night over \(where_)." + moon + science
                          : "Fetching the planetary Kp index."
        }
        let clock = DateFormatter()
        clock.dateFormat = "HH:mm"
        clock.timeZone = TimeZone(identifier: "UTC")
        return "Planetary Kp index \(Self.format(reading.kp)), \(Self.describe(reading.kp)), for the three hours from "
            + "\(clock.string(from: reading.time)) UTC, over \(where_)." + moon + science
    }

    private var kp: Double { override ?? reading?.kp ?? 1.5 }

    init(gpu: GPU) throws {
        let library = try gpu.library(FieldKit.stars + Self.source)
        table = try gpu.kernel(library, "auroraTable")
        landPass = try gpu.kernel(library, "auroraLand")
        skyPass = try gpu.pipeline(library, "auroraSky")
        compose = try gpu.pipeline(library, "auroraCompose")
        lens = try Lens(gpu: gpu)
        curtains = gpu.texture(4096, 4)
        var black: UInt32 = 0
        placeholder = gpu.texture(1, 1, .rgba8Unorm, bytes: &black, bytesPerRow: 4)
        params = (0..<3).map { _ in gpu.buffer(MemoryLayout<Params>.stride) }
        // --variant n shows a night with Kp n % 10, at site n / 10 % 10, for testing.
        override = Snapshot.variant.map { Double(min(max($0 % 10, 0), 9)) }
        site = Snapshot.variant.map { min($0 % 100 / 10, Self.sites.count - 1) } ?? Self.pick(.now)
        pendingSite = site
        shot = Snapshot.variant == nil ? Int.random(in: 0..<Self.shots.count) : 0
        fetch()
        Task { @MainActor in
            photo = await MilkyWay.texture(gpu)
            photoDone = true
        }
    }

    /// Fetches the Kp index, at most every half hour (NOAA publishes a new value every three hours).
    private func fetch() {
        fetched = .now
        Task { @MainActor in
            if let json = await Feed.object(Self.feed, maxAge: 30 * 60), let latest = Self.parse(json) { reading = latest }
            loaded = true
        }
    }

    /// The last entry with a Kp value. The feed has been both an array of objects and an array of rows with a header,
    /// so both are accepted.
    nonisolated static func parse(_ json: Any) -> Reading? {
        guard let rows = json as? [Any] else { return nil }
        let stamp = DateFormatter()
        stamp.locale = Locale(identifier: "en_US_POSIX")
        stamp.timeZone = TimeZone(identifier: "UTC")
        for row in rows.reversed() {
            var kp: Double?
            var tag: String?
            if let o = row as? [String: Any] {
                kp = (o["Kp"] as? NSNumber)?.doubleValue ?? (o["Kp"] as? String).flatMap(Double.init)
                tag = o["time_tag"] as? String
            } else if let a = row as? [Any], a.count >= 2 {
                kp = (a[1] as? NSNumber)?.doubleValue ?? (a[1] as? String).flatMap(Double.init)
                tag = a[0] as? String
            }
            guard let kp, kp.isFinite, (0...9).contains(kp), let tag else { continue }
            for format in ["yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd HH:mm:ss.SSS", "yyyy-MM-dd HH:mm:ss"] {
                stamp.dateFormat = format
                if let time = stamp.date(from: tag) { return Reading(kp: kp, time: time) }
            }
        }
        return nil
    }

    nonisolated static func describe(_ kp: Double) -> String {
        switch kp {
        case ..<2: "a quiet night"
        case ..<4: "an unsettled night"
        case ..<5: "an active night"
        case ..<6: "a minor geomagnetic storm, G1"
        case ..<7: "a moderate geomagnetic storm, G2"
        case ..<8: "a strong geomagnetic storm, G3"
        case ..<9: "a severe geomagnetic storm, G4"
        default: "an extreme geomagnetic storm, G5"
        }
    }

    nonisolated static func format(_ kp: Double) -> String { String(format: "%.1f", kp) }

    /// How lively the aurora is, 0.12 for Kp 0 (still a soft arc: it's a screensaver) to 1 for Kp 7 and up.
    nonisolated static func activity(_ kp: Double) -> Float { Float(min(max((kp + 0.8) / 7.5, 0.2), 1)) }

    /// Geomagnetic latitude in degrees for a centred dipole with its north pole at 80.8° N, 72.7° W (IGRF, 2025).
    nonisolated static func geomagnetic(_ lat: Double, _ lon: Double) -> Double {
        let r = Double.pi / 180, pl = 80.8 * r, pn = -72.7 * r
        let c = sin(lat * r) * sin(pl) + cos(lat * r) * cos(pl) * cos(lon * r - pn)
        return 90 - acos(min(max(c, -1), 1)) / r
    }

    /// How far poleward of a place the bright arcs stand, in km: the oval's equatorward edge sits near
    /// 66.5° − 2.1° × Kp of geomagnetic latitude on the night side, and the brightest arcs a few degrees inside it.
    /// The camera faces them, so they're kept at least 200 km out, where they rise over the mountains.
    nonisolated static func arcDistance(kp: Double, latitude: Double, longitude: Double) -> Float {
        let edge = 66.5 - 2.1 * kp
        let km = (edge + 3 - abs(geomagnetic(latitude, longitude))) * 111
        return Float(min(max(km, 200), 520))
    }

    /// The best place to watch from now: the darkest-skied site with the highest geomagnetic latitude.
    static func pick(_ date: Date) -> Int {
        let alts = sites.map { Astro.horizontal(Astro.sun(date), date, latitude: $0.latitude, longitude: $0.longitude).alt }
        let dark = sites.indices.filter { alts[$0] < -12 }
        if let best = dark.max(by: { abs(geomagnetic(sites[$0].latitude, sites[$0].longitude)) < abs(geomagnetic(sites[$1].latitude, sites[$1].longitude)) }) {
            return best
        }
        return alts.indices.min { alts[$0] < alts[$1] } ?? 0
    }

    nonisolated static func compass(_ az: Double) -> String {
        ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][Int(((az + 22.5) / 45).rounded(.down)) % 8]
    }

    func draw(_ f: Frame) {
        let w = f.target.width, h = f.target.height
        if frames.first?.width != w || frames.first?.height != h {
            frames = [f.gpu.texture(w, h), f.gpu.texture(w, h)]
            light = f.gpu.texture(w, h)
            gbuffers = (0..<2).map { _ in (0..<4).map { _ in f.gpu.texture(w, h) } }
            building = -1
            built = 0
            stale = true
        }
        let t = f.time
        if f.now.timeIntervalSince(fetched) > 30 * 60 { fetch() }
        if f.now.timeIntervalSince(checked) > 60 {
            checked = f.now
            if Snapshot.variant == nil, Astro.horizontal(Astro.sun(f.now), f.now, latitude: Self.sites[site].latitude,
                                                         longitude: Self.sites[site].longitude).alt > -10 {
                pendingSite = Self.pick(f.now)
            }
        }

        // Activity eases to a new Kp over half a minute rather than jumping.
        let goal = Self.activity(kp)
        if level < 0 || loaded != settled {
            level = goal
            settled = loaded
        }
        level += (goal - level) * min(f.dt / 30, 1)
        let act = level

        // Brightness breathes over a minute and a half; substorms flare up every few minutes, more often and more
        // strongly on active nights.
        if t > nextSurge {
            surgeStart = t
            nextSurge = t + .random(in: 150...360) / (0.5 + act)
        }
        let since = t - surgeStart
        let flare = since < 20 ? smooth(0, 20, since) : exp(-(since - 20) / 60)
        brightness = (0.5 + 0.8 * act) * (0.8 + 0.2 * sin(t * 2 * .pi / 90 + f.seed)) * (1 + flare * act)

        // A new shot every eleven minutes, or a new place once this one's sky gets light: the next G-buffer is built
        // in strips over a couple of seconds, then the picture dips to dark and comes back on the new view.
        let wantsNew = (t - shotStart > Self.shotLength || pendingSite != site) && Snapshot.variant == nil
        if wantsNew, building < 0, built == 0 { building = 1 - shown; buildRow = 0 }
        if building < 0, built == 1 {
            dip = max(dip - f.dt / 2, 0)
            if dip == 0 {
                shown = 1 - shown
                built = 0
                site = pendingSite
                described = .distantPast
                shot = (shot + 1) % Self.shots.count
                shotStart = t
            }
        } else {
            dip = min(dip + f.dt / 2.5, 1)
        }

        let s = Self.sites[site]
        let buildSite = building >= 0 ? Self.sites[pendingSite] : s
        let buildShot = building >= 0 ? (shot + 1) % Self.shots.count : shot

        var p = parameters(f, act, s, shot)
        if f.now.timeIntervalSince(described) > 20 { describe(skyTime(f), s) }
        clock += f.dt * (0.4 + 1.3 * act)
        let slot = Int(f.index % 3)

        if stale {
            buildLand(f, set: shown, site: s, shot: shot, rows: 0..<h)
            stale = false
        }
        if building >= 0 {
            let rows = buildRow..<min(buildRow + max(h / 48, 8), h)
            buildLand(f, set: building, site: buildSite, shot: buildShot, rows: rows)
            buildRow = rows.upperBound
            if buildRow >= h { building = -1; built = 1 }
        }

        p.temporal = SIMD4(FieldKit.halton(Int(f.index) % 32 + 1, 2) - 0.5, FieldKit.halton(Int(f.index) % 32 + 1, 3) - 0.5,
                           f.index == 0 ? 1 : 1 - exp(-f.dt / 0.4), 0)
        params[slot].contents().storeBytes(of: p, as: Params.self)
        let u = f.uniforms([act, clock + f.seed * 50, 20 + 100 * act, 0])
        f.cmd.compute(table, 4096, 4, uniforms: u, textures: [curtains], buffers: [params[slot]])
        let now = frames[Int(f.index & 1)], past = frames[Int((f.index + 1) & 1)]
        f.cmd.draw(skyPass, into: now, uniforms: f.uniforms(), textures: [past, curtains, photo ?? placeholder], buffers: [params[slot]])
        let g = gbuffers[shown]
        f.cmd.draw(compose, into: light!, uniforms: f.uniforms(), textures: [now, g[0], g[1], g[2], g[3]], buffers: [params[slot]])
        let fade = dip * dip * (3 - 2 * dip)
        let moonDim = 1 / (1 + 0.035 * p.moonLight.y / max(p.ambient.y, 1e-4))
        lens.develop(f, light!, Lens.Look(exposure: 1.6 * fade * moonDim, white: [0.97, 1, 1.04], bloom: 0.035, halation: 0.025,
                                          vignette: 0.42, aberration: 1.3, grain: 0.75))

        if let synth = f.sound { play(synth, act, f) }
    }

    /// Everything the shaders need about the sky, the light and the land this frame.
    private func parameters(_ f: Frame, _ act: Float, _ s: Site, _ shot: Int) -> Params {
        var p = Params()
        let north = s.latitude >= 0
        let heading = (north ? 0 : 180) + Double(Self.shots[shot])
        let zoom: Float = 1.7
        let pitch = atan(0.55 / zoom)
        p.camera = SIMD4(pitch, zoom, Self.shots[shot] * .pi / 180, 0)

        // The camera's axes as equatorial directions at this moment: the stars turn with the real sky.
        let date = skyTime(f)
        let lst = Astro.lst(date, longitude: s.longitude) * .pi / 180, lat = s.latitude * .pi / 180
        let E = SIMD3(-sin(lst), cos(lst), 0)
        let N = SIMD3(-sin(lat) * cos(lst), -sin(lat) * sin(lst), cos(lat))
        let Z = SIMD3(cos(lat) * cos(lst), cos(lat) * sin(lst), sin(lat))
        let hr = heading * .pi / 180
        let right = E * cos(hr) - N * sin(hr), ahead = E * sin(hr) + N * cos(hr)
        p.east = SIMD4(SIMD3<Float>(right), 0)
        p.ahead = SIMD4(SIMD3<Float>(ahead), 0)
        p.zenith = SIMD4(SIMD3<Float>(Z), 0)
        p.sky = SIMD4(photo == nil ? 0 : 1, 1, 0, 0)

        func local(_ alt: Double, _ az: Double) -> SIMD3<Float> {
            let a = alt * .pi / 180, z = (az - heading) * .pi / 180
            return SIMD3(Float(cos(a) * sin(z)), Float(sin(a)), Float(-cos(a) * cos(z)))
        }
        let sun = Astro.horizontal(Astro.sun(date), date, latitude: s.latitude, longitude: s.longitude)
        p.sun = SIMD4(local(sun.alt, sun.az), Float(sun.alt))
        let moon = Astro.horizontal(Astro.moon(date), date, latitude: s.latitude, longitude: s.longitude)
        let lit = Astro.moonPhase(date).lit
        // Moonlight goes roughly as the lit fraction to the power 1.5 (the full Moon's surge), and fades as it sets.
        let moonGain = Float(pow(lit, 1.5)) * smooth(-1, 6, Float(moon.alt))
        p.moon = SIMD4(local(moon.alt, moon.az), moonGain)

        // Curtains stand where the oval is for this place and Kp; on quiet nights at low geomagnetic latitudes they sit
        // beyond the horizon and only their red tops show.
        let z0 = -Self.arcDistance(kp: Double(kp), latitude: s.latitude, longitude: s.longitude)
        p.distances = SIMD4(z0, z0 - 80 - 60 * act, z0 - 230, z0 + 70)
        p.strengths = SIMD4<Float>(1, 0.55, 0.25 + 0.35 * act, 0.75 * smooth(0.35, 0.8, act)) * brightness
        p.profile = SIMD4(105 - 12 * act, 0.04 + 0.42 * pow(act, 2.5), 0.1 * act + 0.6 * act * act, 28 + 22 * act)
        p.aurora = SIMD4(act, act > 0.35 ? 4 : 3, brightness, clock)

        // Light on the land: the whole night sky, the aurora from the direction of its main arc, and the Moon.
        let d = -z0
        let arcElevation = atan((110 - d * d / 12742) / d)
        let seen = smooth(-0.05, 0.2, arcElevation)
        p.ambient = SIMD4(SIMD3<Float>(0.020, 0.025, 0.036), 0)
        p.glow = SIMD4(SIMD3<Float>(0.20, 1.0, 0.42) * 0.06 * brightness * seen, 0)
        p.glowDir = SIMD4(simd_normalize(SIMD3<Float>(0, max(sin(arcElevation), 0.08), -1)), 0)
        p.moonLight = SIMD4(SIMD3<Float>(0.82, 0.92, 1.0) * 0.3 * moonGain, 0)
        p.land = SIMD4(s.width, s.peaks, s.far, s.seed + Float(shot) * 0.37)
        // Off to one side of the water, a different side each shot, so the mountains frame it unevenly.
        p.sky.z = s.width * (shot % 2 == 0 ? 0.55 : -0.45)
        return p
    }

    /// The moment the sky shows: now, or for `--variant` tests, 11 pm local solar time today at the site, plus the
    /// variant's hundreds digit in hours and its thousands digit in weeks.
    private func skyTime(_ f: Frame) -> Date {
        guard let v = Snapshot.variant else { return f.now }
        let solar = (f.now.timeIntervalSince1970 / 3600 + Self.sites[site].longitude / 15).truncatingRemainder(dividingBy: 24)
        let want = 23 + Double(v / 100 % 10)
        return f.now + (want - solar) * 3600 + Double(v / 1000) * 7 * 86400
    }

    /// The local time and the Moon for the facts.
    private func describe(_ now: Date, _ s: Site) {
        described = .now
        let moon = Astro.horizontal(Astro.moon(now), now, latitude: s.latitude, longitude: s.longitude)
        moonUp = (moon.alt, moon.az, Astro.moonPhase(now).lit)
        let clock = DateFormatter()
        clock.dateFormat = "HH:mm"
        clock.timeZone = TimeZone(identifier: s.zone) ?? .gmt
        localTime = clock.string(from: now)
    }

    /// Ray-marches the mountains and water for some rows of a G-buffer set.
    private func buildLand(_ f: Frame, set: Int, site s: Site, shot: Int, rows: Range<Int>) {
        var p = parameters(f, level, s, shot)
        p.temporal.w = Float(rows.lowerBound)
        let buffer = f.gpu.buffer(MemoryLayout<Params>.stride)
        buffer.contents().storeBytes(of: p, as: Params.self)
        f.cmd.compute(landPass, f.target.width, rows.count, uniforms: f.uniforms(), textures: gbuffers[set], buffers: [buffer])
    }

    private func smooth(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let t = min(max((x - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// Wind over the water, a low pad, and a high shimmer of slightly detuned pairs that swells with the aurora.
    private func play(_ synth: Synth, _ act: Float, _ f: Frame) {
        gust += (Float.random(in: 0...1) - gust) * min(f.dt / 6, 1)
        let swell = min(brightness * (0.4 + 0.6 * act), 1.4)
        synth.noise(0.05 + 0.035 * gust, cutoff: 320 + 380 * gust)
        synth.drone(0, 110, 0.035, pan: -0.2)
        synth.drone(1, 164.8, 0.02, pan: 0.2)
        synth.drone(2, 880, 0.006 + 0.02 * swell, pan: -0.5)
        synth.drone(3, 882.4, 0.006 + 0.02 * swell, pan: 0.5)
        synth.drone(4, 1318.5, 0.003 + 0.014 * swell, pan: -0.3)
        synth.drone(5, 1321.6, 0.003 + 0.014 * swell, pan: 0.3)
        synth.drone(6, 1760 * (1 + 0.002 * sin(f.time * 0.7)), 0.008 * swell * act, pan: 0)
    }

    private static let source = """
    constant float EARTH = 6371.0;

    struct Params {
        float4 camera;   // pitch, zoom, curtain turn
        float4 east, ahead, zenith;
        float4 sky;      // photo ready, unused, valley offset (metres)
        float4 sun;      // direction, altitude (degrees)
        float4 moon;     // direction, brightness
        float4 profile;  // lower border (km), red, pink, green scale height (km)
        float4 aurora;   // activity, curtains, brightness, clock
        float4 ambient, glow, glowDir, moonLight;
        float4 land;     // half-width, peaks, far shore (metres), seed
        float4 temporal; // jitter, history blend, first row
        float4 distances, strengths;
    };

    // The camera looks down -z, tilted up by `pitch`, with x to the right.
    inline float3 camRay(float2 p, float pitch, float zoom) {
        float cp = cos(pitch), sp = sin(pitch);
        return normalize(float3(0, sp, -cp) * zoom + float3(p.x, 0, 0) + p.y * float3(0, cp, sp));
    }
    inline float2 camUV(float3 d, float pitch, float zoom, float2 res) {
        float cp = cos(pitch), sp = sin(pitch);
        float3 f = float3(0, sp, -cp), up = float3(0, cp, sp);
        float2 p = float2(d.x, dot(d, up)) * zoom / max(dot(d, f), 1e-3);
        return float2((p.x * res.y / res.x + 1) * 0.5, (1 - p.y) * 0.5);
    }

    // MARK: The aurora

    // The curtain table: one row per curtain, 4096 samples from 1600 km west to 1600 km east of the camera, each holding
    // the curtain's distance poleward (km, negative), its brightness there, and its fine vertical rays, which a
    // ten-second exposure softens. u.a: activity, clock, fold amplitude (km).
    kernel void auroraTable(constant U &u [[buffer(0)]], constant Params &P [[buffer(1)]],
                            texture2d<float, access::write> out [[texture(0)]], uint2 g [[thread_position_in_grid]]) {
        float x = (float(g.x) + 0.5) / 4096 * 3200 - 1600;
        int k = int(g.y);
        float t = u.a.y, act = u.a.x;
        float meander = (fbm(float2(x / 450 + k * 7.1, t * 0.004 + k * 3.3), 3) - 0.5) * 2;
        float wobble = noise(float2(x / 110 + k * 1.7, t * 0.02 + k));
        float fold = sin(x / 48 + t * 0.05 + 3 * wobble) * smoothstep(0.2, 0.7, noise(float2(x / 260 + 3 + k, t * 0.006)));
        float z = P.distances[k] + meander * (60 + 140 * act) + fold * u.a.z;
        float patch = smoothstep(0.12, 0.75, fbm(float2(x / 260 + k * 4.7 + t * 0.01, t * 0.005 + k), 4));
        patch *= 0.55 + 0.9 * smoothstep(0.3, 0.8, noise(float2(x / 45 + k * 2.3, t * 0.02)));
        float rays = noise(float2(x / 3.5 + t * 0.08, k * 10 + t * 0.01)) * 0.4 + noise(float2(x / 11 - t * 0.03, k * 3 + t * 0.008)) * 0.6;
        rays = mix(1.0, 0.3 + 1.6 * rays * rays, 0.2 + 0.4 * act);
        float ends = smoothstep(1600.0, 1100.0, abs(x));
        out.write(float4(z, patch * P.strengths[k] * ends, rays, 0), g);
    }

    // Light emitted per unit of a sheet at altitude h (km): pink nitrogen at the sharp lower border, oxygen's green
    // (557.7 nm) above it, oxygen's red (630 nm) high up, as a camera renders them.
    inline float3 emission(float h, constant Params &P) {
        float above = h - P.profile.x;
        float base = smoothstep(-2.0, 0.8, above);
        float green = base * (0.7 * exp(-max(above, 0.0) / 6) + 0.3 * exp(-max(above, 0.0) / P.profile.w));
        float red = smoothstep(P.profile.x + 60, P.profile.x + 140, h) * exp(-max(h - P.profile.x - 140, 0.0) / 90) * P.profile.y;
        float pink = exp(-(above + 1.0) * (above + 1.0) / 6) * P.profile.z;
        return float3(0.16, 1.0, 0.34) * green + float3(1.0, 0.03, 0.09) * red * 0.8 + float3(1.0, 0.3, 0.72) * pink * 0.9;
    }

    // Marches the ray between 80 and 420 km altitude (the Earth curving away below it), looks for each place it
    // crosses a curtain, and adds the sheet's light there. A sheet seen edge-on is brighter, as real folds are.
    inline float3 aurora(float3 d, float jitter, constant Params &P, texture2d<float> table) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float dy = d.y;
        float c = (1 - dy * dy) / (2 * EARTH);
        float s0 = (-dy + sqrt(dy * dy + 4 * c * 80)) / (2 * c);
        float s1 = min((-dy + sqrt(dy * dy + 4 * c * 420)) / (2 * c), 2600.0);
        const int N = 24;
        float ds = (s1 - s0) / N;
        float3 acc = 0;
        for (int k = 0; k < 4; k++) {
            if (P.aurora.y < k + 0.5) break;
            float v = (k + 0.5) / 4;
            float sp = s0;
            float4 cp = table.sample(s, float2((sp * d.x + 1600) / 3200, v));
            float gp = sp * d.z - cp.x;
            for (int i = 1; i <= N + 1; i++) {
                float sc = min(s0 + (i - 1 + jitter) * ds, s1);
                float4 cc = table.sample(s, float2((sc * d.x + 1600) / 3200, v));
                float gc = sc * d.z - cc.x;
                // The diffuse glow that surrounds every arc, tens of kilometres thick.
                float hc = sc * dy + c * sc * sc;
                acc += emission(hc, P) * cc.y * exp(-gc * gc / 1800) * ds * 0.004;
                if (gp * gc <= 0 && gp != gc) {
                    float t = gp / (gp - gc);
                    float sm = mix(sp, sc, t);
                    float gm = sm * d.z - table.sample(s, float2((sm * d.x + 1600) / 3200, v)).x;
                    float lo = gp - gm, hi = gm - gc;
                    t = gp * gm < 0 ? t * gp / (abs(lo) < 1e-5 ? 1e-5 : lo) : t + (1 - t) * gm / (abs(hi) < 1e-5 ? 1e-5 : hi);
                    t = saturate(t);
                    float sx = mix(sp, sc, t);
                    float h = sx * dy + c * sx * sx;
                    float slope = abs(gc - gp) / ds;
                    float4 at = table.sample(s, float2((sx * d.x + 1600) / 3200, v));
                    acc += emission(h, P) * at.y * at.z / max(slope, 0.45);
                }
                sp = sc; cp = cc; gp = gc;
            }
        }
        return acc;
    }

    // MARK: The sky

    // The night sky toward local direction d: the Milky Way photograph turned to the place's sidereal time, airglow
    // brightening toward the horizon (van Rhijn), all of it dimmed and reddened by the air it comes through, a trace
    // of twilight, and the Moon with the blue sky and halo its light scatters.
    inline float3 nightSky(float3 d, constant Params &P, texture2d<float> photo, float fp) {
        float s = max(d.y, 0.0);
        float deg = asin(s) * 57.29578;
        float airmass = 1 / (s + 0.50572 * pow(deg + 6.07995, -1.6364));
        float3 through = exp(-float3(0.11, 0.17, 0.30) * airmass);
        float3 eq = P.east.xyz * d.x + P.ahead.xyz * (-d.z) + P.zenith.xyz * d.y;
        float3 stars = P.sky.x > 0.5 ? milkyWay(photo, eq, fp) * 0.6
                                     : starLayer(eq, 420, 0.05, fp, 0.35, 2.3, 0.0022) + starLayer(eq, 110, 0.035, fp, 3.0, 5.7, 0.0022);
        float vr = rsqrt(1 - 0.9705 * (1 - s * s));
        float3 airglow = float3(0.0016, 0.0030, 0.0019) * vr;
        float3 col = (stars + airglow + float3(0.0008, 0.0011, 0.0022)) * through;
        // Light scattered back in near the horizon, so it never goes black, and the aurora's own glow in the air.
        col += float3(0.0030, 0.0036, 0.0040) * (1 - through.g);
        col += P.glow.rgb * 0.06 * exp(-deg / 14);
        float tw = smoothstep(-19.0, -9.0, P.sun.w);
        float3 sh = normalize(float3(P.sun.x, 0, P.sun.z) + 1e-5);
        float toward = max(dot(normalize(float3(d.x, 0, d.z) + 1e-5), sh), 0.0);
        col += tw * tw * (float3(0.002, 0.004, 0.010) * exp(-deg / 20) + float3(0.012, 0.007, 0.003) * toward * toward * exp(-deg / 5));
        if (P.moon.w > 0) {
            float cg = dot(d, P.moon.xyz);
            float3 blue = float3(0.25, 0.5, 1.0) * (0.75 + 0.25 * cg * cg) * (0.45 + 0.55 * vr) * 0.07;
            float3 halo = float3(1.0, 0.94, 0.85) * 0.00012 / max(1.0001 - cg, 0.00015);
            col += P.moon.w * (blue + halo) * mix(float3(1), through, 0.5);
            float r = acos(clamp(cg, -1.0, 1.0));
            if (r < 0.0048) {
                float3 q = d - P.moon.xyz * cg;
                float rho = min(r / 0.0045, 1.0);
                float3 n = normalize(q / max(length(q), 1e-6) * rho - P.moon.xyz * sqrt(1 - rho * rho));
                float day = smoothstep(-0.05, 0.08, dot(n, P.sun.xyz));
                float edge = smoothstep(0.0048, 0.0042, r);
                col += (day * 30 + 0.02) * edge * through;
            }
        }
        return col;
    }

    // The sky and aurora above the horizon, accumulated over frames with sub-pixel jitter, which smooths the march.
    fragment float4 auroraSky(V in [[stage_in]], constant U &u [[buffer(0)]], constant Params &P [[buffer(1)]],
                              texture2d<float> past [[texture(0)]], texture2d<float> table [[texture(1)]],
                              texture2d<float> photo [[texture(2)]]) {
        float2 p = centered(in.uv + P.temporal.xy / u.res, u.res);
        float3 d = camRay(p, P.camera.x, P.camera.y);
        float3 col = 0;
        if (d.y > -0.04) {
            float fp = 2 / (P.camera.y * u.res.y);
            float3 dd = float3(d.x, max(d.y, 0.0), d.z);
            col = nightSky(normalize(dd), P, photo, fp);
            float ct = cos(P.camera.z), st = sin(P.camera.z);
            float3 da = float3(ct * dd.x - st * dd.z, dd.y, st * dd.x + ct * dd.z);
            float jit = 0.02 + 0.96 * hash12(in.pos.xy + float2(u.frame % 64, u.frame % 61) * 17.3);
            // A ten-second exposure softens the curtains: each frame looks along a slightly different line.
            float2 wobble = (hash22(in.pos.xy * 1.3 + float2(u.frame % 59, u.frame % 53) * 7.1) - 0.5) * 3.5;
            float3 dw = camRay(centered(in.uv + (P.temporal.xy + wobble) / u.res, u.res), P.camera.x, P.camera.y);
            dd = float3(dw.x, max(dw.y, 0.0), dw.z);
            float s = max(dd.y, 0.0), deg = asin(s) * 57.29578;
            float3 through = exp(-float3(0.11, 0.17, 0.30) / (s + 0.50572 * pow(deg + 6.07995, -1.6364)));
            col += aurora(normalize(da), jit, P, table) * 0.55 * through;
        }
        float3 prev = past.read(uint2(in.pos.xy)).rgb;
        return float4(mix(prev, col, P.temporal.z), 1);
    }

    // MARK: The land

    // Height above the water in metres at q = (x, z): a fjord winding away down -z between alpine ridges, with a far
    // shore beyond open water and a stony beach under the camera.
    inline float ridges(float2 q, float seed, int oct) {
        float2 s = rot(seed) * q / 3000 + seed * 7.1;
        float m = 0, a = 0.55, w = 1;
        for (int i = 0; i < oct; i++) {
            float n = 1 - abs(noise(s) * 2 - 1);
            n *= n;
            m += a * n * w;
            w = clamp(n * 1.8, 0.0, 1.0);
            s = rot(0.57) * s * 2.03 + 5.3;
            a *= 0.47;
        }
        return m;
    }
    inline float valleyCentre(float z, float hw, float seed) {
        return 0.35 * hw * (sin(z / 5300 + seed) - sin(seed)) + 0.2 * hw * (sin(z / 2300 + 2.7 * seed) - sin(2.7 * seed));
    }
    inline float landHeight(float2 q, constant Params &P, int oct) {
        float hw = P.land.x, peak = P.land.y, far = P.land.z, seed = P.land.w;
        float c = valleyCentre(q.y, hw, seed) + P.sky.z;
        float w = hw * (1 + 0.3 * sin(q.y / 3900 + 3.1 * seed));
        float v = abs(q.x - c) / w;
        float side = smoothstep(0.7, 2.6, v);
        float across = smoothstep(far, far * 1.25, -q.y) * smoothstep(0.2, 0.45, noise(float2(q.x / 9000, seed * 3.1)));
        float wall = max(side, across);
        float h = -35.0;
        if (wall > 0) {
            float m = ridges(q, seed, oct);
            // Scree and crags on every slope, not just the ridgelines.
            float2 r = q / 700 + seed * 3.7;
            float rough = 0, a = 0.5;
            for (int i = 0; i < oct - 2; i++) { rough += a * (noise(r) - 0.5); r = rot(0.9) * r * 2.1 + 1.7; a *= 0.5; }
            float lift = 0.5 + 0.5 * smoothstep(0.7, 4.5, v);
            h = mix(-35.0, 25 + peak * (m * 1.35 * lift + 0.12 * rough * smoothstep(0.7, 1.6, v)), wall);
        }
        // Boulders standing out of the water in front of the camera, each a lumpy dome with its own outline.
        if (q.y > -80 && q.y < -6) {
            float2 cell = floor(q / 7);
            for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
                float2 id = cell + float2(i, j);
                float4 r = float4(hash33(float3(id, seed)), hash12(id * 1.7 + seed));
                if (r.w > 0.38) continue;
                float2 at = (id + 0.2 + 0.6 * r.xy) * 7;
                if (at.y > -9) continue;
                float rad = 0.8 + 2.2 * r.z * r.z;
                float2 dq = q - at;
                float ang = atan2(dq.y, dq.x);
                float outline = rad * (0.8 + 0.25 * sin(ang * 2 + r.x * 9) + 0.12 * sin(ang * 5 + r.y * 7));
                float dd = length(dq) / outline;
                if (dd < 1) h = max(h, -0.8 + (0.6 + 1.1 * r.z) * pow(1 - dd * dd, 0.55) + 0.18 * noise(q * 1.7 + r.xy * 20));
            }
        }
        // The ledge the tripod stands on.
        if (q.y > -3) h = max(h, 1.4 + 0.3 * noise(q * 0.7) - smoothstep(-0.5, -3.0, q.y) * 3);
        return h;
    }
    inline int octavesFor(float footprint) {
        return clamp(int(log2(3000 / max(2 * footprint, 0.5)) / 1.02), 2, 11);
    }

    // Distance along the ray to the land, or -1 if it reaches tmax first.
    inline float marchLand(float3 ro, float3 rd, constant Params &P, float tmax, float pixel) {
        float t = 0.2, lastT = t;
        float top = P.land.y * 1.9 + 60;
        for (int i = 0; i < 360; i++) {
            float3 q = ro + rd * t;
            if (q.y > top && rd.y >= 0) return -1;
            float h = landHeight(q.xz, P, min(octavesFor(t * pixel), 7));
            float gap = q.y - h;
            if (gap < 0) {
                float a = lastT, b = t;
                for (int k = 0; k < 7; k++) {
                    float m = 0.5 * (a + b);
                    float3 qm = ro + rd * m;
                    if (qm.y < landHeight(qm.xz, P, octavesFor(m * pixel))) b = m; else a = m;
                }
                return b;
            }
            lastT = t;
            t += max(gap * 0.4, 0.0015 * t + 0.02);
            if (t > tmax) break;
        }
        return -1;
    }

    struct Surface { float3 n; float albedo; float shadow; };

    // The shape and snow cover of the land where a ray hit it, and whether the Moon can see it.
    inline float3 landNormal(float2 q, float e, int oct, constant Params &P) {
        float hx = landHeight(q + float2(e, 0), P, oct) - landHeight(q - float2(e, 0), P, oct);
        float hz = landHeight(q + float2(0, e), P, oct) - landHeight(q - float2(0, e), P, oct);
        return normalize(float3(-hx, 2 * e, -hz));
    }
    // A slope seen at a grazing angle squeezes a lot of ground into a pixel, so detail finer than that stretched
    // footprint is left out of the normal; otherwise it aliases into streaks.
    inline Surface surface(float3 q, float3 d, float t, constant Params &P, float pixel) {
        float e = max(t * pixel, 0.04);
        float3 rough = landNormal(q.xz, e * 4, max(octavesFor(t * pixel) - 3, 2), P);
        e /= max(abs(dot(rough, d)), 0.12);
        int oct = octavesFor(e);
        Surface s;
        s.n = landNormal(q.xz, e, oct, P);
        float slope = 1 - s.n.y;
        // Snow everywhere but the steepest cliffs; birch scrub darkens the lower slopes; wet stones at the waterline.
        // Rock shows through on steep ground, along bands of strata and in wind-scoured patches; seen at a grazing
        // angle a texture stretches, so none is finer than a few footprints.
        float fine = max(e * 6, 1.0);
        float strata = noise(float2(q.y / 22, (q.x + q.z) / 400 + P.land.w));
        float bare = slope + 0.3 * (fbm(q.xz / max(90.0, fine), 3) - 0.5) + 0.22 * (strata - 0.5) + 0.12 * (noise(q.xz / max(18.0, fine)) - 0.5);
        float snow = smoothstep(0.42, 0.26, bare + 0.08 * (noise(q.xz / max(6.0, fine)) - 0.5));
        float scrub = smoothstep(380.0, 120.0, q.y + 120 * noise(q.xz / 400)) * smoothstep(3.0, 12.0, q.y) * smoothstep(0.55, 0.3, slope);
        float twigs = scrub * (0.3 + 0.35 * noise(q.xz / max(fine, 6.0)));
        // Below the high-tide line the stones are wet, dark and bare.
        float wet = smoothstep(2.6, 1.2, q.y + 0.8 * noise(q.xz / max(fine, 4.0)));
        // Wind-packed snow varies from drift to drift and streaks along the wind; rock is dark and mottled with scree.
        float drift = 0.45 + 0.4 * fbm(q.xz / max(40.0, fine), 4) + 0.25 * (noise(float2(q.x / max(5.0, fine), q.z / max(34.0, fine))) - 0.5);
        float rock = 0.03 + 0.09 * noise(q.xz / max(2.5, fine)) * (0.5 + noise(q.xz / max(14.0, fine)));
        float albedo = mix(mix(rock, 0.62 * drift, snow) * (1 - twigs), 0.03, wet);
        s.albedo = albedo;
        s.shadow = 1;
        if (P.moon.w > 0 && P.moon.y > 0) {
            float3 r = P.moon.xyz;
            float u = 2;
            for (int i = 0; i < 48; i++) {
                float3 p = q + s.n * 1.0 + r * u;
                float gap = p.y - landHeight(p.xz, P, 5);
                s.shadow = min(s.shadow, clamp(gap / (u * 0.02) + 0.5, 0.0, 1.0));
                if (s.shadow <= 0 || p.y > P.land.y * 1.9) break;
                u += max(gap * 0.5, u * 0.05 + 1);
            }
        }
        return s;
    }

    // The land G-buffer, four samples a pixel: g0 = (normal, land coverage), g1 = (distance km, albedo, moon shadow,
    // water coverage), and for the water the same again along the reflected ray: r0, r1 (distance to the water in km).
    kernel void auroraLand(constant U &u [[buffer(0)]], constant Params &P [[buffer(1)]],
                           texture2d<float, access::write> g0 [[texture(0)]], texture2d<float, access::write> g1 [[texture(1)]],
                           texture2d<float, access::write> r0 [[texture(2)]], texture2d<float, access::write> r1 [[texture(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
        uint2 px = uint2(gid.x, gid.y + uint(P.temporal.w));
        if (px.y >= uint(u.res.y)) return;
        float pixel = 2 / (P.camera.y * u.res.y) * 0.5;
        float3 ro = float3(0, 2.4, 0);
        float4 a0 = 0, a1 = 0, b0 = 0, b1 = 0;
        float land = 0, water = 0, rland = 0;
        const float2 offs[4] = { float2(0.125, 0.375), float2(0.625, 0.125), float2(0.875, 0.625), float2(0.375, 0.875) };
        for (int k = 0; k < 4; k++) {
            float2 uv = (float2(px) + offs[k]) / u.res;
            float3 d = camRay(centered(uv, u.res), P.camera.x, P.camera.y);
            float tw = d.y < 0 ? -ro.y / d.y : 1e9;
            float t = marchLand(ro, d, P, min(tw, 90000.0), pixel);
            if (t > 0) {
                Surface s = surface(ro + d * t, d, t, P, pixel);
                a0.xyz += s.n; a1.x += t / 1000; a1.y += s.albedo; a1.z += s.shadow;
                land += 1;
            } else if (d.y < 0) {
                water += 1;
                b1.w += tw / 1000;
                float3 q = ro + d * tw;
                float3 r = float3(d.x, -d.y, d.z);
                float tr = marchLand(q + r * 0.05, r, P, 90000.0, pixel);
                if (tr > 0) {
                    Surface s = surface(q + r * tr, r, tw + tr, P, pixel);
                    b0.xyz += s.n; b1.x += (tw + tr) / 1000; b1.y += s.albedo; b1.z += s.shadow;
                    rland += 1;
                }
            }
        }
        if (land > 0) { a0.xyz = normalize(a0.xyz); a1.xyz /= land; }
        if (rland > 0) { b0.xyz = normalize(b0.xyz); b1.xyz /= rland; }
        if (water > 0) b1.w /= water;
        a0.w = land / 4; a1.w = water / 4; b0.w = water > 0 ? rland / water : 0;
        g0.write(a0, px); g1.write(a1, px); r0.write(b0, px); r1.write(b1, px);
    }

    // Snow and rock lit by the sky, the aurora and the Moon, seen through `km` of air whose own glow is `air`.
    inline float3 litLand(float3 n, float km, float albedo, float shadow, float3 air, constant Params &P) {
        float3 E = P.ambient.rgb * (0.55 + 0.45 * n.y)
                 + P.glow.rgb * (0.3 + 0.7 * max(dot(n, P.glowDir.xyz), 0.0))
                 + P.moonLight.rgb * max(dot(n, P.moon.xyz), 0.0) * shadow;
        float3 through = exp(-km * float3(0.012, 0.018, 0.03));
        return albedo * E * through + air * (1 - through);
    }

    // Puts the land, the water and the sky together. The water is still, with the faintest ripples, so it mirrors the
    // mountains and the aurora; how much it reflects follows Fresnel's law.
    fragment float4 auroraCompose(V in [[stage_in]], constant U &u [[buffer(0)]], constant Params &P [[buffer(1)]],
                                  texture2d<float> sky [[texture(0)]], texture2d<float> g0 [[texture(1)]],
                                  texture2d<float> g1 [[texture(2)]], texture2d<float> r0 [[texture(3)]],
                                  texture2d<float> r1 [[texture(4)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        uint2 px = uint2(in.pos.xy);
        float3 d = camRay(centered(in.uv, u.res), P.camera.x, P.camera.y);
        float4 a0 = g0.read(px), a1 = g1.read(px);
        float land = a0.w, water = a1.w, open = max(1 - land - water, 0.0);
        float2 horizon = camUV(normalize(float3(d.x, 0.015, d.z)), P.camera.x, P.camera.y, u.res);
        float3 air = sky.sample(s, horizon).rgb;
        float3 col = 0;
        if (land > 0) col += land * litLand(a0.xyz, a1.x, a1.y, a1.z, air, P);
        if (open > 0) col += open * sky.sample(s, in.uv).rgb;
        if (water > 0) {
            float km = r1.read(px).w;
            float2 q = (d.xz / max(-d.y, 1e-4)) * 2.4;
            float wave = noise(float2(q.x * 0.6, q.y * 2.2) + float2(0, u.time * 0.25)) - 0.5;
            wave += 0.5 * (noise(float2(q.x * 1.7, q.y * 6.0) - float2(u.time * 0.1, u.time * 0.6)) - 0.5);
            float calm = 0.004 * smoothstep(0.5, 0.0, km);
            float3 r = normalize(float3(d.x, -d.y + wave * calm, d.z));
            float2 ruv = camUV(r, P.camera.x, P.camera.y, u.res);
            float3 reflected = sky.sample(s, ruv).rgb;
            float2 shift = ruv - camUV(float3(d.x, -d.y, d.z), P.camera.x, P.camera.y, u.res);
            float4 b0 = r0.sample(s, in.uv + shift), b1 = r1.sample(s, in.uv + shift);
            if (b0.w > 0) reflected = mix(reflected, litLand(normalize(b0.xyz), b1.x, b1.y, b1.z, air, P), b0.w);
            float c = clamp(-r.y, 0.0, 1.0);
            float fresnel = 0.02 + 0.98 * pow(1 - abs(d.y), 5.0);
            col += water * (reflected * fresnel + float3(0.0004, 0.0006, 0.0007) * (1 - fresnel) * (1 + 0 * c));
        }
        return float4(col, 1);
    }
    """
}
