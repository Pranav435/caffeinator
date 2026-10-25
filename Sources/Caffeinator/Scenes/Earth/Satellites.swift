import AppKit
import Metal
import simd

/// Every active satellite in Celestrak's catalogue, propagated on the GPU in real time over the real Earth and marked
/// like a tracking display: a fine point per satellite with a short hairline tail along its orbit, tinted by orbit
/// class and dimmed in the Earth's shadow. The camera follows the ISS, then pulls back past the geostationary ring.
@MainActor final class Satellites: Scene {
    /// Matches `Sat` in the shader. a: period (minutes), semi-major axis (globe radii), eccentricity, inclination;
    /// b: node, perigee, mean anomaly at the reference time, kind; c: their rates a minute, then drag.
    private struct GPUSat { var a: SIMD4<Float>; var b: SIMD4<Float>; var c: SIMD4<Float> }

    private static let url = URL(string: "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle")!
    private static let names: [Orbit.Kind: String] = [.starlink: "Starlink", .oneweb: "OneWeb", .low: "Other low orbit",
                                                       .medium: "Navigation and medium orbit", .geostationary: "Geostationary",
                                                       .eccentric: "Highly elliptical"]
    /// Linear colours by orbit class; the shader shows them in display space, matching the overlay's key.
    static let colors: [Orbit.Kind: SIMD3<Float>] = [.starlink: [0.3, 0.7, 1.0], .oneweb: [0.45, 0.5, 1.0], .low: [0.8, 0.82, 0.86],
                                                     .medium: [1.0, 0.66, 0.25], .geostationary: [1.0, 0.36, 0.42], .eccentric: [0.66, 0.42, 1.0]]
    private static let segments = 6

    private let globe: Globe
    private let canvas: Globe.Canvas
    private let trails: MTLRenderPipelineState
    private let heads: MTLRenderPipelineState
    private let track: MTLRenderPipelineState
    private let marks: MTLRenderPipelineState
    private var orbits: [Orbit] = []
    private var counts: [Orbit.Kind: Int] = [:]
    private var sats: MTLBuffer
    private var reference = Date()
    private var iss: Int?
    private var trackBuffer: MTLBuffer
    private var trackPoints = 0
    private var trackTime = Date.distantPast
    private var loaded = false
    private var fetching = false
    private var fetched = Date.distantPast
    private var camera: Globe.Camera?
    private var lastOverlay: Float = -1
    private var start: Float
    private var far: Float = 0

    // Sound: who's overhead, and the ISS pass.
    private var overhead = Set<Int>()
    private var scan = 0
    private var pending: [(at: Float, freq: Float, amp: Float, pan: Float, voice: Synth.Voice)] = []
    private var issOver = false

    var scale: Float { 1 }
    var ready: Bool { loaded && globe.ready }

    var facts: String? {
        guard loaded else { return "Loading the satellite catalogue from Celestrak." }
        guard !orbits.isEmpty else { return "The satellite catalogue couldn't be loaded, so only the Earth is shown." }
        var s = "\(EarthText.count(orbits.count)) active satellites, \(EarthText.count(counts[.starlink] ?? 0)) of them Starlink"
        if let i = iss {
            let now = Date()
            let p = Orbit.subpoint(orbits[i].earthFixed(at: now))
            s += "; the ISS is over \(Globe.region(lat: p.lat, lon: p.lon)), \(EarthText.kilometres(p.height)) up"
            let me = (Location.shared.latitude, Location.shared.longitude)
            if issOver {
                s += ", passing over you right now"
            } else {
                let away = Globe.distance((p.lat, p.lon), me)
                s += ", \(EarthText.kilometres(away)) from \(Location.shared.place.isEmpty ? "you" : Location.shared.place)"
            }
        }
        return s + "."
    }

    init(gpu: GPU) throws {
        globe = try Globe.shared(gpu)
        canvas = try Globe.Canvas(gpu: gpu)
        let library = try gpu.library(Globe.metal + Self.source)
        trails = try gpu.pipeline(library, "solid", vertex: "satTrail", blend: .alpha)
        heads = try gpu.pipeline(library, "satHeadFill", vertex: "satHead", blend: .alpha)
        track = try gpu.pipeline(library, "trackFill", vertex: "issTrack", blend: .alpha)
        marks = try gpu.pipeline(library, "markFill", vertex: "mark", blend: .alpha)
        sats = gpu.buffer(16)
        trackBuffer = gpu.buffer(16)
        start = Snapshot.variant.map { [0, 205, 165][$0 % 3] } ?? 0
        Location.shared.request()
        fetch(gpu)
    }

    private func fetch(_ gpu: GPU) {
        guard !fetching else { return }
        fetching = true
        fetched = .now
        Task { @MainActor in
            // Celestrak asks clients to fetch at most a few times a day.
            let data = await Feed.data(Self.url, maxAge: 24 * 3600)
            let parsed = await Task.detached { data.map { Orbit.parse(String(decoding: $0, as: UTF8.self)) } ?? [] }.value
            fetching = false
            if !parsed.isEmpty || orbits.isEmpty { load(parsed, gpu) }
            loaded = true
        }
    }

    private func load(_ list: [Orbit], _ gpu: GPU) {
        orbits = list
        counts = Dictionary(grouping: list, by: \.kind).mapValues(\.count)
        iss = list.firstIndex { $0.name.hasPrefix("ISS (ZARYA)") }
        reference = .now
        let scale = Orbit.radius / 6371
        let gpuSats = list.map { o -> GPUSat in
            let el = o.elements(at: reference)
            let t0 = reference.timeIntervalSince(o.epoch) / 60
            return GPUSat(a: [Float(o.period), Float(o.axis * scale), Float(o.eccentricity), Float(o.inclination)],
                          b: [Float(el.node), Float(el.perigee), Float(el.anomaly), Float(o.kind.rawValue)],
                          c: [Float(o.nodeRate), Float(o.perigeeRate), Float(o.anomalyRate + 2 * o.drag * t0), Float(o.drag)])
        }
        sats = gpu.buffer(gpuSats)
        overhead = []
        trackTime = .distantPast
    }

    func draw(_ f: Frame) {
        if f.now.timeIntervalSince(fetched) > 3600, orbits.isEmpty || f.now.timeIntervalSince(reference) > 12 * 3600 { fetch(f.gpu) }
        let t = f.time + start
        let me = Globe.geo(Location.shared.latitude, Location.shared.longitude)
        let issNow = iss.map { orbits[$0].earthFixed(at: f.now) / 6371 }
        let issSpot = issNow.map { Globe.latLon(SIMD3<Float>($0)) }

        // Near the ISS for two and a half minutes, out past the geostationary ring for two, gliding between.
        let phase = t.truncatingRemainder(dividingBy: 360)
        let s: Float = phase < 150 ? 0 : phase < 195 ? (phase - 150) / 45 : phase < 315 ? 1 : 1 - (phase - 315) / 45
        far = s * s * (3 - 2 * s)
        let nearLat = Float(issSpot?.lat ?? 20) * 0.55 + 12, nearLon = Float(issSpot?.lon ?? 0) + 22
        let farLat = 24 + 5 * sin(t * 0.013), farLon = f.seed * 50 - t * 0.35
        var dLon = (farLon - nearLon).truncatingRemainder(dividingBy: 360)
        if dLon > 180 { dLon -= 360 } else if dLon < -180 { dLon += 360 }
        let distance = exp(log(Float(3.4)) + (log(Float(19)) - log(Float(3.4))) * far)
        let look = Earthquakes.towardDay((Double(nearLat + (farLat - nearLat) * far), Double(nearLon + dLon * far)), sun: Globe.sun(f.now), limit: 72)
        let cam = Globe.Camera(lat: Float(look.lat), lon: Float(look.lon), distance: distance,
                               zoom: 2.3 + 0.3 * far, aspect: f.aspect, date: f.now)
        camera = cam

        let light = canvas.light(for: f)
        let view = canvas.camera(cam)
        var style = Globe.Style()
        style.stars = 0.08
        globe.refresh(f.now)
        globe.draw(f.cmd, into: light, camera: view, style: style, date: f.now, time: f.time)
        canvas.develop(f, Lens.Look(exposure: 0.92, bloom: 0.03, halation: 0.012, vignette: 0.28, aberration: 0.8, grain: 0.35))

        // The tracking display goes over the developed picture.
        let minutes = Float(f.now.timeIntervalSince(reference) / 60)
        let gmst = Float(Astro.rad(Astro.gmst(f.now)))
        let sun = cam.sun
        if issNow != nil, f.now.timeIntervalSince(trackTime) > 1 { updateTrack(f.gpu, f.now) }
        if !orbits.isEmpty {
            let u = f.uniforms([minutes, gmst, f.size.y / 1000, far], SIMD4(sun, 0))
            if trackPoints > 1 {
                f.cmd.draw(track, into: f.target, clear: false, uniforms: u, buffers: [view, trackBuffer], vertices: (trackPoints - 1) * 6)
            }
            f.cmd.draw(trails, into: f.target, clear: false, uniforms: u, buffers: [view, sats],
                       vertices: orbits.count * Self.segments * 2, primitive: .line)
            f.cmd.draw(heads, into: f.target, clear: false, uniforms: u, buffers: [view, sats], vertices: orbits.count, primitive: .point)
        }
        // Markers: the ISS (a reticle) and you (a crosshair).
        var markList: [SIMD4<Float>] = [SIMD4(me * 1.001, 0)]
        if let issNow { markList.append(SIMD4(SIMD3<Float>(issNow), 1)) }
        f.cmd.draw(marks, into: f.target, clear: false, uniforms: f.uniforms([f.size.y / 1000, f.time, far, 0]),
                   buffers: [view, canvas.stream(0, markList)], vertices: 1, instances: markList.count, primitive: .point)

        if let issNow { checkPass(SIMD3<Float>(issNow), me, f) }
        if let synth = f.sound { play(synth, f, me) }
        if f.time - lastOverlay >= 0.1 || f.index == 0 {
            lastOverlay = f.time
            drawOverlay(f, issNow.map { SIMD3<Float>($0) })
        }
    }

    /// The ISS ground track from 45 minutes ago to 50 minutes ahead, a minute apart.
    private func updateTrack(_ gpu: GPU, _ now: Date) {
        guard let i = iss else { return }
        trackTime = now
        let points = stride(from: -45, through: 50, by: 1).map { m -> SIMD4<Float> in
            let p = SIMD3<Float>(orbits[i].earthFixed(at: now.addingTimeInterval(Double(m) * 60)))
            return SIMD4(simd_normalize(p) * 1.0015, Float(m))
        }
        trackBuffer = gpu.buffer(points)
        trackPoints = points.count
    }

    /// Chimes when the ISS climbs more than 10 degrees above your horizon.
    private func checkPass(_ issNow: SIMD3<Float>, _ me: SIMD3<Float>, _ f: Frame) {
        let d = issNow - me
        let elevation = asin(simd_dot(simd_normalize(d), me)) * 180 / .pi
        let over = elevation > 10
        if over, !issOver, f.sound != nil {
            for (k, note) in [880, 1108.7, 1318.5, 1760].enumerated() {
                pending.append((f.time + Float(k) * 0.32, Float(note), 0.07 - Float(k) * 0.01, Float(k) * 0.2 - 0.3, .bell))
            }
        }
        issOver = over
    }

    /// A shimmering pad that brightens with the number of satellites within 25 degrees of straight overhead, a soft
    /// blip as each one arrives there, and the ISS chime.
    private func play(_ synth: Synth, _ f: Frame, _ me: SIMD3<Float>) {
        // Scan a slice of the low orbiters each frame, so every one is checked about once a second.
        if !orbits.isEmpty {
            let slice = max(orbits.count / 30, 1)
            let cap = cos(Float(25) * .pi / 180)
            let east = simd_normalize(SIMD3<Float>(-me.y, me.x, 0) + SIMD3(1e-5, 0, 0))
            for _ in 0..<slice {
                scan = (scan + 1) % orbits.count
                let o = orbits[scan]
                guard o.altitude < 2000 else { continue }
                let p = SIMD3<Float>(o.earthFixed(at: f.now) / 6371)
                let d = simd_normalize(p - me)
                let inCap = simd_dot(d, me) > cap
                if inCap, !overhead.contains(scan) {
                    overhead.insert(scan)
                    if pending.count < 6 {
                        let freq: Float = o.kind == .starlink ? 1318.5 : o.kind == .oneweb ? 987.8 : 1174.7
                        pending.append((f.time + .random(in: 0...0.8), freq, 0.025, simd_dot(d, east) * 0.8, .pluck))
                    }
                } else if !inCap {
                    overhead.remove(scan)
                }
            }
        }
        var keep: [(at: Float, freq: Float, amp: Float, pan: Float, voice: Synth.Voice)] = []
        for p in pending {
            if p.at <= f.time { synth.play(p.voice, p.freq, amp: p.amp, pan: p.pan, length: p.voice == .bell ? 4 : 0.35) } else { keep.append(p) }
        }
        pending = keep
        let busy = min(Float(overhead.count) / 12, 1)
        let shimmer = 0.5 + 0.5 * sin(f.time * 0.23)
        synth.drone(0, 110, 0.07)
        synth.drone(1, 164.8, 0.045, pan: -0.3)
        synth.drone(2, 220.4, 0.03 * (0.6 + 0.4 * shimmer), pan: 0.3)
        synth.drone(3, 277.2, 0.02 + 0.015 * busy, pan: -0.5)
        synth.drone(4, 329.6 + 0.3, 0.012 + 0.012 * busy, pan: 0.5)
        synth.drone(5, 415.3, 0.008 + 0.012 * shimmer * busy, pan: 0.2)
        synth.noise(0.012, cutoff: 2400, crackle: 0)
    }

    // MARK: Overlay

    private func drawOverlay(_ f: Frame, _ issNow: SIMD3<Float>?) {
        guard let cam = camera else { return }
        let drift = CGPoint(x: CGFloat(sin(f.time * 0.007)) * 6, y: CGFloat(cos(f.time * 0.009)) * 4)
        let far = CGFloat(self.far)
        let issInfo = iss.map { Orbit.subpoint(orbits[$0].earthFixed(at: f.now)) }
        let me = Globe.geo(Location.shared.latitude, Location.shared.longitude, 1.001)
        f.overlay.draw { ctx, size in
            let dim = NSColor(white: 1, alpha: 0.42), faint = NSColor(white: 1, alpha: 0.24)
            let left = 44 + drift.x, top = 38 + drift.y
            ctx.text("EVERY SATELLITE", at: CGPoint(x: left, y: top), size: 11, weight: .semibold, color: dim, tracking: 2)
            ctx.text(self.orbits.isEmpty ? "No data" : "\(EarthText.count(self.orbits.count)) active", at: CGPoint(x: left, y: top + 18),
                     size: 22, weight: .light, color: NSColor(white: 1, alpha: 0.6))
            ctx.text(EarthText.utc(f.now), at: CGPoint(x: left, y: top + 48), size: 11, mono: true, color: faint)
            var y = top + 76
            for kind in [Orbit.Kind.starlink, .oneweb, .low, .medium, .geostationary, .eccentric] {
                guard let n = self.counts[kind], n > 0 else { continue }
                ctx.setFillColor(NSColor.earthLinear(Self.colors[kind]!).cgColor)
                ctx.fillEllipse(in: CGRect(x: left, y: y + 3, width: 7, height: 7))
                ctx.text("\(Self.names[kind]!)  \(EarthText.count(n))", at: CGPoint(x: left + 14, y: y), size: 10.5, color: dim)
                y += 17
            }

            // The ISS.
            if let issNow, let info = issInfo, let p = cam.overlay(issNow, size), p.shown > 0.05 {
                let a = CGFloat(p.shown)
                let at = CGPoint(x: p.point.x + 26, y: p.point.y - 34)
                ctx.earthLine(CGPoint(x: p.point.x + 5, y: p.point.y - 5), CGPoint(x: at.x - 3, y: at.y + 9), color: NSColor(white: 1, alpha: 0.4 * a))
                ctx.text("ISS", at: at, size: 12.5, weight: .semibold, color: NSColor(white: 1, alpha: 0.9 * a), tracking: 1)
                ctx.text("\(Int(info.height.rounded())) km · over \(Globe.region(lat: info.lat, lon: info.lon))",
                         at: CGPoint(x: at.x, y: at.y + 16), size: 10.5, color: NSColor(white: 1, alpha: 0.5 * a))
            }
            // You.
            if let p = cam.overlay(me, size), cam.facing(me) > 0.5 {
                let a = 1 - far * 0.7
                let place = Location.shared.place.isEmpty ? "You" : Location.shared.place
                ctx.text(place, at: CGPoint(x: p.point.x + 10, y: p.point.y + 4), size: 10.5, weight: .medium,
                         color: NSColor(srgbRed: 1, green: 0.8, blue: 0.55, alpha: 0.75 * a))
            }
            // Shell labels when pulled back.
            if far > 0.4 {
                let a = (far - 0.4) / 0.6
                let rings: [(String, Double, NSColor)] = [
                    ("GEOSTATIONARY RING · 35,786 KM", 6.62, NSColor.earthLinear(Self.colors[.geostationary]!, alpha: 0.75 * a)),
                    ("GPS · GALILEO · GLONASS · BEIDOU", 4.2, NSColor.earthLinear(Self.colors[.medium]!, alpha: 0.7 * a)),
                ]
                for (label, r, color) in rings {
                    // The point of the ring furthest to the right on screen.
                    var best: (CGPoint, Float)?
                    for k in 0..<72 {
                        let ang = Double(k) * 5
                        guard let p = cam.overlay(Globe.geo(0, ang, r), size) else { continue }
                        if best == nil || p.point.x > best!.0.x { best = (p.point, p.shown) }
                    }
                    guard let (p, _) = best else { continue }
                    let right = p.x + 260 < size.width
                    ctx.text(label, at: CGPoint(x: right ? p.x + 14 : p.x - 14, y: p.y - 6), size: 9.5, weight: .semibold, color: color,
                             anchor: CGPoint(x: right ? 0 : 1, y: 0), tracking: 1.4)
                }
            }
        }
    }

    private static let source = """
    struct Sat { float4 a; float4 b; float4 c; };

    constant float3 kindColors[6] = { float3(0.3, 0.7, 1.0), float3(0.45, 0.5, 1.0), float3(0.8, 0.82, 0.86),
                                      float3(1.0, 0.66, 0.25), float3(1.0, 0.36, 0.42), float3(0.66, 0.42, 1.0) };
    constant float kindLevels[6] = { 0.42, 0.5, 0.42, 0.95, 0.95, 0.9 };

    // Low orbiters thin out as the camera pulls back and they crowd together; the high ones only show once it's outside them.
    inline float kindLevel(int kind, float far) {
        return kindLevels[kind] * (kind < 3 ? 1 - 0.5 * far : smoothstep(0.3, 0.65, far));
    }

    // Earth-fixed position, in globe radii, t minutes after the reference time; gmst in radians.
    float3 satPosition(Sat s, float t, float gmst) {
        float node = s.b.x + s.c.x * t, argp = s.b.y + s.c.y * t;
        float M = fmod(s.b.z + s.c.z * t + s.c.w * t * t, TAU);
        float e = s.a.z, inc = s.a.w;
        float E = M + e * sin(M);
        for (int k = 0; k < 6; k++) E -= (E - e * sin(E) - M) / (1 - e * cos(E));
        float x = s.a.y * (cos(E) - e), y = s.a.y * sqrt(1 - e * e) * sin(E);
        float cw = cos(argp), sw = sin(argp), co = cos(node), so = sin(node), ci = cos(inc), si = sin(inc);
        float xw = x * cw - y * sw, yw = x * sw + y * cw;
        float3 r = float3(xw * co - yw * ci * so, xw * so + yw * ci * co, yw * si);
        float cg = cos(gmst), sg = sin(gmst);
        return float3(cg * r.x + sg * r.y, -sg * r.x + cg * r.y, r.z);
    }

    // Satellites in the Earth's shadow are dimmed, so the shadow shows as a gap in the swarm.
    inline float sunlit(float3 p, float3 sun) {
        float along = dot(p, sun);
        if (along > 0) return 1;
        return mix(0.3, 1.0, smoothstep(0.98, 1.03, length(p - along * sun)));
    }

    // Shells pile up where the view grazes them, at the limb; even that out so the face of the swarm reads too.
    inline float graze(float3 p, constant GlobeCamera &c) {
        return mix(0.6, 1.0, sqrt(abs(dot(normalize(p), normalize(c.eye - p)))));
    }

    // Fade satellites that come too close to the camera.
    inline float nearFade(float3 p, constant GlobeCamera &c) { return smoothstep(0.6, 2.0, dot(p - c.eye, c.basis[2])); }

    // u.a: minutes since the reference time, GMST, pixel scale, how far out the camera is   u.b: sun
    vertex P satTrail(uint vid [[vertex_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                      const device Sat *sats [[buffer(2)]]) {
        const uint segments = \(segments);
        uint i = vid / (segments * 2);
        uint k = (vid % (segments * 2)) / 2 + (vid & 1);
        Sat s = sats[i];
        int kind = int(s.b.w);
        // Tails cover a thirtieth of an orbit, at most three and a half minutes: enough to show which way each one flows.
        float span = min(s.a.x * 0.033, 3.5);
        float back = float(k) / float(segments);
        float t = u.a.x - back * span;
        // Earth turns under the tail too, so trace it in the frame of the moment it was drawn.
        float3 p = satPosition(s, t, u.a.y - back * span * 0.00437527);
        P o;
        o.pos = globeClip(p, c);
        o.size = 1;
        float a = kindLevel(kind, u.a.w) * 0.32 * pow(1 - back, 1.6) * globeShown(p, c) * sunlit(p, u.b.xyz) * nearFade(p, c) * graze(p, c);
        o.color = float4(toSRGB(kindColors[kind]), 1) * a;
        return o;
    }

    struct Head { float4 pos [[position]]; float size [[point_size]]; float4 color; };

    vertex Head satHead(uint i [[vertex_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                        const device Sat *sats [[buffer(2)]]) {
        Sat s = sats[i];
        int kind = int(s.b.w);
        float3 p = satPosition(s, u.a.x, u.a.y);
        Head o;
        o.pos = globeClip(p, c);
        float shown = globeShown(p, c) * nearFade(p, c);
        o.size = (kind == 0 ? 2.6 : 3.2) * u.a.z;
        float a = kindLevel(kind, u.a.w) * shown * sunlit(p, u.b.xyz) * graze(p, c);
        o.color = float4(toSRGB(kindColors[kind]), 1) * saturate(a);
        if (shown <= 0) o.pos = float4(0, 0, -1, 1);
        return o;
    }

    // A crisp point, a pixel and a half across.
    fragment float4 satHeadFill(Head in [[stage_in]], float2 pc [[point_coord]]) {
        float d = length(pc - 0.5) * 2;
        float px = max(fwidth(d), 1e-3);
        return in.color * (1 - smoothstep(0.55 - px, 0.55 + px, d));
    }

    // The ISS ground track as a hairline: the last 45 minutes solid, the next 50 dashed.
    vertex Ribbon issTrack(uint vid [[vertex_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                           const device float4 *pts [[buffer(2)]]) {
        uint seg = vid / 6;
        float4 a = pts[seg], b = pts[seg + 1];
        bool atB = vid % 6 == 2 || vid % 6 == 3 || vid % 6 == 5;
        float age = atB ? b.w : a.w;
        float3 q = atB ? b.xyz : a.xyz;
        float level = age < 0 ? 0.6 * pow(1 + age / 46, 1.2) : 0.45 * (1 - age / 52);
        float alpha = level * globeFacing(q, c) * (1 - u.a.w * 0.8);
        Ribbon o = ribbon(vid % 6, globeClip(a.xyz, c), globeClip(b.xyz, c), 1.0 * u.a.z, float4(0.92, 0.94, 0.96, 1) * alpha, u.res);
        o.along = age;
        return o;
    }

    fragment float4 trackFill(Ribbon in [[stage_in]]) {
        float dash = in.along < 0 ? 1 : step(0.45, fract(in.along * 0.5));
        return in.color * saturate(in.half_width + 0.5 - abs(in.across)) * dash;
    }

    struct Mark { float4 pos [[position]]; float size [[point_size]]; float4 color; float kind; };

    // u.a: pixel scale, time, how far out the camera is
    vertex Mark mark(uint i [[instance_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                     const device float4 *marks [[buffer(2)]]) {
        float4 m = marks[i];
        Mark o;
        o.pos = globeClip(m.xyz, c);
        o.kind = m.w;
        float shown = m.w > 0.5 ? globeShown(m.xyz, c) : globeFacing(m.xyz, c) * (1 - u.a.z * 0.6);
        o.size = (m.w > 0.5 ? 30 : 22) * u.a.x * (1 - 0.45 * u.a.z);
        o.color = (m.w > 0.5 ? float4(0.95, 0.96, 0.97, 1) * 0.9 : float4(1.0, 0.78, 0.55, 1) * 0.85) * shown;
        if (shown <= 0) o.pos = float4(0, 0, -1, 1);
        return o;
    }

    // The ISS gets four corner brackets, like a tracking reticle; you get a small crosshair.
    fragment float4 markFill(Mark in [[stage_in]], float2 pc [[point_coord]]) {
        float2 p = (pc - 0.5) * 2;
        float px = max(fwidth(p.x), 1e-3) * 1.1;
        float2 a = abs(p);
        float ink;
        if (in.kind > 0.5) {
            float frame = max(1 - smoothstep(0.0, px, abs(a.x - 0.82)) , 0.0) * step(a.y, 0.84) * step(0.45, a.y)
                        + max(1 - smoothstep(0.0, px, abs(a.y - 0.82)), 0.0) * step(a.x, 0.84) * step(0.45, a.x);
            float dot_ = 1 - smoothstep(0.1, 0.1 + px, length(p));
            ink = saturate(frame + dot_);
        } else {
            float ring = 1 - smoothstep(0.0, px, abs(length(p) - 0.42));
            float cross_ = (1 - smoothstep(0.0, px, a.x)) * step(0.55, a.y) * step(a.y, 0.95)
                         + (1 - smoothstep(0.0, px, a.y)) * step(0.55, a.x) * step(a.x, 0.95);
            ink = saturate(ring + cross_);
        }
        return in.color * ink;
    }
    """
}
