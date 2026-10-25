import AppKit
import Metal
import simd

/// The USGS feed of the past day's earthquakes, replayed in a minute and a half over the real Earth. Each quake is
/// marked like a seismicity map: a dot coloured by depth and sized by magnitude, and a hairline ring that spreads to a
/// radius set by its energy as it strikes.
@MainActor final class Earthquakes: Scene {
    private struct Quake {
        var lat: Double, lon: Double, depth: Double, mag: Double
        var place: String
        var time: Date
    }

    /// Matches `Quake` in the shader: latitude, longitude, depth, magnitude; moment in replay seconds.
    private struct GPUQuake { var where_: SIMD4<Float>; var when: SIMD4<Float> }

    private static let url = URL(string: "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson")!
    private static let day: Double = 86400
    /// Seconds the 24 hours take to replay, then how long the whole day stays up before it clears and starts again.
    private static let replay: Float = 96, hold: Float = 14, clear: Float = 4

    private let globe: Globe
    private let canvas: Globe.Canvas
    private let rings: MTLRenderPipelineState
    private let dots: MTLRenderPipelineState
    private var quakes: [Quake] = []
    private var buffer: MTLBuffer
    private var loaded = false
    private var fetching = false
    private var fetched = Date.distantPast
    private var windowEnd = Date()
    private var clock: Float
    private var popped = 0
    private var lastOverlay: Float = -1
    private var energy: Float = 0
    private var tokens: Float = 3
    private var camera: Globe.Camera?
    private var view: (lat: Double, lon: Double)?

    var scale: Float { 1 }
    var ready: Bool { loaded && globe.ready }

    var facts: String? {
        guard loaded else { return "Loading the past day's earthquakes from the US Geological Survey." }
        guard let big = quakes.max(by: { $0.mag < $1.mag }) else { return "No earthquake data yet; the USGS feed couldn't be reached." }
        var s = "\(EarthText.count(quakes.count)) earthquakes in the last day; largest magnitude \(EarthText.number(big.mag)), "
        s += "\(Self.spoken(big.place)), \(EarthText.ago(big.time, now: .now))."
        let deep = quakes.filter { $0.depth >= 300 }.count
        if deep > 0 { s += " \(deep == 1 ? "One was" : "\(deep) were") deeper than 300 kilometres." }
        return s
    }

    init(gpu: GPU) throws {
        globe = try Globe.shared(gpu)
        canvas = try Globe.Canvas(gpu: gpu)
        let library = try gpu.library(Globe.metal + Self.source)
        rings = try gpu.pipeline(library, "ribbonFill", vertex: "quakeRing", blend: .alpha)
        dots = try gpu.pipeline(library, "quakeDotFill", vertex: "quakeDot", blend: .alpha)
        buffer = gpu.buffer(16)
        clock = Snapshot.variant.map { Float($0) * 30 } ?? 0
        fetch(gpu)
    }

    private func fetch(_ gpu: GPU) {
        guard !fetching else { return }
        fetching = true
        fetched = .now
        Task { @MainActor in
            let json = await Feed.object(Self.url, maxAge: 300)
            fetching = false
            let parsed = Self.parse(json)
            if !parsed.isEmpty || quakes.isEmpty { quakes = parsed }
            upload(gpu)
            loaded = true
        }
    }

    private static func parse(_ json: Any?) -> [Quake] {
        guard let root = json as? [String: Any], let features = root["features"] as? [[String: Any]] else { return [] }
        return features.compactMap { f in
            guard let p = f["properties"] as? [String: Any], let g = f["geometry"] as? [String: Any],
                  let c = g["coordinates"] as? [Double], c.count >= 2,
                  let mag = (p["mag"] as? NSNumber)?.doubleValue, let ms = (p["time"] as? NSNumber)?.doubleValue,
                  c[0].isFinite, c[1].isFinite, mag.isFinite else { return nil }
            let place = (p["place"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "at \(EarthText.latitude(c[1]))"
            return Quake(lat: c[1], lon: c[0], depth: c.count > 2 && c[2].isFinite ? max(c[2], 0) : 10, mag: mag, place: place,
                         time: Date(timeIntervalSince1970: ms / 1000))
        }.sorted { $0.time < $1.time }
    }

    /// When each quake pops, in replay seconds, for the window ending at `windowEnd`.
    private func moment(_ q: Quake) -> Float {
        Float((q.time.timeIntervalSince(windowEnd) + Self.day) / Self.day) * Self.replay
    }

    private func upload(_ gpu: GPU) {
        let list = quakes.map { q in
            GPUQuake(where_: [Float(q.lat), Float(q.lon), Float(q.depth), Float(q.mag)], when: [moment(q), 0, 0, 0])
        }
        buffer = gpu.buffer(list)
        popped = quakes.filter { moment($0) <= clock }.count
    }

    func draw(_ f: Frame) {
        let cycle = Self.replay + Self.hold + Self.clear
        clock += f.dt
        if clock > cycle {
            clock = 0
            windowEnd = f.now
            upload(f.gpu)
        }
        if f.now.timeIntervalSince(fetched) > 300 { fetch(f.gpu) }

        let aim = steer(f)
        let look = Self.towardDay(aim, sun: Globe.sun(f.now), limit: 95)
        let cam = Globe.Camera(lat: Float(look.lat), lon: Float(look.lon), distance: 3.6, zoom: 2.75, aspect: f.aspect, date: f.now)
        camera = cam
        globe.refresh(f.now)
        let light = canvas.light(for: f)
        let view = canvas.camera(cam)
        let style = Globe.Style()
        globe.draw(f.cmd, into: light, camera: view, style: style, date: f.now, time: f.time)

        let fade = 1 - Globe.smoothstep(Self.replay + Self.hold, cycle, clock)
        canvas.develop(f)
        let u = f.uniforms([clock, fade, f.size.y / 1000, Float(quakes.count)])
        if !quakes.isEmpty {
            f.cmd.draw(dots, into: f.target, clear: false, uniforms: u, buffers: [view, buffer], vertices: 1, instances: quakes.count,
                       primitive: .point)
            f.cmd.draw(rings, into: f.target, clear: false, uniforms: u, buffers: [view, buffer], vertices: 64 * 6, instances: quakes.count)
        }

        // Quakes that popped this frame.
        var fresh: [Quake] = []
        while popped < quakes.count, moment(quakes[popped]) <= clock {
            if clock - moment(quakes[popped]) < 1 { fresh.append(quakes[popped]) }
            popped += 1
        }
        if let synth = f.sound { play(synth, fresh, f) }
        if f.time - lastOverlay >= 0.1 || f.index == 0 {
            lastOverlay = f.time
            drawOverlay(f, fade: fade)
        }
    }

    /// Turns the globe toward the quakes striking in the next few seconds, weighted by size, so the big ones happen in
    /// view. When it's quiet, or the activity is spread around the world, the globe just drifts west to east.
    private func steer(_ f: Frame) -> (lat: Double, lon: Double) {
        var (lat, lon) = view ?? (18, 160 + Double(f.seed) * 37)
        let dt = Double(f.dt)
        var sum = SIMD3<Double>(), total = 0.0
        for q in quakes where abs(moment(q) - clock - 6) < 9 { // from 3 s ago to 15 s ahead
            let w = pow(10, 0.5 * (q.mag - 2))
            let la = q.lat * .pi / 180, lo = q.lon * .pi / 180
            sum += w * SIMD3(cos(la) * cos(lo), cos(la) * sin(lo), sin(la))
            total += w
        }
        let length = (sum * sum).sum().squareRoot()
        let wander = 18 + 14 * sin(Double(f.time) * 0.011 + Double(f.seed))
        if total > 0, length / total > 0.35 {
            let targetLon = atan2(sum.y, sum.x) * 180 / .pi, targetLat = min(max(asin(sum.z / length) * 180 / .pi, -35), 50)
            let turn = (targetLon - lon + 540).truncatingRemainder(dividingBy: 360) - 180
            lon += max(min(turn * (1 - exp(-dt / 5)), 12 * dt), -12 * dt)
            lat += (targetLat - lat) * (1 - exp(-dt / 8))
        } else {
            lon -= 1.1 * dt
            lat += (wander - lat) * (1 - exp(-dt / 8))
        }
        view = (lat, lon.truncatingRemainder(dividingBy: 360))
        return view!
    }

    /// Swings a view direction toward the Sun until it's at most `limit` degrees from it, so some of the day side
    /// is always in the picture.
    static func towardDay(_ aim: (lat: Double, lon: Double), sun: SIMD3<Float>, limit: Float) -> (lat: Double, lon: Double) {
        let v = Globe.geo(aim.lat, aim.lon)
        let angle = acos(max(-1, min(1, simd_dot(v, sun))))
        let most = limit * .pi / 180
        guard angle > most else { return aim }
        let axis = simd_normalize(sun - simd_dot(sun, v) * v)
        let turn = angle - most
        return Globe.latLon(v * cos(turn) + axis * sin(turn))
    }

    /// A thump for each quake as it pops, bigger lower and louder, over a rumble that swells with the energy released.
    private func play(_ synth: Synth, _ fresh: [Quake], _ f: Frame) {
        tokens = min(tokens + f.dt * 3, 4)
        for q in fresh.sorted(by: { $0.mag > $1.mag }) {
            let m = Float(q.mag)
            energy += pow(10, 0.75 * (m - 2)) * 0.02
            guard tokens >= 1 || m >= 4 else { continue }
            tokens -= 1
            let side = sidePan(q)
            let level = min(max((m - 0.5) / 5, 0), 1)
            synth.play(.thump, 150 * pow(2, -(m - 1) / 2.4), amp: 0.035 + 0.11 * level, pan: side, length: 0.5 + level * 1.6)
            if m >= 3.5 { synth.play(.bell, 196 * pow(2, -(m - 3.5) / 3), amp: 0.025 + 0.03 * level, pan: side, length: 3 + level * 3) }
        }
        energy = min(energy * exp(-f.dt / 4), 1)
        synth.noise(0.035 + energy * 0.12, cutoff: 70 + energy * 140)
        synth.drone(0, 41.2, 0.07)
        synth.drone(1, 61.7, 0.035 + energy * 0.04, pan: -0.2)
        synth.drone(2, 82.4, 0.02 + energy * 0.02, pan: 0.2)
        synth.drone(3, 164.8, 0.01 + energy * 0.01, pan: -0.1)
    }

    /// Pans a quake left or right by where it is on screen.
    private func sidePan(_ q: Quake) -> Float {
        guard let p = camera?.project(Globe.geo(q.lat, q.lon)) else { return 0 }
        return max(-0.8, min(0.8, (p.x - 0.5) * 1.6))
    }

    // MARK: Overlay

    private func drawOverlay(_ f: Frame, fade: Float) {
        guard let cam = camera else { return }
        let start = windowEnd.addingTimeInterval(-Self.day)
        let at = start.addingTimeInterval(Double(min(clock, Self.replay) / Self.replay) * Self.day)
        let shownQuakes = quakes.filter { moment($0) <= clock }
        // Label the biggest quakes so far, one per cluster: aftershocks within 300 km count toward their main shock.
        var clusters: [(main: Quake, more: Int)] = []
        for q in shownQuakes.sorted(by: { $0.mag > $1.mag }) where q.mag >= 2.5 {
            if let i = clusters.firstIndex(where: { Globe.distance(($0.main.lat, $0.main.lon), (q.lat, q.lon)) < 300 }) {
                clusters[i].more += 1
            } else if clusters.count < 5 {
                clusters.append((q, 0))
            }
        }
        let drift = CGPoint(x: CGFloat(sin(f.time * 0.007)) * 6, y: CGFloat(cos(f.time * 0.009)) * 4)
        let alpha = CGFloat(fade)
        f.overlay.draw { ctx, size in
            let dim = NSColor(white: 1, alpha: 0.42 * alpha)
            let faint = NSColor(white: 1, alpha: 0.22 * alpha)
            // Labels for the biggest quakes so far, with a hairline to the point.
            var placed: [CGRect] = []
            for (q, more) in clusters {
                guard let p = cam.overlay(Globe.geo(q.lat, q.lon), size) else { continue }
                let facing = cam.shown(Globe.geo(q.lat, q.lon, 1.002)) * Float(alpha)
                guard facing > 0.05 else { continue }
                let a = CGFloat(facing)
                var origin = CGPoint(x: p.point.x + 22, y: p.point.y - 26)
                let title = "M\(EarthText.number(q.mag))  \(Self.short(q.place))"
                let width = CGFloat(title.count) * 6.4 + 20
                while placed.contains(where: { $0.intersects(CGRect(x: origin.x, y: origin.y, width: width, height: 30)) }) { origin.y += 32 }
                placed.append(CGRect(x: origin.x, y: origin.y, width: width, height: 30))
                ctx.earthLine(CGPoint(x: p.point.x + 4, y: p.point.y - 4), CGPoint(x: origin.x - 3, y: origin.y + 8),
                             color: NSColor(white: 1, alpha: 0.3 * a))
                let tint = NSColor.earthLinear(Self.depthColor(q.depth) * 0.9 + 0.1, alpha: 0.9 * a)
                ctx.text(title, at: origin, size: 12, weight: .medium, color: tint)
                let detail = "\(Int(q.depth.rounded())) km deep · \(EarthText.ago(q.time, now: .now))" + (more > 0 ? " · \(more) more nearby" : "")
                ctx.text(detail, at: CGPoint(x: origin.x, y: origin.y + 15), size: 10.5, color: NSColor(white: 1, alpha: 0.45 * a))
            }

            // Header and replay clock.
            let left = 44 + drift.x, top = 38 + drift.y
            ctx.text("EARTHQUAKES · PAST 24 HOURS", at: CGPoint(x: left, y: top), size: 11, weight: .semibold, color: dim, tracking: 2)
            ctx.text(EarthText.utc(at), at: CGPoint(x: left, y: top + 18), size: 22, weight: .light, mono: true, color: NSColor(white: 1, alpha: 0.6))
            ctx.text("\(shownQuakes.count) of \(self.quakes.count) so far", at: CGPoint(x: left, y: top + 48), size: 11, color: faint)

            // Timeline: one tick per quake, height by magnitude, colour by depth; the playhead sweeps across.
            let x0 = 44 + drift.x, x1 = size.width - 44 + drift.x, base = size.height - 40 + drift.y
            ctx.earthLine(CGPoint(x: x0, y: base), CGPoint(x: x1, y: base), color: faint, width: 0.5)
            for q in self.quakes {
                let t = CGFloat(self.moment(q) / Self.replay)
                guard t >= 0, t <= 1 else { continue }
                let x = x0 + (x1 - x0) * t
                let h = 2 + CGFloat(max(q.mag, 0)) * 3.2
                let on = self.moment(q) <= self.clock
                ctx.earthLine(CGPoint(x: x, y: base), CGPoint(x: x, y: base - h),
                             color: NSColor.earthLinear(Self.depthColor(q.depth), alpha: (on ? 0.8 : 0.18) * alpha), width: on ? 1 : 0.75)
            }
            let head = x0 + (x1 - x0) * CGFloat(min(self.clock / Self.replay, 1))
            ctx.earthLine(CGPoint(x: head, y: base + 4), CGPoint(x: head, y: base - 26), color: NSColor(white: 1, alpha: 0.5 * alpha), width: 1)
            ctx.text("24 hours ago", at: CGPoint(x: x0, y: base + 6), size: 10, color: faint)
            ctx.text("now", at: CGPoint(x: x1, y: base + 6), size: 10, color: faint, anchor: CGPoint(x: 1, y: 0))

            // Depth key.
            let keys: [(String, Double)] = [("0", 0), ("35", 35), ("100", 100), ("300", 300), ("700 km", 700)]
            var kx = size.width - 44 + drift.x - 250
            ctx.text("DEPTH", at: CGPoint(x: kx, y: top), size: 10, weight: .semibold, color: faint, tracking: 1.5)
            kx += 52
            for (label, km) in keys {
                ctx.setFillColor(NSColor.earthLinear(Self.depthColor(km)).cgColor)
                ctx.fillEllipse(in: CGRect(x: kx, y: top + 3, width: 7, height: 7))
                ctx.text(label, at: CGPoint(x: kx + 11, y: top), size: 10, color: dim)
                kx += label.count > 3 ? 56 : 38
            }
        }
    }

    /// "169 km NW of Port-Olry, Vanuatu" spoken as "169 kilometres northwest of Port-Olry, Vanuatu".
    nonisolated static func spoken(_ place: String) -> String {
        let words = ["NNE": "north-northeast", "ENE": "east-northeast", "ESE": "east-southeast", "SSE": "south-southeast",
                     "SSW": "south-southwest", "WSW": "west-southwest", "WNW": "west-northwest", "NNW": "north-northwest",
                     "NE": "northeast", "SE": "southeast", "SW": "southwest", "NW": "northwest",
                     "N": "north", "E": "east", "S": "south", "W": "west", "km": "kilometres", "CA": "California"]
        let s = place.split(separator: " ", omittingEmptySubsequences: false).map { part -> String in
            let bare = part.trimmingCharacters(in: .punctuationCharacters)
            guard let w = words[bare] else { return String(part) }
            return part.replacingOccurrences(of: bare, with: w)
        }.joined(separator: " ")
        return s.first?.isNumber == true ? s : "near " + s
    }

    /// The place without its distance, for labels: "Port-Olry, Vanuatu".
    private static func short(_ place: String) -> String {
        guard let r = place.range(of: " of ") else { return place }
        return String(place[r.upperBound...])
    }

    /// Gold for shallow quakes through orange, magenta and violet to blue for the deepest. Matches `depthColor` in Metal.
    static func depthColor(_ km: Double) -> SIMD3<Float> {
        let t = Float(min(max(log(1 + km / 10) / log(71), 0), 1))
        let stops: [(Float, SIMD3<Float>)] = [(0, [1.0, 0.78, 0.32]), (0.35, [1.0, 0.42, 0.16]), (0.56, [0.95, 0.2, 0.48]),
                                               (0.8, [0.58, 0.3, 1.0]), (1, [0.3, 0.55, 1.0])]
        for i in 1..<stops.count where t <= stops[i].0 {
            let (a, ca) = stops[i - 1], (b, cb) = stops[i]
            return ca + (cb - ca) * ((t - a) / (b - a))
        }
        return stops.last!.1
    }

    private static let source = """
    struct Quake { float4 where_; float4 when; };

    // Display colour by depth: pale gold for shallow, through amber and rose to blue-violet for the deepest.
    float3 depthColor(float km) {
        float t = saturate(log(1 + km / 10) / log(71.0));
        float3 c0 = float3(1.0, 0.78, 0.32), c1 = float3(1.0, 0.42, 0.16), c2 = float3(0.95, 0.2, 0.48), c3 = float3(0.58, 0.3, 1.0), c4 = float3(0.3, 0.55, 1.0);
        float3 c = t < 0.35 ? mix(c0, c1, t / 0.35) : t < 0.56 ? mix(c1, c2, (t - 0.35) / 0.21) : t < 0.8 ? mix(c2, c3, (t - 0.56) / 0.24) : mix(c3, c4, (t - 0.8) / 0.2);
        return toSRGB(c);
    }

    // Ring radius in degrees: the energy, 10^(1.5 M), to the power 2/9, so each magnitude step widens it by 10^(1/3).
    inline float ringDegrees(float mag) { return clamp(0.4 * pow(10.0, (mag - 1) / 3), 0.4, 12.0); }
    inline float ringLife(float mag) { return 2.5 + max(mag, 0.0) * 0.9; }

    // A hairline small circle on the surface around each quake as it strikes, 64 pieces.
    // u.a: replay clock, fade, pixel scale, count
    vertex Ribbon quakeRing(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                            constant GlobeCamera &c [[buffer(1)]], const device Quake *qs [[buffer(2)]]) {
        Quake q = qs[iid];
        float mag = q.where_.w;
        float age = u.a.x - q.when.x, life = ringLife(mag);
        uint seg = vid / 6, corner = vid % 6;
        if (age < 0 || age > life || mag < 1.5) { Ribbon o; o.pos = float4(0, 0, -1, 1); o.color = 0; o.across = 0; o.half_width = 1; o.along = 0; return o; }
        bool atB = corner == 2 || corner == 3 || corner == 5;
        float t = age / life;
        float rho = ringDegrees(mag) * (1 - pow(1 - t, 3.0)) * PI / 180;
        float3 n = geo(q.where_.x, q.where_.y, 1.0);
        float3 east = normalize(float3(-n.y, n.x, 0) + float3(1e-5, 0, 0)), north = cross(n, east);
        float a0 = float(seg) / 64 * TAU, a1 = float(seg + 1) / 64 * TAU;
        float3 p0 = (n * cos(rho) + (east * cos(a0) + north * sin(a0)) * sin(rho)) * 1.001;
        float3 p1 = (n * cos(rho) + (east * cos(a1) + north * sin(a1)) * sin(rho)) * 1.001;
        float a = pow(1 - t, 1.6) * (0.3 + 0.6 * saturate((mag - 1.5) / 4)) * u.a.y * globeFacing(atB ? p1 : p0, c);
        return ribbon(corner, globeClip(p0, c), globeClip(p1, c), 1.1 * u.a.z, float4(depthColor(q.where_.z), 1) * a, u.res);
    }

    struct Dot { float4 pos [[position]]; float size [[point_size]]; float3 color; float alpha; };

    vertex Dot quakeDot(uint iid [[instance_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                        const device Quake *qs [[buffer(2)]]) {
        Quake q = qs[iid];
        float mag = q.where_.w;
        float age = u.a.x - q.when.x;
        Dot o;
        float3 p = geo(q.where_.x, q.where_.y, 1.0);
        o.pos = globeClip(p, c);
        float shown = globeFacing(p, c);
        if (age < 0 || shown <= 0) { o.pos = float4(0, 0, -1, 1); o.size = 0; return o; }
        float m = clamp(mag, 0.0, 8.0);
        o.size = (4 + m * 1.5 + 5 * exp(-age * 2.5) * saturate(m / 3)) * u.a.z;
        o.color = depthColor(q.where_.z);
        o.alpha = shown * u.a.y * (0.6 + 0.4 * exp(-age / 30));
        return o;
    }

    // A crisp dot with a thin dark rim, so it reads over cloud as well as sea.
    fragment float4 quakeDotFill(Dot in [[stage_in]], float2 pc [[point_coord]]) {
        float d = length(pc - 0.5) * 2;
        float px = max(fwidth(d), 1e-3);
        float core = 1 - smoothstep(0.5 - px, 0.5 + px, d);
        float rim = (1 - smoothstep(0.78 - px, 0.78 + px, d)) - core;
        return (float4(in.color, 1) * core + float4(0, 0, 0, 0.45) * rim) * in.alpha;
    }
    """
}
