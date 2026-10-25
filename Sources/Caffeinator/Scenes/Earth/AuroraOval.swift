import AppKit
import Metal
import simd

/// NOAA's OVATION aurora forecast seen from above the pole: the forecast probability becomes curtains of green and
/// red light, ray-marched through the upper atmosphere with arcs and rays laid along the magnetic field, over a night
/// globe with city lights, the view line, and a circle at your latitude.
@MainActor final class AuroraOval: Scene {
    private static let ovationURL = URL(string: "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json")!
    private static let kpURL = URL(string: "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")!
    /// The geomagnetic (dipole) north pole, IGRF epoch 2025.
    private static let dipole = Globe.geo(80.8, -72.7)
    /// The polar maps cover 50 degrees around each pole.
    private static let mapSize = 1024

    private let globe: Globe
    private let canvas: Globe.Canvas
    private let mapKernel: MTLComputePipelineState
    private let structureKernel: MTLComputePipelineState
    private let structure: MTLTexture
    private let march: MTLRenderPipelineState
    private let add: MTLRenderPipelineState
    private let lines: MTLRenderPipelineState
    private let marks: MTLRenderPipelineState
    private let maps: [MTLTexture]
    private var ovation: MTLTexture
    private var glow: MTLTexture?
    private var grid: [UInt8] = []
    private var forecast: Date?
    private var kp: Double?
    private var edges: [Double: [Double?]] = [:]
    private var power: [Double: Double] = [1: 0, -1: 0]
    private var viewLines: [Double: MTLBuffer] = [:]
    private var circle: MTLBuffer?
    private var circleLat = 999.0
    private var loaded = false
    private var fetching = false
    private var fetched = Date.distantPast
    private var camera: Globe.Camera?
    private var hemisphere: Float = 1
    private var lastOverlay: Float = -1
    private let start: Float
    private var sparkle: Float = 0

    var scale: Float { 0.7 }
    var ready: Bool { loaded && globe.ready }

    var facts: String? {
        guard loaded else { return "Loading NOAA's aurora forecast." }
        guard let forecast, !grid.isEmpty else { return "The aurora forecast couldn't be loaded, so only the night side of the Earth is shown." }
        var s = "Aurora forecast for \(EarthText.utc(forecast))"
        if let kp { s += ", Kp \(EarthText.number(kp)), \(Self.activity(kp))" }
        let north = viewLatitude(1), south = viewLatitude(-1)
        switch (north, south) {
        case let (n?, s2?): s += ": likely visible above \(EarthText.latitude(n)) and \(EarthText.latitude(s2)) tonight."
        case let (n?, nil): s += ": likely visible above \(EarthText.latitude(n)) tonight."
        case let (nil, s2?): s += ": likely visible above \(EarthText.latitude(s2)) tonight."
        default: s += ": only faint aurora close to the poles."
        }
        let lat = Location.shared.latitude
        let place = Location.shared.place.isEmpty ? "Your location" : Location.shared.place
        if let line = lat >= 0 ? north : south {
            let gap = abs(lat) - abs(line)
            s += " \(place), at \(EarthText.latitude(lat)), "
            s += gap >= 0 ? "is under the oval tonight: look up." : gap > -6 ? "may see it low on the \(lat >= 0 ? "northern" : "southern") horizon."
                : "is \(Int(-gap.rounded())) degrees too far from the pole to see it."
        }
        return s
    }

    init(gpu: GPU) throws {
        globe = try Globe.shared(gpu)
        canvas = try Globe.Canvas(gpu: gpu)
        let library = try gpu.library(Globe.metal + Self.source)
        mapKernel = try gpu.kernel(library, "auroraMap")
        structureKernel = try gpu.kernel(library, "auroraStructure")
        structure = gpu.texture(4096, 8)
        march = try gpu.pipeline(library, "auroraMarch")
        add = try gpu.pipeline(library, "auroraAdd", blend: .add)
        lines = try gpu.pipeline(library, "dashedFill", vertex: "surfaceLine", blend: .alpha)
        marks = try gpu.pipeline(library, "youFill", vertex: "youMark", blend: .alpha)
        maps = (0..<2).map { _ in gpu.texture(Self.mapSize, Self.mapSize, .r16Float) }
        var zero: UInt8 = 0
        ovation = gpu.texture(1, 1, .r8Unorm, bytes: &zero, bytesPerRow: 1)
        start = Snapshot.variant.map { [0, 290, 255][$0 % 3] } ?? 0
        Location.shared.request()
        fetch(gpu)
    }

    private static func activity(_ kp: Double) -> String {
        switch kp {
        case ..<2: "quiet"
        case ..<4: "unsettled"
        case ..<5: "active"
        case ..<6: "a minor storm"
        case ..<7: "a moderate storm"
        case ..<8: "a strong storm"
        case ..<9: "a severe storm"
        default: "an extreme storm"
        }
    }

    private func fetch(_ gpu: GPU) {
        guard !fetching else { return }
        fetching = true
        fetched = .now
        Task { @MainActor in
            async let ov = Feed.object(Self.ovationURL, maxAge: 1800)
            async let k = Feed.object(Self.kpURL, maxAge: 1800)
            let (o, kpJSON) = await (ov, k)
            fetching = false
            if let parsed = Self.parseOvation(o) {
                grid = parsed.grid
                forecast = parsed.time
                upload(gpu)
            }
            kp = Self.parseKp(kpJSON) ?? kp
            loaded = true
        }
    }

    /// The forecast as a 360 × 181 grid of probabilities in percent, row 0 at 90° north, column 0 at 0° east.
    private static func parseOvation(_ json: Any?) -> (grid: [UInt8], time: Date)? {
        guard let root = json as? [String: Any], let coords = root["coordinates"] as? [[NSNumber]] else { return nil }
        var grid = [UInt8](repeating: 0, count: 360 * 181)
        for c in coords where c.count >= 3 {
            let lon = c[0].intValue, lat = c[1].intValue, v = c[2].doubleValue
            guard (0..<360).contains(lon), (-90...90).contains(lat), v.isFinite else { continue }
            grid[(90 - lat) * 360 + lon] = UInt8(min(max(v, 0), 100).rounded())
        }
        let iso = ISO8601DateFormatter()
        let time = (root["Forecast Time"] as? String).flatMap { iso.date(from: $0) } ?? .now
        return (grid, time)
    }

    /// The latest planetary K index; the feed has been both an array of objects and an array of rows with a header.
    private static func parseKp(_ json: Any?) -> Double? {
        guard let rows = json as? [Any] else { return nil }
        for row in rows.reversed() {
            if let d = row as? [String: Any], let v = (d["Kp"] as? NSNumber)?.doubleValue ?? (d["kp_index"] as? NSNumber)?.doubleValue { return v }
            if let a = row as? [Any], a.count > 1, let v = (a[1] as? String).flatMap(Double.init) ?? (a[1] as? NSNumber)?.doubleValue { return v }
        }
        return nil
    }

    private func upload(_ gpu: GPU) {
        grid.withUnsafeBytes { ovation = gpu.texture(360, 181, .r8Unorm, bytes: $0.baseAddress!, bytesPerRow: 360) }
        for h in [1.0, -1.0] {
            // The equatorward edge of a 10 percent chance, for each longitude.
            var edge = (0..<360).map { lon -> Double? in
                for step in 0...50 {
                    let lat = h * Double(40 + step)
                    if grid[(90 - Int(lat)) * 360 + lon] >= 10 { return lat }
                }
                return nil
            }
            // Smooth over 7 degrees of longitude.
            edge = (0..<360).map { i in
                let near = (-3...3).compactMap { edge[(i + $0 + 360) % 360] }
                return near.count >= 4 ? near.reduce(0, +) / Double(near.count) : nil
            }
            edges[h] = edge
            // Break the line where the edge jumps, so it doesn't run along meridians.
            var points: [SIMD4<Float>] = []
            var lines: [SIMD4<Float>] = []
            for lon in 0...360 {
                let previous = lon > 0 ? edge[(lon - 1) % 360] : nil
                if let lat = edge[lon % 360], previous.map({ abs($0 - lat) < 2.5 }) ?? true {
                    points.append(SIMD4(Globe.geo(lat, Double(lon), 1.0008), Float(lon)))
                } else if points.count > 15 {
                    lines += points + [.zero]
                    points = []
                } else {
                    points = []
                }
                if points.isEmpty, let lat = edge[lon % 360] { points.append(SIMD4(Globe.geo(lat, Double(lon), 1.0008), Float(lon))) }
            }
            if points.count > 15 { lines += points }
            viewLines[h] = lines.isEmpty ? nil : gpu.buffer(lines)
            var total = 0.0
            for row in 0..<181 {
                let lat = 90 - Double(row)
                guard lat * h > 0 else { continue }
                for lon in 0..<360 { total += Double(grid[row * 360 + lon]) * cos(lat * .pi / 180) }
            }
            power[h] = total
        }
    }

    /// The median latitude, over the night side, above which there's at least a 10 percent chance of aurora.
    private func viewLatitude(_ h: Double) -> Double? {
        guard let edge = edges[h] else { return nil }
        let night = Astro.subpoint(Astro.sun(.now), .now).lon + 180
        let lats = (-60...60).compactMap { edge[Int((night + Double($0) + 720).truncatingRemainder(dividingBy: 360))] }
        guard lats.count > 20 else { return nil }
        return lats.sorted()[lats.count / 2]
    }

    func draw(_ f: Frame) {
        if f.now.timeIntervalSince(fetched) > 1800 { fetch(f.gpu) }
        let t = f.time + start
        // Four minutes over the north pole, a swing across the night side, two and a half over the south, and back.
        let phase = t.truncatingRemainder(dividingBy: 490)
        let s: Float = phase < 240 ? 0 : phase < 290 ? (phase - 240) / 50 : phase < 440 ? 1 : 1 - (phase - 440) / 50
        let swing = s * s * s * (s * (s * 6 - 15) + 10)
        hemisphere = 1 - 2 * swing
        let sun = Globe.sun(f.now)
        // Look down past each magnetic pole from the night side, so the midnight part of the oval faces us, swaying
        // a little around the pole; swing between hemispheres across the night side.
        func view(_ pole: SIMD3<Float>) -> SIMD3<Float> {
            let away = simd_normalize(-sun - simd_dot(-sun, pole) * pole)
            let tilt = (24 + 4 * sin(t * 0.009)) * Float.pi / 180, sway = 0.45 * sin(t * 0.0037 + f.seed)
            let side = simd_cross(pole, away)
            let dir = simd_normalize(away * cos(sway) + side * sin(sway))
            return simd_normalize(pole * cos(tilt) + dir * sin(tilt))
        }
        let north = view(Self.dipole), south = view(-Self.dipole)
        let angle = acos(min(max(simd_dot(north, south), -1), 1))
        let through = simd_normalize(-sun - simd_dot(-sun, north) * north)
        // Slerp, bowing toward the night side when the two views are nearly opposite.
        var dir = angle > 3.1 ? simd_normalize(north * cos(swing * .pi) + through * sin(swing * .pi))
            : simd_normalize(north * sin((1 - swing) * angle) + south * sin(swing * angle))
        if !dir.x.isFinite { dir = north }
        let spot = Globe.latLon(dir)
        let cam = Globe.Camera(lat: Float(spot.lat), lon: Float(spot.lon), distance: 2.05, zoom: 1.7, aspect: f.aspect, date: f.now)
        camera = cam

        let light = canvas.light(for: f)
        let view = canvas.camera(cam)
        var style = Globe.Style()
        style.lights = 1.2
        style.moonlight = 0.02
        style.nightGlow = 0.006
        globe.refresh(f.now)
        globe.draw(f.cmd, into: light, camera: view, style: style, date: f.now, time: f.time)

        // The aurora: polar maps of the forecast with arcs and rays, then a half-resolution march through the sky.
        if !grid.isEmpty {
            let red = Float(0.35 + min(kp ?? 2, 9) / 9 * 0.9)
            f.cmd.compute(structureKernel, 4096, 4, uniforms: f.uniforms([t, 0, 0, 0]), textures: [structure])
            for (i, h) in [Float(1), -1].enumerated() where i == 0 ? hemisphere > -0.7 : hemisphere < 0.7 {
                f.cmd.compute(mapKernel, Self.mapSize, Self.mapSize, uniforms: f.uniforms([t, h, 0, 0], SIMD4(sun, 0), SIMD4(Self.dipole, 0)),
                              textures: [ovation, structure, maps[i]])
            }
            let w = max(f.target.width / 2, 1), h = max(f.target.height / 2, 1)
            if glow?.width != w || glow?.height != h { glow = f.gpu.texture(w, h) }
            f.cmd.draw(march, into: glow!, uniforms: f.uniforms([t, 0, 0, red], SIMD4(sun, 0), res: SIMD2(Float(w), Float(h))),
                       textures: maps, buffers: [view])
            f.cmd.draw(add, into: light, clear: false, uniforms: f.uniforms(), textures: [glow!])
        }
        canvas.develop(f, Self.look)

        // Over the picture, hairlines on the surface: the view line (10 percent) in each hemisphere, then your latitude.
        let lat = Location.shared.latitude
        if circle == nil || abs(lat - circleLat) > 0.01 {
            circleLat = lat
            circle = f.gpu.buffer((0...360).map { SIMD4(Globe.geo(lat, Double($0), 1.0008), Float($0)) })
        }
        let px = f.size.y / 1000
        if let circle {
            f.cmd.draw(lines, into: f.target, clear: false, uniforms: f.uniforms([px, 1.0, 2, 0.45], [1.0, 0.8, 0.6, 0]),
                       buffers: [view, circle], vertices: 360 * 6)
        }
        let me = SIMD4(Globe.geo(lat, Location.shared.longitude, 1.001), 0)
        f.cmd.draw(marks, into: f.target, clear: false, uniforms: f.uniforms([px, 0, 0, 0]), buffers: [view, canvas.stream(0, [me])], vertices: 1,
                   primitive: .point)

        if let synth = f.sound { play(synth, f) }
        if f.time - lastOverlay >= 0.1 || f.index == 0 {
            lastOverlay = f.time
            drawOverlay(f)
        }
    }

    /// A shimmering pad whose level follows the total auroral power in the hemisphere in view, with an
    /// occasional high glint when it's strong.
    private func play(_ synth: Synth, _ f: Frame) {
        let p = (power[hemisphere >= 0 ? 1 : -1] ?? 0)
        // Probability times cell area: a quiet day sums to about 20,000, a big storm to several hundred thousand.
        let level = Float(min(max(log10(max(p, 1)) - 3.6, 0) / 2, 1))
        let t = f.time
        let chord: [(Float, Float, Float)] = [(146.8, 0.06, -0.2), (220.0, 0.045, 0.25), (293.7, 0.03, -0.4), (329.6, 0.024, 0.4),
                                              (440.0, 0.016, 0), (554.4, 0.01, -0.5)]
        for (i, (freq, amp, pan)) in chord.enumerated() {
            let shimmer = 0.6 + 0.4 * sin(t * (0.11 + Float(i) * 0.037) + Float(i) * 1.7)
            let gain = i < 2 ? 0.5 + 0.5 * level : level
            synth.drone(i, freq * (1 + 0.0015 * sin(t * 0.07 * Float(i + 1))), amp * shimmer * (0.35 + 0.65 * gain), pan: pan)
        }
        synth.noise(0.01 + 0.02 * level, cutoff: 600 + 1400 * level)
        sparkle += f.dt * (0.05 + 0.3 * level)
        if sparkle > 1 {
            sparkle = 0
            let notes: [Float] = [880, 1174.7, 1318.5, 1760]
            synth.play(.bell, notes.randomElement()!, amp: 0.025 + 0.03 * level, pan: .random(in: -0.7...0.7), length: 5)
        }
    }

    private func drawOverlay(_ f: Frame) {
        guard let cam = camera else { return }
        let drift = CGPoint(x: CGFloat(sin(f.time * 0.007)) * 6, y: CGFloat(cos(f.time * 0.009)) * 4)
        f.overlay.draw { ctx, size in
            let dim = NSColor(white: 1, alpha: 0.42), faint = NSColor(white: 1, alpha: 0.24)
            let left = 44 + drift.x, top = 38 + drift.y
            ctx.text("AURORA FORECAST", at: CGPoint(x: left, y: top), size: 11, weight: .semibold, color: dim, tracking: 2)
            ctx.text(self.forecast.map { EarthText.utc($0) } ?? "No forecast", at: CGPoint(x: left, y: top + 18), size: 22, weight: .light,
                     mono: true, color: NSColor(white: 1, alpha: 0.6))
            if let kp = self.kp {
                ctx.text("Kp \(EarthText.number(kp)) · \(Self.activity(kp))", at: CGPoint(x: left, y: top + 48), size: 11, color: faint)
                // A small Kp gauge, 0 to 9.
                let gx = left, gy = top + 68
                for i in 0..<9 {
                    let on = Double(i) < kp
                    let c: NSColor = i >= 5 ? NSColor(srgbRed: 1, green: 0.4, blue: 0.45, alpha: on ? 0.8 : 0.15)
                        : NSColor(srgbRed: 0.45, green: 1, blue: 0.6, alpha: on ? 0.7 : 0.15)
                    ctx.setFillColor(c.cgColor)
                    ctx.fill(CGRect(x: gx + CGFloat(i) * 9, y: gy, width: 6, height: 3))
                }
            }
            // You.
            let lat = Location.shared.latitude, lon = Location.shared.longitude
            if let p = cam.overlay(Globe.geo(lat, lon, 1.001), size), cam.facing(Globe.geo(lat, lon)) > 0.5 {
                let place = Location.shared.place.isEmpty ? "You" : Location.shared.place
                ctx.text("\(place) · \(Int(abs(lat).rounded()))° \(lat >= 0 ? "N" : "S")", at: CGPoint(x: p.point.x + 10, y: p.point.y + 3),
                         size: 10.5, weight: .medium, color: NSColor(srgbRed: 1, green: 0.78, blue: 0.55, alpha: 0.75))
            }
        }
    }

    /// Exposed for the night side, where the aurora and the cities are.
    private static let look = Lens.Look(exposure: 2.2, bloom: 0.04, halation: 0.015, vignette: 0.3, aberration: 0.8, grain: 0.45)

    private static let source = """
    constant float MAXCOLAT = 50.0 * PI / 180;

    // Polar map coordinates for a unit vector: azimuthal equidistant around this hemisphere's pole.
    inline float2 polarUV(float3 n) {
        float colat = acos(clamp(abs(n.z), 0.0, 1.0));
        float2 d = n.xy / max(length(n.xy), 1e-6);
        return 0.5 + 0.5 * (colat / MAXCOLAT) * float2(d.x, -d.y);
    }

    // The aurora's structure depends only on magnetic longitude and time, so it's computed once a frame along a ring:
    // x is magnetic longitude, y one of four discrete arcs. r: how far the arc is pushed across the oval, in km (folds
    // and curls), g: how bright it is here (arcs break up and fade along their length), b: its rays, a: broad surges.
    kernel void auroraStructure(constant U &u [[buffer(0)]], texture2d<float, access::write> out [[texture(0)]],
                                uint2 gid [[thread_position_in_grid]]) {
        float mlon = (float(gid.x) + 0.5) / out.get_width() * TAU - PI;
        float2 ring = float2(cos(mlon), sin(mlon));
        float t = u.a.x, k = float(gid.y);
        float fold = (fbm(float3(ring * 2.2 + k * 3.1, t * 0.008 + k), 3) - 0.5) * 220 + sin(mlon * 3 + t * 0.013 + k * 2) * 35
                   + (fbm(float3(ring * 16 + k, t * 0.02 + 3), 3) - 0.5) * 50;
        float lit = smoothstep(0.38, 0.7, fbm(float3(ring * 2.6 + k * 7.3, t * 0.006 + k * 1.7), 3));
        lit *= 0.5 + 0.5 * noise(float3(ring * 9 + k * 4.1, t * 0.012));
        float rays = (0.4 + 0.6 * noise(float3(ring * 90 + k * 13, t * 0.04))) * (0.7 + 0.3 * noise(float3(ring * 300, t * 0.07)));
        float surge = 0.6 + 0.8 * fbm(float3(ring * 4, t * 0.015 + 7), 3);
        out.write(float4(fold, lit, rays, surge), gid);
    }

    // The forecast in percent at a unit vector.
    inline float chance(texture2d<float> ovation, float3 n) {
        constexpr sampler s(filter::linear, s_address::repeat, t_address::clamp_to_edge);
        float lat = asin(clamp(n.z, -1.0, 1.0)) * 180 / PI, lon = atan2(n.y, n.x) * 180 / PI;
        return ovation.sample(s, float2(fract((lon + 360.5) / 360), (90.5 - lat) / 181)).r * 255;
    }

    // u.a: time, hemisphere   u.b: sun   u.c: dipole axis. How much aurora each point of a polar map shows: a faint
    // diffuse glow where the forecast puts it, and discrete arcs along contours of the forecast (10, 22, 36 and 52
    // percent), so a quiet night has one thin arc and a storm several, folded and broken like real ones.
    kernel void auroraMap(constant U &u [[buffer(0)]], texture2d<float> ovation [[texture(0)]], texture2d<float> structure [[texture(1)]],
                          texture2d<float, access::write> map [[texture(2)]], uint2 gid [[thread_position_in_grid]]) {
        float2 p = (float2(gid) + 0.5) / float2(map.get_width(), map.get_height()) * 2 - 1;
        p.y = -p.y;
        float rr = length(p);
        if (rr > 1) { map.write(float4(0), gid); return; }
        float h = u.a.y;
        float colat = rr * MAXCOLAT;
        float2 d = p / max(rr, 1e-6);
        float3 n = float3(sin(colat) * d.x, sin(colat) * d.y, h * cos(colat));
        float v = chance(ovation, n);
        float diffuse = (v / 18) / (1 + v / 80);

        // The forecast's slope in percent per km, from neighbours 60 km away.
        float3 east = normalize(cross(float3(0, 0, 1), n) + float3(1e-6, 0, 0)), north = cross(n, east);
        const float reach = 60.0 / 6371;
        float2 grad = float2(chance(ovation, normalize(n + east * reach)) - chance(ovation, normalize(n - east * reach)),
                             chance(ovation, normalize(n + north * reach)) - chance(ovation, normalize(n - north * reach))) / 120;
        float slope = max(length(grad), 0.03);

        float3 axis = u.c.xyz * h;
        float3 e1 = normalize(cross(float3(0, 0, 1), axis)), e2 = cross(axis, e1);
        float x = atan2(dot(n, e2), dot(n, e1)) / TAU + 0.5;
        constexpr sampler s(filter::linear, s_address::repeat, t_address::clamp_to_edge);
        float night = smoothstep(0.35, -0.15, dot(n, u.b.xyz));
        const float levels[4] = { 10, 22, 36, 52 };
        float arcs = 0;
        for (int k = 0; k < 4; k++) {
            float4 st = structure.sample(s, float2(x, (float(k) + 0.5) / 8));
            float km = (v - levels[k]) / slope + st.r;
            // A sharp lower edge and a softer glow above it, as in real arcs.
            float arc = 0.75 * exp(-pow(km / 26, 2.0)) + 0.35 * exp(-pow(km / 90, 2.0)) * smoothstep(-30.0, 40.0, km);
            arcs += arc * st.g * st.b * st.a * (1 - 0.15 * float(k));
        }
        map.write(float4(diffuse * 0.06 + arcs * saturate(v / 8) * mix(0.25, 1.0, night)), gid);
    }

    // u.a: time, -, -, red weight   u.b: sun
    fragment float4 auroraMarch(V in [[stage_in]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                                texture2d<float> north [[texture(0)]], texture2d<float> south [[texture(1)]]) {
        float3 rd = globeRay(in.uv, u.res, c);
        float3 ro = c.eye;
        const float r0 = 1 + 85.0 / 6371, r1 = 1 + 420.0 / 6371;
        float2 outer = sphere(ro, rd, float3(0), r1);
        if (outer.y <= 0) return 0;
        float t0 = max(outer.x, 0.0), t1 = outer.y;
        float2 inner = sphere(ro, rd, float3(0), r0);
        if (inner.x > 0) t1 = min(t1, inner.x);
        // About one step per 10 km of path, between 16 and 48.
        int steps = int(clamp((t1 - t0) * 6371 / 10, 16.0, 48.0));
        float dt = (t1 - t0) / steps;
        float jitter = hash12(in.pos.xy);
        constexpr sampler s(filter::linear, address::clamp_to_zero);
        float3 col = 0;
        for (int i = 0; i < steps; i++) {
            float3 q = ro + rd * (t0 + (float(i) + jitter) * dt);
            float r = length(q);
            float km = (r - 1) * 6371;
            float3 n = q / r;
            if (abs(n.z) < cos(MAXCOLAT)) continue;
            float2 uv = polarUV(n);
            float a = n.z >= 0 ? north.sample(s, uv).r : south.sample(s, uv).r;
            if (a <= 0.002) continue;
            float green = smoothstep(88.0, 108.0, km) * exp(-max(km - 108, 0.0) / 40);
            float red = smoothstep(150.0, 230.0, km) * exp(-max(km - 230, 0.0) / 110);
            float violet = exp(-pow((km - 95) / 7, 2.0));
            float lit = smoothstep(-0.12, 0.06, dot(n, u.b.xyz));
            float3 e = float3(0.22, 0.95, 0.5) * green + float3(0.9, 0.2, 0.3) * red * u.a.w * 0.4 + float3(0.55, 0.3, 1.0) * violet * 0.2;
            col += e * a * mix(1.0, 0.25, lit) * dt * 6371;
        }
        // Long grazing paths near the limb pile up light; compress them gently so the green keeps its hue.
        col *= 0.007;
        col = col / (1 + max(col.g, col.r) * 0.9);
        col = mix(float3(dot(col, float3(0.3, 0.6, 0.1))), col, 0.72);
        return float4(col, 0);
    }

    fragment float4 auroraAdd(V in [[stage_in]], texture2d<float> glow [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        return float4(glow.sample(s, in.uv).rgb, 0);
    }

    // u.a: pixel scale, width, dash period in degrees (0 for solid), opacity   u.b: display colour   u.c: sun, and 1 to show only by night
    vertex Ribbon surfaceLine(uint vid [[vertex_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                              const device float4 *pts [[buffer(2)]]) {
        uint seg = vid / 6, corner = vid % 6;
        float4 a = pts[seg], b = pts[seg + 1];
        bool atB = corner == 2 || corner == 3 || corner == 5;
        float3 q = atB ? b.xyz : a.xyz;
        // A zero point marks a break between pieces.
        float gap = (length(a.xyz) < 0.5 || length(b.xyz) < 0.5) ? 0 : 1;
        float night = u.c.w > 0 ? smoothstep(0.15, -0.25, dot(normalize(q), u.c.xyz)) : 1;
        float alpha = u.a.w * globeFacing(q, c) * gap * night;
        Ribbon o = ribbon(corner, globeClip(a.xyz, c), globeClip(b.xyz, c), u.a.y * u.a.x, float4(u.b.rgb, 1) * alpha, u.res);
        o.along = atB ? b.w : a.w;
        if (gap == 0) o.pos = float4(0, 0, -1, 1);
        return o;
    }

    fragment float4 dashedFill(Ribbon in [[stage_in]], constant U &u [[buffer(0)]]) {
        float dash = u.a.z > 0 ? step(0.5, fract(in.along / u.a.z)) : 1;
        return in.color * saturate(in.half_width + 0.5 - abs(in.across)) * dash;
    }

    struct You { float4 pos [[position]]; float size [[point_size]]; float shown; };

    vertex You youMark(uint i [[vertex_id]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                       const device float4 *marks [[buffer(2)]]) {
        You o;
        o.pos = globeClip(marks[i].xyz, c);
        o.shown = globeFacing(marks[i].xyz, c);
        o.size = 22 * u.a.x;
        return o;
    }

    // A small crosshair: a ring with four ticks.
    fragment float4 youFill(You in [[stage_in]], float2 pc [[point_coord]]) {
        float2 p = (pc - 0.5) * 2, a = abs(p);
        float px = max(fwidth(p.x), 1e-3) * 1.1;
        float ring = 1 - smoothstep(0.0, px, abs(length(p) - 0.42));
        float ticks = (1 - smoothstep(0.0, px, a.x)) * step(0.55, a.y) * step(a.y, 0.95)
                    + (1 - smoothstep(0.0, px, a.y)) * step(0.55, a.x) * step(a.x, 0.95);
        return float4(1.0, 0.8, 0.6, 1) * saturate(ring + ticks) * 0.85 * in.shown;
    }
    """
}
