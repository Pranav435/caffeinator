import AppKit
import Metal
import simd

/// The real Moon as it stands in the sky right now: NASA's LRO colour map and LOLA laser-altimeter elevations, turned to
/// the exact libration and lit from the Sun's true direction, so crater shadows lengthen along the real terminator.
/// A slow telescope drifts and zooms from one terminator landmark to the next.
@MainActor final class MoonTonight: Scene {
    static let colorURL = URL(string: "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_4k.tif")!
    static let heightURL = URL(string: "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16_uint.tif")!
    static let creditLine = "Moon: NASA's Scientific Visualization Studio, LRO (LROC colour, LOLA elevation). Stars: ESO/S. Brunier"

    private let gpu: GPU
    private let lens: Lens
    private let light: MTLRenderPipelineState
    private var color: MTLTexture?
    private var height: MTLTexture?
    private var sky: MTLTexture?
    private let blank: MTLTexture
    private var hdr: MTLTexture?
    private var loading = true

    private var view: Lunar.View
    private var viewAt = Date.distantPast
    private var stars: [SIMD4<Float>] = []
    private var starBuffer: MTLBuffer
    private var lookEq: SIMD3<Double> = [1, 0, 0], eastEq: SIMD3<Double> = [0, 1, 0], northEq: SIMD3<Double> = [0, 0, 1]
    private let first: Int
    private var legNames: [Int: String] = [:]
    private var camera = (center: SIMD2<Float>(0, 0), zoom: Float(1))
    private var target: LunarFeature?
    private var lastLeg = -1
    private var overlayAt: Float = -1

    var scale: Float { 0.75 }
    var ready: Bool { !loading }
    var credit: String? { color == nil && height == nil ? nil : Self.creditLine }

    var facts: String? {
        let v = view
        var s = "\(Self.phaseName(v)), \(Self.percent(v.illuminated)) of the disc lit, \(String(format: "%.1f", v.age)) days after new Moon. "
        s += "\(Self.thousands(v.distance)) kilometres away, \(Self.arcText(v.diameter)) wide. "
        s += "Libration tips \(Self.librationText(v)). "
        if let t = target { s += "The telescope is on \(t.name)\(t.note.isEmpty ? "" : ": \(t.note)"). " }
        if color == nil { s += "The NASA maps are not available, so the surface is only sketched. " }
        return s
    }

    init(gpu: GPU) throws {
        self.gpu = gpu
        lens = try Lens(gpu: gpu)
        light = try gpu.pipeline(gpu.library(Self.source), "moonLight")
        var grey: UInt32 = 0xFF80_8080
        blank = gpu.texture(1, 1, .rgba8Unorm_srgb, bytes: &grey, bytesPerRow: 4)
        view = Lunar.view(.now)
        starBuffer = gpu.buffer(64 * MemoryLayout<SIMD4<Float>>.stride)
        first = Snapshot.variant ?? Int.random(in: 0..<1000)
        Task { @MainActor in
            color = await Imagery.texture(Self.colorURL, gpu, maxSize: 4096, maxAge: 365 * 86400)
            height = await Imagery.texture(Self.heightURL, gpu, kind: .gray16, maxSize: 5760, maxAge: 365 * 86400)
            sky = await MilkyWay.texture(gpu)
            loading = false
        }
    }

    // MARK: Wording

    static func percent(_ k: Double) -> String { k > 0.995 ? "99.9 percent" : String(format: "%.1f percent", k * 100) }
    static func thousands(_ x: Double) -> String {
        let f = NumberFormatter()
        f.locale = Locale(identifier: "en_US")
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: Int(x.rounded()))) ?? "\(Int(x))"
    }

    static func phaseName(_ v: Lunar.View) -> String {
        let k = v.illuminated
        if k < 0.02 { return "New Moon" }
        if k > 0.985 { return "Full Moon" }
        if abs(k - 0.5) < 0.03 { return v.waxing ? "First quarter" : "Last quarter" }
        let shape = k < 0.5 ? "crescent" : "gibbous"
        return (v.waxing ? "Waxing " : "Waning ") + shape
    }

    static func arcText(_ degrees: Double) -> String {
        let total = Int((degrees * 3600).rounded())
        return "\(total / 60) arcminutes \(total % 60) arcseconds"
    }

    static func arcShort(_ degrees: Double) -> String {
        let total = Int((degrees * 3600).rounded())
        return String(format: "%d′ %02d″", total / 60, total % 60)
    }

    static func librationText(_ v: Lunar.View) -> String {
        let l = v.libration.x, b = v.libration.y
        let lon = String(format: "%.1f degrees toward the %@ edge", abs(l), l >= 0 ? "Mare Crisium" : "Mare Orientale")
        let lat = String(format: "%.1f degrees toward the %@ pole", abs(b), b >= 0 ? "north" : "south")
        return "\(lon) and \(lat)"
    }

    // MARK: Camera

    /// Landmarks worth a look: visible, lit by a low Sun and not too big for the frame.
    private func candidates() -> [LunarFeature] {
        LunarFeature.all.filter { f in
            let d = f.direction, vis = Self.dot(d, view.toObserver), sun = Self.dot(d, view.toSun)
            return f.kind != .landing && vis > 0.35 && sun > 0.012 && sun < 0.42 && f.size < 650 && f.size > 25 && f.fame <= 3
        }
    }

    private static func dot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double { simd_dot(a, b) }

    private func screen(_ f: LunarFeature) -> SIMD2<Float> {
        let d = f.direction
        return SIMD2(Float(Self.dot(d, view.west)), Float(Self.dot(d, view.north)))
    }

    /// Where the telescope looks at `time`: legs of 85 seconds, each easing from one landmark to the next through a
    /// wide view, with a small tracking wobble.
    private func steer(_ time: Double) {
        let leg = 85.0
        let k = Int(time / leg), s = (time / leg) - Double(k)
        let all = candidates()
        func name(_ n: Int) -> LunarFeature? {
            if let known = legNames[n], let f = LunarFeature.all.first(where: { $0.name == known }) { return f }
            guard !all.isEmpty else { return nil }
            var pick = all[(first &* 7 &+ n &* 5) % all.count]
            if let before = legNames[n - 1], pick.name == before { pick = all[(first &* 7 &+ n &* 5 &+ 1) % all.count] }
            legNames[n] = pick.name
            return pick
        }
        let a = name(k), b = name(k + 1)
        let ease = Float(s * s * (3 - 2 * s))
        func zoomFor(_ f: LunarFeature?) -> Float {
            guard let f else { return 1 }
            return min(max(Float(2 * 1737.4 / (0.8 * f.size * 1.7)), 1.5), 3.4)
        }
        let wide: Float = 1.0
        // Zoom is highest at each landmark, lowest between.
        let near = Float(0.5 + 0.5 * cos(2 * Double.pi * s))
        let za = zoomFor(a), zb = zoomFor(b)
        let zoomNear = za + (zb - za) * ease
        let zoom = wide + (zoomNear - wide) * near * near
        var centerA = SIMD2<Float>(0, 0), centerB = centerA
        if let a { centerA = screen(a) }
        if let b { centerB = screen(b) }
        var center = centerA + (centerB - centerA) * ease
        // Keep the middle of a wide view near the disc's centre, and the zoomed view on the landmark.
        center *= near * near * 0.96 + 0.0
        let t = Float(time)
        center += SIMD2(0.006 * sin(t / 11), 0.005 * sin(t / 17 + 1)) / zoom
        camera = (center, zoom)
        target = s < 0.5 ? a : b
        if k != lastLeg { lastLeg = k }
    }

    // MARK: Draw

    func draw(_ f: Frame) {
        let now = f.now
        if now.timeIntervalSince(viewAt) > 1 || viewAt == .distantPast { refresh(now) }
        if hdr?.width != f.target.width || hdr?.height != f.target.height { hdr = f.gpu.texture(f.target.width, f.target.height) }
        let t = Double(f.time) + Double(f.seed) * 3.7
        steer(t)

        let v = view
        func v4(_ x: SIMD3<Double>, _ w: Float = 0) -> SIMD4<Float> { SIMD4(Float(x.x), Float(x.y), Float(x.z), w) }
        let fit: Float = 0.82
        let rs = fit * camera.zoom
        let angular = Float(asin(Lunar.radius / v.distance))
        // The Moon-lit sunlight is normalised to a well-exposed 1; earthshine is the Earth's phase seen from the Moon.
        let earth = Float(0.011 * pow(1 - v.illuminated, 1.2))
        var params = [v4(v.west), v4(v.north), v4(v.toObserver), v4(v.toSun),
                      SIMD4(camera.center.x, camera.center.y, rs, angular),
                      v4(lookEq), v4(eastEq), v4(northEq),
                      SIMD4(earth, 1, Float(Lunar.rad(v.phaseAngle)), Float(stars.count)),
                      SIMD4(color != nil ? 1 : 0, height != nil ? 1 : 0, sky != nil ? 1 : 0, Float(v.illuminated))]
        let buffer = gpu.buffer(params)
        params.removeAll()
        f.cmd.draw(light, into: hdr!, uniforms: f.uniforms(), textures: [color ?? blank, height ?? blank, sky ?? blank], buffers: [buffer, starBuffer])
        lens.develop(f, hdr!, Lens.Look(exposure: 1.0, white: [1.0, 0.985, 0.95], bloom: 0.05, halation: 0.025, vignette: 0.3, aberration: 0.8, grain: 0.45))

        if f.time - overlayAt > 0.08 { overlayAt = f.time; drawOverlay(f, rs) }
        if let synth = f.sound { play(synth, f) }
    }

    private func refresh(_ now: Date) {
        view = Lunar.view(now)
        viewAt = now
        lookEq = Lunar.unit(view.ra, view.dec)
        eastEq = simd_normalize(simd_cross(SIMD3(0, 0, 1), lookEq))
        northEq = simd_cross(lookEq, eastEq)
        // Catalogue stars within three degrees, as offsets west and north of the Moon in radians.
        var list: [SIMD4<Float>] = []
        for s in SkyCatalog.stars where s.mag < 6.5 {
            let d = Lunar.unit(Double(s.ra), Double(s.dec))
            let c = simd_dot(d, lookEq)
            guard c > 0.9986 else { continue }
            list.append(SIMD4(Float(simd_dot(d, -eastEq) / c), Float(simd_dot(d, northEq) / c), s.mag, s.bv))
            if list.count == 64 { break }
        }
        stars = list
        list.withUnsafeBytes { if let base = $0.baseAddress { starBuffer.contents().copyMemory(from: base, byteCount: $0.count) } }
    }

    // MARK: Sound

    private func play(_ synth: Synth, _ f: Frame) {
        let t = f.time, k = Float(view.illuminated)
        // A slow, open chord on A: the brighter the Moon, the more upper partials ring.
        let ratios: [Float] = [1, 1.5, 2, 3, 4, 5]
        for (i, r) in ratios.enumerated() {
            let breathe = 0.65 + 0.35 * sin(t * (0.04 + 0.013 * Float(i)) + Float(i) * 1.9)
            let amp: Float = (i < 2 ? 0.07 : 0.035 * (0.25 + k)) * breathe / (1 + 0.25 * Float(i))
            synth.drone(i, 55 * r, amp, pan: Float(i % 3 - 1) * 0.35)
        }
        synth.noise(0.004 + 0.002 * k, cutoff: 300)
        // A soft bell each time the telescope settles on a landmark, pitched by its latitude.
        let leg = Int(f.time / 85)
        if leg != chimed, camera.zoom > 1.2 {
            chimed = leg
            let lat = Float(target?.lat ?? 0)
            let notes: [Float] = [220, 247.5, 275, 330, 371.25, 440, 495, 550]
            synth.play(.bell, notes[Int((lat + 90) / 180 * 7.99) % 8], amp: 0.12, pan: 0, length: 5)
        }
    }

    private var chimed = -1

    // MARK: Overlay

    private func drawOverlay(_ f: Frame, _ rs: Float) {
        let v = view, cam = camera
        let drift = SkyKit.drift(Double(f.time) + Double(f.seed) * 50, amount: 12, seed: Double(f.seed))
        let features = LunarFeature.all
        let focus = target
        let hasImages = color != nil
        f.overlay.draw { ctx, size in
            let left = 56 + drift.x, bottom = size.height - 56 + drift.y
            let faint = NSColor(white: 1, alpha: 0.42), mid = NSColor(white: 1, alpha: 0.62)
            ctx.text("THE MOON  ·  \(SkyKit.utc(f.now)) UTC", at: CGPoint(x: left, y: bottom - 118), size: 10.5, weight: .medium, color: faint, tracking: 2)
            ctx.text(Self.phaseName(v), at: CGPoint(x: left - 1, y: bottom - 102), size: 30, weight: .light, color: NSColor(white: 1, alpha: 0.88))
            let rows = [
                String(format: "%.1f %% illuminated  ·  age %.1f days", v.illuminated * 100, v.age),
                "\(Self.thousands(v.distance)) km  ·  \(Self.arcShort(v.diameter)) across",
                String(format: "libration  %+.1f° lon  %+.1f° lat  ·  bright limb PA %.0f°", v.libration.x, v.libration.y, v.brightLimb),
            ]
            for (i, row) in rows.enumerated() { ctx.text(row, at: CGPoint(x: left, y: bottom - 62 + CGFloat(i) * 19), size: 12.5, color: mid) }

            // Instrument line, top left.
            let sourceText = hasImages ? "LRO  ·  LROC colour  ·  LOLA elevation 16 px/°" : "Offline  ·  sketch of the Moon"
            ctx.text(sourceText, at: CGPoint(x: left, y: 52 + drift.y), size: 10.5, weight: .medium, color: faint, tracking: 1.6)

            // Scale bar at the lower right.
            let pointsPerKm = CGFloat(rs) * size.height / 2 / 1737.4
            let nice: [Double] = [10, 20, 50, 100, 200, 500, 1000]
            let want = 150.0 / Double(pointsPerKm)
            let km = nice.last { $0 <= want * 1.4 } ?? 10
            let barX = size.width - 56 - drift.x - CGFloat(km) * pointsPerKm, barY = bottom - 8
            ctx.setStrokeColor(NSColor(white: 1, alpha: 0.55).cgColor)
            ctx.setLineWidth(1)
            ctx.move(to: CGPoint(x: barX, y: barY - 4)); ctx.addLine(to: CGPoint(x: barX, y: barY + 4))
            ctx.move(to: CGPoint(x: barX, y: barY)); ctx.addLine(to: CGPoint(x: barX + CGFloat(km) * pointsPerKm, y: barY))
            ctx.move(to: CGPoint(x: barX + CGFloat(km) * pointsPerKm, y: barY - 4)); ctx.addLine(to: CGPoint(x: barX + CGFloat(km) * pointsPerKm, y: barY + 4))
            ctx.strokePath()
            ctx.text("\(Int(km)) km", at: CGPoint(x: barX + CGFloat(km) * pointsPerKm / 2, y: barY - 18), size: 10.5, weight: .medium, color: mid,
                     anchor: CGPoint(x: 0.5, y: 0), tracking: 1)

            // Names of landmarks near the terminator that are in the frame.
            let aspect = size.width / size.height
            var placed: [CGPoint] = []
            let frameKm = 2 * 1737.4 / Double(rs)
            let ranked = features.filter { $0.kind != .landing || cam.zoom > 2 }.sorted { ($0.fame, -$0.size) < ($1.fame, -$1.size) }
            for ft in ranked {
                guard placed.count < 7, ft.fame <= (cam.zoom > 2.2 ? 4 : cam.zoom > 1.4 ? 3 : 2) else { continue }
                let d = ft.direction
                let vis = simd_dot(d, v.toObserver), sun = simd_dot(d, v.toSun)
                guard vis > 0.12, sun > -0.01, sun < 0.45, ft.size < frameKm * 0.7, ft.size > frameKm * 0.025 || ft.kind == .landing else { continue }
                let sx = Float(simd_dot(d, v.west)) - cam.center.x, sy = Float(simd_dot(d, v.north)) - cam.center.y
                let x = size.width / 2 * (1 + CGFloat(sx * rs) / aspect), y = size.height / 2 * (1 - CGFloat(sy * rs))
                guard x > 120, x < size.width - 120, y > 90, y < size.height - 150 else { continue }
                let p = CGPoint(x: x, y: y)
                if placed.contains(where: { hypot($0.x - p.x, $0.y - p.y) < 70 }) { continue }
                placed.append(p)
                let fade = CGFloat(min(1, max(0, (sun + 0.01) / 0.03)))
                let isFocus = focus?.name == ft.name
                let a = (isFocus ? 0.85 : 0.5) * fade
                ctx.setStrokeColor(NSColor(white: 1, alpha: 0.4 * fade).cgColor)
                ctx.setLineWidth(0.7)
                ctx.move(to: CGPoint(x: x + 6, y: y - 6)); ctx.addLine(to: CGPoint(x: x + 22, y: y - 22)); ctx.addLine(to: CGPoint(x: x + 34, y: y - 22))
                ctx.strokePath()
                ctx.text(ft.name.uppercased(), at: CGPoint(x: x + 38, y: y - 22), size: isFocus ? 11.5 : 10, weight: .medium,
                         color: NSColor(white: 1, alpha: a), anchor: CGPoint(x: 0, y: 0.5), tracking: 1.4)
                if isFocus && !ft.note.isEmpty && ft.kind != .landing {
                    ctx.text(ft.note, at: CGPoint(x: x + 38, y: y - 8), size: 10.5, color: NSColor(white: 1, alpha: 0.5 * fade), anchor: CGPoint(x: 0, y: 0.5))
                }
            }
        }
    }

    // MARK: Shader

    private static let source = """
    struct MoonP { float4 right, up, obs, sun, cam, look, east, north, misc, flags; };
    // cam: centre (Moon radii), screen units per Moon radius, angular radius   misc: earthshine, -, phase angle, star count
    // flags: has colour, has height, has sky, lit fraction

    constexpr sampler lin(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge);
    constexpr sampler lin0(filter::linear, s_address::repeat, t_address::clamp_to_edge);
    constant float MOON_R = 1737.4;

    inline float2 mapUV(float lon, float lat) { return float2(lon / TAU + 0.5, 0.5 - lat / PI); }
    inline float kmOf(float v) { return v * 32.7675 - 10.0; }

    // Cubic B-spline height in km from four bilinear taps, so slopes stay smooth when the map is magnified.
    inline float height(texture2d<float> t, float2 uv) {
        float2 sz = float2(t.get_width(), t.get_height());
        float2 x = uv * sz - 0.5, i = floor(x), f = x - i;
        float2 w0 = (1 - f) * (1 - f) * (1 - f) / 6, w1 = (4 - 6 * f * f + 3 * f * f * f) / 6;
        float2 w2 = (1 + 3 * f + 3 * f * f - 3 * f * f * f) / 6, w3 = f * f * f / 6;
        float2 g0 = w0 + w1, g1 = w2 + w3;
        float2 a = (i - 1 + w1 / g0 + 0.5) / sz, b = (i + 1 + w3 / g1 + 0.5) / sz;
        float v = g0.y * (g0.x * t.sample(lin0, float2(a.x, a.y), level(0)).r + g1.x * t.sample(lin0, float2(b.x, a.y), level(0)).r)
                + g1.y * (g0.x * t.sample(lin0, float2(a.x, b.y), level(0)).r + g1.x * t.sample(lin0, float2(b.x, b.y), level(0)).r);
        return kmOf(v);
    }

    // How much of the Sun reaches a point at the given height, marching toward it over the elevation map.
    inline float sunlight(texture2d<float> t, float lon, float lat, float h0, float3 sunLocal, float lod0) {
        float horizontal = length(sunLocal.xy);
        if (sunLocal.z < 0.002) return 0.0;
        float tanE = sunLocal.z / max(horizontal, 1e-4);
        float2 dir = sunLocal.xy / max(horizontal, 1e-4);
        float cl = max(cos(lat), 0.08);
        float lit = 1.0, d = 0.9;
        for (int i = 0; i < 40; i++) {
            d *= 1.15;
            float ray = h0 + 0.02 + d * tanE - d * d / (2 * MOON_R);
            if (ray > 11.0) break;
            float2 uv = mapUV(lon + d * dir.x / (MOON_R * cl), clamp(lat + d * dir.y / MOON_R, -1.55, 1.55));
            float hs = kmOf(t.sample(lin, uv, level(lod0 + log2(max(d / 1.9, 1.0)) * 0.5)).r);
            lit = min(lit, clamp(0.5 + (ray - hs) / (d * 0.016 + 0.06), 0.0, 1.0));
        }
        return lit;
    }

    fragment float4 moonLight(V in [[stage_in]], constant U &u [[buffer(0)]], constant MoonP &P [[buffer(1)]],
                              constant float4 *stars [[buffer(2)]], texture2d<float> col [[texture(0)]],
                              texture2d<float> hgt [[texture(1)]], texture2d<float> milky [[texture(2)]]) {
        float2 p = centered(in.uv, u.res);
        float rs = P.cam.z;
        float2 q = p / rs + P.cam.xy;
        float pxPerRadius = rs * u.res.y * 0.5;
        float pixAngle = P.cam.w / pxPerRadius;

        // The sky: the Milky Way's glow and catalogue stars at the Moon's own place among them.
        float3 look = P.look.xyz;
        float2 sk = q * P.cam.w;
        float3 dirEq = normalize(look - P.east.xyz * sk.x + P.north.xyz * sk.y);
        float3 skyCol = float3(0.0004);
        if (P.flags.z > 0.5) skyCol += 0.09 * milkyWay(milky, dirEq, pixAngle * 3.0);
        for (int i = 0; i < int(P.misc.w); i++) {
            float4 s = stars[i];
            float2 d = (sk - s.xy) / pixAngle;
            float r2 = dot(d, d);
            float amp = 1.4 * pow(10.0, -0.4 * (s.z - 2.5));
            float3 tint = mix(float3(0.7, 0.8, 1.0), float3(1.0, 0.8, 0.55), smoothstep(-0.1, 1.5, s.w));
            skyCol += tint * amp * (exp(-r2 / 1.3) + 0.02 * exp(-sqrt(r2) / 6.0));
        }

        float r2m = dot(q, q);
        float coverage = clamp((1.0 - sqrt(r2m)) * pxPerRadius + 0.5, 0.0, 1.0);
        float3 outCol = skyCol * (1.0 - coverage);
        // A faint veil of scattered light around the bright disc.
        float rr = sqrt(r2m);
        outCol += P.flags.w * 0.012 * exp(-(rr - 1.0) * 2.6) * step(1.0, rr) * float3(0.9, 0.93, 1.0);
        if (coverage <= 0.0) return float4(outCol, 1);

        float z = sqrt(max(1.0 - r2m, 1e-5));
        float3 n = normalize(q.x * P.right.xyz + q.y * P.up.xyz + z * P.obs.xyz);
        float lon = atan2(n.y, n.x), lat = asin(clamp(n.z, -1.0, 1.0));
        float2 uv = mapUV(lon, lat);

        // Albedo from the LROC map; a mottled grey stands in until it arrives.
        float3 alb;
        if (P.flags.x > 0.5) {
            alb = max(col.sample(lin, uv).rgb, 0.0);
        } else {
            float m = fbm(float3(n * 3.0 + 7.0), 5);
            alb = float3(mix(0.05, 0.16, smoothstep(0.42, 0.62, m))) * float3(1.0, 0.97, 0.93);
        }

        // Fine surface texture beyond the map's resolution: faint craters and regolith.
        float3 sp = n * 900.0;
        float detail = fbm(sp * 0.35, 4) - 0.5;
        float fine = noise(sp * 2.7) - 0.5;
        alb *= 1.0 + 0.16 * detail + 0.05 * fine;

        // Local surface normal from the LOLA elevations.
        float3 east = normalize(cross(float3(0, 0, 1), n));
        float3 north = cross(n, east);
        float3 nl = n;
        float h0 = 0.0;
        float3 sunLocal = float3(dot(P.sun.xyz, east), dot(P.sun.xyz, north), dot(P.sun.xyz, n));
        if (P.flags.y > 0.5) {
            float2 sz = float2(hgt.get_width(), hgt.get_height());
            float cl = max(cos(lat), 0.15);
            float2 du = float2(1.1 / (sz.x * cl), 1.1 / sz.y);
            float he = height(hgt, uv + float2(du.x, 0)) - height(hgt, uv - float2(du.x, 0));
            float hn = height(hgt, uv - float2(0, du.y)) - height(hgt, uv + float2(0, du.y));
            float gx = he / (2 * du.x * TAU * MOON_R * max(cos(lat), 0.02));
            float gy = hn / (2 * du.y * PI * MOON_R);
            nl = normalize(n - gx * east - gy * north);
            h0 = height(hgt, uv);
        }
        // Rough, sub-pixel slopes from the same texture, so crater walls catch light below the map's resolution.
        float bump = 0.55 * (noise(sp * 1.1) - 0.5) + 0.3 * (noise(sp * 3.1) - 0.5);
        float3 tang = normalize(cross(nl, float3(0.3, 0.8, 0.5)));
        nl = normalize(nl + 0.22 * bump * tang + 0.22 * (noise(sp * 1.7 + 9.0) - 0.5) * cross(nl, tang));

        float mu0s = max(dot(n, P.sun.xyz), 0.0);
        float mu0 = clamp(mu0s + 1.35 * (dot(nl, P.sun.xyz) - mu0s), 0.0, 1.0);
        float mu = max(dot(nl, P.obs.xyz), 0.04);
        float lit = (P.flags.y > 0.5) ? sunlight(hgt, lon, lat, h0, sunLocal, 0.0) : step(0.0, sunLocal.z);
        lit = mix(1.0, lit, smoothstep(0.1, 0.4, z)) * step(0.0, sunLocal.z);
        // Lommel-Seeliger scattering with an opposition surge at full Moon.
        float surge = 1.0 + 0.4 * exp(-P.misc.z / 0.09);
        float3 sunlit = alb * 2.0 * mu0 / (mu0 + mu) * lit * surge * float3(1.0, 0.985, 0.95);
        // Earthshine, softly blue, lights every side of the Moon facing us.
        float3 earthlit = alb * P.misc.x * float3(0.8, 0.92, 1.12) * (0.4 + 0.6 * mu);
        float3 moon = (sunlit + earthlit) * 1.35;
        outCol += moon * coverage;
        return float4(outCol, 1);
    }
    """
}
