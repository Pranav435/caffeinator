import AppKit
import Metal
import simd

/// A slow flight over real Martian ground: USGS digital terrain models from MRO's Context Camera and HiRISE, lit by
/// the Sun's true direction for the local solar time at that spot right now, seen through Mars's dusty butterscotch air.
@MainActor final class MarsFlight: Scene {
    private static let hold: Float = 105
    private static let fade: Float = 2.6

    private let gpu: GPU
    private let lens: Lens
    private let light: MTLRenderPipelineState
    private var hdr: MTLTexture?
    private var terrain: MarsTerrain?
    private var site: MarsSite?
    private var next: (site: MarsSite, terrain: MarsTerrain?)?
    private var nextTask: Task<Void, Never>?
    private var order: [Int] = []
    private var tour = 0
    private var siteStart: Float = 0
    private var phase0: Double = 0
    private var transitionAt: Float?
    private var loadingFirst = true
    private var failed = Set<String>()

    private var sun = MarsSun.Position(ls: 0, equationOfTime: 0, declination: 0, localTime: 12, elevation: 40, azimuth: 180)
    private var sunAt = Date.distantPast
    private var camPos = SIMD3<Float>(0, 0, 0)
    private var camHeading: Float = 0
    private var overlayAt: Float = -1
    private var lastTime: Float = 0

    var scale: Float { 0.6 }
    var ready: Bool { !loadingFirst }
    var credit: String? {
        guard let site, terrain?.synthetic == false else { return nil }
        return "Mars: \(site.credit)"
    }

    var facts: String? {
        guard let site else { return "Fetching real terrain of Mars from the USGS Astrogeology archive." }
        if terrain?.synthetic == true { return "Offline, and no Martian terrain is saved yet, so this is an imagined stretch of cratered ground." }
        let p = sun
        var s = "Flying over \(site.name) in \(site.region), \(abs(site.lat).formatted(.number.precision(.fractionLength(1)))) degrees \(site.lat < 0 ? "south" : "north"), "
        s += "\(site.lon.formatted(.number.precision(.fractionLength(1)))) east: \(site.spoken). "
        s += "It is \(MarsSun.clock(p.localTime)) local solar time there, "
        s += p.elevation > 0 ? "with the Sun \(Int(p.elevation.rounded())) degrees above the horizon" : "and the Sun is \(Int((-p.elevation).rounded())) degrees below the horizon"
        s += ", late in \(Self.season(p.ls)) (solar longitude \(Int(p.ls)) degrees). "
        s += "Elevation data from \(site.instrument.replacingOccurrences(of: "  ·  ", with: ", "))."
        return s
    }

    static func season(_ ls: Double) -> String {
        let names = ["northern spring", "northern summer", "northern autumn", "northern winter"]
        return names[min(3, Int(ls / 90))]
    }

    init(gpu: GPU) throws {
        self.gpu = gpu
        lens = try Lens(gpu: gpu)
        light = try gpu.pipeline(gpu.library(Self.source), "marsLight")
        let start = Snapshot.variant ?? Int.random(in: 0..<1000)
        order = (0..<MarsSite.all.count).map { ($0 + start) % MarsSite.all.count }
        Task { @MainActor in await self.begin() }
    }

    // MARK: Sites

    private func begin() async {
        let first = chooseNext(.now)
        let site = MarsSite.all[first]
        var t = await MarsTerrain.load(site, gpu)
        if t == nil { failed.insert(site.id); t = await MarsTerrain.makeFallback(gpu) }
        terrain = t
        self.site = t?.synthetic == true ? MarsSite.all[first] : site
        phase0 = Double(first) * 1.7
        siteStart = lastTime
        loadingFirst = false
    }

    /// The next site to visit: the next in turn that is in daylight, if any are.
    private func chooseNext(_ date: Date) -> Int {
        let n = order.count
        for step in 0..<n {
            let i = order[(tour + step) % n]
            let s = MarsSite.all[i]
            if failed.contains(s.id) { continue }
            if MarsSun.position(date, lat: s.lat, lon: s.lon).elevation > -3 { tour = (tour + step + 1) % n; return i }
        }
        let i = order[tour % n]
        tour = (tour + 1) % n
        return i
    }

    private func prefetch(_ date: Date) {
        guard nextTask == nil, next == nil, !loadingFirst else { return }
        let i = chooseNext(date)
        let s = MarsSite.all[i]
        nextTask = Task { @MainActor in
            let t = await MarsTerrain.load(s, gpu)
            if t == nil { failed.insert(s.id) }
            next = (s, t)
            nextTask = nil
        }
    }

    // MARK: Draw

    func draw(_ f: Frame) {
        lastTime = f.time
        let now = f.now
        if hdr?.width != f.target.width || hdr?.height != f.target.height { hdr = f.gpu.texture(f.target.width, f.target.height) }
        guard terrain != nil, let site else { f.cmd.draw(light, into: f.target, uniforms: f.uniforms(), vertices: 0); return }
        if now.timeIntervalSince(sunAt) > 1 {
            sun = MarsSun.position(now, lat: site.lat, lon: site.lon)
            sunAt = now
        }
        let siteT = f.time - siteStart
        if siteT > Self.hold - 40 { prefetch(now) }

        // Fade out, swap, fade in.
        var fade: Float = 1
        if transitionAt == nil, siteT > Self.hold, let n = next {
            if n.terrain != nil { transitionAt = f.time } else { next = nil }
        }
        if let t = transitionAt {
            let e = f.time - t
            if e < Self.fade { fade = 1 - e / Self.fade }
            else {
                if let n = next, e < Self.fade + 0.05 || self.site?.id != n.site.id {
                    self.terrain = n.terrain; self.site = n.site; siteStart = f.time; phase0 = Double(f.time) * 0.013
                    sun = MarsSun.position(now, lat: n.site.lat, lon: n.site.lon); sunAt = now
                    camPos.y = 0
                    next = nil
                    chimed = false
                }
                let e2 = e - Self.fade
                fade = min(1, e2 / Self.fade)
                if e2 >= Self.fade { transitionAt = nil }
            }
            fade = fade * fade * (3 - 2 * fade)
        }
        guard let terrain = self.terrain, let site = self.site else { return }
        steer(terrain, site, siteT: Double(f.time - siteStart), dt: f.dt)

        let el = Float(sun.elevation * .pi / 180), az = Float(sun.azimuth * .pi / 180)
        let sunDir = SIMD3<Float>(cos(el) * sin(az), sin(el), cos(el) * cos(az))
        let heading = camHeading
        let pitch = Float(site.pitch) + 0.04 * sin(Float(f.time) / 31)
        let fwd = SIMD3<Float>(cos(pitch) * sin(heading), sin(pitch), cos(pitch) * cos(heading))
        let right = simd_normalize(simd_cross(SIMD3(0, 1, 0), fwd))
        let up = simd_cross(fwd, right)
        let sinEl = sin(el)
        let params = [SIMD4(camPos, 1.55), SIMD4(fwd, 0), SIMD4(right, 0), SIMD4(up, 0),
                      SIMD4(terrain.width, terrain.depth, terrain.low, terrain.high),
                      SIMD4(sunDir, sinEl),
                      SIMD4(1 / (Float(site.altitude) * 24), terrain.gsd, terrain.synthetic ? 1 : 0, Float(site.smooth)),
                      SIMD4(Float(f.time), 0, 0, 0)]
        f.cmd.draw(light, into: hdr!, uniforms: f.uniforms(), textures: [terrain.height, terrain.albedo], buffers: [gpu.buffer(params)])
        // Auto exposure: brighter when the Sun is low, as a camera would.
        let day = smoothstep(-0.2, 0.55, sinEl)
        let exposure = (0.9 + 2.4 * (1 - day)) * fade
        lens.develop(f, hdr!, Lens.Look(exposure: exposure, white: [1.0, 0.985, 0.97], bloom: 0.05, halation: 0.02, vignette: 0.32, aberration: 0.9, grain: 0.5))

        if f.time - overlayAt > (transitionAt != nil ? 0.1 : 0.5) { overlayAt = f.time; drawOverlay(f, terrain, site, fade) }
        if let synth = f.sound { play(synth, f, siteT: siteT) }
    }

    private func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    }

    /// Flies a wide, slow oval over the data, keeping the camera well above the highest ground ahead.
    private func steer(_ t: MarsTerrain, _ site: MarsSite, siteT: Double, dt: Float) {
        let period = 760.0
        let theta = phase0 + siteT * 2 * .pi / period
        let ra = 0.8 * Double(t.spread.x), rb = 0.55 * Double(t.spread.y)
        let ex = ra * cos(theta), ez = rb * sin(theta)
        let evx = -ra * sin(theta), evz = rb * cos(theta)
        let ang = Double(t.path.z), ca = cos(ang), sa = sin(ang)
        let x = Double(t.path.x) + ex * ca - ez * sa, z = Double(t.path.y) + ex * sa + ez * ca
        let vx = evx * ca - evz * sa, vz = evx * sa + evz * ca
        // Aim a little ahead of the flight line toward the middle of the data, so the picture stays full of ground.
        let tl = max(hypot(vx, vz), 1e-6)
        let aimX = Double(t.path.x) + vx / tl * 0.4 * Double(t.spread.x), aimZ = Double(t.path.y) + vz / tl * 0.4 * Double(t.spread.x)
        let heading = atan2(aimX - x, aimZ - z) + 0.08 * sin(siteT / 23)
        camHeading = Float(heading)
        var target = t.ground(Float(x), Float(z), radius: 1)
        let ahead = t.ground(Float(x + 3 * Double(site.altitude) * sin(heading)), Float(z + 3 * Double(site.altitude) * cos(heading)), radius: 1)
        target = max(target, ahead - 0.5 * Float(site.altitude)) + Float(site.altitude)
        if camPos.y == 0 { camPos.y = target }
        let rate: Float = target > camPos.y ? 1.2 : 0.2
        camPos.y += (target - camPos.y) * min(1, dt * rate)
        camPos.x = Float(x); camPos.z = Float(z)
    }

    // MARK: Sound

    private var chimed = false

    private func play(_ synth: Synth, _ f: Frame, siteT: Float) {
        let t = f.time
        // A cold, wide drone and a wind that rises and falls over minutes, stronger when the Sun is high.
        let sunny = Float(max(0, min(1, sun.elevation / 50)))
        let base: Float = 49 + Float((site?.id.count ?? 4) % 5) * 2.2
        let ratios: [Float] = [1, 1.5, 2, 3]
        for (i, r) in ratios.enumerated() {
            let breathe = 0.6 + 0.4 * sin(t * (0.03 + 0.011 * Float(i)) + Float(i) * 2.2)
            synth.drone(i, base * r, 0.05 * breathe / (1 + Float(i) * 0.5), pan: Float(i % 2) * 0.5 - 0.25)
        }
        let gust = 0.55 + 0.45 * sin(t / 19) * sin(t / 7.3 + 1)
        synth.noise(0.012 + 0.03 * gust * (0.4 + 0.6 * sunny), cutoff: 220 + 260 * gust)
        if !chimed, siteT > 1.5 {
            chimed = true
            synth.play(.bell, base * 6, amp: 0.09, length: 6)
        }
    }

    // MARK: Overlay

    private func drawOverlay(_ f: Frame, _ t: MarsTerrain, _ site: MarsSite, _ fade: Float) {
        let p = sun
        let drift = SkyKit.drift(Double(f.time) + Double(f.seed) * 50, amount: 12, seed: Double(f.seed))
        let alpha = CGFloat(fade)
        let relief = Int((t.high - t.low).rounded())
        let cam = Int(camPos.y.rounded())
        let synthetic = t.synthetic
        f.overlay.draw { ctx, size in
            let left = 56 + drift.x, bottom = size.height - 56 + drift.y
            let faint = NSColor(white: 1, alpha: 0.42 * alpha), mid = NSColor(white: 1, alpha: 0.62 * alpha)
            ctx.text("MARS FROM ORBIT", at: CGPoint(x: left, y: bottom - 118), size: 10.5, weight: .medium, color: faint, tracking: 2)
            ctx.text(synthetic ? "Imagined ground" : site.name, at: CGPoint(x: left - 1, y: bottom - 102), size: 30, weight: .light,
                     color: NSColor(white: 1, alpha: 0.88 * alpha))
            if !synthetic {
                let ns = site.lat < 0 ? "S" : "N"
                let rows = [
                    String(format: "%@  ·  %.2f° %@  %.2f° E", site.region, abs(site.lat), ns, site.lon),
                    "local solar time \(MarsSun.clock(p.localTime))  ·  Sun \(Int(p.elevation.rounded()))° up, azimuth \(Int(p.azimuth.rounded()))°  ·  Ls \(Int(p.ls))°",
                    "relief \(relief) m  ·  camera \(cam) m  ·  \(site.instrument)",
                ]
                for (i, row) in rows.enumerated() { ctx.text(row, at: CGPoint(x: left, y: bottom - 62 + CGFloat(i) * 19), size: 12.5, color: mid) }
            }
        }
    }

    // MARK: Shader

    private static let source = """
    struct MarsP { float4 cam, fwd, right, up, ext, sun, misc, extra; };
    // cam: position, focal length   ext: width, depth, low, high (metres)   sun: direction, sin(elevation)
    // misc: fog per metre, ground sample distance, synthetic, slope smoothing in cells

    constexpr sampler lin(filter::linear, mip_filter::linear, address::clamp_to_edge);

    // Ground height. Beyond the real data (nodata, filled in on load) it falls away,
    // so no streaks of filled heights stand up against the haze: it drops away to the data's lowest level.
    inline float groundAt(texture2d<float> H, texture2d<float> A, constant MarsP &P, float2 xz, float lod) {
        float2 uv = float2(xz.x / P.ext.x, 1.0 - xz.y / P.ext.y);
        float h = H.sample(lin, uv, level(lod)).r;
        float m = A.sample(lin, uv, level(max(lod, 5.0))).g;
        float inval = 1.0 - smoothstep(0.35, 0.99, m);
        if (inval > 0.0) h = mix(h, 0.0, inval);
        return mix(P.ext.z, P.ext.w, h);
    }

    inline float lodFor(constant MarsP &P, float footprint) { return clamp(log2(max(footprint / P.misc.y, 1.0)), 0.0, 10.0); }

    // The dusty sky: butterscotch near the horizon, a blue glow around a low Sun, the Sun's small white disc.
    inline float3 sky(float3 rd, float3 sd, float sinE) {
        float day = smoothstep(-0.14, 0.32, sinE);
        float h = clamp(rd.y * 2.0 + 0.1, 0.0, 1.0);
        float3 horizon = float3(0.86, 0.60, 0.42), zenith = float3(0.50, 0.33, 0.25);
        float3 c = mix(horizon, zenith, pow(h, 0.55)) * day * 0.9;
        float cs = dot(rd, sd);
        float low = 1.0 - smoothstep(0.02, 0.4, sinE);
        c += float3(0.20, 0.36, 0.78) * exp((cs - 1.0) * 55.0) * 1.1 * low * smoothstep(-0.12, 0.05, sinE);
        c += float3(1.0, 0.72, 0.5) * exp((cs - 1.0) * 7.0) * 0.28 * day;
        c += float3(0.9, 0.95, 1.0) * smoothstep(0.999975, 0.999992, cs) * 30.0 * step(-0.02, sinE);
        // Starlight when the Sun is well down.
        float night = 1.0 - smoothstep(-0.30, -0.08, sinE);
        float2 q = rd.xz / max(rd.y + 0.3, 0.2) * 220.0;
        float st = step(0.9965, hash12(floor(q))) * night * smoothstep(0.0, 0.2, rd.y) * (0.4 + 0.6 * hash12(floor(q) + 3.1));
        return c + st * 0.9 + float3(0.004, 0.005, 0.009) * night;
    }

    fragment float4 marsLight(V in [[stage_in]], constant U &u [[buffer(0)]], constant MarsP &P [[buffer(1)]],
                              texture2d<float> H [[texture(0)]], texture2d<float> A [[texture(1)]]) {
        float2 p = centered(in.uv, u.res);
        float3 ro = P.cam.xyz;
        float3 rd = normalize(p.x * P.right.xyz + p.y * P.up.xyz + P.cam.w * P.fwd.xyz);
        float3 sd = P.sun.xyz;
        float sinE = P.sun.w;
        float pxAngle = 2.0 / (P.cam.w * u.res.y);
        float W = P.ext.x, D = P.ext.y;
        float3 skyC = sky(rd, sd, sinE);

        // March over the elevation map.
        float t = 20.0, tPrev = 20.0;
        bool hit = false;
        float far = 150000.0;
        for (int i = 0; i < 300; i++) {
            float3 q = ro + rd * t;
            float lod = lodFor(P, t * pxAngle);
            float g = groundAt(H, A, P, q.xz, lod);
            float d = q.y - g;
            if (d < 0.0) { hit = true; break; }
            if (q.y > P.ext.w + 20.0 && rd.y >= 0.0) break;
            tPrev = t;
            t += max(d * 0.5, t * pxAngle * 0.35);
            if (t > far) break;
        }
        float3 col = skyC;
        if (hit) {
            float lo = tPrev, hi = t;
            for (int k = 0; k < 7; k++) {
                float mid = 0.5 * (lo + hi);
                float3 q = ro + rd * mid;
                if (q.y < groundAt(H, A, P, q.xz, lodFor(P, mid * pxAngle))) hi = mid; else lo = mid;
            }
            t = 0.5 * (lo + hi);
            float3 q = ro + rd * t;
            float foot = t * pxAngle;
            float eps = max(P.misc.y * P.misc.w, foot * 1.4);
            float lod = lodFor(P, eps);
            float hxm = groundAt(H, A, P, q.xz - float2(eps, 0), lod), hxp = groundAt(H, A, P, q.xz + float2(eps, 0), lod);
            float hzm = groundAt(H, A, P, q.xz - float2(0, eps), lod), hzp = groundAt(H, A, P, q.xz + float2(0, eps), lod);
            float3 n = normalize(float3(hxm - hxp, 2.0 * eps, hzm - hzp));

            float2 uv = float2(q.x / W, 1.0 - q.z / D);
            float a = A.sample(lin, uv, level(lodFor(P, max(foot, P.misc.y * P.misc.w * 0.6)))).r;
            float valid = smoothstep(0.7, 0.99, A.sample(lin, uv, level(max(lodFor(P, foot), 5.0))).g);
            // Fine surface texture below the data's resolution, fading with distance.
            float near = 1.0 - smoothstep(30.0 * P.misc.y, 300.0 * P.misc.y, t);
            float2 fq = q.xz / (3.0 * P.misc.y);
            float grain = fbm(fq, 4) - 0.5;
            a = clamp(a + near * 0.16 * grain * (0.5 + a), 0.0, 1.0);
            // Martian dust and sand: bright ochre where dust has settled, dark basaltic brown-grey where it has blown clear.
            float3 dark = float3(0.070, 0.058, 0.054), bright = float3(0.56, 0.35, 0.21);
            float3 albedo = mix(dark, bright, smoothstep(0.0, 0.95, pow(a, 1.25)));

            // Light: sunlight dimmed and reddened by the dusty air, soft shadows from the terrain, and bright sky fill.
            float3 tau = float3(0.32, 0.48, 0.70);
            float3 sunTrans = exp(-tau / max(sinE, 0.025)) * step(-0.05, sinE);
            float mu0 = max(dot(n, sd), 0.0);
            float shadow = 1.0;
            if (sinE > 0.0 && mu0 > 0.0) {
                float s = max(eps * 2.0, 4.0);
                float res = 1.0;
                float3 so = q + n * eps * 0.5;
                for (int j = 0; j < 40; j++) {
                    float3 sp = so + sd * s;
                    if (sp.y > P.ext.w + 10.0) break;
                    float d = sp.y - groundAt(H, A, P, sp.xz, lodFor(P, s * 0.02 + eps));
                    res = min(res, 7.0 * d / s);
                    if (res < 0.01) break;
                    s += clamp(d * 0.7, P.misc.y * 0.6, 0.2 * s + P.misc.y);
                    if (s > 14000.0) break;
                }
                shadow = smoothstep(0.0, 1.0, clamp(res, 0.0, 1.0));
            } else if (sinE <= 0.0) { shadow = 0.0; }
            float day = smoothstep(-0.14, 0.32, sinE);
            float3 skyFill = mix(float3(0.50, 0.34, 0.26), float3(0.80, 0.56, 0.42), 0.5) * (0.35 + 0.65 * (0.5 + 0.5 * n.y)) * day * 0.5;
            float3 direct = float3(1.0, 0.93, 0.82) * 2.3 * sunTrans * mu0 * shadow;
            float3 night = float3(0.010, 0.012, 0.018) * (1.0 - day);
            float3 lit = albedo * (direct + skyFill + night);

            // Aerial perspective: distant ground sinks into the butterscotch haze, and the data's edges dissolve into it.
            float dist = t;
            float tr = exp(-dist * P.misc.x);
            float edge = smoothstep(0.0, 0.22 * min(W, D), min(min(q.x, W - q.x), min(q.z, D - q.z)));
            float3 haze = skyC;
            float keep = tr * edge * valid;
            col = mix(haze, lit, keep);
            if (hit && (q.x < 0.0 || q.x > W || q.z < 0.0 || q.z > D)) col = haze;
        }
        return float4(col, 1);
    }
    """
}
