import AppKit
import Metal
import simd

/// The pulsar map from the Pioneer plaques and the Voyager Golden Record, rebuilt in 3D over a faint Milky Way:
/// lines from the Sun to fourteen pulsars, each as long as the pulsar is far, each marked in binary with its period
/// in units of the hydrogen line, and each pulsar flickering at its true rate. Two at a time come into focus, clicking.
@MainActor final class Pulsars: Scene {
    struct Pulsar {
        let name: String
        let nickname: String?
        /// Period in seconds, galactic longitude and latitude in degrees, distance in kiloparsecs.
        let period: Double
        let l: Double, b: Double
        let distance: Double
    }

    /// The plaque's fourteen pulsars with today's values from the ATNF Pulsar Catalogue, version 2.9.0
    /// (Manchester et al. 2005, https://www.atnf.csiro.au/research/pulsar/psrcat/): P0, Gl, Gb and DIST, the
    /// best distance estimate (from parallax or associations where known, else the YMW16 model of the dispersion measure).
    static let all = [
        Pulsar(name: "B1727-47", nickname: nil, period: 0.829946, l: 342.565, b: -7.669, distance: 0.70),
        Pulsar(name: "B1451-68", nickname: nil, period: 0.263377, l: 313.868, b: -8.543, distance: 0.43),
        Pulsar(name: "B1240-64", nickname: nil, period: 0.388485, l: 302.051, b: -1.532, distance: 2.0),
        Pulsar(name: "B0833-45", nickname: "Vela", period: 0.0893283, l: 263.552, b: -2.787, distance: 0.28),
        Pulsar(name: "B0950+08", nickname: nil, period: 0.253065, l: 228.908, b: 43.697, distance: 0.261),
        Pulsar(name: "B0823+26", nickname: nil, period: 0.530661, l: 196.964, b: 31.743, distance: 0.50),
        Pulsar(name: "B0531+21", nickname: "Crab", period: 0.0333924, l: 184.557, b: -5.784, distance: 2.0),
        Pulsar(name: "B0525+21", nickname: nil, period: 3.745539, l: 183.856, b: -6.896, distance: 1.215),
        Pulsar(name: "B0329+54", nickname: nil, period: 0.714520, l: 144.995, b: -1.221, distance: 1.68),
        Pulsar(name: "B2217+47", nickname: nil, period: 0.538469, l: 98.385, b: -7.598, distance: 2.383),
        Pulsar(name: "B2016+28", nickname: nil, period: 0.557953, l: 68.099, b: -3.983, distance: 0.98),
        Pulsar(name: "B1933+16", nickname: nil, period: 0.358745, l: 52.436, b: -2.093, distance: 3.7),
        Pulsar(name: "B1929+10", nickname: nil, period: 0.226519, l: 47.382, b: -3.885, distance: 0.31),
        Pulsar(name: "B1642-03", nickname: nil, period: 0.387692, l: 14.114, b: 26.062, distance: 4.0),
    ]

    /// Pairs brought into focus together: about equally far, so the camera can come close enough to read both, and
    /// pointing different ways. The first, Vela and the Crab, is heard over a wide view of the whole galaxy.
    private static let pairs = [[3, 6], [3, 12], [4, 1], [8, 7], [6, 2], [5, 0], [10, 9], [11, 13]]

    /// The period of the hydrogen 21 cm line, the plaque's unit of time.
    static let hydrogen = 1 / 1_420_405_751.768

    /// A period written as the plaque writes it: a whole number of hydrogen periods, in binary, most significant bit first.
    static func bits(_ period: Double) -> [Bool] {
        var n = UInt64((period / hydrogen).rounded())
        var out: [Bool] = []
        while n > 0 { out.insert(n & 1 == 1, at: 0); n >>= 1 }
        return out
    }

    static func position(_ p: Pulsar) -> SIMD3<Double> {
        let l = Astro.rad(p.l), b = Astro.rad(p.b)
        return SIMD3(cos(b) * cos(l), cos(b) * sin(l), sin(b)) * p.distance
    }

    static func lightYears(_ kpc: Double) -> Int {
        let ly = kpc * 3261.56
        let unit = pow(10, floor(log10(ly)) - 1)
        return Int((ly / unit).rounded() * unit)
    }

    static func describe(_ period: Double) -> String {
        if period < 0.1 { return "\(SpaceKit.number(period * 1000, decimals: 1)) milliseconds" }
        if period < 1 { return "\(Int((period * 1000).rounded())) milliseconds" }
        return "\(SpaceKit.number(period, decimals: 2)) seconds"
    }

    private let discPipe: MTLRenderPipelineState
    private let linePipe: MTLRenderPipelineState
    private let glowPipe: MTLRenderPipelineState
    private let tonePipe: MTLRenderPipelineState
    private let labelPipe: MTLRenderPipelineState
    private let bloom: Bloom
    private let segments: MTLBuffer
    private let segmentCount: Int
    private let rings: [Ring]
    private var light: MTLTexture?
    /// The galaxy behind, smooth enough for half resolution.
    private var backdrop: MTLTexture?
    private var labels: Labels?

    private var pair: Int
    private var since: Float = 0
    private var focus: [Float]
    private var target = SIMD3<Double>.zero
    private var distance = 9.0
    private var counts: [Int]
    /// Seconds since the scene began, kept in double precision so the fast pulsars keep time for days.
    private var clock = 0.0

    var scale: Float { 1 }
    var facts: String? {
        let lines = Self.pairs[pair].map { i -> String in
            let p = Self.all[i]
            let name = "PSR \(p.name.replacingOccurrences(of: "-", with: "\u{2212}"))"
            let title = p.nickname.map { "\(name), the \($0)" } ?? "\(name), a"
            return "\(title) pulsar: spins every \(Self.describe(p.period)), \(SpaceKit.number(Double(Self.lightYears(p.distance)))) light-years away"
        }
        let wide = pair == 0 ? "The pulsar map from the Pioneer plaques: 14 pulsars, the Sun 26,700 light-years from the galactic centre. " : ""
        return wide + lines.joined(separator: ". ") + "."
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(SpaceKit.metal + Self.source)
        discPipe = try gpu.pipeline(library, "galaxyDisc")
        linePipe = try gpu.pipeline(library, "lineFragment", vertex: "plaqueVertex", blend: .add)
        glowPipe = try gpu.pipeline(library, "glowFragment", vertex: "beaconVertex", blend: .add)
        tonePipe = try gpu.pipeline(library, "pulsarTone")
        labelPipe = try gpu.pipeline(library, "labelFragment", vertex: "labelVertex", blend: .alpha)
        bloom = try Bloom(gpu: gpu)
        rings = (0..<4).map { _ in Ring(gpu: gpu, bytes: 2048) }
        pair = (Snapshot.variant ?? Int.random(in: 0..<Self.pairs.count)) % Self.pairs.count
        focus = [Float](repeating: 0, count: Self.all.count)
        counts = [Int](repeating: 0, count: Self.all.count)

        // Segments in kiloparsecs from the Sun: start, end, then (pulsar index, kind, 0, bit).
        // Kind 0 is a pulsar's line, 1 a binary tick, 2 the long line toward the galactic centre.
        var segs: [SIMD4<Float>] = []
        func f3(_ v: SIMD3<Double>) -> SIMD3<Float> { SIMD3(Float(v.x), Float(v.y), Float(v.z)) }
        for (i, p) in Self.all.enumerated() {
            let end = Self.position(p)
            segs += [.zero, SIMD4(f3(end), 0), SIMD4(Float(i), 0, 0, 0)]
            let bits = Self.bits(p.period)
            for (k, bit) in bits.enumerated() {
                let t = 0.22 + 0.72 * Double(k) / Double(max(bits.count - 1, 1))
                segs += [SIMD4(f3(end * t), Float(t)), SIMD4(f3(end), 0), SIMD4(Float(i), 1, 0, bit ? 1 : 0)]
            }
        }
        segs += [.zero, SIMD4(8.2, 0, 0, 0), SIMD4(-1, 2, 0, 0)]
        segmentCount = segs.count / 3
        segments = gpu.buffer(segs)
        target = Self.focusPoint(pair)
        distance = Self.focusDistance(pair)
    }

    private static func focusPoint(_ index: Int) -> SIMD3<Double> {
        if index == 0 { return SIMD3(4.5, 0, 0) }
        let pair = pairs[index]
        return pair.map { position(all[$0]) }.reduce(.zero, +) / Double(pair.count) * 0.45
    }

    private static func focusDistance(_ index: Int) -> Double {
        if index == 0 { return 27 }
        let far = pairs[index].map { all[$0].distance }.max() ?? 2
        return min(max(2.8 * far + 0.7, 1.3), 12)
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        clock += Double(f.dt)
        since += f.dt
        if since > 26 {
            since = 0
            pair = (pair + 1) % Self.pairs.count
        }
        let focused = Self.pairs[pair]
        for i in focus.indices {
            let goal: Float = focused.contains(i) ? 1 : 0
            focus[i] += (goal - focus[i]) * min(f.dt * 1.2, 1)
        }
        // The camera eases toward the pair in focus, closer for nearby pulsars so their binary stays legible.
        let ease = Double(min(f.dt * 0.35, 1))
        target += (Self.focusPoint(pair) - target) * ease
        distance += (Self.focusDistance(pair) - distance) * ease

        let t = Double(f.time) + Double(f.seed) * 50
        let yaw = t * 2 * .pi / 720
        let elevation = Astro.rad(44 + 14 * sin(t * 0.006))
        let eye = target + SIMD3(cos(elevation) * cos(yaw), cos(elevation) * sin(yaw), sin(elevation)) * distance
        let forward = simd_normalize(target - eye)
        let right = simd_normalize(simd_cross(forward, SIMD3(0, 0, 1)))
        let up = simd_cross(right, forward)
        let density = f.size.y / 1000
        let f3 = { (v: SIMD3<Double>) in SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        let view = f.uniforms(SIMD4(f3(eye), 2.0), SIMD4(f3(right), 0), SIMD4(f3(up), 0), SIMD4(f3(forward), density))

        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
            backdrop = f.gpu.texture(f.target.width / 2, f.target.height / 2)
            labels = makeLabels(f)
        }
        let light = light!

        // Each pulsar's brightness this frame. Pulses are widened to at least 40 ms, and the faster a pulsar spins
        // the shallower its flicker, so nothing strobes: the Crab is a steady glow with a faint shimmer.
        let now = clock
        var beacons: [SIMD4<Float>] = []
        for (i, p) in Self.all.enumerated() {
            let phase = (now / p.period + Double(i) * 0.37).truncatingRemainder(dividingBy: 1)
            let width = max(0.04 / p.period, 0.03)
            let d = min(phase, 1 - phase) / width
            let shape = exp(-d * d * 0.5)
            let rate = 1 / p.period
            let depth = rate <= 3 ? 0.85 : 0.85 * pow(3 / rate, 1.5)
            let pulse = Float(1 - depth + depth * shape)
            beacons.append(SIMD4(f3(Self.position(p)), Float(i)))
            beacons.append(SIMD4(pulse, focus[i], Float(depth), 0))
        }
        beacons.append(SIMD4(0, 0, 0, -1))
        beacons.append(SIMD4(1, 1, 0, 0))
        let beaconBuffer = rings[0].next(beacons)
        let focusBuffer = rings[1].next(focus)

        f.cmd.draw(discPipe, into: backdrop!, uniforms: view)
        f.cmd.draw(linePipe, into: light, uniforms: view, buffers: [segments, focusBuffer],
                   vertices: 4, instances: segmentCount, primitive: .triangleStrip)
        f.cmd.draw(glowPipe, into: light, clear: false, uniforms: view, buffers: [beaconBuffer],
                   vertices: 4, instances: beacons.count / 2, primitive: .triangleStrip)
        let glow = bloom.glow(f.cmd, light)
        f.cmd.draw(tonePipe, into: f.target, uniforms: f.uniforms(), textures: [light, glow, backdrop!])

        if let labels { drawLabels(f, labels, eye: eye, right: right, up: up, forward: forward, density: density) }
        if let synth = f.sound { play(synth, f, focused, eye: eye, right: right) }
    }

    private func makeLabels(_ f: Frame) -> Labels {
        var items: [(String, Labels.Style)] = []
        for p in Self.all {
            items.append((Self.title(p), Labels.Style(size: 10.5, weight: .medium, tracking: 0.5)))
            items.append((Self.short(p.period), Labels.Style(size: 9.5, weight: .regular, tracking: 0.3)))
        }
        items.append(("SUN", Labels.Style(size: 10, weight: .semibold, tracking: 2)))
        items.append(("TO THE GALACTIC CENTRE", Labels.Style(size: 9.5, weight: .regular, tracking: 1.5)))
        return Labels(gpu: f.gpu, density: f.overlay.scale * CGFloat(scale), items)
    }

    private static func title(_ p: Pulsar) -> String {
        let name = p.name.replacingOccurrences(of: "-", with: "\u{2212}")
        return p.nickname.map { "\(name) \($0)" } ?? name
    }

    private static func short(_ period: Double) -> String {
        period < 1 ? "\(SpaceKit.number(period * 1000, decimals: period < 0.1 ? 1 : 0)) ms" : "\(SpaceKit.number(period, decimals: 2)) s"
    }

    private func drawLabels(_ f: Frame, _ labels: Labels, eye: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>,
                            forward: SIMD3<Double>, density: Float) {
        func pixel(_ q: SIMD3<Double>) -> SIMD2<Float>? {
            let d = q - eye
            let z = (d * forward).sum()
            guard z > 0.1 else { return nil }
            let p = SIMD2((d * right).sum(), (d * up).sum()) * 2 / z
            return SIMD2(Float((p.x / Double(f.aspect) * 0.5 + 0.5) * Double(f.size.x)), Float((0.5 - p.y * 0.5) * Double(f.size.y)))
        }
        var quads: [LabelQuad] = []
        var taken: [SIMD4<Float>] = []
        // Labels in focus first; the rest only where they don't collide.
        func free(_ q: LabelQuad) -> Bool {
            let x0 = q.pos.x - q.anchor.x * q.rect.z, y0 = q.pos.y - q.anchor.y * q.rect.w
            let box = SIMD4(x0 - 4, y0 - 2, x0 + q.rect.z + 4, y0 + q.rect.w + 2)
            if taken.contains(where: { $0.x < box.z && box.x < $0.z && $0.y < box.w && box.y < $0.w }) { return false }
            taken.append(box)
            return true
        }
        if let at = pixel(.zero), let q = labels.quad("SUN", at: at + SIMD2(0, 12 * density), anchor: [0.5, 0], color: .premultiplied(1, 0.9, 0.6, 0.6)),
           free(q) {
            quads.append(q)
        }
        for i in Self.all.indices.sorted(by: { focus[$0] > focus[$1] }) {
            let p = Self.all[i]
            guard let at = pixel(Self.position(p)) else { continue }
            let a = 0.3 + 0.62 * focus[i]
            if let q = labels.quad(Self.title(p), at: at + SIMD2(0, -10 * density), anchor: [0.5, 1], color: .premultiplied(0.95, 0.88, 0.7, a)),
               focus[i] > 0.5 || free(q) {
                if focus[i] > 0.5 { _ = free(q) }
                quads.append(q)
            }
            if focus[i] > 0.02, let q = labels.quad(Self.short(p.period), at: at + SIMD2(0, 9 * density), anchor: [0.5, 0],
                                                    color: .premultiplied(0.75, 0.85, 1.0, 0.7 * focus[i])) {
                _ = free(q)
                quads.append(q)
            }
        }
        if let at = pixel(SIMD3(pair == 0 ? 6 : 2.6, 0, 0)), let q = labels.quad("TO THE GALACTIC CENTRE", at: at + SIMD2(0, -8 * density), anchor: [0.5, 1],
                                                                color: .premultiplied(0.9, 0.85, 0.75, 0.35)) {
            quads.append(q)
        }
        f.cmd.draw(labelPipe, into: f.target, clear: false, uniforms: f.uniforms(SIMD4(labels.size.x, labels.size.y, 0, 0)),
                   textures: [labels.texture], buffers: [rings[2].next(quads)], vertices: 4, instances: quads.count, primitive: .triangleStrip)
    }

    // MARK: Sound

    /// The pulsars in focus click at their real rates, like the radio recordings: Vela eleven times a second, the
    /// Crab a thirty-hertz buzz. Each also hums its spin rate raised by octaves into the low register.
    private func play(_ synth: Synth, _ f: Frame, _ focused: [Int], eye: SIMD3<Double>, right: SIMD3<Double>) {
        let now = clock
        for (slot, i) in focused.enumerated() {
            let p = Self.all[i]
            let count = Int(now / p.period)
            let pan = Float(max(-0.7, min(0.7, ((Self.position(p) - eye) * right).sum() * 0.25)))
            if count != counts[i] {
                let fast = p.period < 0.1
                let level = focus[i] * (fast ? 0.03 : 0.07)
                if level > 0.004 {
                    synth.play(.click, fast ? 900 : 1500 + Float(slot) * 400, amp: level, pan: pan, length: fast ? 0.012 : 0.03)
                }
            }
            counts[i] = count
            var hum = 1 / p.period
            while hum < 110 { hum *= 2 }
            synth.drone(slot, Float(hum), 0.03 * focus[i], pan: pan)
        }
        synth.noise(0.01, cutoff: 180)
    }

    // MARK: Shaders

    private static let source = """
    // a: eye, zoom   b: right   c: up   d: forward, pixel density. Distances in kiloparsecs, Sun at the origin,
    // x toward the galactic centre, z toward the north galactic pole.
    inline float3 viewOf(float3 q, constant U &u) {
        float3 d = q - u.a.xyz;
        float z = max(dot(d, u.d.xyz), 0.01);
        float2 p = float2(dot(d, u.b.xyz), dot(d, u.c.xyz)) * u.a.w / z;
        return float3((p.x / (u.res.x / u.res.y) * 0.5 + 0.5) * u.res.x, (0.5 - p.y * 0.5) * u.res.y, z);
    }

    // The Milky Way's disc seen from outside: four logarithmic spiral arms with knots of young blue stars and pink
    // nebulae, dust lanes along their inner edges, and the old yellow bar and bulge, the Sun between the Sagittarius
    // and Perseus arms 8.2 kiloparsecs out.
    fragment float4 galaxyDisc(V in [[stage_in]], constant U &u [[buffer(0)]]) {
        float2 p = (in.uv * 2 - 1) * float2(u.res.x / u.res.y, -1) / u.a.w;
        float3 rd = normalize(u.d.xyz + p.x * u.b.xyz + p.y * u.c.xyz);
        float3 c = float3(0.0012, 0.0014, 0.0026);
        if (abs(rd.z) > 1e-4) {
            float t = -u.a.z / rd.z;
            if (t > 0) {
                float2 q = (u.a.xyz + rd * t).xy - float2(8.2, 0);   // galactocentric
                float R = length(q);
                float theta = atan2(q.y, q.x);
                // Two major arms and two minor ones, wavering and broken up by clumps of star formation.
                float pitch = 0.24;
                float2 w = q * 0.9;
                float clump = fbm(w * 1.3, 5);
                float phase = theta - log(max(R, 0.3)) / pitch + 0.5 * (fbm(q * 0.35 + 3.0, 3) - 0.5);
                float k = PI - log(10.0) / pitch;
                float turn = (phase - k) / (PI / 2) + 0.5;
                float s = fract(turn) - 0.5;
                float major = fmod(abs(floor(turn)), 2.0) < 0.5 ? 1.0 : 0.4;
                float width = 0.35 + 0.05 * R;
                float arm = major * exp(-pow(s * (PI / 2) * R * sin(pitch) / width, 2.0)) * (0.55 + 0.9 * fbm(q * 0.7 + 9.0, 3));
                float dust = major * exp(-pow((s + 0.08) * (PI / 2) * R * sin(pitch) / (width * 0.4), 2.0));
                float disc = exp(-R / 4.5) * smoothstep(17.0, 10.0, R);
                float bulge = exp(-length(float2(dot(q, float2(0.87, 0.5)), dot(q, float2(-0.5, 0.87)) * 2.4)) / 1.1);
                float3 young = float3(0.55, 0.68, 1.0) * arm * (0.4 + 1.2 * clump * clump) * disc * 2.2;
                float3 old = float3(1.0, 0.78, 0.5) * (disc * 0.45 + bulge * 2.4);
                float nebula = smoothstep(0.66, 0.8, fbm(w * 3.1 + 5.0, 4)) * arm;
                float3 col = (young + old * (0.5 + 0.5 * clump)) * (1 - 0.6 * dust * smoothstep(1.5, 4.0, R));
                col += float3(1.0, 0.35, 0.5) * nebula * disc * 1.4;
                float fade = exp(-t / 90.0) * smoothstep(0.0, 0.08, abs(rd.z));
                c += col * 0.09 * fade;
            }
        }
        return float4(c, 1);
    }

    // b: buffer 1 holds segments (start, end, info), buffer 2 the focus of each pulsar.
    vertex LineOut plaqueVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                const device float4 *seg [[buffer(1)]], const device float *focus [[buffer(2)]]) {
        float4 a = seg[3 * iid], b = seg[3 * iid + 1], info = seg[3 * iid + 2];
        int kind = int(info.y);
        float f = info.x >= 0 ? focus[int(info.x)] : 0.0;
        float d = u.d.w;
        float3 pa = viewOf(a.xyz, u), pb = viewOf(b.xyz, u);
        float4 gold = float4(1.0, 0.86, 0.62, 0);
        if (kind == 0) {
            return lineQuad(pa.xy, pb.xy, 1.15 * d, gold * (0.16 + 0.45 * f), vid, u.res);
        }
        if (kind == 2) {
            float3 far = viewOf(float3(8.2, 0, 0), u);
            return lineQuad(pa.xy, far.xy, 1.0 * d, gold * 0.12, vid, u.res);
        }
        // A binary digit: a tick across the line, long for one and short for zero, sized to the line on screen.
        float2 dir = pb.xy - viewOf(float3(0), u).xy;
        float len = length(dir);
        float2 n = len > 1e-3 ? float2(-dir.y, dir.x) / len : float2(0, 1);
        float spacing = len * 0.72 / 30.0;
        float h = clamp(spacing * 0.9, 1.5 * d, 6.0 * d) * (info.w > 0.5 ? 1.0 : 0.4);
        return lineQuad(pa.xy - n * h, pa.xy + n * h, 1.0 * d, gold * (0.13 + 0.55 * f), vid, u.res);
    }

    // Pulsars and the Sun. Buffer: position and index (-1 for the Sun), then pulse, focus and flicker depth.
    vertex GlowOut beaconVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                const device float4 *bs [[buffer(1)]]) {
        float4 a = bs[2 * iid], b = bs[2 * iid + 1];
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        float3 p = viewOf(a.xyz, u);
        float d = u.d.w;
        GlowOut o;
        o.uv = corner;
        if (a.w < 0) {
            float r = 14 * d;
            o.pos = pixelToClip(p.xy + corner * r, u.res);
            o.color = float4(1.0, 0.85, 0.55, 0) * 1.6;
            o.extra = float4(30, 9, 0.08, 0);
            return o;
        }
        float pulse = b.x, focus = b.y;
        float r = (12 + 10 * focus) * d;
        o.pos = pixelToClip(p.xy + corner * r, u.res);
        float3 tint = mix(float3(0.65, 0.8, 1.0), float3(0.85, 0.92, 1.0), focus);
        o.color = float4(tint * pulse * (0.8 + 1.1 * focus), 0);
        o.extra = float4(55, 7, 0.06 + 0.1 * focus, 0);
        return o;
    }

    fragment float4 pulsarTone(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                               texture2d<float> glow [[texture(1)]], texture2d<float> back [[texture(2)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = back.sample(s, in.uv).rgb + light.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.6;
        return float4(film(c), 1);
    }
    """
}
