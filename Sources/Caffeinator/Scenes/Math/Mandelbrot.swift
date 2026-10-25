import Foundation
import Metal

/// A continuous zoom sixteen trillion times into the Mandelbrot set. Floats run out of digits a million times in, so
/// it uses perturbation theory: one reference orbit in double-double precision on the CPU, and per-pixel float offsets
/// from it on the GPU, rebased onto the reference's start whenever they would lose precision (Zhuoran's method).
/// Keyframes, each √2 deeper than the last, are rendered a band of rows per frame; the display zooms between them.
@MainActor final class Mandelbrot: Scene {
    /// A place to zoom into: either an exact point, or a region the refiner explores for a deep point of its own.
    struct Target: Sendable {
        let name: String
        let re: MandelbrotDD
        let im: MandelbrotDD
        /// Width of the region the refiner starts from; zero for exact points.
        var region: Double = 0
    }

    static let targets = [
        Target(name: "the Seahorse Valley", re: MandelbrotDD("-0.743643887037158704752191506114774"), im: MandelbrotDD("0.131825904205311970493132056385139")),
        Target(name: "the Elephant Valley", re: MandelbrotDD("0.2850"), im: MandelbrotDD("0.0110"), region: 0.04),
        Target(name: "the Triple Spiral Valley", re: MandelbrotDD("-0.0880"), im: MandelbrotDD("0.6540"), region: 0.04),
        Target(name: "the filaments of the period-five bulb", re: MandelbrotDD("-0.5600"), im: MandelbrotDD("0.6400"), region: 0.04),
        Target(name: "the Seahorse Valley", re: MandelbrotDD("-0.7453"), im: MandelbrotDD("0.1127"), region: 0.04),
        Target(name: "the valley between the period-two and period-four bulbs", re: MandelbrotDD("-1.2560"), im: MandelbrotDD("0.0380"), region: 0.04),
        Target(name: "the northern antenna", re: MandelbrotDD("-0.1600"), im: MandelbrotDD("1.0350"), region: 0.04),
        Target(name: "the period-doubling cascade", re: MandelbrotDD("-1.3900"), im: MandelbrotDD("0.0200"), region: 0.02),
    ]

    private static let look = Lens.Look(exposure: 1.9, white: [1.03, 1, 0.95], bloom: 0.03, halation: 0.03, vignette: 0.5, aberration: 1.4, grain: 0.6)
    private static let startWidth = 3.4
    private static let finalWidth = 2e-13
    private static let zoomSeconds = 205.0
    private static let holdSeconds = 7.0
    private static let fadeSeconds = 3.0

    private let escape: MTLComputePipelineState
    private let shade: MTLRenderPipelineState
    private let expose: MTLRenderPipelineState
    private let bloom: Bloom
    private let lens: Lens
    private let gpu: GPU
    private var hdr: MTLTexture?
    private var shot: MTLTexture?
    private var keys: [MTLTexture] = []
    private let stats: [MTLBuffer]
    private var slotLevel = [-1, -1, -1]
    private var slotCap = [0, 0, 0]
    private var rendering = -1
    private var done = -1
    private var rowsDone = 0
    private var reference: MTLBuffer?
    private var referenceLength = 0

    private var index: Int
    private var target: Target?
    private var place = ""
    private var region = ""
    private var angle: Float = 0
    private var hue: Float = 0
    private var palette = 0
    private var upcoming: Target?
    private var rng: UInt64

    private var zoomTime = 0.0
    private var holdTime = 0.0
    private var fade: Float = 0
    private var level = 0.0
    private var finalLevel = 0.0
    private var deepest = 0
    private var detail: Float = 0
    private var nextBell: Float = 2
    private var noteStep = 0

    var scale: Float { 0.75 }
    var ready: Bool { target != nil }

    var facts: String? {
        guard target != nil else { return "Finding a deep point in the Mandelbrot set" }
        let zoom = Self.startWidth / width(level)
        var s = zoom < 1.5 ? "The whole Mandelbrot set; zooming toward \(region), at \(place)"
                           : "Magnified \(Self.words(zoom)) times, in \(region) near \(place)"
        if deepest > 0 { s += "; the slowest points here take \(deepest.formatted()) iterations to escape" }
        return s
    }

    init(gpu: GPU) throws {
        self.gpu = gpu
        let library = try gpu.library(Self.source)
        escape = try gpu.kernel(library, "escapeTime")
        shade = try gpu.pipeline(library, "mandelShade")
        expose = try gpu.pipeline(library, "mandelExpose")
        bloom = try Bloom(gpu: gpu)
        lens = try Lens(gpu: gpu)
        stats = (0..<3).map { _ in gpu.buffer(32) }
        rng = UInt64.random(in: 1...UInt64.max)
        if let v = Snapshot.variant { rng = UInt64(v + 1) &* 0x9E37_79B9_7F4A_7C15 }
        index = (Snapshot.variant ?? Int.random(in: 0..<Self.targets.count)) % Self.targets.count
        prepare(Self.targets[index])
    }

    // MARK: Targets

    /// Starts the zoom on an exact point, or refines a region into one in the background first.
    private func prepare(_ t: Target) {
        if t.region == 0 {
            begin(t)
        } else {
            refine(t) { [weak self] found in self?.begin(found) }
        }
    }

    private func refine(_ t: Target, then: @escaping @MainActor (Target) -> Void) {
        let seed = next(), final = Self.finalWidth
        Task { @MainActor in
            let found = await Task.detached(priority: .utility) { MandelbrotRefiner.refine(t, seed: seed, final: final) }.value
            then(found ?? Self.targets[0])
        }
    }

    private func begin(_ t: Target) {
        target = t
        let orbit = MandelbrotRefiner.orbit(t.re, t.im, limit: 60_000)
        reference = gpu.buffer(orbit.map { SIMD2<Float>(Float($0.x), Float($0.y)) })
        referenceLength = orbit.count
        place = Self.complex(t.re.value, t.im.value)
        region = t.name
        angle = Float(Double(next() % 1000) / 1000 * 2 * .pi)
        hue = Float(next() % 1000) / 1000
        palette = Int(next() % 3)
        slotLevel = [-1, -1, -1]
        rendering = -1
        done = -1
        zoomTime = 0
        holdTime = 0
        level = 0
        deepest = 0
        finalLevel = 2 * log2(Self.startWidth / Self.finalWidth)
        // Look for the next place while this one plays.
        index = (index + 1) % Self.targets.count
        upcoming = nil
        let n = Self.targets[index]
        if n.region == 0 { upcoming = n } else { refine(n) { [weak self] in self?.upcoming = $0 } }
    }

    private func next() -> UInt64 {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return rng
    }

    private func width(_ level: Double) -> Double { Self.startWidth * pow(2, -level / 2) }

    // MARK: Frame

    func draw(_ f: Frame) {
        if hdr?.width != f.target.width || hdr?.height != f.target.height {
            hdr = f.gpu.texture(f.target.width, f.target.height)
            shot = f.gpu.texture(f.target.width, f.target.height)
            keys = (0..<3).map { _ in f.gpu.texture(f.target.width, f.target.height, .rg32Float) }
            slotLevel = [-1, -1, -1]
            rendering = -1
            done = -1
        }
        guard target != nil, let reference else {
            f.cmd.draw(expose, into: shot!, uniforms: f.uniforms(), textures: [hdr!, hdr!])
            lens.develop(f, shot!, Self.look)
            f.sound?.reset()
            return
        }
        let w = f.target.width, h = f.target.height

        // The first two keyframes are shallow and cheap: render them whole. Then one level at a time, a band of rows
        // per frame, finishing well before the zoom needs it.
        if done < 1 {
            for l in 0..<2 { start(l); render(f, rows: h, reference) }
        }
        if rendering < 0, done + 1 <= Int(level) + 2, Double(done + 1) <= finalLevel + 1 { start(done + 1) }
        if rendering >= 0 {
            let perLevel = Self.zoomSeconds * 0.95 / finalLevel
            render(f, rows: max(Int(ceil(Double(h) * Double(f.dt) / (perLevel * 0.6))), 4), reference)
        }

        // Zoom, stalling at the last complete keyframe if the GPU falls behind.
        let wanted = finalLevel * Self.ease(min(zoomTime + Double(f.dt), Self.zoomSeconds) / Self.zoomSeconds, 0.05)
        if wanted < Double(done) - 0.0001 || zoomTime >= Self.zoomSeconds {
            zoomTime = min(zoomTime + Double(f.dt), Self.zoomSeconds)
            level = min(wanted, Double(done) - 0.0001)
        }
        if zoomTime >= Self.zoomSeconds { holdTime += Double(f.dt) }
        let ending = holdTime > Self.holdSeconds
        fade = ending ? max(fade - f.dt / Float(Self.fadeSeconds), 0) : min(fade + f.dt / 2, 1)
        if ending, fade == 0, let up = upcoming {
            begin(up)
            return
        }

        let k = Int(level)
        let outer = slot(k), inner = slot(k + 1)
        let v = width(level)
        if outer >= 0 {
            let s = stats[outer].contents().assumingMemoryBound(to: UInt32.self)
            deepest = Int(s[2])
            detail = Float(s[4]) / Float(max(s[0], 1))
        }
        let u = f.uniforms([Float(v / width(Double(k))), Float(v / width(Double(k + 1))), fade, Float(slotCap[max(outer, 0)])],
                           [hue + f.time * 0.004, Float(level), Float(palette), 0],
                           [Float(w), Float(h), 0, 0])
        if outer >= 0, inner >= 0 {
            f.cmd.draw(shade, into: hdr!, uniforms: u, textures: [keys[outer], keys[inner]])
        }
        let soft = bloom.glow(f.cmd, hdr!, levels: 2)
        f.cmd.draw(expose, into: shot!, uniforms: u, textures: [hdr!, soft])
        lens.develop(f, shot!, Self.look)

        if let synth = f.sound { play(synth, f) }
    }

    private func slot(_ level: Int) -> Int {
        level <= done && slotLevel[level % 3] == level ? level % 3 : -1
    }

    private func start(_ l: Int) {
        let s = l % 3
        let previous = stats[(l + 1) % 3].contents().assumingMemoryBound(to: UInt32.self)
        // Raise the iteration cap with depth, from what the keyframe two levels up needed.
        let seen = slotLevel[(l + 1) % 3] == l - 2 ? Int(previous[2]) : 0
        // Points still orbiting past 15,000 steps sit on the rim of a minibrot and read as part of it.
        slotCap[s] = min(max(400 + l * 40, seen * 3 / 2 + 300), 15_000)
        slotLevel[s] = l
        rendering = l
        rowsDone = 0
        let c = stats[s].contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<8 { c[i] = 0 }
        c[3] = .max
    }

    private func render(_ f: Frame, rows: Int, _ reference: MTLBuffer) {
        guard rendering >= 0 else { return }
        let s = rendering % 3
        let key = keys[s]
        let n = min(rows, key.height - rowsDone)
        let pixel = width(Double(rendering)) / Double(key.width)
        let u = f.uniforms([Float(pixel), Float(rowsDone), Float(slotCap[s]), Float(referenceLength)],
                           [cos(angle), sin(angle), Float(key.width), Float(key.height)])
        f.cmd.compute(escape, key.width, n, uniforms: u, textures: [key], buffers: [reference, stats[s]])
        rowsDone += n
        if rowsDone >= key.height {
            done = rendering
            rendering = -1
        }
    }

    /// Trapezoidal speed: eases in and out over the first and last `a` of the way, constant in between.
    static func ease(_ u: Double, _ a: Double) -> Double {
        let u = min(max(u, 0), 1)
        let p = u < a ? u * u / (2 * a) : u > 1 - a ? 1 - a - (1 - u) * (1 - u) / (2 * a) : u - a / 2
        return p / (1 - a)
    }

    // MARK: Sound

    /// A Shepard pad: octave-spaced partials glide down as the zoom deepens, fading in at the top and out at the bottom,
    /// so the pitch seems to fall forever. Detail in the picture adds a shimmer and the odd high bell.
    private func play(_ synth: Synth, _ f: Frame) {
        let fall = Float(level) / 16
        let base = -fall - (-fall).rounded(.down)
        for i in 0..<5 {
            let octave = Float(i) + base
            let freq = 55 * pow(2, octave)
            let x = (octave - 2.4) / 1.3
            let amp = exp(-x * x) * 0.07 * fade
            synth.drone(i, freq, amp, pan: Float(i % 2 == 0 ? -0.25 : 0.25))
            if i < 2 { synth.drone(5 + i, freq * 3, amp * 0.25, pan: i == 0 ? 0.4 : -0.4) }
        }
        let shimmer = min(detail * 3, 1) * fade
        synth.drone(7, 55 * pow(2, 4 + base) * 1.5 * (1 + 0.003 * sin(f.time * 5.1)), 0.018 * shimmer)
        synth.noise(0.02 * fade, cutoff: 200 + 600 * shimmer)
        nextBell -= f.dt
        if nextBell < 0 {
            nextBell = 1.8 + (1 - shimmer) * 4 + Float(next() % 100) / 50
            guard fade > 0.5 else { return }
            let scale: [Float] = [1, 9.0 / 8, 5.0 / 4, 3.0 / 2, 5.0 / 3, 2]
            noteStep = (noteStep + Int(next() % 3) + 1) % scale.count
            synth.play(.bell, 55 * pow(2, 3 + base) * scale[noteStep], amp: 0.03 + 0.04 * shimmer,
                       pan: Float(Int(next() % 200) - 100) / 140, length: 3)
        }
    }

    // MARK: Words

    static func words(_ x: Double) -> String {
        let names: [(Double, String)] = [(1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")]
        for (n, name) in names where x >= n {
            let v = x / n
            return (v < 10 ? String(format: "%.1f", v) : String(format: "%.0f", v)) + " " + name
        }
        return String(format: "%.0f", x)
    }

    static func complex(_ re: Double, _ im: Double) -> String {
        let r = String(format: "%.6f", abs(re)), i = String(format: "%.6f", abs(im))
        return (re < 0 ? "minus " : "") + r + (im < 0 ? " minus " : " plus ") + i + "i"
    }

    private static let source = """
    inline float2 cmul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

    // One band of a keyframe. a: pixel size, first row, iteration cap, reference length   b: rotation cos, sin, size
    // Each texel keeps the smooth iteration count (negative inside the set), and, packed as halves, the distance to the
    // set in pixels and the slope lighting from the derivative.
    kernel void escapeTime(constant U &u [[buffer(0)]], const device float2 *ref [[buffer(1)]],
                           device atomic_uint *stats [[buffer(2)]], texture2d<float, access::write> out [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
        uint2 p = uint2(gid.x, gid.y + uint(u.a.y));
        bool valid = p.x < uint(u.b.z) && p.y < uint(u.b.w);
        float ps = u.a.x;
        float2 o = (float2(p) + 0.5 - u.b.zw * 0.5) * ps;
        float2 dc = float2(o.x * u.b.x - o.y * u.b.y, o.x * u.b.y + o.y * u.b.x);
        int cap = int(u.a.z), last = int(u.a.w) - 1;
        float2 d = 0, der = 0, z = 0;
        int m = 0, n = 0;
        bool escaped = false;
        if (valid) {
            for (; n < cap; n++) {
                float2 Z = ref[m];
                z = Z + d;
                // The derivative is kept in pixels, so it stays in float range however deep we go.
                der = 2 * cmul(z, der) + float2(ps, 0);
                // Past 1e30 pixels the point is on the set for any purpose; keep the direction, cap the size.
                float dm = abs(der.x) + abs(der.y);
                if (dm > 1e30) der *= 1e30 / dm;
                d = cmul(2 * Z + d, d) + dc;
                m++;
                z = ref[m] + d;
                float r2 = dot(z, z);
                if (r2 > 1e6) { escaped = true; n++; break; }
                // Rebase: when the pixel's orbit comes nearer zero than the reference, restart the reference from zero.
                if (r2 < dot(d, d) || m >= last) { d = z; m = 0; }
            }
        }
        float nu = -1, de = 0, slope = 0;
        if (escaped) {
            float r = length(z), lr = log(r);
            nu = float(n) + 1 - log2(lr);
            float big = max(max(abs(der.x), abs(der.y)), 1e-30);
            float2 dn = der / big;
            de = r * lr / (length(dn) * big);
            float2 w = cmul(z, float2(dn.x, -dn.y));
            slope = dot(w, float2(0.7071, 0.7071)) / max(length(w), 1e-30);
        }
        if (valid) out.write(float4(nu, as_type<float>(half2(half(log2(max(de, 1e-4))), half(slope))), 0, 0), p);
        uint esc = simd_sum(uint(valid && escaped));
        uint inside = simd_sum(uint(valid && !escaped));
        uint mx = simd_max(escaped ? uint(n) : 0u);
        uint mn = simd_min(escaped ? uint(n) : 0xffffffffu);
        uint edge = simd_sum(uint(escaped && de < 1.0));
        if (simd_is_first()) {
            if (esc) atomic_fetch_add_explicit(&stats[0], esc, memory_order_relaxed);
            if (inside) atomic_fetch_add_explicit(&stats[1], inside, memory_order_relaxed);
            atomic_fetch_max_explicit(&stats[2], mx, memory_order_relaxed);
            atomic_fetch_min_explicit(&stats[3], mn, memory_order_relaxed);
            if (edge) atomic_fetch_add_explicit(&stats[4], edge, memory_order_relaxed);
        }
    }

    struct Texel { float nu, de, slope; };

    inline Texel fetch(texture2d<float> t, int2 q) {
        q = clamp(q, int2(0), int2(t.get_width() - 1, t.get_height() - 1));
        float2 v = t.read(uint2(q)).xy;
        half2 h = as_type<half2>(v.y);
        return Texel{v.x, exp2(float(h.x)), float(h.y)};
    }

    // Reflectance of a thin transparent film of thickness `t` nanometres on bright metal: two-beam interference, which
    // washes toward grey as the orders pile up. This is where anodised titanium and oil on steel get their colours.
    inline float3 thinFilm(float t, float wash) {
        const float3 lambda = float3(640.0, 545.0, 455.0);
        float3 ph = 4 * M_PI_F * 1.45 * t / lambda;
        float3 c = 0.5 - 0.5 * cos(ph) * exp(-t * wash);
        return pow(c, 2.2);
    }

    // The plate's height at a texel: the set is a pocket, and its rim rises over a few pixels to the surface.
    inline float plate(Texel t, float px) {
        return t.nu < 0 ? 0.0 : 1 - exp(-t.de * px / 6.0);
    }

    // Light off one texel of the plate: thin-film colour on the surface, black anodised in the pocket, a softbox and a
    // lacquer highlight reflected in the bevels. The film's thickness follows the escape time, and the colour shifts
    // with the angle the bevel is seen from.
    float3 tint(texture2d<float> t, int2 q, float px, float2 uv, constant U &u) {
        Texel c = fetch(t, q);
        float h0 = plate(fetch(t, q + int2(1, 0)), px) - plate(fetch(t, q - int2(1, 0)), px);
        float h1 = plate(fetch(t, q + int2(0, 1)), px) - plate(fetch(t, q - int2(0, 1)), px);
        float3 n = normalize(float3(-h0 * 2.2, -h1 * 2.2, 1));
        float3 r = reflect(float3(0, 0, -1), n);
        const float3 key = float3(-0.5, -0.62, 0.6) * 1.0;
        float3 L = normalize(key);
        float box = smoothstep(0.55, 0.93, dot(r, L));
        float sky = 0.5 + 0.5 * saturate(-r.y);
        float pool = 0.5 + 0.7 * exp(-dot(uv - float2(0.38, 0.36), uv - float2(0.38, 0.36)) * 3.5);
        float env = (0.10 * sky + 1.5 * box) * pool;
        float spec = pow(saturate(dot(r, L)), 60.0) * 0.5;
        int p = int(u.b.z);
        float thick = 300 + (p == 1 ? 40 : p == 2 ? 95 : 0) + 50 * sqrt(max(c.nu, 0.0)) + 60 * u.b.x;
        thick /= max(n.z, 0.6);
        if (!(c.nu >= 0)) {
            float pit = 0.02 + 0.05 * thinFilm(520 + 60 * u.b.x, 0.0).g;
            return pit * (0.15 + env) + spec * 0.35;
        }
        float3 film = thinFilm(thick, 0.0006) * 1.6;
        // Bare brushed metal away from the set; the film colours only where the relief is.
        float3 metal = float3(0.62, 0.60, 0.57) * (0.92 + 0.08 * noise(float2(q) * float2(0.35, 5.0)));
        float w = exp(-c.de * px / 18.0) * 0.6;
        return mix(metal, film, w) * env + spec;
    }

    float3 sampleKey(texture2d<float> t, float2 q, float px, float2 uv, constant U &u) {
        float2 i = floor(q), f = q - i;
        int2 b = int2(i);
        float3 c00 = tint(t, b, px, uv, u), c10 = tint(t, b + int2(1, 0), px, uv, u);
        float3 c01 = tint(t, b + int2(0, 1), px, uv, u), c11 = tint(t, b + int2(1, 1), px, uv, u);
        return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    }

    // a: view width over the outer and inner keyframes' widths, fade   c: size
    fragment float4 mandelShade(V in [[stage_in]], constant U &u [[buffer(0)]],
                                texture2d<float> outer [[texture(0)]], texture2d<float> inner [[texture(1)]]) {
        float2 size = u.c.xy;
        float2 o = in.pos.xy - size * 0.5;
        float2 qi = o * u.a.y + size * 0.5 - 0.5;
        float2 edge = min(qi, size - 1 - qi);
        float wi = smoothstep(1.0, 12.0, min(edge.x, edge.y));
        float3 c = 0;
        if (wi < 1) c = sampleKey(outer, o * u.a.x + size * 0.5 - 0.5, 1 / u.a.x, in.uv, u) * (1 - wi);
        if (wi > 0) c += sampleKey(inner, qi, 1 / u.a.y, in.uv, u) * wi;
        return float4(c, 1);
    }

    // Shallow depth of field: the plate is focused along a tilted band, and falls into the blurred copy either side.
    fragment float4 mandelExpose(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> hdr [[texture(0)]],
                                 texture2d<float> soft [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float d = abs((in.uv.y - 0.52) + 0.22 * (in.uv.x - 0.5));
        float m = smoothstep(0.12, 0.5, d);
        float3 c = mix(hdr.sample(s, in.uv).rgb, soft.sample(s, in.uv).rgb, m);
        return float4(c * u.a.z, 1);
    }
    """
}

/// A double-double number: about 32 significant digits, enough for a reference orbit ten trillion times in.
struct MandelbrotDD: Sendable {
    var hi: Double
    var lo: Double

    init(_ hi: Double, _ lo: Double = 0) {
        self.hi = hi
        self.lo = lo
    }

    /// Parses a decimal such as "-0.7436438870371587047", keeping every digit up to the format's precision.
    init(_ text: String) {
        var v = MandelbrotDD(0), places = 0, point = false
        for ch in text {
            if ch == "." { point = true } else if let d = ch.wholeNumberValue {
                v = v * MandelbrotDD(10) + MandelbrotDD(Double(d))
                if point { places += 1 }
            }
        }
        var p = MandelbrotDD(1)
        for _ in 0..<places { p = p * MandelbrotDD(10) }
        v = v / p
        self = text.hasPrefix("-") ? MandelbrotDD(-v.hi, -v.lo) : v
    }

    var value: Double { hi + lo }

    private static func quick(_ a: Double, _ b: Double) -> MandelbrotDD {
        let s = a + b
        return MandelbrotDD(s, b - (s - a))
    }

    static func + (a: MandelbrotDD, b: MandelbrotDD) -> MandelbrotDD {
        let s = a.hi + b.hi
        let v = s - a.hi
        let e = (a.hi - (s - v)) + (b.hi - v)
        return quick(s, e + a.lo + b.lo)
    }

    static func - (a: MandelbrotDD, b: MandelbrotDD) -> MandelbrotDD { a + MandelbrotDD(-b.hi, -b.lo) }

    static func * (a: MandelbrotDD, b: MandelbrotDD) -> MandelbrotDD {
        let p = a.hi * b.hi
        let e = (-p).addingProduct(a.hi, b.hi)
        return quick(p, e + a.hi * b.lo + a.lo * b.hi)
    }

    static func / (a: MandelbrotDD, b: MandelbrotDD) -> MandelbrotDD {
        let q1 = a.hi / b.hi
        let r = a - b * MandelbrotDD(q1)
        let q2 = r.hi / b.hi
        let r2 = r - b * MandelbrotDD(q2)
        return quick(q1, q2) + MandelbrotDD(r2.hi / b.hi)
    }
}

/// Finds deep points worth zooming into, and computes reference orbits. Pure functions, run off the main actor.
enum MandelbrotRefiner {
    /// The orbit of c = re + im·i in double-double, rounded to doubles, until it escapes or reaches `limit`.
    static func orbit(_ re: MandelbrotDD, _ im: MandelbrotDD, limit: Int) -> [SIMD2<Double>] {
        var out: [SIMD2<Double>] = [.zero]
        out.reserveCapacity(min(limit, 4096) + 1)
        var x = MandelbrotDD(0), y = MandelbrotDD(0)
        for _ in 0..<limit {
            let xx = x * x, yy = y * y, xy = x * y
            x = xx - yy + re
            y = xy + xy + im
            let v = SIMD2(x.value, y.value)
            out.append(v)
            if v.x * v.x + v.y * v.y > 1e6 { break }
        }
        return out
    }

    /// Smooth escape count of the point offset (dx, dy) from the reference, or -1 if it hasn't escaped by `cap`.
    static func probe(_ ref: UnsafeBufferPointer<SIMD2<Double>>, _ dx: Double, _ dy: Double, cap: Int) -> Double {
        var x = 0.0, y = 0.0, m = 0
        let last = ref.count - 1
        for n in 0..<cap {
            let zx = ref[m].x, zy = ref[m].y
            let tx = 2 * zx + x, ty = 2 * zy + y
            (x, y) = (tx * x - ty * y + dx, tx * y + ty * x + dy)
            m += 1
            let fx = ref[m].x + x, fy = ref[m].y + y
            let r2 = fx * fx + fy * fy
            if r2 > 1e6 { return Double(n + 2) - log2(log(r2) / 2) }
            if r2 < x * x + y * y || m >= last { (x, y, m) = (fx, fy, 0) }
        }
        return -1
    }

    /// Walks down from the region's width to `final`, each step four times deeper, moving the target to a point
    /// whose escape count is high for its neighbourhood but not the highest: those lead into spirals and embedded
    /// Julia sets, where the very highest lead into the slow cusps of minibrots.
    static func refine(_ t: Mandelbrot.Target, seed: UInt64, final: Double) -> Mandelbrot.Target? {
        var rng = seed | 1
        func random() -> UInt64 {
            rng ^= rng << 13
            rng ^= rng >> 7
            rng ^= rng << 17
            return rng
        }
        for _ in 0..<4 {
            var re = t.re, im = t.im
            var width = t.region
            var good = true
            let gw = 32, gh = 20
            var median = 0.0
            while width > final {
                let ref = orbit(re, im, limit: 14_000)
                let window = width * 0.35
                var counts: [(Double, Int, Int)] = []
                ref.withUnsafeBufferPointer { r in
                    for j in 0..<gh {
                        for i in 0..<gw {
                            let dx = (Double(i) + 0.5 - Double(gw) / 2) / Double(gw) * window
                            let dy = (Double(j) + 0.5 - Double(gh) / 2) / Double(gw) * window
                            let n = probe(r, dx, dy, cap: 14_000)
                            if n >= 0 { counts.append((n, i, j)) }
                        }
                    }
                }
                if counts.count < gw * gh / 3 { good = false; break }
                counts.sort { $0.0 < $1.0 }
                median = counts[counts.count / 2].0
                let band = counts[(counts.count * 85 / 100)...(counts.count * 95 / 100)]
                let pick = band[band.startIndex + Int(random() % UInt64(band.count))]
                re = re + MandelbrotDD((Double(pick.1) + 0.5 - Double(gw) / 2) / Double(gw) * window)
                im = im + MandelbrotDD((Double(pick.2) + 0.5 - Double(gh) / 2) / Double(gw) * window)
                width /= 4
            }
            // Very slow points cost too much to draw, and usually mean a dull cusp.
            if good, median < 6000 { return Mandelbrot.Target(name: t.name, re: re, im: im) }
        }
        return nil
    }
}
