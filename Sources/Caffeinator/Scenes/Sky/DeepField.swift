import AppKit
import Metal
import simd

/// Real Webb and Hubble images from ESA's archives, one at a time: each drifts in a slow Ken Burns move for three
/// minutes, labelled like a figure in a paper (telescope, instrument, filters with their wavelengths and assigned
/// colours, a scale bar in light-years, north and east, field of view, date and credit), then dissolves into the next.
/// The drone takes its colour from the picture: warm, dark and bright images each sound different.
@MainActor final class DeepField: Scene {
    /// One image on screen, from when it started.
    private struct Shot {
        let index: Int
        let texture: MTLTexture
        let start: Float
        /// Its slot in the statistics buffer.
        let slot: Int
    }

    /// The colour statistics of a picture that shape its sound.
    private struct Tone { var red: Float = 0.3, green: Float = 0.3, blue: Float = 0.3, light: Float = 0.1, bright: Float = 0.02, contrast: Float = 0.1 }

    static let hold: Float = 180
    static let fade: Float = 12

    private let gpu: GPU
    private let pass: MTLRenderPipelineState
    private let statsPass: MTLComputePipelineState
    private let stats: MTLBuffer
    private let blank: MTLTexture
    private let first: Int
    private var current: Shot?
    private var incoming: Shot?
    /// A finished download waiting for its turn.
    private var waiting: (index: Int, texture: MTLTexture, slot: Int)?
    private var loading = false
    private var failed: Set<Int> = []
    private var attempted = false
    private var slots = 0
    private var overlayKey = ""
    private var overlayAt: Float = -10
    private var nextBell: Float = 6
    /// Scene time of the last frame, and the earliest time to try another download after one failed.
    private var clock: Float = 0
    private var retry: Float = 0

    var scale: Float { 1 }
    var ready: Bool { attempted }

    init(gpu: GPU) throws {
        self.gpu = gpu
        let library = try gpu.library(SkyKit.metal + Self.source)
        pass = try gpu.pipeline(library, "plateView")
        statsPass = try gpu.kernel(library, "plateStats")
        stats = gpu.buffer(2 * 8 * MemoryLayout<Float>.stride)
        var black: UInt32 = 0
        blank = gpu.texture(1, 1, .rgba8Unorm_srgb, bytes: &black, bytesPerRow: 4)
        first = (Snapshot.variant ?? Int.random(in: 0..<Self.plates.count)) % Self.plates.count
        load(first)
    }

    var facts: String? {
        guard let shot = incoming ?? current else {
            return attempted ? "Offline, and no Webb or Hubble images are saved yet, so this is only a faint star field for now."
                             : "Fetching a Webb or Hubble image from ESA's archive."
        }
        let p = Self.plates[shot.index]
        let telescope = p.webb ? "Webb" : "Hubble"
        let lines = p.filters.compactMap(\.feature)
        var bands = "\(p.instrument), \(Self.spokenCount(p.filters.count)) filters from \(Self.spokenWavelength(p.filters.map(\.microns).min()!, p.webb)) to \(Self.spokenWavelength(p.filters.map(\.microns).max()!, p.webb))"
        if let feature = lines.first { bands += ", one of them tuned to \(feature)" }
        return "\(p.spoken) \(telescope)'s \(bands). \(p.name), \(p.distanceText), released \(p.released)."
    }

    var credit: String? {
        guard let shot = incoming ?? current else { return nil }
        let p = Self.plates[shot.index]
        return "\(p.webb ? "ESA/Webb" : "ESA/Hubble") image, CC BY 4.0: \(p.credit)"
    }

    private static func spokenCount(_ n: Int) -> String {
        ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][min(n, 10)]
    }

    private static func spokenWavelength(_ microns: Double, _ webb: Bool) -> String {
        webb ? String(format: microns < 10 ? "%.1f microns" : "%.0f microns", microns) : "\(Int((microns * 1000).rounded())) nanometres"
    }

    // MARK: Loading

    /// Fetches one image (never more than one at a time), then measures its colours for the sound.
    private func load(_ index: Int) {
        loading = true
        let plate = Self.plates[index]
        let gpu = self.gpu
        Task { @MainActor [weak self] in
            let texture = await Imagery.texture(plate.url, gpu, maxSize: 4096, maxAge: 365 * 86400)
            guard let self else { return }
            loading = false
            attempted = true
            guard let texture else {
                failed.insert(index)
                retry = clock + 5
                return
            }
            let slot = slots % 2
            slots += 1
            if let cmd = gpu.queue.makeCommandBuffer() {
                cmd.compute(statsPass, 1, uniforms: U(a: [Float(slot), 0, 0, 0]), textures: [texture], buffers: [stats])
                cmd.commit()
            }
            waiting = (index, texture, slot)
        }
    }

    /// The image after `index` that hasn't failed this round.
    private func following(_ index: Int) -> Int? {
        let n = Self.plates.count
        return (1...n).map { (index + $0) % n }.first { !failed.contains($0) || $0 == index }
    }

    private func advance(_ t: Float) {
        if current == nil, let w = waiting {
            current = Shot(index: w.index, texture: w.texture, start: t, slot: w.slot)
            waiting = nil
        }
        if let c = current, incoming == nil, let w = waiting, t - c.start > Self.hold {
            incoming = Shot(index: w.index, texture: w.texture, start: t, slot: w.slot)
            waiting = nil
        }
        if let i = incoming, t - i.start > Self.fade {
            current = i
            incoming = nil
        }
        // Fetch the next image a little after this one has settled, so only one download runs at a time. Offline,
        // only saved images load; when every other one has failed, try them all again a few minutes later.
        guard !loading, waiting == nil, incoming == nil, t >= retry else { return }
        if let c = current, t - c.start > 20 {
            if let next = following(c.index), next != c.index {
                load(next)
            } else {
                failed.removeAll()
                retry = t + 300
            }
        } else if current == nil, attempted, failed.count < Self.plates.count, let next = following(first) {
            load(next)
        }
    }

    // MARK: Framing

    /// The part of the image on screen, as centre and extent in texture coordinates, for a shot `t` seconds in.
    /// The move eases in and out; if the next picture is late it simply drifts back the way it came.
    private func view(_ shot: Shot, _ t: Float, aspect: Double) -> SIMD4<Float> {
        let p = Self.plates[shot.index]
        let progress = Double(t - shot.start) / Double(Self.hold + Self.fade)
        let e = 0.5 - 0.5 * cos(.pi * progress)
        let k = p.from + (p.to - p.from) * e
        let a = Double(shot.texture.width) / Double(shot.texture.height)
        var ex = 1.0, ey = 1.0
        if a > aspect { ex = aspect / a } else { ey = a / aspect }
        ex /= k.z
        ey /= k.z
        let cx = min(max(k.x, ex / 2), 1 - ex / 2), cy = min(max(k.y, ey / 2), 1 - ey / 2)
        return SIMD4(Float(cx), Float(cy), Float(ex), Float(ey))
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        clock = f.time
        advance(f.time)
        let aspect = Double(f.aspect)
        var a = SIMD4<Float>(0.5, 0.5, 1, 1), b = a
        var w: Float = 0
        if let c = current { a = view(c, f.time, aspect: aspect) }
        if let i = incoming {
            b = view(i, f.time, aspect: aspect)
            let x = min(max((f.time - i.start) / Self.fade, 0), 1)
            w = x * x * (3 - 2 * x)
        }
        let ta = current?.texture ?? blank, tb = incoming?.texture ?? blank
        let u = f.uniforms(a, b, [Float(ta.width), Float(ta.height), Float(tb.width), Float(tb.height)],
                           [w, current == nil ? 0 : 1, incoming == nil ? 0 : 1, 0])
        f.cmd.draw(pass, into: f.target, uniforms: u, textures: [ta, tb])
        drawOverlay(f, w)
        if let synth = f.sound { play(synth, f, w) }
    }

    // MARK: Sound

    private func tone(_ shot: Shot?) -> Tone {
        guard let shot else { return Tone() }
        let p = stats.contents().assumingMemoryBound(to: Float.self) + shot.slot * 8
        guard p[3] > 0 else { return Tone() }
        return Tone(red: p[0], green: p[1], blue: p[2], light: p[3], bright: p[4], contrast: p[5])
    }

    /// A drone on A whose chord follows the picture: warm images lean on the major third, cool ones on the ninth,
    /// darker ones sit lower and fuller, bright ones open the upper partials, and starry ones ring soft bells.
    private func play(_ synth: Synth, _ f: Frame, _ w: Float) {
        let a = tone(current), b = tone(incoming)
        func mix(_ x: Float, _ y: Float) -> Float { x + (y - x) * w }
        let sum = max(mix(a.red + a.green + a.blue, b.red + b.green + b.blue), 1e-4)
        let red = mix(a.red, b.red) / sum, green = mix(a.green, b.green) / sum, blue = mix(a.blue, b.blue) / sum
        let light = min(mix(a.light, b.light) * 3, 1), bright = mix(a.bright, b.bright), contrast = mix(a.contrast, b.contrast)
        let warm = max(0, min(1, (red - blue) * 3 + 0.5))
        let t = f.time
        let swell = 0.5 + 0.5 * sin(t * 2 * .pi / 23)
        let slow = 0.5 + 0.5 * sin(t * 2 * .pi / 41 + 1.3)
        synth.drone(0, 55, 0.05 + 0.03 * (1 - light), pan: -0.1)
        synth.drone(1, 82.5, 0.03 * (0.5 + red * 1.5) * (0.8 + 0.2 * slow), pan: 0.2)
        synth.drone(2, 110 * 1.001, 0.026, pan: -0.25)
        synth.drone(3, 137.5, 0.02 * warm * (0.7 + 0.3 * swell), pan: 0.3)
        synth.drone(4, 165, 0.016 * (0.5 + green * 1.5), pan: -0.35)
        synth.drone(5, 220, 0.01 * (0.4 + blue * 1.8) * (0.6 + 0.4 * slow), pan: 0.4)
        synth.drone(6, 247.5, 0.009 * (1 - warm) * swell, pan: -0.45)
        synth.drone(7, 330, 0.004 + 0.008 * light * (0.5 + contrast * 2), pan: 0.15)
        synth.noise(0.01 + 0.02 * light, cutoff: 180 + 700 * blue + 300 * light)
        // Stars as the occasional soft bell from the pentatonic scale on A.
        if t > nextBell {
            let rate = 0.04 + 0.6 * min(bright * 8, 1)
            nextBell = t + Float.random(in: 0.6...1.4) / rate
            if current != nil {
                let notes: [Float] = [440, 493.9, 554.4, 659.3, 740, 880]
                synth.play(.bell, notes.randomElement()!, amp: 0.012 + 0.01 * light, pan: Float.random(in: -0.7...0.7), length: 7)
            }
        }
    }

    // MARK: Overlay

    /// A label set like a figure in a paper: thin, small, and quiet, in the corners.
    private func drawOverlay(_ f: Frame, _ w: Float) {
        let shown = w < 0.5 ? current : incoming
        let alpha: Float = {
            guard let shown else { return 0 }
            let since = f.time - shown.start
            let intro = min(max((since - 1) / 3, 0), 1)
            return incoming == nil ? intro : (w < 0.5 ? 1 - w * 2 : w * 2 - 1)
        }()
        let view = shown.map { self.view($0, f.time, aspect: Double(f.aspect)) }
        let p = shown.map { Self.plates[$0.index] }
        let size = f.overlay.size
        let pointsPerArcmin = p.flatMap { p in view.map { Double(size.width) / (Double($0.z) * p.field.x) } } ?? 1
        let bar = p.map { Self.bar($0, pointsPerArcmin: pointsPerArcmin) }
        let drift = SkyKit.drift(Double(f.time) + Double(f.seed) * 50, amount: 12, seed: Double(f.seed))
        let key = "\(shown?.index ?? -1) \(Int(alpha * 20)) \(Int((bar?.length ?? 0) * 2)) \(Int(drift.x)) \(Int(drift.y))"
        guard key != overlayKey, f.time - overlayAt > 0.1 || overlayKey.isEmpty else { return }
        overlayKey = key
        overlayAt = f.time
        f.overlay.draw { ctx, size in
            guard let p, let bar, alpha > 0 else { return }
            let a = CGFloat(alpha)
            let shadow = NSShadow()
            shadow.shadowColor = NSColor(white: 0, alpha: 0.75)
            shadow.shadowBlurRadius = 7
            shadow.set()
            let left = 54 + drift.x, bottom = size.height - 50 + drift.y
            func white(_ x: CGFloat) -> NSColor { NSColor(white: 1, alpha: x * a) }

            // Left: telescope and instrument, name, what and how far, the filter legend, the credit.
            ctx.text("\(p.telescope.uppercased())  ·  \(p.instrument.uppercased())", at: CGPoint(x: left, y: bottom - 136), size: 10,
                     weight: .medium, color: white(0.46), tracking: 2)
            ctx.text(p.name, at: CGPoint(x: left - 1, y: bottom - 122), size: 28, weight: .light, color: white(0.9))
            ctx.text("\(p.what)  ·  \(p.distanceText)", at: CGPoint(x: left, y: bottom - 84), size: 12.5, color: white(0.62))
            let column: CGFloat = 52
            for (i, filter) in p.filters.enumerated() {
                let x = left + CGFloat(i) * column
                let c = filter.color.rgb
                ctx.setFillColor(NSColor(srgbRed: c.x, green: c.y, blue: c.z, alpha: 0.85 * a).cgColor)
                ctx.fill(CGRect(x: x, y: bottom - 53, width: column - 10, height: 2))
                ctx.text(filter.name, at: CGPoint(x: x, y: bottom - 47), size: 9.5, weight: .medium, mono: true, color: white(0.62))
                ctx.text(Self.wavelength(filter.microns, p.webb), at: CGPoint(x: x, y: bottom - 34), size: 9.5, color: white(0.5))
            }
            ctx.text(p.credit, at: CGPoint(x: left, y: bottom - 11), size: 9, color: white(0.44))

            // Right: north and east, the scale bar, the field and date.
            let right = size.width - 56 - drift.x
            let barY = bottom - 30
            ctx.setStrokeColor(white(0.7).cgColor)
            ctx.setLineWidth(1)
            ctx.move(to: CGPoint(x: right - bar.length, y: barY - 3))
            ctx.addLine(to: CGPoint(x: right - bar.length, y: barY))
            ctx.addLine(to: CGPoint(x: right, y: barY))
            ctx.addLine(to: CGPoint(x: right, y: barY - 3))
            ctx.strokePath()
            ctx.text(bar.label, at: CGPoint(x: right - bar.length / 2, y: barY - 18), size: 10, color: white(0.62), anchor: CGPoint(x: 0.5, y: 0))
            let field = String(format: "%@′ × %@′ field", Self.trim(p.field.x), Self.trim(p.field.y))
            ctx.text("\(field)  ·  released \(p.released)", at: CGPoint(x: right, y: bottom - 11), size: 9.5, color: white(0.46),
                     anchor: CGPoint(x: 1, y: 0))
            if let north = p.north {
                let o = CGPoint(x: right - 6, y: barY - 50)
                let n = north * .pi / 180
                // Screen y grows downward; east is 90° anticlockwise from north on the sky.
                let up = CGVector(dx: sin(n), dy: -cos(n)), east = CGVector(dx: -cos(n), dy: -sin(n))
                ctx.setStrokeColor(white(0.55).cgColor)
                ctx.setLineWidth(0.8)
                ctx.setLineJoin(.miter)
                for (v, len) in [(up, 26.0), (east, 18.0)] {
                    let tip = CGPoint(x: o.x + v.dx * len, y: o.y + v.dy * len)
                    ctx.move(to: o)
                    ctx.addLine(to: tip)
                    // A small open arrowhead.
                    for side in [-1.0, 1.0] {
                        ctx.move(to: tip)
                        ctx.addLine(to: CGPoint(x: tip.x - v.dx * 4 + v.dy * 2.4 * side, y: tip.y - v.dy * 4 - v.dx * 2.4 * side))
                    }
                }
                ctx.strokePath()
                ctx.text("N", at: CGPoint(x: o.x + up.dx * 34, y: o.y + up.dy * 34), size: 9, weight: .medium, color: white(0.6),
                         anchor: CGPoint(x: 0.5, y: 0.5))
                ctx.text("E", at: CGPoint(x: o.x + east.dx * 26, y: o.y + east.dy * 26), size: 9, weight: .medium, color: white(0.5),
                         anchor: CGPoint(x: 0.5, y: 0.5))
            }
        }
    }

    /// A round-numbered scale bar between about 70 and 170 points long: light-years at the object, or arcseconds for
    /// a deep field.
    static func bar(_ p: Plate, pointsPerArcmin: Double) -> (length: CGFloat, label: String) {
        let nice: [Double] = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
                              100_000, 200_000, 500_000, 1e6, 2e6]
        let target = 130 / pointsPerArcmin // arcminutes
        if let scale = p.scale {
            let ly = nice.last { $0 <= target * scale } ?? nice[0]
            let arcmin = ly / scale
            let angle = arcmin >= 1 ? "\(trim(arcmin))′" : "\(Int((arcmin * 60).rounded()))″"
            let number = ly >= 1 ? SkyKit.grouped(Int(ly)) : trim(ly)
            return (CGFloat(arcmin * pointsPerArcmin), "\(number) light-year\(ly == 1 ? "" : "s")  ·  \(angle)")
        }
        let arcsec = nice.last { $0 <= target * 60 } ?? 1
        return (CGFloat(arcsec / 60 * pointsPerArcmin), "\(SkyKit.grouped(Int(arcsec))) arcseconds")
    }

    static func wavelength(_ microns: Double, _ webb: Bool) -> String {
        webb ? String(format: microns < 10 ? "%.2f µm" : "%.1f µm", microns) : "\(Int((microns * 1000).rounded())) nm"
    }

    /// 4.20 → "4.2", 7.0 → "7".
    static func trim(_ x: Double) -> String {
        var s = String(format: "%.2f", x)
        while s.hasSuffix("0") { s.removeLast() }
        if s.hasSuffix(".") { s.removeLast() }
        return s
    }

    private static let source = """
    // a, b: centre and extent (texture coordinates) of images A and B   c: their sizes   d: B's weight, has A, has B
    inline float3 plateSample(texture2d<float> t, float2 screen, float4 view, float2 size, float2 res) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
        float2 uv = view.xy + (screen - 0.5) * view.zw;
        float footprint = view.z * size.x / res.x;
        if (footprint <= 1.0) return catmull(t, s, uv).rgb;
        // Shrinking: four rotated-grid taps from the level just finer than the pixel, sharp without shimmer.
        float lod = max(log2(footprint) - 1.0, 0.0);
        float2 px = view.zw / res;
        float3 c = t.sample(s, uv + px * float2(0.125, 0.375), level(lod)).rgb;
        c += t.sample(s, uv + px * float2(-0.375, 0.125), level(lod)).rgb;
        c += t.sample(s, uv + px * float2(0.375, -0.125), level(lod)).rgb;
        c += t.sample(s, uv + px * float2(-0.125, -0.375), level(lod)).rgb;
        return c * 0.25;
    }

    fragment float4 plateView(V in [[stage_in]], constant U &u [[buffer(0)]],
                              texture2d<float> A [[texture(0)]], texture2d<float> B [[texture(1)]]) {
        float3 c;
        if (u.d.y > 0.5) {
            c = plateSample(A, in.uv, u.a, u.c.xy, u.res);
            if (u.d.z > 0.5) c = mix(c, plateSample(B, in.uv, u.b, u.c.zw, u.res), u.d.x);
        } else {
            // Nothing yet: a faint field of stars.
            float2 px = in.uv * u.res;
            c = float3(0.0012, 0.0014, 0.002) + starfield(px + u.seed * 91, u.res.y / 140, 0.08, u.res.y / 1000, u.time) * 0.25;
        }
        // A gentle fall-off at the corners, and a soft shade behind the labels that deepens where the picture is bright.
        float2 v = in.uv - 0.5;
        c *= 1 - 0.18 * dot(v, v);
        float aspect = u.res.x / u.res.y;
        float2 q = (in.uv - float2(0.0, 1.0)) * float2(aspect, 1) * float2(1.3, 3.2);
        float2 r = (in.uv - float2(1.0, 1.0)) * float2(aspect, 1) * float2(2.8, 3.6);
        float shade = exp(-dot(q, q) * 1.5) + 0.8 * exp(-dot(r, r) * 1.5);
        if (shade > 0.01 && u.d.y > 0.5) {
            constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
            float2 uv = u.a.xy + (in.uv - 0.5) * u.a.zw;
            float3 around = A.sample(s, uv, level(log2(max(u.a.z * u.c.x / u.res.x, 1.0) * 48))).rgb;
            float lum = dot(around, float3(0.2126, 0.7152, 0.0722));
            c *= 1 - shade * (0.18 + 0.5 * smoothstep(0.03, 0.35, lum));
        }
        return float4(toSRGB(max(c, 0.0)), 1);
    }

    // The colour of a picture for its sound: mean linear red, green, blue and brightness, the share of bright pixels,
    // and the spread of brightness. Written to slot u.a.x.
    kernel void plateStats(constant U &u [[buffer(0)]], device float *out [[buffer(1)]], texture2d<float> t [[texture(0)]],
                           uint id [[thread_position_in_grid]]) {
        if (id > 0) return;
        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
        float lod = log2(max(float(max(t.get_width(), t.get_height())) / 64.0, 1.0));
        float3 sum = 0;
        float l1 = 0, l2 = 0, bright = 0;
        for (int j = 0; j < 64; j++) for (int i = 0; i < 64; i++) {
            float3 c = t.sample(s, (float2(i, j) + 0.5) / 64, level(lod)).rgb;
            float l = dot(c, float3(0.2126, 0.7152, 0.0722));
            sum += c;
            l1 += l;
            l2 += l * l;
            bright += l > 0.3 ? 1 : 0;
        }
        float n = 64 * 64;
        device float *o = out + int(u.a.x) * 8;
        float mean = l1 / n;
        o[0] = sum.r / n; o[1] = sum.g / n; o[2] = sum.b / n; o[3] = max(mean, 1e-4);
        o[4] = bright / n; o[5] = sqrt(max(l2 / n - mean * mean, 0.0));
    }
    """
}
