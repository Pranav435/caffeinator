import AppKit
import Metal
import simd

/// NASA SDO's latest images of the Sun, dissolving from one wavelength to the next while the disc slowly turns and
/// breathes. The channels climb from the visible surface to flare plasma at ten million kelvin, then the magnetic field.
/// GOES X-ray flux runs along the bottom as a light curve.
@MainActor final class SunToday: Scene {
    struct Channel {
        let code: String
        let label: String
        let spoken: String
        let source: String
        let shows: String
        let fact: String
        let tint: SIMD3<Float>
        let hmi: Bool
        /// Pitch of the swell that greets this channel, as a ratio over the drone.
        let degree: Float
    }

    static let channels = [
        Channel(code: "HMIIC", label: "HMI Continuum", spoken: "visible light", source: "Photosphere · 5,800 K",
                shows: "Sunspots on the visible surface", fact: "the photosphere at 5,800 kelvin, with its sunspots",
                tint: [1.0, 0.5, 0.12], hmi: true, degree: 1),
        Channel(code: "1600", label: "AIA 1600 Å", spoken: "1600 ångströms", source: "Ultraviolet · 10,000 to 100,000 K",
                shows: "Upper photosphere and the magnetic network",
                fact: "ultraviolet from the upper photosphere and transition region, tracing the bright magnetic network",
                tint: [0.85, 0.8, 0.3], hmi: false, degree: 9.0 / 8),
        Channel(code: "0304", label: "AIA 304 Å", spoken: "304 ångströms", source: "Helium II · 50,000 K",
                shows: "Chromosphere, prominences and filaments",
                fact: "helium at 50,000 kelvin in the chromosphere, with prominences and filaments",
                tint: [1.0, 0.28, 0.1], hmi: false, degree: 5.0 / 4),
        Channel(code: "0171", label: "AIA 171 Å", spoken: "171 ångströms", source: "Iron IX · 600,000 K",
                shows: "Coronal loops and the quiet corona", fact: "iron at 600,000 kelvin traces coronal loops",
                tint: [1.0, 0.72, 0.2], hmi: false, degree: 3.0 / 2),
        Channel(code: "0193", label: "AIA 193 Å", spoken: "193 ångströms", source: "Iron XII · 1.2 million K",
                shows: "The corona; coronal holes show dark",
                fact: "iron at 1.2 million kelvin lights the corona, and coronal holes show dark",
                tint: [0.95, 0.62, 0.35], hmi: false, degree: 5.0 / 3),
        Channel(code: "0211", label: "AIA 211 Å", spoken: "211 ångströms", source: "Iron XIV · 2 million K",
                shows: "Hot corona over active regions", fact: "iron at 2 million kelvin glows over active regions",
                tint: [0.8, 0.45, 0.85], hmi: false, degree: 2),
        Channel(code: "0131", label: "AIA 131 Å", spoken: "131 ångströms", source: "Iron VIII and XXI · 0.4 and 10 million K",
                shows: "Flaring regions", fact: "iron at 400,000 kelvin, and at 10 million kelvin where flares burn",
                tint: [0.2, 0.8, 0.85], hmi: false, degree: 9.0 / 4),
        Channel(code: "HMIB", label: "HMI Magnetogram", spoken: "the magnetogram", source: "Line-of-sight magnetic field",
                shows: "White points toward us, black away",
                fact: "the magnetic field, white where it points toward us and black where it points away",
                tint: [0.4, 0.45, 0.55], hmi: true, degree: 3.0 / 4),
    ]

    /// Low-degree p-modes of the Sun (BiSON), in microhertz, sped up 100,000 times. The first two are
    /// radial order 21, degree 0 and order 20, degree 2: 9 µHz apart, so they beat about once a second.
    private static let modes: [(freq: Float, amp: Float, pan: Float)] = [
        (3033.8, 0.055, -0.25), (3024.7, 0.05, 0.25), (3098.2, 0.011, -0.6), (2963.4, 0.011, 0.6), (3168.6, 0.008, -0.4), (3159.0, 0.008, 0.4),
    ]

    private let gpu: GPU
    private let light: MTLRenderPipelineState
    private let finish: MTLRenderPipelineState
    private let bloom: Bloom
    private let blank: MTLTexture
    private var hdr: MTLTexture?
    private var textures: [String: MTLTexture] = [:]
    private var images: [String: Data] = [:]
    private var stamps: [String: Date] = [:]
    private var xray: [(time: Date, flux: Double)] = []
    private var pending = 0
    private var fetched = Date.distantPast
    private let start: Int
    private var shown: (a: Int, b: Int, w: Float) = (0, 0, 0)
    private var overlayKey = ""
    private var overlayAt: Float = -1

    var scale: Float { 1 }
    var ready: Bool { pending == 0 && fetched != .distantPast }

    var facts: String? {
        let c = Self.channels[shown.w < 0.5 ? shown.a : shown.b]
        var s: String
        if textures.isEmpty {
            s = pending > 0 ? "Fetching the Sun's latest images from NASA's Solar Dynamics Observatory."
                            : "Offline, and no SDO images are saved yet, so this is a drawing of the Sun."
        } else if let stamp = stamps[c.code] {
            s = "SDO, \(c.spoken) at \(SkyKit.utc(stamp)) UTC: \(c.fact)."
        } else {
            s = "SDO's latest image in \(c.spoken): \(c.fact)."
        }
        if let now = xray.last {
            s += " GOES X-ray flux \(Self.flareClass(now.flux))"
            if let peak = xray.max(by: { $0.flux < $1.flux }), Self.flareClass(peak.flux) != Self.flareClass(now.flux) {
                s += "; the strongest in the last 6 hours was \(Self.flareClass(peak.flux)) at \(SkyKit.utc(peak.time)) UTC"
            }
            s += "."
        }
        return s
    }

    init(gpu: GPU) throws {
        self.gpu = gpu
        let library = try gpu.library(SkyKit.metal + Self.source)
        light = try gpu.pipeline(library, "sunLight")
        finish = try gpu.pipeline(library, "sunFinish")
        bloom = try Bloom(gpu: gpu)
        var black: UInt32 = 0
        blank = gpu.texture(1, 1, .rgba8Unorm_srgb, bytes: &black, bytesPerRow: 4)
        start = (Snapshot.variant ?? Int.random(in: 0..<Self.channels.count)) % Self.channels.count
        shown = (start, start, 0)
    }

    /// The rows of SWPC's X-ray feed. A copy has been seen with stray bytes after the array, so parse up to its end.
    nonisolated static func xrayRows(_ data: Data?) -> [[String: Any]]? {
        guard var data else { return nil }
        if let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] { return rows }
        guard let end = data.range(of: Data("}]".utf8)) else { return nil }
        data = data.prefix(upTo: end.upperBound)
        return try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
    }

    /// The GOES class of an X-ray flux in watts per square metre, such as "C1.2".
    nonisolated static func flareClass(_ flux: Double) -> String {
        let classes: [(String, Double)] = [("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)]
        guard flux > 0 else { return "A0.0" }
        let i = classes.firstIndex { flux >= $0.1 } ?? classes.count - 1
        var (letter, base) = classes[i]
        var value = (flux / base * 10).rounded() / 10
        if value >= 10, i > 0 { (letter, base) = classes[i - 1]; value = (flux / base * 10).rounded() / 10 }
        return letter + String(format: "%.1f", value)
    }

    /// Glyph templates for the digits 0 to 9 in the caption SDO prints under each AIA image: 5 × 8 cells of ink, one hex
    /// digit each, averaged from real captions.
    private nonisolated static let digits: [[Double]] = """
    0000048556a2008f000ab20094c5880053000000;0000045aaa555ff000ff000ff000ff0000000000;00000585674201b001b503c408c2003555500000;\
    0000004577006a10025a0000ab65890252000000;0000000390019d03a3d0856e5002d00000000000;0000047553695524403b0000ab757a0253000000;\
    000003855683001fc589c000a5c5890053000000;000005555800057003c000c200a5000300000000;00010565579a15a5baa5f100cb635a0155000000;\
    0000038558c300a5a03d05a7c3a57a0053000000
    """.split(separator: ";").map { $0.compactMap { $0.hexDigitValue.map { Double($0) / 15 } } }

    /// Reads the time printed in the bottom corner of an AIA image ("2026-10-24 10:36:58 UT") by matching each digit
    /// against the templates. Returns nil if the caption doesn't look as expected.
    nonisolated static func stamp(_ data: Data) -> Date? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil), image.width >= 512 else { return nil }
        // Drawn at 1024 pixels whatever size SDO serves; the caption sits in the bottom 5%.
        let w = 1024, h = 1024, top = 975
        var gray = [UInt8](repeating: 0, count: w * h)
        let drawn = gray.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { return nil }
        func ink(_ x: Int, _ y: Int) -> Bool { gray[y * w + x] > 140 }
        let rows = (top..<h).filter { y in (0..<700).contains { ink($0, y) } }
        guard let y0 = rows.first, let y1 = rows.last, y1 - y0 >= 10 else { return nil }
        var spans: [(Int, Int)] = []
        var start: Int?
        for x in 0..<700 {
            let on = (y0...y1).contains { ink(x, $0) }
            if on, start == nil { start = x }
            if !on, let s = start { spans.append((s, x - 1)); start = nil }
        }
        // The last two glyphs are "UT"; the 18 before them are "2026-10-24" and "10:36:58".
        guard spans.count >= 20 else { return nil }
        let glyphs = Array(spans[(spans.count - 20)..<(spans.count - 2)])
        guard [4, 7].allSatisfy({ glyphs[$0].1 - glyphs[$0].0 >= 7 }), [12, 15].allSatisfy({ glyphs[$0].1 - glyphs[$0].0 <= 5 }) else { return nil }
        func read(_ g: (Int, Int)) -> Int {
            var cells = [Double](repeating: 0, count: 40)
            let gw = g.1 - g.0 + 1, gh = y1 - y0 + 1
            for j in 0..<8 {
                for i in 0..<5 {
                    let ya = y0 + j * gh / 8, yb = max(y0 + (j + 1) * gh / 8, ya + 1)
                    let xa = g.0 + i * gw / 5, xb = max(g.0 + (i + 1) * gw / 5, xa + 1)
                    var sum = 0.0
                    for y in ya..<yb { for x in xa..<xb { sum += Double(gray[y * w + x]) / 255 } }
                    cells[j * 5 + i] = sum / Double((yb - ya) * (xb - xa))
                }
            }
            let distance = digits.map { t in zip(t, cells).reduce(0) { $0 + abs($1.0 - $1.1) } }
            return distance.indices.min { distance[$0] < distance[$1] } ?? 0
        }
        let v = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17].map { read(glyphs[$0]) }
        func number(_ r: Range<Int>) -> Int { r.reduce(0) { $0 * 10 + v[$1] } }
        let text = String(format: "%04d%02d%02d_%02d%02d%02d", number(0..<4), number(4..<6), number(6..<8), number(8..<10), number(10..<12), number(12..<14))
        return parseStamp(text)
    }

    nonisolated static func parseStamp(_ text: String) -> Date? {
        let digits = text
        let patterns = [#"(\d{4})-(\d{2})-(\d{2})\s*(\d{2}):(\d{2}):(\d{2})"#, #"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"#]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern),
                  let m = regex.firstMatch(in: digits, range: NSRange(digits.startIndex..., in: digits)) else { continue }
            let v = (1...6).compactMap { Range(m.range(at: $0), in: digits).flatMap { Int(digits[$0]) } }
            guard v.count == 6 else { continue }
            var c = Calendar(identifier: .gregorian)
            c.timeZone = TimeZone(identifier: "UTC")!
            if let date = c.date(from: DateComponents(year: v[0], month: v[1], day: v[2], hour: v[3], minute: v[4], second: v[5])),
               abs(date.timeIntervalSinceNow) < 5 * 86400 { return date }
        }
        return nil
    }

    private func refresh() {
        fetched = .now
        pending += 2
        Task { @MainActor [weak self] in
            let downloads = await withTaskGroup(of: (String, Data?).self) { group in
                for c in Self.channels {
                    let code = c.code
                    group.addTask {
                        let url = URL(string: "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_\(code).jpg")!
                        return (code, await Feed.data(url, maxAge: 900))
                    }
                }
                var out: [String: Data] = [:]
                for await (code, data) in group { if let data { out[code] = data } }
                return out
            }
            guard let changed = self.map({ scene in downloads.filter { scene.images[$0.key] != $0.value } }) else { return }
            for (code, data) in changed {
                let pixels = await Task.detached(priority: .utility) { SkyPixels.decode(data) }.value
                guard let self else { return }
                if let pixels, let t = gpu.mipmapped(pixels) {
                    textures[code] = t
                    images[code] = data
                }
            }
            // AIA images carry their time in the caption; HMI's is in a small text file beside them.
            var read = await Task.detached(priority: .utility) {
                changed.filter { !$0.key.hasPrefix("HMI") }.compactMapValues { Self.stamp($0) }
            }.value
            let hmi = await Feed.data(URL(string: "https://sdo.gsfc.nasa.gov/assets/img/latest/hmitimes.txt")!, maxAge: 900)
            if let time = hmi.flatMap({ String(data: $0, encoding: .utf8) }).flatMap(Self.parseStamp) {
                read["HMIIC"] = time
                read["HMIB"] = time
            }
            guard let self else { return }
            stamps.merge(read) { $1 }
            overlayKey = ""
            pending -= 1
        }
        Task { @MainActor [weak self] in
            let url = URL(string: "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json")!
            let rows = Self.xrayRows(await Feed.data(url, maxAge: 900))
            guard let self else { return }
            defer { pending -= 1 }
            let iso = ISO8601DateFormatter()
            let points = (rows ?? []).compactMap { row -> (Date, Double)? in
                guard row["energy"] as? String == "0.1-0.8nm", let flux = row["flux"] as? Double, flux > 0,
                      let tag = row["time_tag"] as? String, let time = iso.date(from: tag) else { return nil }
                return (time, flux)
            }.sorted { $0.0 < $1.0 }
            if !points.isEmpty { xray = points.map { (time: $0.0, flux: $0.1) } }
            overlayKey = ""
        }
    }

    func draw(_ f: Frame) {
        if f.now.timeIntervalSince(fetched) > 900 { refresh() }
        if hdr?.width != f.target.width || hdr?.height != f.target.height { hdr = f.gpu.texture(f.target.width, f.target.height) }

        // Which channels are showing: each holds for 24 s, then dissolves into the next over 7 s.
        let order = Self.channels.indices.filter { textures[Self.channels[$0].code] != nil }
        let period: Float = 31, hold: Float = 24
        let pos = f.time / period
        var a = start, b = start, progress: Float = 0
        if !order.isEmpty {
            let first = order.firstIndex { $0 >= start } ?? 0
            let k = Int(pos) + first
            a = order[k % order.count]
            b = order[(k + 1) % order.count]
            progress = order.count > 1 ? max(0, (pos - floor(pos)) * period - hold) / (period - hold) : 0
        }
        let w = progress * progress * (3 - 2 * progress)
        shown = (a, b, w)
        let ca = Self.channels[a], cb = Self.channels[b]

        // A slow turn (one revolution in two hours), a breathing zoom and a gentle wander.
        let t = Double(f.time) + Double(f.seed) * 37
        let angle = Float(t * 2 * .pi / 7200)
        let radius = Float(0.6 + 0.1 * (0.5 - 0.5 * cos(t * 2 * .pi / 260)))
        let center = SIMD2<Float>(Float(0.07 * sin(t / 97)), Float(0.035 * sin(t / 131 + 1)))
        // SDO sees the Sun 959.63 arcseconds wide at 1 AU; AIA's 1024 images are 2.4″ per pixel, HMI's 2.016″.
        let au = Float(simd_length(Astro.sunEcliptic(f.now)))
        let arcsec = 959.63 / max(au, 0.9)
        func size(_ c: Channel) -> Float { arcsec / (c.hmi ? 2.016 : 2.4) / 1024 }
        let u = f.uniforms([angle, radius, center.x, center.y],
                           [progress, size(ca), size(cb), order.isEmpty ? 1 : 0],
                           SIMD4(ca.tint, ca.hmi ? 1 : 0), SIMD4(cb.tint, cb.hmi ? 1 : 0))
        let ta = textures[ca.code] ?? blank, tb = textures[cb.code] ?? blank
        f.cmd.draw(light, into: hdr!, uniforms: u, textures: [ta, tb])
        let glow = bloom.glow(f.cmd, hdr!)
        f.cmd.draw(finish, into: f.target, uniforms: u, textures: [hdr!, glow])

        drawOverlay(f, ca, cb, w)
        if let synth = f.sound { play(synth, f, cb, progress) }
    }

    private func play(_ synth: Synth, _ f: Frame, _ next: Channel, _ progress: Float) {
        let t = f.time
        for (i, m) in Self.modes.enumerated() {
            // Modes are excited by convection at random, so each one's strength wanders.
            let wander = 0.7 + 0.3 * sin(t * (0.05 + 0.011 * Float(i)) + Float(i) * 2.1)
            synth.drone(i, m.freq / 10, m.amp * wander, pan: m.pan)
        }
        synth.drone(6, 3033.8 / 20, 0.03)
        let swell = progress > 0 ? sin(Float.pi * progress) : 0
        synth.drone(7, 3033.8 / 20 * next.degree * 2, 0.055 * swell * swell)
        // X-ray flux as a faint hiss: A class is almost silent, an X flare a soft rush.
        let flux = xray.last?.flux ?? 1e-7
        let level = Float(max(0, min(1, (log10(flux) + 8.5) / 5)))
        synth.noise(0.004 + 0.05 * level * level, cutoff: 900 + 2500 * level)
    }

    // MARK: Overlay

    private func drawOverlay(_ f: Frame, _ a: Channel, _ b: Channel, _ w: Float) {
        let fade = w < 0.5 ? 1 - w * 2 : w * 2 - 1
        let c = w < 0.5 ? a : b
        let key = "\(c.code)\(Int(fade * 12))"
        guard key != overlayKey || f.time - overlayAt > 20 else { return }
        guard f.time - overlayAt > 0.1 || overlayKey.isEmpty else { return }
        overlayKey = key
        overlayAt = f.time
        let drift = SkyKit.drift(Double(f.time) + Double(f.seed) * 50, amount: 14, seed: Double(f.seed))
        let stamp = stamps[c.code]
        let points = xray
        f.overlay.draw { ctx, size in
            let left = 56 + drift.x, bottom = size.height - 58 + drift.y
            let alpha = CGFloat(fade)
            let tint = NSColor(srgbRed: CGFloat(c.tint.x) * 0.5 + 0.5, green: CGFloat(c.tint.y) * 0.5 + 0.5,
                               blue: CGFloat(c.tint.z) * 0.5 + 0.5, alpha: 0.85 * alpha)
            let when = stamp.map { "SDO  ·  \(SkyKit.utc($0)) UTC" } ?? "SDO  ·  LATEST"
            ctx.text(when, at: CGPoint(x: left, y: bottom - 92), size: 10.5, weight: .medium,
                     color: NSColor(white: 1, alpha: 0.42 * alpha), tracking: 2)
            ctx.text(c.label, at: CGPoint(x: left - 1, y: bottom - 76), size: 30, weight: .light,
                     color: NSColor(white: 1, alpha: 0.88 * alpha))
            ctx.text(c.source, at: CGPoint(x: left, y: bottom - 36), size: 14, color: tint)
            ctx.text(c.shows, at: CGPoint(x: left, y: bottom - 16), size: 12.5, color: NSColor(white: 1, alpha: 0.5 * alpha))
            if points.count > 1 { Self.lightCurve(ctx, points, right: size.width - 56 - drift.x, bottom: bottom) }
        }
    }

    /// The GOES X-ray light curve on a log scale with the flare classes marked.
    private static func lightCurve(_ ctx: CGContext, _ points: [(time: Date, flux: Double)], right: CGFloat, bottom: CGFloat) {
        let width: CGFloat = 240, height: CGFloat = 70
        let x0 = right - width - 14, y0 = bottom - height
        let t0 = points.first!.time.timeIntervalSince1970, t1 = max(points.last!.time.timeIntervalSince1970, t0 + 1)
        let low = -8.0, high = -3.0
        func y(_ flux: Double) -> CGFloat {
            y0 + height * CGFloat(1 - (min(max(log10(max(flux, 1e-9)), low), high) - low) / (high - low))
        }
        ctx.setStrokeColor(NSColor(white: 1, alpha: 0.1).cgColor)
        ctx.setLineWidth(0.5)
        for i in 0...5 {
            let yy = y(pow(10, low + Double(i)))
            ctx.move(to: CGPoint(x: x0, y: yy))
            ctx.addLine(to: CGPoint(x: x0 + width, y: yy))
        }
        ctx.strokePath()
        for (i, letter) in ["A", "B", "C", "M", "X"].enumerated() {
            ctx.text(letter, at: CGPoint(x: x0 + width + 8, y: y(pow(10, low + Double(i) + 0.5))), size: 9, weight: .medium,
                     color: NSColor(white: 1, alpha: 0.38), anchor: CGPoint(x: 0, y: 0.5))
        }
        ctx.setStrokeColor(NSColor(srgbRed: 1, green: 0.72, blue: 0.4, alpha: 0.8).cgColor)
        ctx.setLineWidth(1.1)
        ctx.setLineJoin(.round)
        for (i, p) in points.enumerated() {
            let x = x0 + width * CGFloat((p.time.timeIntervalSince1970 - t0) / (t1 - t0))
            if i == 0 { ctx.move(to: CGPoint(x: x, y: y(p.flux))) } else { ctx.addLine(to: CGPoint(x: x, y: y(p.flux))) }
        }
        ctx.strokePath()
        let last = points.last!
        ctx.setFillColor(NSColor(srgbRed: 1, green: 0.8, blue: 0.55, alpha: 0.95).cgColor)
        ctx.fillEllipse(in: CGRect(x: x0 + width - 2.5, y: y(last.flux) - 2.5, width: 5, height: 5))
        ctx.text("GOES X-RAY FLUX  ·  6 HOURS", at: CGPoint(x: x0, y: y0 - 56), size: 10.5, weight: .medium,
                 color: NSColor(white: 1, alpha: 0.42), tracking: 2)
        ctx.text(flareClass(last.flux), at: CGPoint(x: x0 - 1, y: y0 - 40), size: 22, weight: .light,
                 color: NSColor(white: 1, alpha: 0.85))
    }

    private static let source = """
    // a: rotation, disc radius (screen units), centre   b: dissolve progress, disc radius in image A and B (uv), no images
    // c: tint A, A is HMI   d: tint B, B is HMI
    float3 disc(texture2d<float> t, float2 q, float size, float hmi, float texel) {
        float r = length(q);
        float2 uv = 0.5 + q * size * float2(1, -1);
        float3 c = picture(t, uv, texel).rgb;
        // Keep AIA's corona but fade it before the square edge of the image and its printed caption; HMI sees only the disc.
        float edge = hmi > 0.5 ? 1 - smoothstep(1.0, 1.03, r) : 1 - smoothstep(1.06, 1.22, r);
        if (hmi < 0.5) edge *= 1 - smoothstep(0.935, 0.95, uv.y) * (1 - smoothstep(0.55, 0.6, uv.x));
        return c * edge * (hmi > 0.5 ? 0.8 : 1.1);
    }

    fragment float4 sunLight(V in [[stage_in]], constant U &u [[buffer(0)]],
                             texture2d<float> A [[texture(0)]], texture2d<float> B [[texture(1)]]) {
        float2 p = centered(in.uv, u.res);
        float R = u.a.y;
        float2 q = rot(u.a.x) * ((p - u.a.zw) / R);
        float r = length(q);
        float pxPerRadius = R * u.res.y * 0.5;
        float3 c;
        float hmiW;
        float3 tint;
        if (u.b.w > 0.5) {
            // No images at all: a limb-darkened disc with granulation, so the scene still says "Sun".
            float mu = sqrt(max(1 - r * r, 0.0));
            float g = 0.85 + 0.15 * fbm(q * 40 + u.time * 0.02, 4);
            c = float3(1.0, 0.45, 0.1) * (0.4 + 0.6 * mu) * g * (1 - smoothstep(0.995, 1.005, r)) * 0.9;
            tint = float3(1.0, 0.5, 0.15);
            hmiW = 1;
        } else {
            float3 a = disc(A, q, u.b.y, u.c.w, pxPerRadius / (u.b.y * A.get_width()));
            float3 b = disc(B, q, u.b.z, u.d.w, pxPerRadius / (u.b.z * B.get_width()));
            // The next wavelength blooms outward from the middle through a soft, turbulent front.
            float n = 0.5 * saturate(r / 1.3) + 0.5 * fbm(q * 2.2 + 5.3, 4);
            float e = u.b.x * 1.35 - 0.17;
            float w = 1 - smoothstep(e - 0.13, e + 0.13, n);
            c = mix(a, b, w);
            float k = u.b.x * u.b.x * (3 - 2 * u.b.x);
            tint = mix(u.c.xyz, u.d.xyz, k);
            hmiW = mix(u.c.w, u.d.w, k);
        }
        // A soft halo beyond the limb carries the colour of the light, strongest where the image has none.
        float beyond = max(r - 1, 0.0);
        float halo = (0.09 * exp(-beyond * 7) + 0.012 * exp(-beyond * 2.2)) * smoothstep(0.985, 1.02, r);
        halo *= mix(0.55, 1.0, hmiW) * mix(1.0, smoothstep(1.0, 1.25, r) * 0.7 + 0.3, 1 - hmiW);
        c += tint * tint * halo;
        // Faint stars far behind, turning with the Sun.
        c += starfield(q * pxPerRadius, 9, 0.08, 1.1, u.time) * 0.25 * smoothstep(1.2, 1.6, r);
        return float4(c, 1);
    }

    fragment float4 sunFinish(V in [[stage_in]], constant U &u [[buffer(0)]],
                              texture2d<float> hdr [[texture(0)]], texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = hdr.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.22;
        // Gentle shoulder instead of a hard clip, then out to display.
        c = select(c, 0.82 + 0.18 * tanh((c - 0.82) / 0.18), c > 0.82);
        float2 v = in.uv - 0.5;
        c *= 1 - 0.35 * dot(v, v);
        return float4(toSRGB(max(c, 0.0)), 1);
    }
    """
}
