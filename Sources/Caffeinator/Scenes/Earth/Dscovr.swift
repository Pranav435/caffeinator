import AppKit
import ImageIO
import Metal
import simd

/// Today's photographs of the whole Earth from NASA's EPIC camera on DSCOVR, a million and a half kilometres away at
/// the Sun-Earth L1 point, where the Earth is always lit face-on. The day's frames play as one slow rotation: each
/// is reprojected onto the turning globe with its own centroid and orientation, and neighbours blend where they
/// overlap, so the real clouds drift and the continents roll past without a cut.
@MainActor final class Dscovr: Scene {
    private struct Shot {
        var date: Date
        var lat: Double, lon: Double
        /// Earth to DSCOVR and Earth to Sun in J2000 equatorial kilometres.
        var craft: SIMD3<Double>, sun: SIMD3<Double>
        var texture: MTLTexture
        /// Shares of the disc that are cloud, open ocean and land or ice, for the soundscape.
        var cloud: Float, ocean: Float, land: Float
    }

    /// What the playhead shows right now, for the annotations and facts.
    private struct State {
        var date: Date, lat: Double, lon: Double, distance: Double, angle: Double
        var subsolar: (lat: Double, lon: Double)
        var index: Int, count: Int
        var cloud: Float, ocean: Float, land: Float
    }

    private static let api = URL(string: "https://epic.gsfc.nasa.gov/api/natural")!
    /// Seconds each photograph's step of the rotation takes.
    private static let interval: Float = 9
    /// Half the field of view of the sky behind, as a focal length in centered units.
    private static let skyZoom: Float = 2.4
    private static let radius: Float = 0.7

    private let gpu: GPU
    private let globe: Globe
    private let canvas: Globe.Canvas
    private let pass: MTLRenderPipelineState
    private var shots: [Shot] = []
    private var sky: MTLTexture?
    private var settled = 0
    private var failed = false
    private var loading = false
    private var checked = Date.distantPast
    private var day = ""
    private var head: Float
    private var state: State?
    private var camera: Globe.Camera?
    private var lastOverlay: Float = -1
    private var lastBell: Float = 0
    private var cloudy: Float = 0, oceanic: Float = 0, landed: Float = 0

    var scale: Float { 1 }
    var ready: Bool { settled >= 2 && (shots.count >= 3 || failed || !loading) }
    var credit: String? { shots.isEmpty ? Globe.credit : "Earth: NASA EPIC team, DSCOVR · Milky Way: ESO/S. Brunier" }

    var facts: String? {
        guard let s = state else {
            return failed ? "NASA's EPIC photographs couldn't be reached, so this is a rendering of the Earth lit by today's Sun."
                          : "Loading today's photographs of the whole Earth from NASA's EPIC camera."
        }
        let place = Globe.region(lat: s.lat, lon: s.lon)
        return "DSCOVR is \(EarthText.kilometres(s.distance)) from Earth, at the Sun-Earth L1 point. This photograph from "
            + "\(EarthText.utc(s.date)) on \(Self.dayText(s.date)) is centred on \(place), at \(Self.spoken(s.lat, "north", "south")) "
            + "and \(Self.spoken(s.lon, "east", "west")). It is picture \(s.index + 1) of \(s.count) that day."
    }

    init(gpu: GPU) throws {
        self.gpu = gpu
        globe = try Globe.shared(gpu)
        canvas = try Globe.Canvas(gpu: gpu)
        pass = try gpu.pipeline(gpu.library(Globe.metal + Self.source), "epicFill")
        head = Float(Snapshot.variant ?? 0) * 3.5
        Task { @MainActor in
            sky = await MilkyWay.texture(gpu)
            settled += 1
        }
        load()
    }

    // MARK: Pictures

    private struct Entry { var date: Date, lat: Double, lon: Double, craft: SIMD3<Double>, sun: SIMD3<Double>, url: URL }

    private static func parse(_ json: Any?) -> [Entry] {
        guard let list = json as? [[String: Any]] else { return [] }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        func vector(_ any: Any?) -> SIMD3<Double>? {
            guard let d = any as? [String: Any], let x = (d["x"] as? NSNumber)?.doubleValue, let y = (d["y"] as? NSNumber)?.doubleValue,
                  let z = (d["z"] as? NSNumber)?.doubleValue else { return nil }
            return SIMD3(x, y, z)
        }
        return list.compactMap { e in
            guard let name = e["image"] as? String, let stamp = e["date"] as? String, let date = formatter.date(from: stamp),
                  let c = e["centroid_coordinates"] as? [String: Any], let lat = (c["lat"] as? NSNumber)?.doubleValue,
                  let lon = (c["lon"] as? NSNumber)?.doubleValue, let craft = vector(e["dscovr_j2000_position"]),
                  let sun = vector(e["sun_j2000_position"]), lat.isFinite, lon.isFinite, simd_length(craft) > 1e5,
                  let id = e["identifier"] as? String, id.count >= 8, id.allSatisfy(\.isNumber),
                  name.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" }) else { return nil }
            let y = id.prefix(4), m = id.dropFirst(4).prefix(2), d = id.dropFirst(6).prefix(2)
            guard let url = URL(string: "https://epic.gsfc.nasa.gov/archive/natural/\(y)/\(m)/\(d)/jpg/\(name).jpg") else { return nil }
            return Entry(date: date, lat: lat, lon: lon, craft: craft, sun: sun, url: url)
        }.sorted { $0.date < $1.date }
    }

    /// Fetches the latest day's list, then its photographs one at a time.
    private func load() {
        guard !loading else { return }
        loading = true
        checked = .now
        Task { @MainActor in
            let entries = Self.parse(await Feed.object(Self.api, maxAge: 1800))
            let name = entries.first.map { Self.dayText($0.date) } ?? ""
            if entries.isEmpty || name == day && shots.count == entries.count {
                if entries.isEmpty && shots.isEmpty { failed = true }
                loading = false
                settled = max(settled, 2)
                return
            }
            var fresh: [Shot] = []
            for e in entries {
                guard let texture = await Imagery.texture(e.url, gpu, maxSize: 2048, maxAge: 30 * 86400) else { continue }
                let stats = await Self.statistics(e.url)
                fresh.append(Shot(date: e.date, lat: e.lat, lon: e.lon, craft: e.craft, sun: e.sun, texture: texture,
                                  cloud: stats.0, ocean: stats.1, land: stats.2))
                if fresh.count >= 2 { shots = fresh; day = name }
                if fresh.count == 3 { settled = 2 }
            }
            if fresh.count >= 2 { shots = fresh; day = name }
            if shots.isEmpty { failed = true }
            loading = false
            settled = max(settled, 2)
        }
    }

    /// Shares of the Earth's disc that are cloud, ocean and land, from a small copy of the picture.
    private static func statistics(_ url: URL) async -> (Float, Float, Float) {
        guard let data = await Feed.data(url, maxAge: 30 * 86400) else { return (0.4, 0.5, 0.1) }
        return await Task.detached(priority: .utility) { measure(data) }.value
    }

    private nonisolated static func measure(_ data: Data) -> (Float, Float, Float) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                                                          kCGImageSourceThumbnailMaxPixelSize: 64] as CFDictionary)
        else { return (0.4, 0.5, 0.1) }
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        let ok = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return (0.4, 0.5, 0.1) }
        var cloud = 0, ocean = 0, land = 0, total = 0
        let radius = Float(w) * 0.34
        for y in 0..<h {
            for x in 0..<w where hypot(Float(x) - Float(w) / 2, Float(y) - Float(h) / 2) < radius {
                let i = (y * w + x) * 4
                let r = Float(bytes[i]) / 255, g = Float(bytes[i + 1]) / 255, b = Float(bytes[i + 2]) / 255
                total += 1
                if min(r, min(g, b)) > 0.5 { cloud += 1 } else if b > r * 1.25 { ocean += 1 } else { land += 1 }
            }
        }
        let n = Float(max(total, 1))
        return (Float(cloud) / n, Float(ocean) / n, Float(land) / n)
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        if f.now.timeIntervalSince(checked) > 3 * 3600 { load() }
        globe.refresh(f.now)
        head += f.dt / Self.interval
        let light = canvas.light(for: f)
        if shots.count >= 2 {
            drawPhotograph(f, light)
        } else {
            drawFallback(f, light)
        }
        canvas.develop(f, Self.look)
        if f.time - lastOverlay >= 0.1 || f.index == 0 {
            lastOverlay = f.time
            drawOverlay(f)
        }
        if let synth = f.sound { play(synth, f) }
    }

    /// A camera filming the sunlit Earth with nothing added: no vignette to speak of, and the faintest grain, since the
    /// pictures already carry their own.
    private static let look = Lens.Look(exposure: 1.35, bloom: 0.02, halation: 0.008, vignette: 0.2, aberration: 0.4, grain: 0.22)

    /// Where on the timeline the playhead is: the shot it has just left, the one it is heading for, how far between.
    private func segment() -> (a: Int, b: Int, t: Float) {
        let n = shots.count
        let s = head.truncatingRemainder(dividingBy: Float(n))
        let i = min(Int(s < 0 ? s + Float(n) : s), n - 1)
        return (i, (i + 1) % n, min(max(s - Float(i), 0), 1))
    }

    /// The camera that looked at the Earth for a shot: its centroid, turned to put the north of the ecliptic up, as EPIC
    /// holds it. Matches the picture's own frame, so the photograph maps onto the sphere.
    private func axes(_ s: Shot) -> (lat: Float, lon: Float, roll: Float) {
        let g = Astro.rad(Astro.gmst(s.date))
        let eps = Astro.rad(Astro.obliquity(Astro.dayNumber(s.date)))
        let e = SIMD3(0, -sin(eps), cos(eps))
        let north = SIMD3(cos(g) * e.x + sin(g) * e.y, -sin(g) * e.x + cos(g) * e.y, e.z)
        let a = Astro.rad(s.lat), o = Astro.rad(s.lon)
        let east = SIMD3(-sin(o), cos(o), 0.0)
        let up = SIMD3(-sin(a) * cos(o), -sin(a) * sin(o), cos(a))
        return (Float(s.lat), Float(s.lon), Float(atan2(-simd_dot(north, east), simd_dot(north, up))))
    }

    private func drawPhotograph(_ f: Frame, _ light: MTLTexture) {
        let seg = segment()
        let a = shots[seg.a], b = shots[seg.b]
        let ease = seg.t * seg.t * (3 - 2 * seg.t)
        // Interpolated view: the shortest arc between the centroids, the shortest turn between the rolls.
        let ca = axes(a), cb = axes(b)
        let va = Globe.geo(Double(ca.lat), Double(ca.lon)), vb = Globe.geo(Double(cb.lat), Double(cb.lon))
        let angle = acos(max(-1, min(1, simd_dot(va, vb))))
        let v = angle < 1e-4 ? va : (va * sin((1 - seg.t) * angle) + vb * sin(seg.t * angle)) / sin(angle)
        let at = Globe.latLon(v)
        var dRoll = cb.roll - ca.roll
        dRoll -= 2 * .pi * (dRoll / (2 * .pi)).rounded()
        let roll = ca.roll + dRoll * seg.t
        var gap = b.date.timeIntervalSince(a.date)
        if gap <= 0 { gap += 86400 }
        let date = a.date.addingTimeInterval(gap * Double(seg.t))
        let cam = Globe.Camera(lat: Float(at.lat), lon: Float(at.lon), distance: 240, zoom: Self.skyZoom, roll: roll, aspect: f.aspect, date: date)
        camera = cam

        func shot(_ s: Shot) -> [SIMD4<Float>] {
            let c = axes(s)
            let k = Globe.Camera(lat: c.lat, lon: c.lon, distance: 240, zoom: 1, roll: c.roll, aspect: 1, date: s.date)
            // Disc radius as a share of the picture's width, from the angular size of the Earth against EPIC's field.
            let share = Float(atan(6371 / simd_length(s.craft)) * 180 / .pi / 0.6253)
            return [SIMD4(k.basis.columns.0, 0), SIMD4(k.basis.columns.1, 0), SIMD4(-k.basis.columns.2, share)]
        }
        let shotBuffer = canvas.stream(1, shot(a) + shot(b))
        let u = f.uniforms([Self.radius, sky == nil ? 0 : 0.3, ease, 1], [Float(a.texture.width), 0, 0, 0])
        f.cmd.draw(pass, into: light, uniforms: u, textures: [a.texture, b.texture, sky ?? a.texture], buffers: [canvas.camera(cam), shotBuffer])

        let mix = { (x: Float, y: Float) in x + (y - x) * ease }
        let craft = simd_mix(a.craft, b.craft, SIMD3(repeating: Double(ease)))
        let sun = simd_mix(a.sun, b.sun, SIMD3(repeating: Double(ease)))
        let g = Astro.rad(Astro.gmst(date))
        let sunFixed = SIMD3(cos(g) * sun.x + sin(g) * sun.y, -sin(g) * sun.x + cos(g) * sun.y, sun.z)
        let sub = Globe.latLon(SIMD3<Float>(sunFixed / simd_length(sunFixed)))
        let nearest = ease < 0.5 ? seg.a : seg.b
        state = State(date: date, lat: at.lat, lon: at.lon, distance: simd_length(craft),
                      angle: acos(max(-1, min(1, simd_dot(craft, sun) / (simd_length(craft) * simd_length(sun))))) * 180 / .pi,
                      subsolar: sub, index: nearest, count: shots.count,
                      cloud: mix(a.cloud, b.cloud), ocean: mix(a.ocean, b.ocean), land: mix(a.land, b.land))
    }

    /// Without photographs: the shared globe seen from the Sun's side, lit by the real Sun.
    private func drawFallback(_ f: Frame, _ light: MTLTexture) {
        let s = Astro.subpoint(Astro.sun(f.now), f.now)
        let cam = Globe.Camera(lat: Float(s.lat), lon: Float(s.lon), distance: 14, zoom: Self.radius * (14 * 14 - 1).squareRoot(),
                               aspect: f.aspect, date: f.now)
        camera = cam
        globe.draw(f.cmd, into: light, camera: canvas.camera(cam), style: Globe.Style(), date: f.now, time: f.time)
        state = nil
    }

    // MARK: Sound

    /// Ocean sets the depth of the pad, cloud the air and shimmer over it, land the warmth and the bells.
    private func play(_ synth: Synth, _ f: Frame) {
        let k = Float(1 - exp(-f.dt / 3))
        if let s = state {
            cloudy += (s.cloud - cloudy) * k
            oceanic += (s.ocean - oceanic) * k
            landed += (s.land - landed) * k
        }
        synth.noise(0.012 + 0.05 * cloudy, cutoff: 260 + 700 * cloudy)
        synth.drone(0, 73.4, 0.05)
        synth.drone(1, 110, 0.03 + 0.04 * oceanic, pan: -0.2)
        synth.drone(2, 146.8, 0.012 + 0.04 * oceanic, pan: 0.2)
        synth.drone(3, 185, 0.004 + 0.04 * landed, pan: -0.1)
        synth.drone(4, 220, 0.006 + 0.012 * oceanic)
        synth.drone(5, 880, 0.001 + 0.01 * cloudy, pan: 0.3)
        synth.drone(6, 1108.7, 0.0005 + 0.006 * cloudy, pan: -0.3)
        // A bell from D major pentatonic every few seconds, more often over land and higher where the picture is bright.
        guard state != nil, f.time - lastBell > 7 - 3.5 * landed else { return }
        lastBell = f.time
        let notes: [Float] = [293.7, 329.6, 370, 440, 493.9, 587.3, 659.3, 740]
        let i = min(Int(Float.random(in: 0..<1) * 4 + cloudy * 4), notes.count - 1)
        synth.play(.bell, notes[i], amp: 0.02 + 0.02 * landed, pan: Float.random(in: -0.5...0.5), length: 5)
    }

    // MARK: Overlay

    private static func dayText(_ date: Date) -> String {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        let p = c.dateComponents([.year, .month, .day], from: date)
        let months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        return "\(p.day ?? 1) \(months[((p.month ?? 1) - 1) % 12]) \(p.year ?? 2026)"
    }

    private static func isoDay(_ date: Date) -> String {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        let p = c.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", p.year ?? 0, p.month ?? 0, p.day ?? 0)
    }

    private static func spoken(_ x: Double, _ pos: String, _ neg: String) -> String {
        let n = Int(abs(x).rounded())
        return "\(n) degree\(n == 1 ? "" : "s") \(x >= 0 ? pos : neg)"
    }

    private static func degrees(_ x: Double, _ pos: String, _ neg: String) -> String {
        String(format: "%.1f° %@", abs(x), x >= 0 ? pos : neg)
    }

    private func drawOverlay(_ f: Frame) {
        guard let s = state, let cam = camera else { f.overlay.clear(); return }
        let times = shots.map(\.date)
        let seg = segment()
        let drift = CGPoint(x: CGFloat(sin(f.time * 0.007)) * 6, y: CGFloat(cos(f.time * 0.009)) * 4)
        let sub = s.subsolar
        f.overlay.draw { ctx, size in
            let dim = NSColor(white: 1, alpha: 0.42), faint = NSColor(white: 1, alpha: 0.22), hair = NSColor(white: 1, alpha: 0.3)
            let left = 44 + drift.x, top = 38 + drift.y
            ctx.text("WHOLE EARTH · DSCOVR / EPIC", at: CGPoint(x: left, y: top), size: 11, weight: .semibold, color: dim, tracking: 2)
            ctx.text(EarthText.utc(s.date), at: CGPoint(x: left, y: top + 18), size: 22, weight: .light, mono: true,
                     color: NSColor(white: 1, alpha: 0.6))
            ctx.text(Self.isoDay(s.date), at: CGPoint(x: left, y: top + 48), size: 11, mono: true, color: faint)
            let rows = [("CENTROID", "\(Self.degrees(s.lat, "N", "S"))  \(Self.degrees(s.lon, "E", "W"))"),
                        ("RANGE", String(format: "%d,%03d,000 km", Int(s.distance) / 1_000_000, Int(s.distance) / 1000 % 1000)),
                        ("SUN–EARTH–CRAFT", String(format: "%.1f°", s.angle)),
                        ("SUBSOLAR", "\(Self.degrees(sub.lat, "N", "S"))  \(Self.degrees(sub.lon, "E", "W"))")]
            for (i, row) in rows.enumerated() {
                let y = top + 78 + CGFloat(i) * 16
                ctx.text(row.0, at: CGPoint(x: left, y: y), size: 9.5, weight: .semibold, color: faint, tracking: 1.2)
                ctx.text(row.1, at: CGPoint(x: left + 118, y: y), size: 10.5, mono: true, color: dim)
            }

            // The Earth's disc: its poles and the Sun's point, and a dimension line under it.
            let c = CGPoint(x: size.width / 2, y: size.height / 2)
            let r = CGFloat(Self.radius) * size.height / 2
            func screen(_ v: SIMD3<Float>) -> (p: CGPoint, front: Bool) {
                let x = simd_dot(v, cam.basis.columns.0), y = simd_dot(v, cam.basis.columns.1)
                return (CGPoint(x: c.x + CGFloat(x) * r, y: c.y - CGFloat(y) * r), simd_dot(v, -cam.basis.columns.2) > 0)
            }
            let pole = screen([0, 0, 1])
            let away = CGPoint(x: pole.p.x - c.x, y: pole.p.y - c.y)
            let norm = max(hypot(away.x, away.y), 1)
            let u = CGPoint(x: away.x / norm, y: away.y / norm)
            ctx.earthLine(CGPoint(x: c.x + u.x * (r + 8), y: c.y + u.y * (r + 8)), CGPoint(x: c.x + u.x * (r + 22), y: c.y + u.y * (r + 22)), color: hair)
            ctx.text("N", at: CGPoint(x: c.x + u.x * (r + 32), y: c.y + u.y * (r + 32)), size: 10, weight: .medium, color: dim,
                     anchor: CGPoint(x: 0.5, y: 0.5))
            if pole.front {
                ctx.setStrokeColor(hair.cgColor)
                ctx.setLineWidth(0.75)
                ctx.strokeEllipse(in: CGRect(x: pole.p.x - 3, y: pole.p.y - 3, width: 6, height: 6))
            }
            let sun = screen(Globe.geo(sub.lat, sub.lon))
            if sun.front {
                let q = sun.p
                ctx.setStrokeColor(NSColor(red: 1, green: 0.85, blue: 0.55, alpha: 0.55).cgColor)
                ctx.setLineWidth(0.75)
                ctx.strokeEllipse(in: CGRect(x: q.x - 5, y: q.y - 5, width: 10, height: 10))
                ctx.fillEllipse(in: CGRect(x: q.x - 0.75, y: q.y - 0.75, width: 1.5, height: 1.5))
                ctx.text("subsolar point", at: CGPoint(x: q.x + 10, y: q.y - 5), size: 9.5, color: NSColor(red: 1, green: 0.88, blue: 0.65, alpha: 0.5))
            }
            let by = c.y + r + 36
            ctx.earthLine(CGPoint(x: c.x - r, y: by), CGPoint(x: c.x + r, y: by), color: hair, width: 0.5)
            for x in [c.x - r, c.x + r] { ctx.earthLine(CGPoint(x: x, y: by - 4), CGPoint(x: x, y: by + 4), color: hair, width: 0.5) }
            ctx.text("12,742 km", at: CGPoint(x: c.x, y: by + 8), size: 10, mono: true, color: faint, anchor: CGPoint(x: 0.5, y: 0))

            // The day's pictures as ticks along the bottom, with the playhead.
            let x0 = 44 + drift.x, x1 = size.width - 44 + drift.x, base = size.height - 40 + drift.y
            ctx.earthLine(CGPoint(x: x0, y: base), CGPoint(x: x1, y: base), color: faint, width: 0.5)
            let n = times.count
            func place(_ i: Float) -> CGFloat { x0 + (x1 - x0) * CGFloat(i / Float(max(n, 1))) }
            for i in 0..<n {
                let near = abs(Float(i) - (Float(seg.a) + seg.t)) < 0.5
                ctx.earthLine(CGPoint(x: place(Float(i) + 0.5), y: base), CGPoint(x: place(Float(i) + 0.5), y: base - 8),
                              color: NSColor(white: 1, alpha: near ? 0.6 : 0.28), width: 1)
            }
            let px = place(Float(seg.a) + 0.5 + seg.t)
            ctx.earthLine(CGPoint(x: px, y: base + 4), CGPoint(x: px, y: base - 22), color: NSColor(white: 1, alpha: 0.5), width: 1)
            if let first = times.first, let last = times.last {
                ctx.text(EarthText.utc(first), at: CGPoint(x: x0, y: base + 6), size: 10, mono: true, color: faint)
                ctx.text(EarthText.utc(last), at: CGPoint(x: x1, y: base + 6), size: 10, mono: true, color: faint, anchor: CGPoint(x: 1, y: 0))
            }
        }
    }

    // MARK: Shader

    private static let source = """
    // Catmull-Rom through nine bilinear taps: sharper than bilinear when the picture is magnified.
    float3 sharp(texture2d<float> t, float2 uv) {
        constexpr sampler s(filter::linear, address::clamp_to_zero);
        float2 size = float2(t.get_width(), t.get_height());
        float2 x = uv * size, tc = floor(x - 0.5) + 0.5, f = x - tc;
        float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
        float2 w12 = w1 + w2;
        float2 t0 = (tc - 1.0) / size, t12 = (tc + w2 / w12) / size, t3 = (tc + 2.0) / size;
        float3 r = 0;
        r += t.sample(s, float2(t0.x, t0.y), level(0)).rgb * w0.x * w0.y;
        r += t.sample(s, float2(t12.x, t0.y), level(0)).rgb * w12.x * w0.y;
        r += t.sample(s, float2(t3.x, t0.y), level(0)).rgb * w3.x * w0.y;
        r += t.sample(s, float2(t0.x, t12.y), level(0)).rgb * w0.x * w12.y;
        r += t.sample(s, float2(t12.x, t12.y), level(0)).rgb * w12.x * w12.y;
        r += t.sample(s, float2(t3.x, t12.y), level(0)).rgb * w3.x * w12.y;
        r += t.sample(s, float2(t0.x, t3.y), level(0)).rgb * w0.x * w3.y;
        r += t.sample(s, float2(t12.x, t3.y), level(0)).rgb * w12.x * w3.y;
        r += t.sample(s, float2(t3.x, t3.y), level(0)).rgb * w3.x * w3.y;
        return max(r, 0.0);
    }

    // u.a: Earth radius on screen (centered units), stars, blend toward the second picture
    // buffer 2: for each picture, the right and up axes and the direction toward the camera, Earth-fixed; w of the last
    // is the disc's radius as a share of the picture's width.
    fragment float4 epicFill(V in [[stage_in]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                             constant float4 *shots [[buffer(2)]], texture2d<float> pa [[texture(0)]], texture2d<float> pb [[texture(1)]],
                             texture2d<float> sky [[texture(2)]]) {
        float2 p = centered(in.uv, u.res) / u.a.x;
        float r = length(p);
        float z = sqrt(max(1 - r * r, 0.0));
        float3 right = c.basis[0], up = c.basis[1], toward = -c.basis[2];
        float3 n = p.x * right + p.y * up + z * toward;
        float3 unit = r > 1 ? n / r : n;

        float3 photo = 0;
        float total = 0;
        for (int k = 0; k < 2; k++) {
            float3 fr = shots[k * 3].xyz, fu = shots[k * 3 + 1].xyz, ft = shots[k * 3 + 2].xyz;
            float share = shots[k * 3 + 2].w;
            float2 q = float2(dot(n, fr), dot(n, fu)) * share;
            float facing = dot(unit, ft);
            float w = (k == 0 ? 1 - u.a.z : u.a.z) * smoothstep(-0.02, 0.5, facing) + 1e-5;
            photo += w * sharp(k == 0 ? pa : pb, float2(0.5 + q.x, 0.5 - q.y));
            total += w;
        }
        photo /= total;
        photo *= smoothstep(1.07, 1.0, r);

        float3 rd = globeRay(in.uv, u.res, c);
        float spread = 2 / (u.res.y * c.zoom);
        float3 stars = u.a.y > 0 ? milkyWay(sky, globeSky(rd, c), spread) * u.a.y : float3(0);
        float cover = saturate(dot(photo, float3(0.33)) * 5);
        return float4(photo + stars * (1 - cover), 1);
    }
    """
}
