import AppKit
import Metal
import simd

/// The whole Earth as it looks right now from the geostationary weather satellite nearest the Mac, from CIRA's SLIDER
/// GeoColor imagery: true colour by day, city lights and infrared clouds by night. The last few hours play as a slow
/// time-lapse so clouds stream and the line between day and night sweeps across, while the view drifts in toward
/// the most striking weather.
@MainActor final class LiveEarth: Scene {
    struct Satellite: Sendable {
        let slug: String
        let name: String
        let longitude: Double
        /// SLIDER zoom level to fetch: 2^zoom tiles a side (678 to 688 pixels each for GOES and Himawari, 464 for Meteosat).
        let zoom: Int
        /// The disc's radius as a fraction of the image width.
        let disc: Double
    }

    nonisolated static let satellites = [
        Satellite(slug: "goes-19", name: "GOES-19", longitude: -75.2, zoom: 1, disc: 0.5),
        Satellite(slug: "goes-18", name: "GOES-18", longitude: -137.2, zoom: 1, disc: 0.5),
        Satellite(slug: "himawari", name: "Himawari-9", longitude: 140.7, zoom: 1, disc: 0.4935),
        Satellite(slug: "meteosat-0deg", name: "Meteosat", longitude: 0, zoom: 2, disc: 0.486),
        Satellite(slug: "meteosat-9", name: "Meteosat-9", longitude: 45.5, zoom: 2, disc: 0.486),
    ]

    /// Frames in the loop, and the minutes between them.
    nonisolated static let frames = 8
    nonisolated static let spacing = 30
    /// Size of each composed frame in pixels.
    nonisolated static let size = 1440

    private struct Shot {
        let stamp: Int64
        let time: Date
        let luma: MTLTexture
        let chroma: MTLTexture
        let brightness: Float
        /// Bright, structured cloud worth drifting toward, in disc units (x east, y north).
        let targets: [SIMD2<Double>]
    }

    private struct Times: Decodable { let timestamps_int: [Int64] }

    private let gpu: GPU
    private let view: MTLRenderPipelineState
    private let convert: MTLComputePipelineState
    private let blank: MTLTexture
    private var sat: Satellite
    private let pinned: Bool
    private var shots: [Shot] = []
    private var fetched = Date.distantPast
    private var attempted = false
    private var busy = false
    private var shown = -1
    private var shownNewest = false
    private var lastLoop = -1
    private var focus = 0

    var scale: Float { 1 }
    var ready: Bool { attempted && !busy }

    var facts: String? {
        guard let newest = shots.last else {
            return attempted ? "Offline, and no \(sat.name) images are saved yet, so the Earth is dark for now."
                             : "Fetching the latest \(sat.name) images of the whole Earth."
        }
        let span = newest.time.timeIntervalSince(shots[0].time) / 3600
        var s = "\(sat.name) GeoColor, latest image \(SkyKit.utc(newest.time)) UTC"
        if shots.count > 1 { s += ", looping the last \(Self.hours(span))" }
        s += ". " + Self.describe(newest.time, satellite: sat.longitude).capitalizedFirst + "."
        if !newest.targets.isEmpty, let place = Self.place(newest.targets[focus % newest.targets.count], satellite: sat) {
            s += " The view drifts toward the clouds over \(place)."
        }
        return s
    }

    init(gpu: GPU) throws {
        self.gpu = gpu
        let library = try gpu.library(SkyKit.metal + Self.source)
        view = try gpu.pipeline(library, "earthView")
        convert = try gpu.kernel(library, "earthToYUV")
        var zero: UInt16 = 0x8080
        blank = gpu.texture(1, 1, .rg8Unorm, bytes: &zero, bytesPerRow: 2)
        Location.shared.request()
        if let v = Snapshot.variant {
            sat = Self.satellites[v % Self.satellites.count]
            pinned = true
        } else {
            sat = Self.nearest(Location.shared.longitude)
            pinned = false
        }
    }

    /// The satellite whose sub-satellite point is closest in longitude.
    nonisolated static func nearest(_ longitude: Double) -> Satellite {
        satellites.min { gap($0.longitude, longitude) < gap($1.longitude, longitude) }!
    }

    private nonisolated static func gap(_ a: Double, _ b: Double) -> Double {
        let d = abs(a - b).truncatingRemainder(dividingBy: 360)
        return min(d, 360 - d)
    }

    nonisolated static func date(_ stamp: Int64) -> Date? {
        let s = String(stamp)
        guard s.count == 14 else { return nil }
        let v = [0, 4, 6, 8, 10, 12].map { Int(s.dropFirst($0).prefix($0 == 0 ? 4 : 2)) ?? -1 }
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c.date(from: DateComponents(year: v[0], month: v[1], day: v[2], hour: v[3], minute: v[4], second: v[5]))
    }

    /// The frames to show: the newest ones on a half-hour grid, so most survive from one refresh to the next.
    nonisolated static func pick(_ stamps: [Int64], count: Int = frames) -> [Int64] {
        var out: [Int64] = []
        for stamp in stamps.sorted(by: >) {
            guard let d = date(stamp) else { continue }
            let minute = Int(d.timeIntervalSince1970 / 60) % spacing
            guard minute == 0, !out.contains(where: { abs((date($0) ?? d).timeIntervalSince(d)) < 600 }) else { continue }
            out.append(stamp)
            if out.count == count { break }
        }
        return out
    }

    nonisolated static func tiles(_ stamp: Int64, _ sat: Satellite) -> [URL] {
        let s = String(stamp)
        let day = "\(s.prefix(4))/\(s.dropFirst(4).prefix(2))/\(s.dropFirst(6).prefix(2))"
        let n = 1 << sat.zoom
        return (0..<n * n).map { i in
            let path = String(format: "%@/%@---full_disk/geocolor/%@/%02d/%03d_%03d.png", day, sat.slug, s, sat.zoom, i / n, i % n)
            return URL(string: "https://slider.cira.colostate.edu/data/imagery/" + path)!
        }
    }

    private static func hours(_ h: Double) -> String {
        let half = (h * 2).rounded() / 2
        if half <= 1 { return "hour" }
        return half == half.rounded() ? "\(Int(half)) hours" : "\(Int(half)) and a half hours"
    }

    // MARK: Loading

    private func refresh() {
        fetched = .now
        if !pinned, Self.nearest(Location.shared.longitude).slug != sat.slug {
            sat = Self.nearest(Location.shared.longitude)
            shots = []
        }
        guard !busy else { return }
        busy = true
        let sat = self.sat
        Task { @MainActor [weak self] in
            let url = URL(string: "https://slider.cira.colostate.edu/data/json/\(sat.slug)/full_disk/geocolor/latest_times.json")!
            let list = await Feed.json(Times.self, url, maxAge: 600)
            let want = Self.pick(list?.timestamps_int ?? [])
            if let self, sat.slug == self.sat.slug, !want.isEmpty {
                shots.removeAll { !want.contains($0.stamp) }
                Self.prune(keeping: want.flatMap { Self.tiles($0, sat) })
            }
            // Newest first, so the present shows up as soon as possible. Holding the scene only while a frame loads
            // lets it go promptly when the screensaver ends.
            for stamp in want {
                guard let scene = self, sat.slug == scene.sat.slug else { break }
                guard !scene.shots.contains(where: { $0.stamp == stamp }), let shot = await scene.load(stamp, sat),
                      sat.slug == scene.sat.slug else { continue }
                scene.shots.append(shot)
                scene.shots.sort { $0.stamp < $1.stamp }
                scene.attempted = true
            }
            self?.busy = false
            self?.attempted = true
        }
    }

    private func load(_ stamp: Int64, _ sat: Satellite) async -> Shot? {
        guard let time = Self.date(stamp) else { return nil }
        let urls = Self.tiles(stamp, sat)
        let data = await withTaskGroup(of: (Int, Data?).self) { group in
            for (i, url) in urls.enumerated() {
                // Published tiles never change, so a week-old copy is as good as new.
                group.addTask { (i, await Feed.data(url, maxAge: 7 * 86400)) }
            }
            var out = [Data?](repeating: nil, count: urls.count)
            for await (i, d) in group { out[i] = d }
            return out
        }
        let tiles = data.compactMap { $0 }
        guard tiles.count == urls.count else { return nil }
        let disc = sat.disc
        guard let (pixels, grid) = await Task.detached(priority: .utility, operation: { Self.compose(tiles, size: Self.size) }).value
        else { return nil }
        let analysis = Self.analyse(grid, disc: disc)
        guard let (luma, chroma) = upload(pixels) else { return nil }
        return Shot(stamp: stamp, time: time, luma: luma, chroma: chroma, brightness: analysis.brightness, targets: analysis.targets)
    }

    /// Lays the tiles out into one square image, plus a 64 × 64 thumbnail for finding the weather.
    nonisolated static func compose(_ tiles: [Data], size: Int) -> (SkyPixels, [UInt8])? {
        let n = Int(Double(tiles.count).squareRoot())
        var bytes = [UInt8](repeating: 0, count: size * size * 4)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let ok = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: size, height: size, bitsPerComponent: 8, bytesPerRow: size * 4,
                                      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.interpolationQuality = .high
            let step = Double(size) / Double(n)
            for (i, data) in tiles.enumerated() {
                guard let src = CGImageSourceCreateWithData(data as CFData, nil),
                      let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return false }
                let row = i / n, col = i % n
                ctx.draw(image, in: CGRect(x: Double(col) * step, y: Double(n - 1 - row) * step, width: step, height: step))
            }
            return true
        }
        guard ok else { return nil }
        let pixels = SkyPixels(width: size, height: size, bytes: bytes)
        var grid = [UInt8](repeating: 0, count: 64 * 64 * 4)
        let small = pixels.bytes.withUnsafeBytes { raw -> Bool in
            guard let provider = CGDataProvider(data: Data(raw) as CFData),
                  let image = CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4, space: space,
                                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider,
                                      decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return false }
            return grid.withUnsafeMutableBytes { g -> Bool in
                guard let ctx = CGContext(data: g.baseAddress, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 256, space: space,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
                ctx.interpolationQuality = .medium
                ctx.draw(image, in: CGRect(x: 0, y: 0, width: 64, height: 64))
                return true
            }
        }
        return small ? (pixels, grid) : (pixels, [])
    }

    /// Finds bright, textured, colourless patches (cloud rather than desert) away from the limb, and the average brightness.
    nonisolated static func analyse(_ grid: [UInt8], disc: Double) -> (brightness: Float, targets: [SIMD2<Double>]) {
        guard grid.count == 64 * 64 * 4 else { return (0.3, []) }
        var white = [Double](repeating: 0, count: 64 * 64)
        var sum = 0.0, count = 0.0
        for i in 0..<64 * 64 {
            let r = Double(grid[i * 4]) / 255, g = Double(grid[i * 4 + 1]) / 255, b = Double(grid[i * 4 + 2]) / 255
            let mx = max(r, g, b), mn = min(r, g, b)
            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            white[i] = lum * max(0, 1 - 2.5 * (mx - mn))
            sum += lum
            count += 1
        }
        func position(_ x: Int, _ y: Int) -> SIMD2<Double> {
            let u = (Double(x) + 0.5) / 64 - 0.5, v = 0.5 - (Double(y) + 0.5) / 64
            return SIMD2(u / disc, v / disc)
        }
        var scored: [(Double, SIMD2<Double>)] = []
        for y in 3..<61 {
            for x in 3..<61 {
                let d = position(x, y)
                guard simd_length(d) < 0.62 else { continue }
                var m = 0.0, m2 = 0.0
                for j in -3...3 { for i in -3...3 { let v = white[(y + j) * 64 + x + i]; m += v; m2 += v * v } }
                m /= 49
                let sd = max(m2 / 49 - m * m, 0).squareRoot()
                scored.append((m * sd * (1 - 0.4 * simd_length(d)), d))
            }
        }
        var targets: [SIMD2<Double>] = []
        for (_, d) in scored.sorted(by: { $0.0 > $1.0 }) where targets.allSatisfy({ simd_length($0 - d) > 0.4 }) {
            targets.append(d)
            if targets.count == 3 { break }
        }
        return (Float(sum / max(count, 1)), targets)
    }

    /// Converts a frame to luma and half-size chroma textures with mipmaps: a third of the memory of RGBA, so a long
    /// loop of full-resolution frames fits comfortably.
    private func upload(_ p: SkyPixels) -> (MTLTexture, MTLTexture)? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: p.width, height: p.height, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = gpu.device.hasUnifiedMemory ? .shared : .managed
        guard let source = gpu.device.makeTexture(descriptor: d) else { return nil }
        p.bytes.withUnsafeBytes {
            source.replace(region: MTLRegionMake2D(0, 0, p.width, p.height), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: p.width * 4)
        }
        func plane(_ format: MTLPixelFormat, _ w: Int, _ h: Int) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: true)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return gpu.device.makeTexture(descriptor: d)
        }
        guard let luma = plane(.r8Unorm, p.width, p.height), let chroma = plane(.rg8Unorm, p.width / 2, p.height / 2),
              let cmd = gpu.queue.makeCommandBuffer() else { return nil }
        cmd.compute(convert, p.width / 2, p.height / 2, uniforms: U(), textures: [source, luma, chroma])
        if let blit = cmd.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: luma)
            blit.generateMipmaps(for: chroma)
            blit.endEncoding()
        }
        cmd.commit()
        return (luma, chroma)
    }

    /// Feed keeps everything it downloads; tiles that fall out of the loop are deleted here so the cache stays small.
    private static func prune(keeping urls: [URL]) {
        let key = "LiveEarth.tiles"
        let keep = Set(urls.map(\.absoluteString))
        let old = UserDefaults.standard.stringArray(forKey: key) ?? []
        for s in old where !keep.contains(s) {
            guard let url = URL(string: s) else { continue }
            var hash: UInt64 = 0xcbf2_9ce4_8422_2325
            for byte in s.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100_0000_01b3 }
            let name = String(hash, radix: 16) + "." + (url.pathExtension.isEmpty ? "bin" : url.pathExtension)
            try? FileManager.default.removeItem(at: Feed.folder.appendingPathComponent(name))
        }
        UserDefaults.standard.set(Array(keep), forKey: key)
    }

    // MARK: Geography

    /// Latitude and longitude seen at a point of the disc (x east, y north, in disc radii), or nil off the planet.
    /// The geostationary projection from the GOES-R product guide, on the GRS80 ellipsoid.
    nonisolated static func geo(_ p: SIMD2<Double>, satellite lon0: Double) -> (lat: Double, lon: Double)? {
        let H = 42164.16, req = 6378.137, rpol = 6356.752
        let k = asin(req / H)
        let x = p.x * k, y = p.y * k
        let e = req * req / (rpol * rpol)
        let a = sin(x) * sin(x) + cos(x) * cos(x) * (cos(y) * cos(y) + e * sin(y) * sin(y))
        let b = -2 * H * cos(x) * cos(y), c = H * H - req * req
        let disc = b * b - 4 * a * c
        guard disc >= 0 else { return nil }
        let rs = (-b - disc.squareRoot()) / (2 * a)
        let sx = rs * cos(x) * cos(y), sy = -rs * sin(x), sz = rs * cos(x) * sin(y)
        let lat = atan(e * sz / ((H - sx) * (H - sx) + sy * sy).squareRoot()) * 180 / .pi
        var lon = lon0 - atan(sy / (H - sx)) * 180 / .pi
        if lon > 180 { lon -= 360 } else if lon < -180 { lon += 360 }
        return (lat, lon)
    }

    private static func place(_ p: SIMD2<Double>, satellite: Satellite) -> String? {
        guard let g = geo(p, satellite: satellite.longitude) else { return nil }
        return SkyPlaces.nearest(g.lat, g.lon)?.name
    }

    /// Where it's sunrise, sunset, night and midday on the visible half of the Earth.
    static func describe(_ date: Date, satellite lon0: Double) -> String {
        let sun = Astro.subpoint(Astro.sun(date), date)
        let visible = SkyPlaces.all.filter { SkyPlaces.angle($0.lat, $0.lon, 0, lon0) < 68 }
        func altitude(_ p: SkyPlaces.Place) -> Double { 90 - SkyPlaces.angle(p.lat, p.lon, sun.lat, sun.lon) }
        func morning(_ p: SkyPlaces.Place) -> Bool {
            var d = (p.lon - sun.lon).truncatingRemainder(dividingBy: 360)
            if d > 180 { d -= 360 } else if d < -180 { d += 360 }
            return d < 0
        }
        var parts: [String] = []
        let edge = visible.filter { abs(altitude($0)) < 6 && !$0.big }
        if let p = edge.filter(morning).min(by: { abs(altitude($0)) < abs(altitude($1)) }) { parts.append("sunrise over \(p.name)") }
        if let p = edge.filter({ !morning($0) }).min(by: { abs(altitude($0)) < abs(altitude($1)) }) { parts.append("sunset over \(p.name)") }
        if let p = visible.filter({ $0.big && altitude($0) < -20 }).min(by: { altitude($0) < altitude($1) }) {
            parts.append("night across \(p.name)")
        }
        if let p = visible.filter({ altitude($0) > 55 }).max(by: { altitude($0) < altitude($1) }) { parts.append("midday over \(p.name)") }
        if parts.isEmpty { parts.append(SkyPlaces.angle(0, lon0, sun.lat, sun.lon) < 90 ? "the whole disc in daylight" : "the whole disc in night") }
        return parts.joined(separator: ", ")
    }

    // MARK: Drawing

    func draw(_ f: Frame) {
        if f.now.timeIntervalSince(fetched) > 600 { refresh() }
        let n = shots.count
        // The loop: 3 seconds a frame, a 7-second hold on the newest, then a slow dissolve back to the oldest.
        let step: Float = 3, hold: Float = 7, back: Float = 4
        let cycle = Float(max(n - 1, 0)) * step + hold + back
        let phase = n > 1 ? f.time.truncatingRemainder(dividingBy: cycle) : 0
        var a = max(n - 1, 0), b = a, w: Float = 0
        if n > 1 {
            if phase < Float(n - 1) * step {
                let x = phase / step
                a = Int(x)
                b = a + 1
                let t = x - Float(a)
                w = t * 0.4 + 0.6 * t * t * (3 - 2 * t)
            } else if phase > Float(n - 1) * step + hold {
                a = n - 1
                b = 0
                let t = (phase - Float(n - 1) * step - hold) / back
                w = t * t * (3 - 2 * t)
            }
        }
        let loop = n > 1 ? Int(f.time / cycle) : 0

        // Every two and a half minutes the view eases in toward a patch of weather, lingers, and eases back out.
        let t = Double(f.time) + Double(f.seed) * 13
        let period = 150.0
        let k = (t / period).truncatingRemainder(dividingBy: 1) * period
        func ease(_ x: Double) -> Double { let c = min(max(x, 0), 1); return c * c * (3 - 2 * c) }
        let zoom = ease((k - 45) / 22) * (1 - ease((k - 118) / 24))
        let targets = shots.last?.targets ?? []
        focus = Int(t / period)
        let target = targets.isEmpty ? SIMD2<Double>(0, 0) : targets[focus % targets.count]
        let radius = 0.86 * (1 + 0.5 * zoom)
        let wander = SIMD2(0.05 * sin(t / 83), 0.03 * sin(t / 61 + 1))
        let focus = target * zoom * 0.75
        let center = -focus * radius + wander * (1 - zoom * 0.6)

        // The Sun's direction in the satellite's view (x east, y north, z toward the satellite), for the glow on the limb.
        let when = n > 0 ? shots[w < 0.5 ? a : b].time : f.now
        let sun = Astro.subpoint(Astro.sun(when), when)
        let (lat, dl) = (sun.lat * .pi / 180, (sun.lon - sat.longitude) * .pi / 180)
        let s = SIMD3<Float>(Float(cos(lat) * sin(dl)), Float(sin(lat)), Float(cos(lat) * cos(dl)))

        let u = f.uniforms([w, Float(radius), Float(center.x), Float(center.y)],
                           [Float(sat.disc), n == 0 ? 1 : 0, Float(Self.size), 0],
                           SIMD4(s, 0))
        let A = n > 0 ? shots[a] : nil, B = n > 0 ? shots[b] : nil
        f.cmd.draw(view, into: f.target, uniforms: u,
                   textures: [A?.luma ?? blank, A?.chroma ?? blank, B?.luma ?? blank, B?.chroma ?? blank])

        let current = w < 0.5 ? a : b
        if current != shown || n == 0 && shown != -2 {
            shown = n == 0 ? -2 : current
            drawOverlay(f, current)
        }
        if let synth = f.sound {
            play(synth, f, n > 0 ? mix(shots[a].brightness, shots[b].brightness, w) : 0.2, s)
            if loop != lastLoop {
                lastLoop = loop
                shownNewest = false
            }
            if n > 1, current == n - 1, !shownNewest {
                shownNewest = true
                synth.play(.bell, 220, amp: 0.05, pan: -0.2, length: 5)
                synth.play(.bell, 330, amp: 0.03, pan: 0.2, length: 5)
            }
        }
    }

    private func mix(_ a: Float, _ b: Float, _ w: Float) -> Float { a + (b - a) * w }

    /// A deep oceanic pad on A: the swell of the surf follows a slow wave, brighter partials come in as more of the
    /// disc is in daylight, and cloudier frames open the filter a little.
    private func play(_ synth: Synth, _ f: Frame, _ brightness: Float, _ sun: SIMD3<Float>) {
        let day = max(0, min(1, 0.5 + sun.z * 0.6))
        let swell = 0.5 + 0.5 * sin(f.time * 2 * .pi / 11)
        let slow = 0.5 + 0.5 * sin(f.time * 2 * .pi / 37 + 1)
        synth.drone(0, 55, 0.075, pan: -0.15)
        synth.drone(1, 82.41, 0.05 * (0.8 + 0.2 * slow), pan: 0.2)
        synth.drone(2, 110, 0.04)
        synth.drone(3, 164.81, 0.022 * (0.6 + 0.4 * swell), pan: -0.3)
        synth.drone(4, 220.0 * 1.0015, 0.008 + 0.01 * day, pan: 0.35)
        synth.drone(5, 277.18, 0.012 * day * slow, pan: -0.4)
        synth.drone(6, 329.63, 0.006 + 0.006 * (1 - day), pan: 0.4)
        synth.noise(0.02 + 0.035 * swell * swell, cutoff: 220 + 260 * swell + 400 * brightness)
    }

    private func drawOverlay(_ f: Frame, _ current: Int) {
        let sat = self.sat
        let times = shots.map(\.time)
        let seed = Double(f.seed)
        let drift = SkyKit.drift(Double(f.time) + seed * 40, amount: 12, seed: seed)
        f.overlay.draw { ctx, size in
            let left = 54 + drift.x, bottom = size.height - 54 + drift.y
            let shadow = NSShadow()
            shadow.shadowColor = NSColor(white: 0, alpha: 0.7)
            shadow.shadowBlurRadius = 8
            shadow.set()
            let lon = sat.longitude == 0 ? "0°" : String(format: "%.1f° %@", abs(sat.longitude), sat.longitude < 0 ? "W" : "E")
            ctx.text("\(sat.name.uppercased())  ·  GEOCOLOR  ·  \(lon)", at: CGPoint(x: left, y: bottom - 74), size: 10.5,
                     weight: .medium, color: NSColor(white: 1, alpha: 0.5), tracking: 2)
            guard current >= 0, current < times.count else {
                ctx.text("Waiting for the satellite", at: CGPoint(x: left, y: bottom - 56), size: 22, weight: .light,
                         color: NSColor(white: 1, alpha: 0.6))
                return
            }
            let time = times[current]
            let clock = NSAttributedString(string: "\(SkyKit.utc(time)) UTC", attributes: [
                .font: NSFont.monospacedDigitSystemFont(ofSize: 26, weight: .light), .foregroundColor: NSColor(white: 1, alpha: 0.88)])
            clock.draw(at: CGPoint(x: left - 1, y: bottom - 58))
            let ago = Int((times.last!.timeIntervalSince(time) / 60).rounded())
            let label = current == times.count - 1 ? "latest"
                : ago >= 60 ? "\(ago / 60) h\(ago % 60 == 0 ? "" : " \(ago % 60) min") earlier" : "\(ago) min earlier"
            ctx.text(label, at: CGPoint(x: left + clock.size().width + 10, y: bottom - 49), size: 12.5, color: NSColor(white: 1, alpha: 0.55))
            // A hairline timeline with a tick per frame; the one showing is brighter.
            let width: CGFloat = 220
            let y = bottom - 10
            ctx.setStrokeColor(NSColor(white: 1, alpha: 0.22).cgColor)
            ctx.setLineWidth(0.5)
            ctx.move(to: CGPoint(x: left, y: y))
            ctx.addLine(to: CGPoint(x: left + width, y: y))
            ctx.strokePath()
            guard let first = times.first, let last = times.last else { return }
            let span = max(last.timeIntervalSince(first), 1)
            for (i, t) in times.enumerated() {
                let x = left + width * CGFloat(t.timeIntervalSince(first) / span)
                let on = i == current
                ctx.setStrokeColor(NSColor(white: 1, alpha: on ? 0.95 : 0.35).cgColor)
                ctx.setLineWidth(on ? 1.5 : 0.75)
                ctx.move(to: CGPoint(x: x, y: y - (on ? 7 : 4)))
                ctx.addLine(to: CGPoint(x: x, y: y))
                ctx.strokePath()
            }
        }
    }

    private static let source = """
    // a: frame blend, disc radius on screen, disc centre   b: disc radius in the image, no frames, image size   c: Sun direction
    float3 earthRGB(texture2d<float> Y, texture2d<float> C, float2 uv, float texel) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_zero);
        float y = picture(Y, uv, texel).r;
        float2 c = C.sample(s, uv).rg - 0.5;
        float3 rgb = float3(y + 1.5748 * c.y, y - 0.1873 * c.x - 0.4681 * c.y, y + 1.8556 * c.x);
        // GeoColor is a little flat; lift saturation slightly.
        rgb = saturate(mix(float3(y), rgb, 1.14));
        return fromSRGB(rgb);
    }

    fragment float4 earthView(V in [[stage_in]], constant U &u [[buffer(0)]],
                              texture2d<float> YA [[texture(0)]], texture2d<float> CA [[texture(1)]],
                              texture2d<float> YB [[texture(2)]], texture2d<float> CB [[texture(3)]]) {
        float2 p = centered(in.uv, u.res);
        float R = u.a.y;
        float2 d = (p - u.a.zw) / R;
        float r = length(d);
        float px = R * u.res.y * 0.5;
        float3 sun = u.c.xyz;

        // Space: a deep blue-black with faint stars.
        float3 col = float3(0.0015, 0.002, 0.004) * (1.2 - 0.5 * length(p));
        col += starfield(in.uv * u.res + u.seed * 97, 7, 0.07, 1.0, u.time) * 0.18;

        // The planet.
        if (r < 1.01 && u.b.y < 0.5) {
            float2 iuv = 0.5 + d * u.b.x * float2(1, -1);
            float texel = px / (u.b.x * u.b.z);
            float3 e = mix(earthRGB(YA, CA, iuv, texel), earthRGB(YB, CB, iuv, texel), u.a.x);
            float m = 1 - smoothstep(1 - 1.5 / px, 1 + 0.5 / px, r);
            col = mix(col, e, m);
        } else if (r < 1.0) {
            // No pictures yet: a dark globe, lit only by its rim.
            col = float3(0.004, 0.006, 0.012);
        }

        // The atmosphere: a thin bright shell on the day side, deep blue on the night side, warm along the terminator,
        // with forward scattering when the Sun is behind the planet.
        float2 n = r > 0 ? d / r : float2(0, 1);
        float lit = dot(float3(n, 0), sun);
        float day = smoothstep(-0.35, 0.3, lit);
        float dusk = exp(-lit * lit * 18) * 0.6;
        float behind = pow(saturate(dot(n, normalize(sun.xy + 1e-4))) * saturate(-sun.z), 3.0);
        float3 air = mix(float3(0.03, 0.06, 0.16), float3(0.35, 0.62, 1.0), day) + float3(1.0, 0.45, 0.15) * dusk * 0.6;
        air += float3(1.0, 0.75, 0.5) * behind * 0.8;
        float beyond = max(r - 1, 0.0);
        float shell = (exp(-beyond * 45) * 0.55 + exp(-beyond * 9) * 0.07) * smoothstep(0.985, 1.0, r);
        float rim = exp(-(1 - min(r, 1.0)) * 70) * 0.18 * (r <= 1 ? 1 : 0);
        col += air * (shell + rim) * (0.35 + 0.65 * day + behind);

        // Shoulder, vignette (a little deeper behind the caption), out to display.
        col = select(col, 0.85 + 0.15 * tanh((col - 0.85) / 0.15), col > 0.85);
        float2 v = in.uv - 0.5;
        col *= 1 - 0.3 * dot(v, v);
        col *= 1 - 0.45 * exp(-length((in.uv - float2(0.0, 1.0)) * float2(u.res.x / u.res.y, 1) * float2(2.2, 3.2)));
        return float4(toSRGB(max(col, 0.0)), 1);
    }

    // One thread per 2 × 2 block: full-resolution luma, averaged chroma (BT.709, on the sRGB-encoded values).
    kernel void earthToYUV(constant U &u [[buffer(0)]], texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> Y [[texture(1)]], texture2d<float, access::write> C [[texture(2)]],
                           uint2 g [[thread_position_in_grid]]) {
        float2 c = 0;
        for (uint j = 0; j < 2; j++) for (uint i = 0; i < 2; i++) {
            uint2 q = g * 2 + uint2(i, j);
            float3 rgb = src.read(q).rgb;
            float y = dot(rgb, float3(0.2126, 0.7152, 0.0722));
            Y.write(float4(y), q);
            c += float2((rgb.b - y) / 1.8556, (rgb.r - y) / 1.5748);
        }
        C.write(float4(c * 0.25 + 0.5, 0, 0), g);
    }
    """
}

private extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
