import AppKit
import ImageIO
import Metal
import simd

/// The Earth shared by the Live Earth scenes, made from real imagery: NASA's Blue Marble Next Generation for the
/// current month (land, sea ice, snow), yesterday's clouds cut out of the VIIRS satellites' global mosaic, NASA's
/// Black Marble city lights, and ESO's Milky Way behind. `draw` ray-traces it lit by the real Sun through a physically
/// based atmosphere (Rayleigh and Mie scattering, ozone), with clouds 6 km up casting soft shadows, sunglint on the
/// sea and airglow over the night-side limb. Natural Earth's outlines serve as the water mask, and as the look until
/// the photographs arrive.
///
/// Coordinates are Earth-fixed, in Earth radii: x toward 0° 0°, y toward 0° 90° E, z toward the north pole.
/// Scenes compile `Globe.metal` in front of their own source to project points the same way the globe does.
@MainActor final class Globe {
    struct Style {
        var lights: Float = 1
        var atmosphere: Float = 1
        /// The Milky Way behind the Earth. Photographs exposed for the day side barely show it.
        var stars: Float = 0.15
        /// The green airglow line above the night-side limb.
        var airglow: Float = 1
        var glint: Float = 1
        var clouds: Float = 1
        /// Moonlight on the night side as a share of sunlight, at full Moon. Real moonlight is about 1/400,000 of
        /// sunlight; this exaggerates it the way a long exposure of the night side would.
        var moonlight: Float = 0.01
        /// The night sky's own faint light on the night side (starlight and airglow), as a share of sunlight.
        var nightGlow: Float = 0.004
    }

    static let credit = "Earth: NASA Blue Marble, Black Marble and GIBS/VIIRS · Milky Way: ESO/S. Brunier"
    static let maskWidth = 4096, maskHeight = 2048
    private static let lightsURL = URL(string: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_01deg.jpg")!
    /// NASA's 2002 cloud composite: the clouds when yesterday's can't be had, and over the polar night where VIIRS sees nothing.
    private static let oldCloudsURL = URL(string: "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_2048.jpg")!
    /// Blue Marble Next Generation (topography and bathymetry), one record per month of 2004.
    private static let months = [73580, 73605, 73630, 73655, 73701, 73726, 73751, 73776, 73801, 73826, 73884, 73909]
    /// Shared while any Earth scene holds it, so the textures go when the screensaver ends.
    private static weak var instance: Globe?

    /// Land coverage, 0 (sea) to 1 (land), with mipmaps.
    let land: MTLTexture
    private(set) var surface: MTLTexture?
    private(set) var clouds: MTLTexture?
    private(set) var lights: MTLTexture?
    private(set) var sky: MTLTexture?
    /// The day of the VIIRS passes the clouds come from, or nil for NASA's old cloud map.
    private(set) var cloudDay: Date?
    private var settled = 0
    private var cloudsChecked = Date.distantPast
    private var cloudsLoading = false
    private let blank: MTLTexture
    private let gpu: GPU
    private let pass: MTLRenderPipelineState
    private let opacity: MTLComputePipelineState
    private let finishClouds: MTLComputePipelineState

    /// True once every photograph has loaded or failed.
    var ready: Bool { settled >= 4 }

    static func shared(_ gpu: GPU) throws -> Globe {
        if let instance { return instance }
        let globe = try Globe(gpu: gpu)
        instance = globe
        return globe
    }

    private init(gpu: GPU) throws {
        self.gpu = gpu
        let library = try gpu.library(Self.metal + Self.passes)
        pass = try gpu.pipeline(library, "globeSurface")
        opacity = try gpu.kernel(library, "cloudOpacity")
        finishClouds = try gpu.kernel(library, "cloudFinish")
        let w = Self.maskWidth, h = Self.maskHeight
        let pixels = Self.rasterise(w, h)
        land = pixels.withUnsafeBytes { Self.mipmapped(gpu, w, h, $0.baseAddress!) }
        var zero: UInt8 = 0
        blank = gpu.texture(1, 1, .r8Unorm, bytes: &zero, bytesPerRow: 1)
        let year = 365 * 86400.0
        Task { @MainActor in
            surface = await Imagery.texture(Self.blueMarble(.now), gpu, maxSize: 5400, maxAge: year)
            settled += 1
        }
        Task { @MainActor in
            lights = await Imagery.texture(Self.lightsURL, gpu, kind: .gray, maxSize: 3600, maxAge: year)
            settled += 1
        }
        Task { @MainActor in
            sky = await MilkyWay.texture(gpu)
            settled += 1
        }
        Task { @MainActor in
            await loadClouds(.now)
            settled += 1
        }
    }

    static func blueMarble(_ date: Date) -> URL {
        let month = Calendar(identifier: .gregorian).component(.month, from: date)
        let id = months[(month + 11) % 12]
        return URL(string: String(format: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/%d/world.topo.bathy.2004%02d.3x5400x2700.jpg", id, month))!
    }

    /// A day's global mosaic of VIIRS true colour from Suomi NPP, NOAA-20 and NOAA-21, from NASA GIBS.
    static func viirs(_ day: Date) -> URL {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        let d = c.dateComponents([.year, .month, .day], from: day)
        let layers = ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"].map { $0 + "_CorrectedReflectance_TrueColor" }.joined(separator: ",")
        let date = String(format: "%04d-%02d-%02d", d.year ?? 2026, d.month ?? 1, d.day ?? 1)
        return URL(string: "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
            + "&LAYERS=\(layers)&STYLES=&CRS=EPSG:4326&BBOX=-90,-180,90,180&WIDTH=4096&HEIGHT=2048&FORMAT=image/jpeg&TIME=\(date)")!
    }

    /// Clouds from the most recent complete day of VIIRS passes (a day's mosaic fills in a few hours after midnight
    /// UTC), stepping back a day if one comes back mostly empty, else NASA's old cloud map.
    private func loadClouds(_ now: Date) async {
        cloudsLoading = true
        cloudsChecked = now
        let old = await Imagery.texture(Self.oldCloudsURL, gpu, kind: .gray, maxSize: 2048, maxAge: 365 * 86400)
        let base = await Imagery.texture(Self.blueMarble(now), gpu, maxSize: 5400, maxAge: 365 * 86400)
        var day = now.addingTimeInterval(-30 * 3600)
        for _ in 0..<3 {
            let url = Self.viirs(day)
            if let data = await Feed.data(url, maxAge: 60 * 86400),
               await Task.detached(priority: .utility, operation: { Self.coverage(data) }).value > 0.6,
               let mosaic = await Imagery.texture(url, gpu, maxSize: 4096, maxAge: 60 * 86400) {
                clouds = extract(mosaic, base: base, old: old, subsolar: Float(Astro.sun(day).dec))
                cloudDay = day
                cloudsLoading = false
                return
            }
            day = day.addingTimeInterval(-86400)
        }
        if clouds == nil, let old { clouds = extract(nil, base: nil, old: old, subsolar: 0) }
        cloudsLoading = false
    }

    /// Fetches newer clouds once a new day's mosaic should be complete. Scenes call this now and then.
    func refresh(_ now: Date) {
        guard !cloudsLoading, now.timeIntervalSince(cloudsChecked) > 3600, let day = cloudDay,
              Self.viirs(now.addingTimeInterval(-30 * 3600)) != Self.viirs(day) else { return }
        Task { @MainActor in await loadClouds(now) }
    }

    /// Share of the mosaic between 60° S and 60° N that has data.
    private nonisolated static func coverage(_ data: Data) -> Double {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                                                          kCGImageSourceThumbnailMaxPixelSize: 256] as CFDictionary)
        else { return 0 }
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h)
        let ok = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return 0 }
        let rows = (h / 6)..<(h * 5 / 6)
        let filled = rows.reduce(0) { sum, y in sum + (0..<w).filter { bytes[y * w + $0] > 6 }.count }
        return Double(filled) / Double(rows.count * w)
    }

    /// Turns a VIIRS mosaic into cloud opacity against the cloud-free Blue Marble, without the sunglint, on the GPU.
    private func extract(_ mosaic: MTLTexture?, base: MTLTexture?, old: MTLTexture?, subsolar: Float) -> MTLTexture? {
        let w = mosaic?.width ?? old?.width ?? 1, h = mosaic?.height ?? old?.height ?? 1
        func make(_ format: MTLPixelFormat) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: true)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return gpu.device.makeTexture(descriptor: d)
        }
        guard let raw = make(.rgba16Float), let out = make(.r8Unorm), let cmd = gpu.queue.makeCommandBuffer() else { return nil }
        // Without a mosaic, every pixel counts as missing and takes the old map.
        cmd.compute(opacity, w, h, uniforms: U(a: [base != nil ? 1 : 0, old != nil ? 1 : 0, 0, 0]),
                    textures: [mosaic ?? blank, base ?? blank, old ?? blank, raw])
        if let blit = cmd.makeBlitCommandEncoder() { blit.generateMipmaps(for: raw); blit.endEncoding() }
        cmd.compute(finishClouds, w, h, uniforms: U(a: [subsolar, 0, 0, 0]), textures: [raw, land, out])
        if let blit = cmd.makeBlitCommandEncoder() { blit.generateMipmaps(for: out); blit.endEncoding() }
        cmd.commit()
        return out
    }

    /// Ray-traces the globe, its atmosphere and the sky around it into `target` as linear HDR light.
    func draw(_ cmd: MTLCommandBuffer, into target: MTLTexture, camera: MTLBuffer, style s: Style, date: Date, time: Float) {
        let moon = Astro.subpoint(Astro.moon(date), date)
        let phase = Float(Astro.moonPhase(date).lit)
        let u = U(res: SIMD2(Float(target.width), Float(target.height)), time: time,
                  a: [s.lights, s.atmosphere, s.stars, s.airglow],
                  b: [s.glint, clouds == nil ? 0 : s.clouds, surface == nil ? 0 : 1, sky == nil ? 0 : 1],
                  c: SIMD4(Self.geo(moon.lat, moon.lon), s.moonlight * phase), d: [time, s.nightGlow, 0, 0])
        cmd.draw(pass, into: target, uniforms: u, textures: [surface ?? blank, clouds ?? blank, lights ?? blank, land, sky ?? blank],
                 buffers: [camera])
    }

    /// Land and sea at a quarter of a degree, for looking up on the CPU.
    private static let coarse = rasterise(1440, 720)

    /// Whether a point is land, from the 1:50m outlines at a quarter of a degree.
    static func isLand(lat: Double, lon: Double) -> Bool {
        let w = 1440, h = 720
        let x = Int(((lon + 180) / 360 * Double(w)).rounded(.down)), y = Int(((90 - lat) / 180 * Double(h)).rounded(.down))
        return coarse[min(max(y, 0), h - 1) * w + (x % w + w) % w] > 127
    }

    /// A rough place name: an ocean or sea ("the South Atlantic") or a landmass ("Africa").
    static func region(lat: Double, lon: Double) -> String {
        if isLand(lat: lat, lon: lon) {
            if lat < -60 { return "Antarctica" }
            if lat > 59, lon > -75, lon < -10 { return "Greenland" }
            if lon < -30 { return lat > 12 || (lat > 7 && lon < -77) ? "North America" : "South America" }
            if lon > 110, lon < 156, lat < -10 { return "Australia" }
            if lon > 165, lat < -33 { return "New Zealand" }
            if lon > -20, lon < 52, lat < 37, lat > -36, !(lon > 34 && lat > 30) && !(lon > 42 && lat > 12) { return "Africa" }
            if lon < 45, lat > 35 { return lon > 26 && lat < 42 ? "Turkey" : "Europe" }
            if lon > 95, lon < 141, lat < 8 { return "Southeast Asia" }
            return "Asia"
        }
        if lat > 66 { return "the Arctic Ocean" }
        if lat < -60 { return "the Southern Ocean" }
        if lat > 30, lat < 46, lon > -6, lon < 36 { return "the Mediterranean" }
        if lat > 9, lat < 22, lon > -88, lon < -60 { return "the Caribbean" }
        if lat > 18, lat < 31, lon > -98, lon < -81 { return "the Gulf of Mexico" }
        if lat > 51, lat < 66, lon > -95, lon < -77 { return "Hudson Bay" }
        if lat > 0, lat < 23, lon > 105, lon < 121 { return "the South China Sea" }
        let indian = lat < 30 && ((lat < -10 && lon > 20 && lon < 147) || (lon > 32 && lon < 100))
        if indian { return "the Indian Ocean" }
        let pacific = lon > 100 || lon < (lat > 9 ? (lat > 18 ? -100 : -84) : -70)
        return (lat >= 0 ? "the North " : "the South ") + (pacific ? "Pacific" : "Atlantic")
    }

    // MARK: Geometry

    nonisolated static func geo(_ lat: Double, _ lon: Double, _ r: Double = 1) -> SIMD3<Float> {
        let a = lat * .pi / 180, o = lon * .pi / 180
        return SIMD3(Float(r * cos(a) * cos(o)), Float(r * cos(a) * sin(o)), Float(r * sin(a)))
    }

    nonisolated static func latLon(_ q: SIMD3<Float>) -> (lat: Double, lon: Double) {
        let n = simd_normalize(q)
        return (Double(asin(max(-1, min(1, n.z)))) * 180 / .pi, Double(atan2(n.y, n.x)) * 180 / .pi)
    }

    /// Unit vector from the Earth's centre toward the Sun, Earth-fixed.
    nonisolated static func sun(_ date: Date) -> SIMD3<Float> {
        let s = Astro.subpoint(Astro.sun(date), date)
        return geo(s.lat, s.lon)
    }

    /// Great-circle distance in kilometres.
    nonisolated static func distance(_ a: (lat: Double, lon: Double), _ b: (lat: Double, lon: Double)) -> Double {
        let p = geo(a.lat, a.lon), q = geo(b.lat, b.lon)
        return Double(acos(max(-1, min(1, simd_dot(p, q))))) * 6371
    }

    /// Mirrors `GlobeCamera` in `Globe.metal`: a perspective camera in Earth-fixed coordinates.
    struct Camera {
        /// Right, up and forward.
        var basis: simd_float3x3
        var eye: SIMD3<Float>
        var sun: SIMD3<Float>
        /// Where the Earth's centre sits on screen, in centered units (y runs -1 to 1, +y up).
        var shift: SIMD2<Float>
        /// Focal length in centered units.
        var zoom: Float
        var aspect: Float
        /// Greenwich sidereal angle in radians, which turns the sky with the Earth.
        var gmst: Float

        /// Looks at the Earth's centre from `distance` radii above a latitude and longitude, north up (over a pole,
        /// the longitude's meridian points down), turned by `roll` radians, with the Sun and stars of `date`.
        init(lat: Float, lon: Float, distance: Float, zoom: Float, roll: Float = 0, shift: SIMD2<Float> = .zero,
             aspect: Float, date: Date) {
            let a = lat * .pi / 180, o = lon * .pi / 180
            let out = SIMD3(cos(a) * cos(o), cos(a) * sin(o), sin(a))
            let f = -out
            let east = SIMD3(-sin(o), cos(o), 0)
            let up = simd_cross(east, f)
            let r = east * cos(roll) + up * sin(roll), u = up * cos(roll) - east * sin(roll)
            basis = simd_float3x3(r, u, f)
            eye = out * distance
            sun = Globe.sun(date)
            gmst = Float(Astro.rad(Astro.gmst(date)))
            self.shift = shift
            self.zoom = zoom
            self.aspect = aspect
        }

        /// Screen position (uv, 0 to 1 from the top left) and distance along the view, or nil behind the camera.
        func project(_ q: SIMD3<Float>) -> SIMD3<Float>? {
            let d = q - eye
            let z = simd_dot(d, basis.columns.2)
            guard z > 1e-3 else { return nil }
            let p = SIMD2(simd_dot(d, basis.columns.0), simd_dot(d, basis.columns.1)) * zoom / z + shift
            return SIMD3(p.x / aspect * 0.5 + 0.5, 0.5 - p.y * 0.5, z)
        }

        /// 1 when nothing of the Earth lies between the camera and q, fading to 0 just behind the limb.
        func shown(_ q: SIMD3<Float>) -> Float {
            let d = q - eye, l = simd_length(d), rd = d / l
            let b = -simd_dot(eye, rd)
            let m = simd_length(eye + rd * b)
            guard m < 1, b < l else { return 1 }
            let entry = b - (1 - m * m).squareRoot()
            if entry > l - 2e-3 { return 1 }
            return 1 - min(max((1 - m) / 0.012, 0), 1)
        }

        /// For points on the surface: 1 on the near side, fading out at the limb. Matches `globeFacing`.
        func facing(_ q: SIMD3<Float>) -> Float {
            let x = simd_dot(simd_normalize(q), simd_normalize(eye - q))
            return Globe.smoothstep(-0.02, 0.06, x)
        }

        /// Projects onto the overlay in points, with the fade for points behind the Earth.
        func overlay(_ q: SIMD3<Float>, _ size: CGSize) -> (point: CGPoint, shown: Float)? {
            guard let p = project(q) else { return nil }
            return (CGPoint(x: CGFloat(p.x) * size.width, y: CGFloat(p.y) * size.height), shown(q))
        }

        /// The Earth's apparent radius on the overlay, in points.
        func radius(_ size: CGSize) -> CGFloat {
            let d = simd_length(eye)
            return CGFloat(zoom / (d * d - 1).squareRoot() * 0.5) * size.height
        }
    }

    /// Per-scene render state: an HDR light buffer, the camera it's filmed with, and a small ring of camera buffers so
    /// the CPU never rewrites one the GPU may still be reading.
    @MainActor final class Canvas {
        private let gpu: GPU
        let lens: Lens
        private var ring: [MTLBuffer]
        private var slot = 0
        private(set) var light: MTLTexture?

        init(gpu: GPU) throws {
            self.gpu = gpu
            lens = try Lens(gpu: gpu)
            ring = (0..<3).map { _ in gpu.buffer(MemoryLayout<Camera>.stride) }
        }

        /// The light buffer for this frame, sized to the target.
        func light(for f: Frame) -> MTLTexture {
            if light?.width != f.target.width || light?.height != f.target.height { light = gpu.texture(f.target.width, f.target.height) }
            return light!
        }

        func camera(_ c: Camera) -> MTLBuffer {
            slot = (slot + 1) % ring.count
            var c = c
            memcpy(ring[slot].contents(), &c, MemoryLayout<Camera>.stride)
            return ring[slot]
        }

        private var streams: [Int: (buffers: [MTLBuffer], slot: Int)] = [:]

        /// A buffer holding `items`, from a ring of three per `key`, for small arrays that change every frame.
        func stream<T>(_ key: Int, _ items: [T]) -> MTLBuffer {
            let length = max(items.count * MemoryLayout<T>.stride, 16)
            var ring = streams[key] ?? ([], 0)
            ring.slot = (ring.slot + 1) % 3
            if ring.buffers.count < 3 || ring.buffers[ring.slot].length < length {
                let fresh = gpu.buffer(length * 2)
                if ring.buffers.count < 3 { ring.buffers.append(fresh); ring.slot = ring.buffers.count - 1 } else { ring.buffers[ring.slot] = fresh }
            }
            let buffer = ring.buffers[ring.slot]
            items.withUnsafeBytes { if let base = $0.baseAddress { memcpy(buffer.contents(), base, $0.count) } }
            streams[key] = ring
            return buffer
        }

        /// Develops the light into the frame's target like a photograph. Instrument marks go on top after this.
        func develop(_ f: Frame, _ look: Lens.Look = Canvas.look) {
            guard let light else { return }
            lens.develop(f, light, look)
        }

        /// A camera in orbit: clean optics, a little grain.
        static let look = Lens.Look(exposure: 1.25, bloom: 0.03, halation: 0.012, vignette: 0.28, aberration: 0.8, grain: 0.35)
    }

    // MARK: Rasterising

    /// Unpacks `shapes` into rings of (longitude, latitude) in degrees, flagged when they're water.
    static func rings() -> [(water: Bool, points: [SIMD2<Double>])] {
        guard let data = Data(base64Encoded: shapes, options: .ignoreUnknownCharacters) else { return [] }
        let bytes = [UInt8](data)
        var i = 0
        func varint() -> UInt64 {
            var v: UInt64 = 0, shift: UInt64 = 0
            while i < bytes.count {
                let b = bytes[i]
                i += 1
                v |= UInt64(b & 0x7f) << shift
                if b < 0x80 { break }
                shift += 7
            }
            return v
        }
        func zigzag() -> Int64 { let v = varint(); return Int64(bitPattern: v >> 1) ^ -Int64(bitPattern: v & 1) }
        var out: [(Bool, [SIMD2<Double>])] = []
        var x: Int64 = 0, y: Int64 = 0
        let count = Int(varint())
        for _ in 0..<count where i < bytes.count {
            let head = varint()
            var points: [SIMD2<Double>] = []
            for _ in 0..<Int(head >> 1) {
                x += zigzag()
                y += zigzag()
                points.append(SIMD2(Double(x) / 100, Double(y) / 100))
            }
            out.append((head & 1 == 1, points))
        }
        return out
    }

    /// Fills land white and water black, antialiased, row 0 at the north pole.
    private static func rasterise(_ w: Int, _ h: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: w * h)
        pixels.withUnsafeMutableBytes { buffer in
            guard let ctx = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            ctx.setShouldAntialias(true)
            ctx.scaleBy(x: CGFloat(w) / 360, y: CGFloat(h) / 180)
            ctx.translateBy(x: 180, y: 90)
            let all = rings()
            for water in [false, true] {
                ctx.setFillColor(gray: water ? 0 : 1, alpha: 1)
                for ring in all where ring.water == water && ring.points.count > 2 {
                    ctx.beginPath()
                    ctx.addLines(between: ring.points.map { CGPoint(x: $0.x, y: $0.y) })
                    ctx.closePath()
                    ctx.fillPath()
                }
            }
        }
        return pixels
    }

    private static func mipmapped(_ gpu: GPU, _ w: Int, _ h: Int, _ bytes: UnsafeRawPointer) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: w, height: h, mipmapped: true)
        d.usage = [.shaderRead, .renderTarget]
        d.storageMode = gpu.device.hasUnifiedMemory ? .shared : .managed
        let t = gpu.device.makeTexture(descriptor: d)!
        t.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: bytes, bytesPerRow: w)
        if let cmd = gpu.queue.makeCommandBuffer(), let blit = cmd.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: t)
            blit.endEncoding()
            cmd.commit()
        }
        return t
    }
}
