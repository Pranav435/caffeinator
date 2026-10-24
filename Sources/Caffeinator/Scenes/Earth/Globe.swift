import AppKit
import Metal
import simd

/// The Earth shared by the Live Earth scenes. Natural Earth land and lakes are rasterised once into an
/// equirectangular coverage mask whose mipmaps give crisp, screen-constant coastlines at any zoom; NASA's Black Marble
/// adds city lights on the night side when it can be fetched. `draw` ray-traces the globe lit by the real Sun.
///
/// Coordinates are Earth-fixed, in Earth radii: x toward 0° 0°, y toward 0° 90° E, z toward the north pole.
/// Scenes compile `Globe.metal` in front of their own source to project points the same way the globe does.
@MainActor final class Globe {
    struct Style {
        var coast: Float = 1
        var land: Float = 1
        var graticule: Float = 1
        /// City lights on the night side.
        var lights: Float = 1
        var atmosphere: Float = 1
        var stars: Float = 0
        /// Brightness of the unlit hemisphere's land and ocean.
        var night: Float = 1
        /// Sunlight glinting off the ocean.
        var glint: Float = 1
        var coastColor: SIMD3<Float> = [0.32, 0.68, 1.0]
        /// Graticule spacing in degrees.
        var spacing: Float = 15
        /// 1 for real sunlight; 0 shows the whole globe as night.
        var daylight: Float = 1
    }

    static let maskWidth = 4096, maskHeight = 2048
    private static let lightsURL = URL(string: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_01deg.jpg")!
    /// Shared while any Earth scene holds it, so the textures go when the screensaver ends.
    private static weak var instance: Globe?

    /// Land coverage, 0 (sea) to 1 (land), with mipmaps.
    let land: MTLTexture
    /// Night lights (stored as the square root of linear radiance), once Black Marble has arrived.
    private(set) var lights: MTLTexture?
    /// True once the city lights have loaded or failed.
    private(set) var lightsSettled = false
    private let blank: MTLTexture
    private let mask: [UInt8]
    private static let maskStep = 4
    private let surface: MTLRenderPipelineState
    private let tone: MTLRenderPipelineState

    static func shared(_ gpu: GPU) throws -> Globe {
        if let instance { return instance }
        let globe = try Globe(gpu: gpu)
        instance = globe
        return globe
    }

    private init(gpu: GPU) throws {
        let library = try gpu.library(Self.metal + Self.passes)
        surface = try gpu.pipeline(library, "globeSurface")
        tone = try gpu.pipeline(library, "globeFinish")
        let w = Self.maskWidth, h = Self.maskHeight
        let pixels = Self.rasterise(w, h)
        land = pixels.withUnsafeBytes { Self.mipmapped(gpu, w, h, $0.baseAddress!) }
        // A coarse copy for looking up land and sea on the CPU.
        let s = Self.maskStep
        mask = (0..<(h / s)).flatMap { y in (0..<(w / s)).map { x in pixels[(y * s + s / 2) * w + x * s + s / 2] } }
        var zero: UInt8 = 0
        blank = gpu.texture(1, 1, .r8Unorm, bytes: &zero, bytesPerRow: 1)
        Task { @MainActor in
            if let data = await Feed.data(Self.lightsURL, maxAge: 90 * 86400) {
                let decoded = await Task.detached { Self.lightLevels(data) }.value
                if let (bytes, w, h) = decoded {
                    lights = bytes.withUnsafeBytes { Self.mipmapped(gpu, w, h, $0.baseAddress!) }
                }
            }
            lightsSettled = true
        }
    }

    /// Ray-traces the globe, its atmosphere and the space around it into `target` as linear HDR light.
    func draw(_ cmd: MTLCommandBuffer, into target: MTLTexture, camera: MTLBuffer, style s: Style, time: Float) {
        let u = U(res: SIMD2(Float(target.width), Float(target.height)), time: time,
                  a: [s.coast, s.land, s.graticule, s.lights], b: [s.atmosphere, s.stars, s.night, s.glint],
                  c: SIMD4(s.coastColor, 0), d: [s.spacing, s.daylight, 0, 0])
        cmd.draw(surface, into: target, uniforms: u, textures: [land, lights ?? blank], buffers: [camera])
    }

    /// Tone maps the HDR light plus its bloom into the scene's display-ready target.
    func finish(_ f: Frame, light: MTLTexture, glow: MTLTexture, bloom: Float = 0.6, exposure: Float = 1) {
        f.cmd.draw(tone, into: f.target, uniforms: f.uniforms([bloom, exposure, 0, 0]), textures: [light, glow])
    }

    /// Whether a point is land, from the 1:50m outlines at a quarter of a degree.
    func isLand(lat: Double, lon: Double) -> Bool {
        let w = Self.maskWidth / Self.maskStep, h = Self.maskHeight / Self.maskStep
        let x = Int(((lon + 180) / 360 * Double(w)).rounded(.down)), y = Int(((90 - lat) / 180 * Double(h)).rounded(.down))
        return mask[min(max(y, 0), h - 1) * w + (x % w + w) % w] > 127
    }

    /// A rough place name: an ocean or sea ("the South Atlantic") or a landmass ("Africa").
    func region(lat: Double, lon: Double) -> String {
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

    static func geo(_ lat: Double, _ lon: Double, _ r: Double = 1) -> SIMD3<Float> {
        let a = lat * .pi / 180, o = lon * .pi / 180
        return SIMD3(Float(r * cos(a) * cos(o)), Float(r * cos(a) * sin(o)), Float(r * sin(a)))
    }

    static func latLon(_ q: SIMD3<Float>) -> (lat: Double, lon: Double) {
        let n = simd_normalize(q)
        return (Double(asin(max(-1, min(1, n.z)))) * 180 / .pi, Double(atan2(n.y, n.x)) * 180 / .pi)
    }

    /// Unit vector from the Earth's centre toward the Sun, Earth-fixed.
    static func sun(_ date: Date) -> SIMD3<Float> {
        let s = Astro.subpoint(Astro.sun(date), date)
        return geo(s.lat, s.lon)
    }

    /// Great-circle distance in kilometres.
    static func distance(_ a: (lat: Double, lon: Double), _ b: (lat: Double, lon: Double)) -> Double {
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

        /// Looks at the Earth's centre from `distance` radii above a latitude and longitude, north up (over a pole,
        /// the longitude's meridian points down), turned by `roll` radians.
        init(lat: Float, lon: Float, distance: Float, zoom: Float, roll: Float = 0, shift: SIMD2<Float> = .zero,
             aspect: Float, sun: SIMD3<Float>) {
            let a = lat * .pi / 180, o = lon * .pi / 180
            let out = SIMD3(cos(a) * cos(o), cos(a) * sin(o), sin(a))
            let f = -out
            let east = SIMD3(-sin(o), cos(o), 0)
            let up = simd_cross(east, f)
            let r = east * cos(roll) + up * sin(roll), u = up * cos(roll) - east * sin(roll)
            basis = simd_float3x3(r, u, f)
            eye = out * distance
            self.sun = sun
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
    }

    /// Per-scene render state: an HDR light buffer, its bloom, and a small ring of camera buffers so the CPU never
    /// rewrites one the GPU may still be reading.
    @MainActor final class Canvas {
        private let gpu: GPU
        private let bloom: Bloom
        private var ring: [MTLBuffer]
        private var slot = 0
        private(set) var light: MTLTexture?

        init(gpu: GPU) throws {
            self.gpu = gpu
            bloom = try Bloom(gpu: gpu)
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

        func finish(_ f: Frame, globe: Globe, bloom strength: Float = 0.6, exposure: Float = 1) {
            guard let light else { return }
            globe.finish(f, light: light, glow: bloom.glow(f.cmd, light), bloom: strength, exposure: exposure)
        }
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

    /// Black Marble's 0.1 degree mosaic as night-light levels: linear radiance above the faint land and ice
    /// background, stored as its square root for precision in 8 bits.
    private nonisolated static func lightLevels(_ data: Data) -> ([UInt8], Int, Int)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let w = image.width, h = image.height
        var gray = [UInt8](repeating: 0, count: w * h)
        let ok = gray.withUnsafeMutableBytes { buffer -> Bool in
            guard let ctx = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }
        let table: [UInt8] = (0..<256).map { i in
            let v = Double(i) / 255
            let linear = v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
            let level = max(linear - 0.075, 0) / 0.925
            return UInt8((level.squareRoot() * 255).rounded())
        }
        return (gray.map { table[Int($0)] }, w, h)
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
