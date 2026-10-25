import CoreGraphics
import Foundation
import ImageIO
import Metal

/// One site's terrain on the GPU: elevation and image, both mipmapped, plus a coarse copy for steering the camera.
final class MarsTerrain {
    let height: MTLTexture
    let albedo: MTLTexture
    /// Ground extent in metres (east-west, north-south) and the elevation range of the height texture.
    let width: Float, depth: Float
    let low: Float, high: Float
    let gsd: Float
    /// Highest ground in each cell of a coarse grid (metres), row 0 at the north edge.
    let coarse: [Float]
    let coarseSize: Int
    let synthetic: Bool
    /// The data's centre, the direction of its long axis (radians from east toward north) and its spread along and across it.
    let path: SIMD4<Float>
    let spread: SIMD2<Float>

    init(height: MTLTexture, albedo: MTLTexture, width: Float, depth: Float, low: Float, high: Float, gsd: Float, coarse: [Float],
         coarseSize: Int, synthetic: Bool, path: SIMD4<Float>, spread: SIMD2<Float>) {
        self.path = path; self.spread = spread
        self.height = height; self.albedo = albedo; self.width = width; self.depth = depth; self.low = low; self.high = high
        self.gsd = gsd; self.coarse = coarse; self.coarseSize = coarseSize; self.synthetic = synthetic
    }

    /// Ground height under `(x, z)` (metres east, north of the south-west corner), highest within a window of `radius` cells.
    func ground(_ x: Float, _ z: Float, radius: Int = 1) -> Float {
        let n = coarseSize
        let cx = Int(max(0, min(1, x / width)) * Float(n - 1)), cy = Int(max(0, min(1, 1 - z / depth)) * Float(n - 1))
        var m = -Float.infinity
        for dy in -radius...radius { for dx in -radius...radius {
            m = max(m, coarse[max(0, min(n - 1, cy + dy)) * n + max(0, min(n - 1, cx + dx))])
        } }
        return m
    }

    // MARK: Decoding

    struct Grid: Sendable {
        var w: Int, h: Int
        var height: [Float]
        var albedo: [Float]
        var gsd: Float
        var synthetic = false
    }

    /// Reads the first band of a float or 16-bit TIFF as floats; nodata becomes NaN.
    nonisolated static func plane(_ data: Data) -> (w: Int, h: Int, v: [Float])? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
              let bytes = img.dataProvider?.data as Data? else { return nil }
        let w = img.width, h = img.height, bpc = img.bitsPerComponent, row = img.bytesPerRow, step = img.bitsPerPixel / 8
        var out = [Float](repeating: .nan, count: w * h)
        let big = img.bitmapInfo.rawValue & CGBitmapInfo.byteOrderMask.rawValue == CGBitmapInfo.byteOrder16Big.rawValue
        bytes.withUnsafeBytes { raw in
            for y in 0..<h { for x in 0..<w {
                let o = y * row + x * step
                guard o + bpc / 8 <= raw.count else { continue }
                if bpc == 32 {
                    let f = raw.loadUnaligned(fromByteOffset: o, as: Float.self)
                    out[y * w + x] = f.isNaN || f < -1e4 ? .nan : f
                } else if bpc == 16 {
                    var u = raw.loadUnaligned(fromByteOffset: o, as: UInt16.self)
                    if big { u = u.byteSwapped }
                    out[y * w + x] = u == 0 ? .nan : Float(u)
                } else if bpc == 8 {
                    let u = raw[o]
                    out[y * w + x] = u == 0 ? .nan : Float(u)
                }
            } }
        }
        return (w, h, out)
    }

    /// Averages blocks of `f` cells, skipping nodata.
    nonisolated static func shrink(_ p: (w: Int, h: Int, v: [Float]), _ f: Int) -> (w: Int, h: Int, v: [Float]) {
        guard f > 1 else { return p }
        let w = p.w / f, h = p.h / f
        var out = [Float](repeating: .nan, count: w * h)
        for y in 0..<h { for x in 0..<w {
            var s: Float = 0, n: Float = 0
            for j in 0..<f { for i in 0..<f {
                let v = p.v[(y * f + j) * p.w + x * f + i]
                if !v.isNaN { s += v; n += 1 }
            } }
            if n >= Float(f * f) / 2 { out[y * w + x] = s / n }
        } }
        return (w, h, out)
    }

    nonisolated static func decode(dtm: Data, ortho: Data, gsd: Double) -> Grid? {
        guard var d = plane(dtm), var o = plane(ortho), d.w == o.w, d.h == o.h else { return nil }
        let f = max(1, (max(d.w, d.h) + 4095) / 4096)
        d = shrink(d, f); o = shrink(o, f)
        return Grid(w: d.w, h: d.h, height: d.v, albedo: o.v, gsd: Float(gsd) * Float(f))
    }

    /// A believable stand-in when nothing can be downloaded: rolling cratered ground, 30 km square.
    nonisolated static func fallback() -> Grid {
        let n = 512
        var h = [Float](repeating: 0, count: n * n), a = h
        func hash(_ x: Int, _ y: Int) -> Float {
            var s = UInt32(truncatingIfNeeded: x &* 374761393 &+ y &* 668265263); s = (s ^ (s >> 13)) &* 1274126177
            return Float((s ^ (s >> 16)) & 0xFFFF) / 65535
        }
        func noise(_ x: Float, _ y: Float) -> Float {
            let ix = Int(floor(x)), iy = Int(floor(y)), fx = x - floor(x), fy = y - floor(y)
            let u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy)
            return (hash(ix, iy) * (1 - u) + hash(ix + 1, iy) * u) * (1 - v) + (hash(ix, iy + 1) * (1 - u) + hash(ix + 1, iy + 1) * u) * v
        }
        for y in 0..<n { for x in 0..<n {
            var v: Float = 0, amp: Float = 1, fr: Float = 4
            for _ in 0..<6 { v += amp * noise(Float(x) / Float(n) * fr, Float(y) / Float(n) * fr); amp *= 0.5; fr *= 2 }
            h[y * n + x] = v * 900
            a[y * n + x] = 0.35 + 0.4 * noise(Float(x) / 40, Float(y) / 40) + 0.2 * (v - 1) + 0.1 * hash(x, y)
        } }
        // A few craters.
        for k in 0..<14 {
            let cx = hash(k, 1) * Float(n), cy = hash(k, 2) * Float(n), r = 10 + hash(k, 3) * 40
            for y in max(0, Int(cy - r * 1.4))..<min(n, Int(cy + r * 1.4)) { for x in max(0, Int(cx - r * 1.4))..<min(n, Int(cx + r * 1.4)) {
                let d = hypot(Float(x) - cx, Float(y) - cy) / r
                let bowl = d < 1 ? -(1 - d * d) * r * 5 : 0
                let rim = exp(-pow((d - 1) * 4, 2)) * r * 1.6
                h[y * n + x] += bowl + rim
            } }
        }
        var g = Grid(w: n, h: n, height: h, albedo: a, gsd: 60)
        g.synthetic = true
        return g
    }

    /// Fills nodata along rows, then down columns.
    nonisolated static func fill(_ v: inout [Float], _ w: Int, _ h: Int) {
        for y in 0..<h {
            var last = Float.nan
            for x in 0..<w { if v[y * w + x].isNaN { if !last.isNaN { v[y * w + x] = last } } else { last = v[y * w + x] } }
            last = .nan
            for x in stride(from: w - 1, through: 0, by: -1) { if v[y * w + x].isNaN { if !last.isNaN { v[y * w + x] = last } } else { last = v[y * w + x] } }
        }
        for x in 0..<w {
            var last = Float.nan
            for y in 0..<h { if v[y * w + x].isNaN { if !last.isNaN { v[y * w + x] = last } } else { last = v[y * w + x] } }
            last = .nan
            for y in stride(from: h - 1, through: 0, by: -1) { if v[y * w + x].isNaN { if !last.isNaN { v[y * w + x] = last } } else { last = v[y * w + x] } }
        }
        for i in 0..<v.count where v[i].isNaN { v[i] = 0 }
    }

    /// Quantised bytes ready for upload.
    struct Packed: Sendable {
        var w: Int, h: Int
        var height: Data      // UInt16
        var albedo: Data      // UInt8 pairs: image, then 255 where there is data
        var low: Float, high: Float
        var gsd: Float
        var coarse: [Float]
        var synthetic: Bool
        var path: SIMD4<Float>
        var spread: SIMD2<Float>
    }

    nonisolated static func pack(_ g: Grid) -> Packed {
        let n = g.w * g.h
        var valid = [Bool](repeating: false, count: n)
        var hv = g.height
        var av = [Float]()
        av.reserveCapacity(n)
        for i in 0..<n {
            let ok = !g.height[i].isNaN && !g.albedo[i].isNaN
            valid[i] = ok
            if ok { av.append(g.albedo[i]) }
        }
        // Only nodata connected to the frame's border counts as outside; voids inside the data are filled in.
        var outside = [Bool](repeating: false, count: n)
        var queue = [Int]()
        queue.reserveCapacity(n / 4)
        func seed(_ i: Int) { if !valid[i], !outside[i] { outside[i] = true; queue.append(i) } }
        for x in 0..<g.w { seed(x); seed((g.h - 1) * g.w + x) }
        for y in 0..<g.h { seed(y * g.w); seed(y * g.w + g.w - 1) }
        var head = 0
        while head < queue.count {
            let i = queue[head]; head += 1
            let x = i % g.w, y = i / g.w
            if x > 0 { seed(i - 1) }
            if x < g.w - 1 { seed(i + 1) }
            if y > 0 { seed(i - g.w) }
            if y < g.h - 1 { seed(i + g.w) }
        }
        // Keep a margin inside the data: its last cells are ragged, so they count as outside too.
        let margin = Int(0.07 * Float(min(g.w, g.h)))
        var dist = [Int16](repeating: 0, count: n)
        let cap = Int16(400)
        for y in 0..<g.h { for x in 0..<g.w {
            let i = y * g.w + x
            if outside[i] { dist[i] = 0; continue }
            var d = cap
            if x == 0 || y == 0 { d = 1 } else { d = min(d, dist[i - 1] + 1, dist[i - g.w] + 1) }
            dist[i] = d
        } }
        for y in stride(from: g.h - 1, through: 0, by: -1) { for x in stride(from: g.w - 1, through: 0, by: -1) {
            let i = y * g.w + x
            if outside[i] { continue }
            var d = dist[i]
            if x == g.w - 1 || y == g.h - 1 { d = min(d, 1) } else { d = min(d, dist[i + 1] + 1, dist[i + g.w] + 1) }
            dist[i] = d
        } }
        for i in 0..<n { valid[i] = !outside[i] && Int(dist[i]) > margin }
        fill(&hv, g.w, g.h)
        var av2 = g.albedo
        fill(&av2, g.w, g.h)
        // Robust range for the image and exact range for the heights.
        var lo: Float = .infinity, hi: Float = -.infinity
        for i in 0..<n where !g.height[i].isNaN { lo = min(lo, g.height[i]); hi = max(hi, g.height[i]) }
        if !(hi > lo) { lo = 0; hi = 1 }
        av.sort()
        let a0 = av.isEmpty ? 0 : av[av.count / 100], a1 = av.isEmpty ? 1 : av[min(av.count - 1, av.count * 997 / 1000)]
        var hb = Data(count: n * 2), ab = Data(count: n * 2)
        hb.withUnsafeMutableBytes { hr in ab.withUnsafeMutableBytes { ar in
            let hp = hr.bindMemory(to: UInt16.self), ap = ar.bindMemory(to: UInt8.self)
            for i in 0..<n {
                hp[i] = UInt16(max(0, min(65535, (hv[i] - lo) / (hi - lo) * 65535)))
                ap[2 * i] = UInt8(max(0, min(1, (av2[i] - a0) / max(a1 - a0, 1e-6))) * 255)
                ap[2 * i + 1] = valid[i] ? 255 : 0
            }
        } }
        // Centre and long axis of the valid data, for steering the flight over it.
        var sx = 0.0, sz = 0.0, sxx = 0.0, szz = 0.0, sxz = 0.0, cnt = 0.0
        for j in stride(from: 0, to: g.h, by: 3) { for i in stride(from: 0, to: g.w, by: 3) where valid[j * g.w + i] {
            let x = (Double(i) + 0.5) * Double(g.gsd), z = (Double(g.h - j) - 0.5) * Double(g.gsd)
            sx += x; sz += z; sxx += x * x; szz += z * z; sxz += x * z; cnt += 1
        } }
        cnt = max(cnt, 1)
        let mx = sx / cnt, mz = sz / cnt
        let vxx = sxx / cnt - mx * mx, vzz = szz / cnt - mz * mz, vxz = sxz / cnt - mx * mz
        let angle = 0.5 * atan2(2 * vxz, vxx - vzz)
        let tr = (vxx + vzz) / 2, dif = sqrt(max(((vxx - vzz) / 2) * ((vxx - vzz) / 2) + vxz * vxz, 0))
        let s1 = sqrt(max(tr + dif, 1)), s2 = sqrt(max(tr - dif, 1))
        let c = 48
        var coarse = [Float](repeating: -.infinity, count: c * c)
        for y in 0..<g.h { for x in 0..<g.w {
            let k = (y * c / g.h) * c + x * c / g.w
            coarse[k] = max(coarse[k], hv[y * g.w + x])
        } }
        return Packed(w: g.w, h: g.h, height: hb, albedo: ab, low: lo, high: hi, gsd: g.gsd, coarse: coarse.map { $0.isFinite ? $0 : lo }, synthetic: g.synthetic,
                      path: SIMD4(Float(mx), Float(mz), Float(angle), 0), spread: SIMD2(Float(s1), Float(s2)))
    }

    @MainActor static func upload(_ p: Packed, _ gpu: GPU) -> MarsTerrain? {
        func make(_ format: MTLPixelFormat, _ bytes: Data, _ bpp: Int) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: p.w, height: p.h, mipmapped: true)
            d.usage = .shaderRead
            d.storageMode = .private
            let row = p.w * bpp
            guard let t = gpu.device.makeTexture(descriptor: d),
                  let staging = bytes.withUnsafeBytes({ gpu.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
                  let cmd = gpu.queue.makeCommandBuffer(), let blit = cmd.makeBlitCommandEncoder() else { return nil }
            blit.copy(from: staging, sourceOffset: 0, sourceBytesPerRow: row, sourceBytesPerImage: row * p.h,
                      sourceSize: MTLSize(width: p.w, height: p.h, depth: 1), to: t, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin())
            blit.generateMipmaps(for: t)
            blit.endEncoding()
            cmd.commit()
            return t
        }
        guard let h = make(.r16Unorm, p.height, 2), let a = make(.rg8Unorm, p.albedo, 2) else { return nil }
        return MarsTerrain(height: h, albedo: a, width: Float(p.w) * p.gsd, depth: Float(p.h) * p.gsd, low: p.low, high: p.high, gsd: p.gsd,
                           coarse: p.coarse, coarseSize: 48, synthetic: p.synthetic, path: p.path, spread: p.spread)
    }

    /// Downloads, decodes and uploads a site's terrain; nil when it can't be had.
    @MainActor static func load(_ site: MarsSite, _ gpu: GPU) async -> MarsTerrain? {
        guard let dtm = await Feed.data(site.dtm, maxAge: 365 * 86400) else { return nil }
        guard let ortho = await Feed.data(site.ortho, maxAge: 365 * 86400) else { return nil }
        let gsd = site.gsd
        let packed = await Task.detached(priority: .utility) { () -> Packed? in
            guard let g = decode(dtm: dtm, ortho: ortho, gsd: gsd) else { return nil }
            return pack(g)
        }.value
        return packed.flatMap { upload($0, gpu) }
    }

    @MainActor static func makeFallback(_ gpu: GPU) async -> MarsTerrain? {
        let packed = await Task.detached(priority: .utility) { pack(fallback()) }.value
        return upload(packed, gpu)
    }
}
