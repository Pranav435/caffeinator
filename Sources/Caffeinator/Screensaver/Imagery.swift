import CoreGraphics
import Foundation
import ImageIO
import Metal

/// Photographs and maps from NASA, ESA and ESO for the realistic scenes. Each is downloaded once into `Feed`'s cache,
/// decoded off the main thread and uploaded as a mipmapped texture, so it stays sharp and calm at any zoom. A texture is
/// shared while any scene holds it: four displays of globes decode the Blue Marble once.
@MainActor enum Imagery {
    enum Kind: Hashable, Sendable {
        /// A photograph, sampled as linear light (the texture is sRGB).
        case color
        /// One channel of data, such as a mask or an 8-bit height map, sampled as 0...1.
        case gray
        /// A 16-bit height map (PNG or TIFF), sampled as 0...1.
        case gray16
    }

    private final class Held {
        weak var texture: MTLTexture?
        init(_ texture: MTLTexture) { self.texture = texture }
    }

    /// Carries a finished texture out of its loading task; only the main actor touches it.
    private final class Loaded: @unchecked Sendable {
        let texture: MTLTexture?
        init(_ texture: MTLTexture?) { self.texture = texture }
    }

    private static var held: [String: Held] = [:]
    private static var pending: [String: Task<Loaded, Never>] = [:]

    /// The image at `url` as a texture at most `maxSize` pixels on its long side, or nil when it can't be had (offline
    /// with nothing cached, or not an image). Published imagery rarely changes, so a cached copy serves for `maxAge`.
    static func texture(_ url: URL, _ gpu: GPU, kind: Kind = .color, maxSize: Int = 8192,
                        maxAge: TimeInterval = 90 * 86400) async -> MTLTexture? {
        let key = "\(kind) \(maxSize) \(url.absoluteString)"
        if let texture = held[key]?.texture { return texture }
        if let task = pending[key] { return await task.value.texture }
        let task = Task { @MainActor () -> Loaded in
            guard let data = await Feed.data(url, maxAge: maxAge),
                  let pixels = await Task.detached(priority: .utility, operation: { decode(data, kind, maxSize) }).value
            else { return Loaded(nil) }
            return Loaded(upload(pixels, gpu))
        }
        pending[key] = task
        let texture = await task.value.texture
        pending[key] = nil
        held = held.filter { $0.value.texture != nil }
        if let texture { held[key] = Held(texture) }
        return texture
    }

    /// Decoded pixels, rows padded to 256 bytes for the blit.
    struct Pixels: Sendable {
        let kind: Kind
        let width: Int
        let height: Int
        let bytesPerRow: Int
        let bytes: Data
    }

    /// Decodes image data, shrinking it to `maxSize` while decoding when it's larger.
    nonisolated static func decode(_ data: Data, _ kind: Kind, _ maxSize: Int) -> Pixels? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let w0 = props?[kCGImagePropertyPixelWidth] as? Int ?? 0, h0 = props?[kCGImagePropertyPixelHeight] as? Int ?? 0
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: maxSize,
                                        kCGImageSourceCreateThumbnailWithTransform: true]
        let image = max(w0, h0) > maxSize ? CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
                                          : CGImageSourceCreateImageAtIndex(source, 0, nil)
        guard let image else { return nil }
        let w = image.width, h = image.height
        let bpp = kind == .color ? 4 : kind == .gray ? 1 : 2
        let row = (w * bpp + 255) & ~255
        // Data keeps its values: gray images draw into their own colour space so nothing is gamma-converted.
        let gray = image.colorSpace?.model == .monochrome ? image.colorSpace! : CGColorSpaceCreateDeviceGray()
        let space = kind == .color ? CGColorSpace(name: CGColorSpace.sRGB)! : gray
        let info = switch kind {
        case .color: CGImageAlphaInfo.premultipliedLast.rawValue
        case .gray: CGImageAlphaInfo.none.rawValue
        case .gray16: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue
        }
        var bytes = Data(count: row * h)
        let ok = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: kind == .gray16 ? 16 : 8,
                                      bytesPerRow: row, space: space, bitmapInfo: info) else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? Pixels(kind: kind, width: w, height: h, bytesPerRow: row, bytes: bytes) : nil
    }

    private static func upload(_ p: Pixels, _ gpu: GPU) -> MTLTexture? {
        let format: MTLPixelFormat = switch p.kind {
        case .color: .rgba8Unorm_srgb
        case .gray: .r8Unorm
        case .gray16: .r16Unorm
        }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: p.width, height: p.height, mipmapped: true)
        d.usage = .shaderRead
        d.storageMode = .private
        guard let texture = gpu.device.makeTexture(descriptor: d),
              let staging = p.bytes.withUnsafeBytes({ gpu.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let cmd = gpu.queue.makeCommandBuffer(), let blit = cmd.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: staging, sourceOffset: 0, sourceBytesPerRow: p.bytesPerRow, sourceBytesPerImage: p.bytesPerRow * p.height,
                  sourceSize: MTLSize(width: p.width, height: p.height, depth: 1), to: texture, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin())
        blit.generateMipmaps(for: texture)
        blit.endEncoding()
        cmd.commit()
        return texture
    }
}

/// The real sky behind the space scenes: ESO's 360-degree photograph of the Milky Way by Serge Brunier (eso0932a,
/// CC BY 4.0). Sample it in Metal with `milkyWay(texture, direction, spread)` from the shared header.
enum MilkyWay {
    static let url = URL(string: "https://cdn.eso.org/images/large/eso0932a.jpg")!
    static let credit = "Milky Way: ESO/S. Brunier"

    @MainActor static func texture(_ gpu: GPU) async -> MTLTexture? { await Imagery.texture(url, gpu, maxSize: 6000, maxAge: 365 * 86400) }
}
