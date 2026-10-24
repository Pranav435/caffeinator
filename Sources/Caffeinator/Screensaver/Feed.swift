import AppKit

/// Downloads live data with an on-disk cache. A fresh enough copy is used as is; otherwise it fetches, and if the
/// network fails it falls back to the last copy however old, so scenes still work offline.
enum Feed {
    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 25
        // In Low Data Mode the scenes make do with what's cached.
        config.allowsConstrainedNetworkAccess = false
        config.httpAdditionalHeaders = ["User-Agent": "Caffeinator/1 (+https://github.com/Pranav435/caffeinator)"]
        return URLSession(configuration: config)
    }()

    static let folder: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let url = base.appendingPathComponent("io.github.pranav435.caffeinator/feeds", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()

    /// The body of `url`, at most `maxAge` seconds old when the network allows.
    static func data(_ url: URL, maxAge: TimeInterval) async -> Data? {
        let file = folder.appendingPathComponent(name(url))
        let age = (try? file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate).map { -$0.timeIntervalSinceNow }
        if let age, age < maxAge, let cached = try? Data(contentsOf: file) { return cached }
        if let (data, response) = try? await session.data(from: url), (response as? HTTPURLResponse)?.statusCode == 200, !data.isEmpty {
            try? data.write(to: file, options: .atomic)
            return data
        }
        return try? Data(contentsOf: file)
    }

    static func json<T: Decodable>(_ type: T.Type, _ url: URL, maxAge: TimeInterval) async -> T? {
        guard let data = await data(url, maxAge: maxAge) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }

    /// Untyped JSON, for feeds whose shape doesn't suit Codable.
    static func object(_ url: URL, maxAge: TimeInterval) async -> Any? {
        guard let data = await data(url, maxAge: maxAge) else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
    }

    static func image(_ url: URL, maxAge: TimeInterval) async -> CGImage? {
        guard let data = await data(url, maxAge: maxAge), let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// A file name from the URL: FNV-1a hash plus the original extension.
    private static func name(_ url: URL) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in url.absoluteString.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100_0000_01b3 }
        let ext = url.pathExtension.isEmpty ? "bin" : url.pathExtension
        return String(hash, radix: 16) + "." + ext
    }
}

extension GPU {
    /// Uploads a CGImage as an sRGB-decoded RGBA8 texture.
    func texture(_ image: CGImage) -> MTLTexture? {
        let w = image.width, h = image.height
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let bytes = ctx.data else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return texture(w, h, .rgba8Unorm_srgb, bytes: bytes, bytesPerRow: w * 4)
    }
}
