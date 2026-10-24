import AppKit
import Metal

/// RGBA8 pixels decoded off the main thread, ready to upload.
struct SkyPixels: Sendable {
    let width: Int
    let height: Int
    var bytes: [UInt8]

    /// Decodes image data, drawn into `width` × `height` when given (otherwise its own size).
    nonisolated static func decode(_ data: Data, width: Int? = nil, height: Int? = nil) -> SkyPixels? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let w = width ?? image.width, h = height ?? image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        let ok = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? SkyPixels(width: w, height: h, bytes: bytes) : nil
    }

    /// Average sRGB brightness 0...1 of the pixels inside a centred circle of `radius` (a fraction of the width).
    func brightness(radius: Double = 0.45, step: Int = 7) -> Double {
        var sum = 0.0, n = 0.0
        let c = Double(width) / 2, r2 = pow(radius * Double(width), 2)
        for y in stride(from: 0, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) where pow(Double(x) - c, 2) + pow(Double(y) - c, 2) < r2 {
                let i = (y * width + x) * 4
                sum += Double(Int(bytes[i]) + Int(bytes[i + 1]) + Int(bytes[i + 2])) / 765
                n += 1
            }
        }
        return n > 0 ? sum / n : 0
    }
}

extension GPU {
    /// Uploads pixels as an sRGB texture with a full mip chain, so it stays smooth when drawn smaller.
    func mipmapped(_ p: SkyPixels) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm_srgb, width: p.width, height: p.height, mipmapped: true)
        d.usage = [.shaderRead]
        d.storageMode = device.hasUnifiedMemory ? .shared : .managed
        guard let t = device.makeTexture(descriptor: d) else { return nil }
        p.bytes.withUnsafeBytes {
            t.replace(region: MTLRegionMake2D(0, 0, p.width, p.height), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: p.width * 4)
        }
        if let cmd = queue.makeCommandBuffer(), let blit = cmd.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: t)
            blit.endEncoding()
            cmd.commit()
        }
        return t
    }
}

/// Small helpers the Sky scenes share: wording for VoiceOver, slow drift against burn-in, and Metal snippets.
enum SkyKit {
    /// "10:36" in UTC.
    static func utc(_ date: Date) -> String {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        let parts = c.dateComponents([.hour, .minute], from: date)
        return String(format: "%02d:%02d", parts.hour ?? 0, parts.minute ?? 0)
    }

    /// 1792836097 → "1,792,836,097".
    static func grouped(_ n: Int) -> String {
        let s = String(abs(n))
        var out = ""
        for (i, ch) in s.enumerated() {
            if i > 0, (s.count - i) % 3 == 0 { out += "," }
            out.append(ch)
        }
        return (n < 0 ? "-" : "") + out
    }

    /// Azimuth in degrees (from north through east) as a spoken direction, such as "south-east".
    static func compass(_ az: Double) -> String {
        let names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
        let i = Int((((az.truncatingRemainder(dividingBy: 360) + 360).truncatingRemainder(dividingBy: 360) + 22.5) / 45)) % 8
        return names[i]
    }

    /// A slow Lissajous wander of up to `amount` points, so text and edges never sit still for hours.
    static func drift(_ t: Double, amount: Double, seed: Double = 0) -> CGPoint {
        CGPoint(x: amount * sin(t / 173 + seed * 3.1), y: amount * 0.7 * sin(t / 241 + seed * 1.7))
    }

    /// Metal helpers shared by the Sky scenes.
    static let metal = """
    // Catmull-Rom bicubic filtering in nine bilinear taps: sharper than bilinear when an image is magnified.
    inline float4 catmull(texture2d<float> t, sampler s, float2 uv) {
        float2 size = float2(t.get_width(), t.get_height());
        float2 sp = uv * size;
        float2 p1 = floor(sp - 0.5) + 0.5;
        float2 f = sp - p1;
        float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
        float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
        float2 w3 = f * f * (-0.5 + 0.5 * f);
        float2 w12 = w1 + w2;
        float2 t0 = (p1 - 1) / size, t3 = (p1 + 2) / size, t12 = (p1 + w2 / w12) / size;
        float4 c = 0;
        c += t.sample(s, float2(t0.x, t0.y), level(0)) * w0.x * w0.y;
        c += t.sample(s, float2(t12.x, t0.y), level(0)) * w12.x * w0.y;
        c += t.sample(s, float2(t3.x, t0.y), level(0)) * w3.x * w0.y;
        c += t.sample(s, float2(t0.x, t12.y), level(0)) * w0.x * w12.y;
        c += t.sample(s, float2(t12.x, t12.y), level(0)) * w12.x * w12.y;
        c += t.sample(s, float2(t3.x, t12.y), level(0)) * w3.x * w12.y;
        c += t.sample(s, float2(t0.x, t3.y), level(0)) * w0.x * w3.y;
        c += t.sample(s, float2(t12.x, t3.y), level(0)) * w12.x * w3.y;
        c += t.sample(s, float2(t3.x, t3.y), level(0)) * w3.x * w3.y;
        return max(c, 0.0);
    }

    // Samples an image sharply when it's magnified and with mipmaps when it's shrunk. `texel` is screen pixels per image texel.
    inline float4 picture(texture2d<float> t, float2 uv, float texel) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_zero);
        return texel > 1.1 ? catmull(t, s, uv) : t.sample(s, uv);
    }

    // A sparse star field: one chance of a star per `cell` pixels, each a soft dot about `size` pixels across.
    inline float3 starfield(float2 px, float cell, float density, float size, float time) {
        float2 id0 = floor(px / cell);
        float3 c = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            float2 id = id0 + float2(i, j);
            float h = hash12(id * 1.37 + 11.3);
            if (h > density) continue;
            float2 at = (id + hash22(id + 3.1)) * cell;
            float d = length(px - at) / size;
            float mag = pow(hash12(id + 7.7), 8.0);
            float tw = 0.85 + 0.15 * sin(time * (0.3 + h * 1.5) + h * 40);
            float3 tint = mix(float3(1.0, 0.8, 0.62), float3(0.72, 0.82, 1.0), hash12(id + 2.2));
            c += tint * (0.03 + mag * 1.2) * tw * exp(-d * d * 2.5);
        }
        return c;
    }
    """
}
