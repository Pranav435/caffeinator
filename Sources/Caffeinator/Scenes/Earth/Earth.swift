import AppKit

@MainActor extension SceneKind {
    static let earthScenes: [SceneKind] = [
        SceneKind(id: "earthquakes", name: "Earthquakes", family: .earth,
                  about: "A slowly turning globe replays the last day's earthquakes in a minute and a half: each is marked like a seismicity map, a small dot sized by magnitude and coloured by depth, gold for shallow through violet for deep, with a hairline ring that spreads to a radius set by its energy. The Earth beneath is a satellite photograph, with real clouds, the day and night sides and city lights.",
                  make: Earthquakes.init, gain: 0.35),
        SceneKind(id: "satellites", name: "Every Satellite", family: .earth,
                  about: "Every active satellite, about sixteen thousand, moves around the Earth in real time: each drawn as a fine point, Starlink's shells in pale cyan, navigation satellites in gold, the geostationary ring in rose, and the International Space Station with its ground track.",
                  make: Satellites.init),
        SceneKind(id: "aurora-oval", name: "Aurora Oval", family: .earth,
                  about: "The Earth seen from above the north magnetic pole, and at times the south, with NOAA's aurora forecast drawn as soft green and red curtains over a photographic Earth, city lights on the night side, a pink line at the oval's edge and a dashed circle at your latitude.",
                  make: AuroraOval.init, gain: 1.8),
        SceneKind(id: "dscovr", name: "Whole Earth", family: .earth,
                  about: "Today's real photographs of the whole Earth from NASA's EPIC camera, on the DSCOVR spacecraft a million and a half kilometres away, where the Sun is always behind it. The day's thirteen pictures turn slowly into one rotation, white swirls of cloud over deep blue sea and tan land, with thin labels for the time, the camera's range, and the point under the Sun.",
                  make: Dscovr.init),
        SceneKind(id: "dsn", name: "Deep Space Network", family: .space,
                  about: "NASA's three Deep Space Network sites, Goldstone, Madrid and Canberra, on a small Earth, with beams reaching out to the spacecraft they're talking to right now, from the Moon to Voyager, and photons crawling along each link.",
                  make: DeepSpace.init, gain: 1.2),
    ]
}

/// Numbers, distances and times written the way VoiceOver reads them well.
enum EarthText {
    static func count(_ n: Int) -> String { n.formatted(.number.grouping(.automatic)) }

    static func number(_ x: Double, digits: Int = 1) -> String {
        x.formatted(.number.precision(.fractionLength(0...digits)).grouping(.automatic))
    }

    /// "418 kilometres", "25.1 billion kilometres".
    static func kilometres(_ km: Double) -> String {
        switch km {
        case 1e9...: "\(number(km / 1e9)) billion kilometres"
        case 1e6...: "\(number(km / 1e6)) million kilometres"
        case 1e4...: "\(count(Int((km / 100).rounded()) * 100)) kilometres"
        default: "\(count(Int(km.rounded()))) kilometres"
        }
    }

    /// "46 hours", "12 minutes", "1.3 seconds".
    static func duration(_ seconds: Double) -> String {
        switch seconds {
        case 7200...: "\(number(seconds / 3600, digits: seconds > 36000 ? 0 : 1)) hours"
        case 3600..<7200: "1 hour \(Int((seconds - 3600) / 60)) minutes"
        case 120...: "\(Int((seconds / 60).rounded())) minutes"
        case 10...: "\(Int(seconds.rounded())) seconds"
        default: "\(number(seconds)) seconds"
        }
    }

    static func ago(_ date: Date, now: Date) -> String {
        let s = max(now.timeIntervalSince(date), 0)
        switch s {
        case ..<90: return "just now"
        case ..<3600: return "\(Int((s / 60).rounded())) minutes ago"
        case ..<5400: return "an hour ago"
        default: return "\(Int((s / 3600).rounded())) hours ago"
        }
    }

    /// "70 degrees north".
    static func latitude(_ lat: Double) -> String {
        "\(Int(abs(lat).rounded())) degrees \(lat >= 0 ? "north" : "south")"
    }

    /// "14:05 UTC".
    static func utc(_ date: Date) -> String {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        let parts = c.dateComponents([.hour, .minute], from: date)
        return String(format: "%02d:%02d UTC", parts.hour ?? 0, parts.minute ?? 0)
    }
}

/// Small drawing helpers for the scenes' overlays.
extension CGContext {
    /// A hairline from a to b.
    @MainActor func earthLine(_ a: CGPoint, _ b: CGPoint, color: NSColor, width: CGFloat = 0.75) {
        setStrokeColor(color.cgColor)
        setLineWidth(width)
        move(to: a)
        addLine(to: b)
        strokePath()
    }
}

extension NSColor {
    /// A colour from linear 0...1 components, as the shaders use, for overlay text that matches.
    static func earthLinear(_ c: SIMD3<Float>, alpha: CGFloat = 1) -> NSColor {
        func s(_ x: Float) -> CGFloat {
            let v = max(min(x, 1), 0)
            return CGFloat(v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1 / 2.4) - 0.055)
        }
        return NSColor(srgbRed: s(c.x), green: s(c.y), blue: s(c.z), alpha: alpha)
    }
}

extension Globe {
    nonisolated static func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let t = min(max((x - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }
}
