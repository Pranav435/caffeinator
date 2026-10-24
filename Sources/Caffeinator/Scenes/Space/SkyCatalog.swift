import Foundation

/// The embedded star catalogue from `StarData`, decoded once.
enum SkyCatalog {
    struct Star { var ra, dec, mag, bv: Float }
    struct Name { var name: String; var ra, dec: Float; var rank: Int }

    static func inflate(_ base64: String) -> Data {
        guard let packed = Data(base64Encoded: base64),
              let data = try? (packed as NSData).decompressed(using: .zlib) as Data else { return Data() }
        return data
    }

    private static func point(_ b: [UInt8], _ i: Int) -> (ra: Float, dec: Float) {
        let r = UInt16(b[i]) | UInt16(b[i + 1]) << 8, d = UInt16(b[i + 2]) | UInt16(b[i + 3]) << 8
        return (Float(r) / 65536 * 360, Float(d) / 65535 * 180 - 90)
    }

    static let stars: [Star] = {
        let b = [UInt8](inflate(StarData.stars))
        return stride(from: 0, to: b.count - 5, by: 6).map { i in
            let p = point(b, i)
            return Star(ra: p.ra, dec: p.dec, mag: Float(b[i + 4]) / 25 - 2, bv: Float(b[i + 5]) / 80 - 0.5)
        }
    }()

    /// Constellation stick figures as polylines of (right ascension, declination) in degrees.
    static let lines: [[SIMD2<Float>]] = {
        let b = [UInt8](inflate(StarData.lines))
        var out: [[SIMD2<Float>]] = []
        var i = 0
        while i < b.count {
            let n = Int(b[i])
            i += 1
            guard i + n * 4 <= b.count else { break }
            out.append((0..<n).map { k in let p = point(b, i + k * 4); return SIMD2(p.ra, p.dec) })
            i += n * 4
        }
        return out
    }()

    private static func parse(_ s: String) -> [[Substring]] { s.split(separator: ";").map { $0.split(separator: ",") } }

    static let constellations: [Name] = parse(StarData.names).compactMap { f in
        guard f.count >= 4, let ra = Float(f[1]), let dec = Float(f[2]) else { return nil }
        return Name(name: String(f[0]), ra: ra, dec: dec, rank: Int(f[3]) ?? 3)
    }

    /// Brightest first, leaving out any that would crowd a brighter neighbour's label.
    static let starNames: [Name] = {
        var kept: [Name] = []
        for f in parse(StarData.starNames) {
            guard f.count >= 3, let ra = Float(f[1]), let dec = Float(f[2]) else { continue }
            let v = unit(ra: Double(ra), dec: Double(dec))
            if kept.contains(where: { (unit(ra: Double($0.ra), dec: Double($0.dec)) * v).sum() > cos(Astro.rad(3)) }) { continue }
            kept.append(Name(name: String(f[0]), ra: ra, dec: dec, rank: 0))
        }
        return kept
    }()

    /// 720 × 360 half-degree cells of galactic longitude and latitude, 0 to 5.
    static let milkyWay: [UInt8] = {
        let b = [UInt8](inflate(StarData.milkyWay))
        return b.count == 720 * 360 ? b : [UInt8](repeating: 0, count: 720 * 360)
    }()

    /// Rows of the rotation from equatorial (J2000) to galactic coordinates.
    static let galactic: [SIMD3<Double>] = [[-0.0548755604, -0.8734370902, -0.4838350155],
                                            [0.4941094279, -0.4448296300, 0.7469822445],
                                            [-0.8676661490, -0.1980763734, 0.4559837762]]

    static func unit(ra: Double, dec: Double) -> SIMD3<Double> {
        let a = Astro.rad(ra), d = Astro.rad(dec)
        return SIMD3(cos(d) * cos(a), cos(d) * sin(a), sin(d))
    }

    /// Colour temperature of a star from its B−V index (Ballesteros 2012).
    static func temperature(bv: Double) -> Double { 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62)) }

    /// The same black body fit as the shader header, normalised to the brightest channel.
    static func blackbody(_ kelvin: Double) -> SIMD3<Float> {
        let t = min(max(kelvin, 1000), 40000) / 100
        let r = t <= 66 ? 1 : min(max(1.29293618606 * pow(t - 60, -0.1332047592), 0), 1)
        let g = t <= 66 ? min(max(0.39008157876 * log(t) - 0.63184144378, 0), 1) : min(max(1.12989086089 * pow(t - 60, -0.0755148492), 0), 1)
        let b = t >= 66 ? 1 : (t <= 19 ? 0 : min(max(0.54320678911 * log(t - 10) - 1.19625408914, 0), 1))
        return SIMD3(Float(r), Float(g), Float(b))
    }
}

/// The major annual meteor showers, after the International Meteor Organization's calendar.
struct Shower {
    let name: String
    let adjective: String
    /// Peak date, and the first and last days of activity, as (month, day).
    let peak: (Int, Int)
    let start: (Int, Int)
    let end: (Int, Int)
    /// Radiant at the peak (degrees, J2000) and its daily drift.
    let ra: Double, dec: Double, dRa: Double, dDec: Double
    /// Zenithal hourly rate at the peak, and how fast it falls off: ZHR × 10^(−slope × days), before and after.
    let zhr: Double, rise: Double, fall: Double
    /// Entry speed, km/s.
    let speed: Double

    static let all = [
        Shower(name: "Quadrantids", adjective: "Quadrantid", peak: (1, 3), start: (12, 28), end: (1, 12),
               ra: 230, dec: 49, dRa: 0.8, dDec: -0.2, zhr: 80, rise: 1.8, fall: 1.8, speed: 41),
        Shower(name: "Lyrids", adjective: "Lyrid", peak: (4, 22), start: (4, 14), end: (4, 30),
               ra: 271, dec: 34, dRa: 1.1, dDec: 0, zhr: 18, rise: 0.25, fall: 0.35, speed: 49),
        Shower(name: "Eta Aquariids", adjective: "Eta Aquariid", peak: (5, 6), start: (4, 19), end: (5, 28),
               ra: 338, dec: -1, dRa: 0.9, dDec: 0.35, zhr: 50, rise: 0.08, fall: 0.08, speed: 66),
        Shower(name: "Perseids", adjective: "Perseid", peak: (8, 12), start: (7, 17), end: (8, 24),
               ra: 48, dec: 58, dRa: 1.35, dDec: 0.12, zhr: 100, rise: 0.2, fall: 0.4, speed: 59),
        Shower(name: "Orionids", adjective: "Orionid", peak: (10, 21), start: (10, 2), end: (11, 7),
               ra: 95, dec: 16, dRa: 0.7, dDec: 0.1, zhr: 20, rise: 0.12, fall: 0.12, speed: 66),
        Shower(name: "Leonids", adjective: "Leonid", peak: (11, 17), start: (11, 6), end: (11, 30),
               ra: 152, dec: 22, dRa: 0.7, dDec: -0.35, zhr: 15, rise: 0.25, fall: 0.3, speed: 71),
        Shower(name: "Geminids", adjective: "Geminid", peak: (12, 14), start: (12, 4), end: (12, 20),
               ra: 112, dec: 33, dRa: 1.0, dDec: -0.15, zhr: 150, rise: 0.39, fall: 0.81, speed: 35),
    ]

    /// Days from this shower's nearest peak, in the user's calendar.
    func days(from date: Date, calendar: Calendar = .current) -> Double {
        let year = calendar.component(.year, from: date)
        let peaks = (year - 1...year + 1).compactMap { calendar.date(from: DateComponents(year: $0, month: peak.0, day: peak.1, hour: 12)) }
        return peaks.map { date.timeIntervalSince($0) / 86400 }.min { abs($0) < abs($1) } ?? 1000
    }

    /// Days before and after the peak that the shower is active.
    var window: (before: Double, after: Double) {
        func ordinal(_ md: (Int, Int)) -> Double { Double([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][md.0 - 1] + md.1) }
        var before = ordinal(peak) - ordinal(start), after = ordinal(end) - ordinal(peak)
        if before < 0 { before += 365 }
        if after < 0 { after += 365 }
        return (before, after)
    }

    /// The zenithal hourly rate today, or nil when the shower isn't active.
    func rate(on date: Date) -> Double? {
        let d = days(from: date)
        guard d >= -window.before - 0.5, d <= window.after + 0.5 else { return nil }
        return zhr * pow(10, -(d < 0 ? rise : fall) * abs(d))
    }

    func radiant(on date: Date) -> (ra: Double, dec: Double) {
        let d = days(from: date)
        return (Astro.norm(ra + dRa * d), dec + dDec * d)
    }

    static func active(on date: Date) -> (Shower, Double)? {
        all.compactMap { s in s.rate(on: date).map { (s, $0) } }.filter { $0.1 >= 1 }.max { $0.1 < $1.1 }
    }
}
