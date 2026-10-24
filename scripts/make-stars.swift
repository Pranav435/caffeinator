// Builds Sources/Caffeinator/Scenes/Space/StarData.swift, the night sky's embedded catalogue, from d3-celestial
// (https://github.com/ofrohn/d3-celestial, BSD 3-clause, Copyright (c) 2015 Olaf Frohn): stars to magnitude 6
// (XHIP), constellation lines and names (IAU), bright star names, and the Milky Way outline in five brightness
// steps (Jose R. Vieira's Milky Way Outline Catalog), rasterised here in galactic coordinates.
// Run from the repo root: swift scripts/make-stars.swift [folder with the JSON files; downloads them otherwise]
import Foundation

let base = "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/"
let folder = CommandLine.arguments.dropFirst().first

func load(_ name: String) -> [String: Any] {
    let data: Data
    if let folder {
        data = try! Data(contentsOf: URL(fileURLWithPath: folder).appendingPathComponent(name))
    } else {
        data = try! Data(contentsOf: URL(string: base + name)!)
    }
    return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
}

func features(_ json: [String: Any]) -> [[String: Any]] { json["features"] as! [[String: Any]] }
func number(_ v: Any?) -> Double? {
    if let d = v as? Double { return d }
    if let s = v as? String { return Double(s) }
    return nil
}

var bytes = Data()
func u8(_ v: Int) { bytes.append(UInt8(clamping: v)) }
func u16(_ v: Int) { let x = UInt16(clamping: v); bytes.append(UInt8(x & 0xff)); bytes.append(UInt8(x >> 8)) }
/// Right ascension 0..<360 and declination -90...90 as two 16-bit fractions.
func point(_ ra: Double, _ dec: Double) {
    let r = (ra.truncatingRemainder(dividingBy: 360) + 360).truncatingRemainder(dividingBy: 360)
    u16(Int((r / 360 * 65536).rounded()) & 0xffff)
    u16(Int(((dec + 90) / 180 * 65535).rounded()))
}
func packed() -> String {
    let z = try! (bytes as NSData).compressed(using: .zlib) as Data
    bytes = Data()
    return z.base64EncodedString()
}

// Stars, brightest first: position, magnitude (-2...8 in 1/25ths), B-V (-0.5...2.5 in 1/80ths).
let starList = features(load("stars.6.json")).compactMap { f -> (Double, Double, Double, Double, Int)? in
    let c = (f["geometry"] as! [String: Any])["coordinates"] as! [Double]
    let p = f["properties"] as! [String: Any]
    guard let mag = number(p["mag"]) else { return nil }
    return (c[0], c[1], mag, number(p["bv"]) ?? 0.6, f["id"] as? Int ?? 0)
}.sorted { $0.2 < $1.2 }
for s in starList {
    point(s.0, s.1)
    u8(Int(((s.2 + 2) * 25).rounded()))
    u8(Int(((s.3 + 0.5) * 80).rounded()))
}
let stars = packed()

// Proper names of the brightest stars.
let starNames = load("starnames.json")
let named = starList.prefix(40).compactMap { s -> String? in
    guard let n = (starNames[String(s.4)] as? [String: Any])?["name"] as? String, !n.isEmpty, n != "Toliman" else { return nil }
    return String(format: "%@,%.3f,%.3f", n, (s.0 + 360).truncatingRemainder(dividingBy: 360), s.1)
}.joined(separator: ";")

// Constellation stick figures as polylines: a count, then the points.
for f in features(load("constellations.lines.json")) {
    for line in (f["geometry"] as! [String: Any])["coordinates"] as! [[[Double]]] {
        u8(line.count)
        for p in line { point(p[0], p[1]) }
    }
}
let lines = packed()

// Constellation names with their label positions and IAU rank (1 = most prominent).
let names = features(load("constellations.json")).map { f -> String in
    let p = f["properties"] as! [String: Any]
    let c = (f["geometry"] as! [String: Any])["coordinates"] as! [Double]
    return String(format: "%@,%.2f,%.2f,%@", p["name"] as! String, (c[0] + 360).truncatingRemainder(dividingBy: 360), c[1],
                  p["rank"] as? String ?? "3")
}.joined(separator: ";")

// The Milky Way: for each half-degree cell of galactic longitude and latitude, how many of the five nested
// brightness outlines contain it. Rings are tested even-odd with rays from the south galactic pole.
let toGalactic: [[Double]] = [[-0.0548755604, -0.8734370902, -0.4838350155],
                              [0.4941094279, -0.4448296300, 0.7469822445],
                              [-0.8676661490, -0.1980763734, 0.4559837762]]
func galactic(_ ra: Double, _ dec: Double) -> (l: Double, b: Double) {
    let a = ra * .pi / 180, d = dec * .pi / 180
    let v = [cos(d) * cos(a), cos(d) * sin(a), sin(d)]
    let g = toGalactic.map { $0[0] * v[0] + $0[1] * v[1] + $0[2] * v[2] }
    let l = atan2(g[1], g[0]) * 180 / .pi
    return ((l + 360).truncatingRemainder(dividingBy: 360), asin(max(-1, min(1, g[2]))) * 180 / .pi)
}
let width = 720, height = 360
var levels = [UInt8](repeating: 0, count: width * height)
for f in features(load("mw.json")) {
    var crossings = [[Double]](repeating: [], count: width)
    for polygon in (f["geometry"] as! [String: Any])["coordinates"] as! [[[[Double]]]] {
        for ring in polygon {
            let g = ring.map { galactic($0[0], $0[1]) }
            for i in 0..<g.count {
                let p = g[i], q = g[(i + 1) % g.count]
                var dl = q.l - p.l
                if dl > 180 { dl -= 360 }
                if dl < -180 { dl += 360 }
                guard dl != 0 else { continue }
                // Columns whose centres fall between the two longitudes, the short way round.
                let lo = min(p.l, p.l + dl), hi = max(p.l, p.l + dl)
                var c = Int(ceil(lo * 2 - 0.5))
                while Double(c) + 0.5 < hi * 2 {
                    let lc = (Double(c) + 0.5) / 2
                    let t = (lc - p.l) / dl
                    crossings[((c % width) + width) % width].append(p.b + (q.b - p.b) * t)
                    c += 1
                }
            }
        }
    }
    for x in 0..<width {
        let cs = crossings[x].sorted()
        for y in 0..<height {
            let b = -90 + (Double(y) + 0.5) / 2
            if cs.filter({ $0 < b }).count % 2 == 1 { levels[y * width + x] += 1 }
        }
    }
}
bytes = Data(levels)
let milkyWay = packed()

let out = """
// Generated by scripts/make-stars.swift; do not edit. Data from d3-celestial by Olaf Frohn
// (https://github.com/ofrohn/d3-celestial, BSD 3-clause licence, Copyright (c) 2015 Olaf Frohn), which draws on
// XHIP (Anderson & Francis 2012), the IAU constellation pages, and Jose R. Vieira's Milky Way Outline Catalog.

/// The night sky's catalogue. Binary fields are zlib-compressed and base64-encoded; positions are two little-endian
/// 16-bit fractions of right ascension (0 to 360 degrees) and declination (-90 to 90).
enum StarData {
    /// \(starList.count) stars to magnitude 6, brightest first: position, then magnitude as (mag + 2) × 25 and B−V as (bv + 0.5) × 80.
    static let stars = "\(stars)"
    /// Constellation stick figures: polylines, each a point count then its points.
    static let lines = "\(lines)"
    /// Constellation names, label right ascension and declination, and IAU rank.
    static let names = "\(names)"
    /// Proper names of the brightest stars, with right ascension and declination.
    static let starNames = "\(named)"
    /// The Milky Way in galactic coordinates: \(width) × \(height) half-degree cells from longitude 0 and latitude -90,
    /// each the number (0 to 5) of nested brightness outlines that contain it.
    static let milkyWay = "\(milkyWay)"
}

"""
try! out.write(toFile: "Sources/Caffeinator/Scenes/Space/StarData.swift", atomically: true, encoding: .utf8)
print("\(starList.count) stars, \(out.utf8.count / 1024) KB")
