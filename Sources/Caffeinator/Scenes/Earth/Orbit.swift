import Foundation

/// One satellite from a NORAD two-line element set, propagated with SGP4's secular terms: the Brouwer mean motion,
/// J2 and J4 drift of the node and perigee, and drag through the first derivative of the mean motion. Short-period
/// terms are left out, which costs a few kilometres, well under a pixel at globe scale.
struct Orbit: Sendable {
    enum Kind: Int, Sendable { case starlink, oneweb, low, medium, geostationary, eccentric }

    let name: String
    let epoch: Date
    let inclination: Double     // radians
    let eccentricity: Double
    let node: Double            // right ascension of the ascending node at epoch, radians
    let perigee: Double         // argument of perigee at epoch, radians
    let anomaly: Double         // mean anomaly at epoch, radians
    let motion: Double          // Brouwer mean motion, radians a minute
    let axis: Double            // semi-major axis, Earth equatorial radii (WGS-72)
    let nodeRate: Double, perigeeRate: Double, anomalyRate: Double   // radians a minute
    let drag: Double            // half the mean motion's derivative, radians a minute squared

    static let ke = 0.0743669161331734, k2 = 5.413080e-4, k4 = 0.62098875e-6
    /// WGS-72 equatorial radius, km, the unit SGP4 works in.
    static let radius = 6378.135

    /// Orbital period in minutes.
    var period: Double { 2 * .pi / anomalyRate }
    /// Mean altitude above the equator, km.
    var altitude: Double { (axis - 1) * Self.radius }

    var kind: Kind {
        if name.hasPrefix("STARLINK") { return .starlink }
        if name.hasPrefix("ONEWEB") { return .oneweb }
        if eccentricity > 0.25 { return .eccentric }
        if period > 1300, period < 1600 { return .geostationary }
        return altitude < 2000 ? .low : .medium
    }

    /// Parses a three-line set: a name, then lines 1 and 2.
    init?(name: String, _ line1: String, _ line2: String) {
        let l1 = Array(line1.utf8), l2 = Array(line2.utf8)
        guard l1.count >= 61, l2.count >= 63, l1[0] == UInt8(ascii: "1"), l2[0] == UInt8(ascii: "2") else { return nil }
        func field(_ l: [UInt8], _ a: Int, _ b: Int) -> String { String(decoding: l[a..<b], as: UTF8.self).trimmingCharacters(in: .whitespaces) }
        func number(_ l: [UInt8], _ a: Int, _ b: Int) -> Double? { Double(field(l, a, b)) }
        guard let yy = Int(field(l1, 18, 20)), let day = number(l1, 20, 32), let ndot = number(l1, 33, 43),
              let inc = number(l2, 8, 16), let raan = number(l2, 17, 25), let ecc = Double("0." + field(l2, 26, 33)),
              let argp = number(l2, 34, 42), let m = number(l2, 43, 51), let revs = number(l2, 52, 63), revs > 0.05, ecc < 0.99 else { return nil }
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        guard let jan1 = c.date(from: DateComponents(year: yy < 57 ? 2000 + yy : 1900 + yy, month: 1, day: 1)) else { return nil }
        self.name = name.trimmingCharacters(in: .whitespaces)
        epoch = jan1.addingTimeInterval((day - 1) * 86400)
        let rad = Double.pi / 180
        inclination = inc * rad
        eccentricity = ecc
        node = raan * rad
        perigee = argp * rad
        anomaly = m * rad
        drag = ndot * 2 * .pi / (1440 * 1440)

        // SGP4 initialisation: undo the Kozai mean motion, then the secular rates.
        let no = revs * 2 * .pi / 1440
        let cosi = cos(inclination), theta2 = cosi * cosi, theta4 = theta2 * theta2
        let x3thm1 = 3 * theta2 - 1, beta2 = 1 - ecc * ecc, beta = beta2.squareRoot()
        let a1 = pow(Self.ke / no, 2.0 / 3)
        let del1 = 1.5 * Self.k2 * x3thm1 / (a1 * a1 * beta * beta2)
        let ao = a1 * (1 - del1 * (1.0 / 3 + del1 * (1 + 134.0 / 81 * del1)))
        let delo = 1.5 * Self.k2 * x3thm1 / (ao * ao * beta * beta2)
        motion = no / (1 + delo)
        axis = ao / (1 - delo)
        let pinvsq = 1 / (axis * axis * beta2 * beta2)
        let temp1 = 3 * Self.k2 * pinvsq * motion, temp2 = temp1 * Self.k2 * pinvsq, temp3 = 1.25 * Self.k4 * pinvsq * pinvsq * motion
        anomalyRate = motion + 0.5 * temp1 * beta * x3thm1 + 0.0625 * temp2 * beta * (13 - 78 * theta2 + 137 * theta4)
        perigeeRate = -0.5 * temp1 * (1 - 5 * theta2) + 0.0625 * temp2 * (7 - 114 * theta2 + 395 * theta4) + temp3 * (3 - 36 * theta2 + 49 * theta4)
        nodeRate = -temp1 * cosi + (0.5 * temp2 * (4 - 19 * theta2) + 2 * temp3 * (3 - 7 * theta2)) * cosi
    }

    /// Mean elements at a moment: node, argument of perigee and mean anomaly in radians, wrapped to 0..<2π.
    func elements(at date: Date) -> (node: Double, perigee: Double, anomaly: Double) {
        let t = date.timeIntervalSince(epoch) / 60
        func wrap(_ x: Double) -> Double { let r = x.truncatingRemainder(dividingBy: 2 * .pi); return r < 0 ? r + 2 * .pi : r }
        return (wrap(node + nodeRate * t), wrap(perigee + perigeeRate * t), wrap(anomaly + anomalyRate * t + drag * t * t))
    }

    /// Position in the TEME frame, km.
    func teme(at date: Date) -> SIMD3<Double> {
        let el = elements(at: date)
        let e = eccentricity
        var E = el.anomaly + e * sin(el.anomaly)
        for _ in 0..<8 { E -= (E - e * sin(E) - el.anomaly) / (1 - e * cos(E)) }
        let x = axis * (cos(E) - e), y = axis * (1 - e * e).squareRoot() * sin(E)
        let cw = cos(el.perigee), sw = sin(el.perigee), co = cos(el.node), so = sin(el.node)
        let ci = cos(inclination), si = sin(inclination)
        let xw = x * cw - y * sw, yw = x * sw + y * cw
        return SIMD3(xw * co - yw * ci * so, xw * so + yw * ci * co, yw * si) * Self.radius
    }

    /// Earth-fixed position, km: TEME turned by Greenwich sidereal time.
    func earthFixed(at date: Date) -> SIMD3<Double> {
        let r = teme(at: date), g = Astro.rad(Astro.gmst(date))
        return SIMD3(cos(g) * r.x + sin(g) * r.y, -sin(g) * r.x + cos(g) * r.y, r.z)
    }

    /// Latitude and longitude below the satellite (geocentric), and its height above the WGS-84 ellipsoid in km.
    static func subpoint(_ p: SIMD3<Double>) -> (lat: Double, lon: Double, height: Double) {
        let r = (p * p).sum().squareRoot()
        let lat = asin(p.z / r), lon = atan2(p.y, p.x)
        let surface = 6378.137 * (1 - sin(lat) * sin(lat) / 298.257)
        return (lat * 180 / .pi, lon * 180 / .pi, r - surface)
    }

    /// Every three-line set in a Celestrak TLE file.
    static func parse(_ text: String) -> [Orbit] {
        let lines = text.split(whereSeparator: \.isNewline).map(String.init)
        var out: [Orbit] = []
        out.reserveCapacity(lines.count / 3)
        var i = 0
        while i + 2 < lines.count {
            if lines[i + 1].hasPrefix("1 "), lines[i + 2].hasPrefix("2 "), let o = Orbit(name: lines[i], lines[i + 1], lines[i + 2]) {
                out.append(o)
                i += 3
            } else {
                i += 1
            }
        }
        return out
    }
}
