import Foundation
import simd

/// The Moon as seen from Earth, after Jean Meeus's "Astronomical Algorithms" (2nd edition): its position from the main
/// terms of ELP-2000/82 (chapter 47, about 10″), the Sun (chapter 25), nutation (chapter 22), the phase and bright limb
/// (chapter 48), and the optical and physical libration, axis and selenographic Sun (chapter 53). Angles are degrees.
enum Lunar {
    /// TT − UTC: 37 leap seconds plus 32.184 s.
    static let ttMinusUTC = 69.184
    static let radius = 1737.4
    static let synodicMonth = 29.530589

    struct Position {
        /// Geocentric ecliptic longitude and latitude of date (without nutation), and distance in km.
        var longitude: Double
        var latitude: Double
        var distance: Double
    }

    /// Everything the scene shows, for one moment and one observer.
    struct View {
        let date: Date
        /// Topocentric right ascension and declination, equinox J2000, degrees.
        let ra: Double
        let dec: Double
        /// Distance from the observer, km.
        let distance: Double
        /// Fraction of the disc that's lit, phase angle, and whether the Moon is waxing.
        let illuminated: Double
        let phaseAngle: Double
        let waxing: Bool
        /// Position angle of the midpoint of the bright limb, from celestial north through east.
        let brightLimb: Double
        /// Libration in longitude and latitude: the selenographic point at the centre of the disc.
        let libration: SIMD2<Double>
        /// Position angle of the Moon's north pole, from celestial north through east.
        let axis: Double
        /// Selenographic longitude and latitude of the point with the Sun overhead.
        let subsolar: SIMD2<Double>
        /// Days since the last new Moon.
        let age: Double
        /// Body-fixed unit vectors (x toward 0° longitude, z north) of the observer's line of sight back toward them,
        /// sky west (screen right) and celestial north (screen up); and toward the Sun.
        let toObserver: SIMD3<Double>
        let west: SIMD3<Double>
        let north: SIMD3<Double>
        let toSun: SIMD3<Double>

        /// Selenographic colongitude of the Sun: the longitude of the morning terminator, 0° at first quarter.
        var colongitude: Double { norm(90 - subsolar.x) }
        /// Apparent angular diameter, degrees.
        var diameter: Double { 2 * deg(asin(Lunar.radius / distance)) }
    }

    // MARK: Time

    static func julianDay(_ date: Date) -> Double { date.timeIntervalSince1970 / 86400 + 2_440_587.5 }

    /// Julian centuries of Terrestrial Time since J2000.
    static func centuries(_ date: Date) -> Double { (julianDay(date) + ttMinusUTC / 86400 - 2_451_545.0) / 36525 }

    // MARK: The Moon (chapter 47)

    struct Arguments {
        var L, D, M, Mp, F, E, T: Double
        init(_ T: Double) {
            self.T = T
            L = 218.3164477 + 481267.88123421 * T - 0.0015786 * T * T + T * T * T / 538841 - T * T * T * T / 65_194_000
            D = 297.8501921 + 445267.1114034 * T - 0.0018819 * T * T + T * T * T / 545868 - T * T * T * T / 113_065_000
            M = 357.5291092 + 35999.0502909 * T - 0.0001536 * T * T + T * T * T / 24_490_000
            Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T * T + T * T * T / 69699 - T * T * T * T / 14_712_000
            F = 93.2720950 + 483202.0175233 * T - 0.0036539 * T * T - T * T * T / 3_526_000 + T * T * T * T / 863_310_000
            E = 1 - 0.002516 * T - 0.0000074 * T * T
        }
        /// Longitude of the ascending node of the Moon's mean orbit.
        var node: Double { 125.0445479 - 1934.1362891 * T + 0.0020754 * T * T + T * T * T / 467441 - T * T * T * T / 60_616_000 }
    }

    /// Table 47.A: multiples of D, M, M′, F, then Σl (10⁻⁶ °) and Σr (m).
    private static let longitudeTerms: [(Int8, Int8, Int8, Int8, Int32, Int32)] = [
        (0, 0, 1, 0, 6288774, -20905355), (2, 0, -1, 0, 1274027, -3699111), (2, 0, 0, 0, 658314, -2955968),
        (0, 0, 2, 0, 213618, -569925), (0, 1, 0, 0, -185116, 48888), (0, 0, 0, 2, -114332, -3149), (2, 0, -2, 0, 58793, 246158),
        (2, -1, -1, 0, 57066, -152138), (2, 0, 1, 0, 53322, -170733), (2, -1, 0, 0, 45758, -204586), (0, 1, -1, 0, -40923, -129620),
        (1, 0, 0, 0, -34720, 108743), (0, 1, 1, 0, -30383, 104755), (2, 0, 0, -2, 15327, 10321), (0, 0, 1, 2, -12528, 0),
        (0, 0, 1, -2, 10980, 79661), (4, 0, -1, 0, 10675, -34782), (0, 0, 3, 0, 10034, -23210), (4, 0, -2, 0, 8548, -21636),
        (2, 1, -1, 0, -7888, 24208), (2, 1, 0, 0, -6766, 30824), (1, 0, -1, 0, -5163, -8379), (1, 1, 0, 0, 4987, -16675),
        (2, -1, 1, 0, 4036, -12831), (2, 0, 2, 0, 3994, -10445), (4, 0, 0, 0, 3861, -11650), (2, 0, -3, 0, 3665, 14403),
        (0, 1, -2, 0, -2689, -7003), (2, 0, -1, 2, -2602, 0), (2, -1, -2, 0, 2390, 10056), (1, 0, 1, 0, -2348, 6322),
        (2, -2, 0, 0, 2236, -9884), (0, 1, 2, 0, -2120, 5751), (0, 2, 0, 0, -2069, 0), (2, -2, -1, 0, 2048, -4950),
        (2, 0, 1, -2, -1773, 4130), (2, 0, 0, 2, -1595, 0), (4, -1, -1, 0, 1215, -3958), (0, 0, 2, 2, -1110, 0),
        (3, 0, -1, 0, -892, 3258), (2, 1, 1, 0, -810, 2616), (4, -1, -2, 0, 759, -1897), (0, 2, -1, 0, -713, -2117),
        (2, 2, -1, 0, -700, 2354), (2, 1, -2, 0, 691, 0), (2, -1, 0, -2, 596, 0), (4, 0, 1, 0, 549, -1423), (0, 0, 4, 0, 537, -1117),
        (4, -1, 0, 0, 520, -1571), (1, 0, -2, 0, -487, -1739), (2, 1, 0, -2, -399, 0), (0, 0, 2, -2, -381, -4421),
        (1, 1, 1, 0, 351, 0), (3, 0, -2, 0, -340, 0), (4, 0, -3, 0, 330, 0), (2, -1, 2, 0, 327, 0), (0, 2, 1, 0, -323, 1165),
        (1, 1, -1, 0, 299, 0), (2, 0, 3, 0, 294, 0), (2, 0, -1, -2, 0, 8752),
    ]

    /// Table 47.B: multiples of D, M, M′, F, then Σb (10⁻⁶ °).
    private static let latitudeTerms: [(Int8, Int8, Int8, Int8, Int32)] = [
        (0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, 1, -1, 277693), (2, 0, 0, -1, 173237), (2, 0, -1, 1, 55413),
        (2, 0, -1, -1, 46271), (2, 0, 0, 1, 32573), (0, 0, 2, 1, 17198), (2, 0, 1, -1, 9266), (0, 0, 2, -1, 8822),
        (2, -1, 0, -1, 8216), (2, 0, -2, -1, 4324), (2, 0, 1, 1, 4200), (2, 1, 0, -1, -3359), (2, -1, -1, 1, 2463),
        (2, -1, 0, 1, 2211), (2, -1, -1, -1, 2065), (0, 1, -1, -1, -1870), (4, 0, -1, -1, 1828), (0, 1, 0, 1, -1794),
        (0, 0, 0, 3, -1749), (0, 1, -1, 1, -1565), (1, 0, 0, 1, -1491), (0, 1, 1, 1, -1475), (0, 1, 1, -1, -1410),
        (0, 1, 0, -1, -1344), (1, 0, 0, -1, -1335), (0, 0, 3, 1, 1107), (4, 0, 0, -1, 1021), (4, 0, -1, 1, 833),
        (0, 0, 1, -3, 777), (4, 0, -2, 1, 671), (2, 0, 0, -3, 607), (2, 0, 2, -1, 596), (2, -1, 1, -1, 491),
        (2, 0, -2, 1, -451), (0, 0, 3, -1, 439), (2, 0, 2, 1, 422), (2, 0, -3, -1, 421), (2, 1, -1, 1, -366),
        (2, 1, 0, 1, -351), (4, 0, 0, 1, 331), (2, -1, 1, 1, 315), (2, -2, 0, -1, 302), (0, 0, 1, 3, -283),
        (2, 1, 1, -1, -229), (1, 1, 0, -1, 223), (1, 1, 0, 1, 223), (0, 1, -2, -1, -220), (2, 1, -1, -1, -220),
        (1, 0, 1, 1, -185), (2, -1, -2, -1, 181), (0, 1, 2, 1, -177), (4, 0, -2, -1, 176), (4, -1, -1, -1, 166),
        (1, 0, 1, -1, -164), (4, 0, 1, -1, 132), (1, 0, -1, -1, -119), (4, -1, 0, -1, 115), (2, -2, 0, 1, 107),
    ]

    /// Geocentric position of the Moon referred to the mean equinox of date.
    static func moon(_ T: Double) -> Position {
        let a = Arguments(T)
        var l = 0.0, r = 0.0, b = 0.0
        for (d, m, mp, f, sl, sr) in longitudeTerms {
            let arg = rad(Double(d) * a.D + Double(m) * a.M + Double(mp) * a.Mp + Double(f) * a.F)
            let e = abs(m) == 1 ? a.E : abs(m) == 2 ? a.E * a.E : 1
            l += Double(sl) * e * sin(arg)
            r += Double(sr) * e * cos(arg)
        }
        for (d, m, mp, f, sb) in latitudeTerms {
            let arg = rad(Double(d) * a.D + Double(m) * a.M + Double(mp) * a.Mp + Double(f) * a.F)
            let e = abs(m) == 1 ? a.E : abs(m) == 2 ? a.E * a.E : 1
            b += Double(sb) * e * sin(arg)
        }
        let A1 = 119.75 + 131.849 * T, A2 = 53.09 + 479264.290 * T, A3 = 313.45 + 481266.484 * T
        l += 3958 * sinD(A1) + 1962 * sinD(a.L - a.F) + 318 * sinD(A2)
        b += -2235 * sinD(a.L) + 382 * sinD(A3) + 175 * sinD(A1 - a.F) + 175 * sinD(A1 + a.F) + 127 * sinD(a.L - a.Mp)
            - 115 * sinD(a.L + a.Mp)
        return Position(longitude: norm(a.L + l / 1e6), latitude: b / 1e6, distance: 385000.56 + r / 1000)
    }

    // MARK: The Sun (chapter 25) and nutation (chapter 22)

    /// The Sun's geometric longitude (mean equinox of date) and distance in AU.
    static func sun(_ T: Double) -> (longitude: Double, distance: Double) {
        let L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
        let M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
        let e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T
        let C = (1.914602 - 0.004817 * T - 0.000014 * T * T) * sinD(M) + (0.019993 - 0.000101 * T) * sinD(2 * M) + 0.000289 * sinD(3 * M)
        let R = 1.000001018 * (1 - e * e) / (1 + e * cosD(M + C))
        // Aberration, which belongs to the apparent position but not to nutation.
        return (norm(L0 + C - 0.00569), R)
    }

    /// Nutation in longitude and obliquity, and the mean obliquity, degrees.
    static func nutation(_ T: Double) -> (dPsi: Double, dEps: Double, eps0: Double) {
        let omega = 125.04452 - 1934.136261 * T, L = 280.4665 + 36000.7698 * T, Lp = 218.3165 + 481267.8813 * T
        let dPsi = (-17.20 * sinD(omega) - 1.32 * sinD(2 * L) - 0.23 * sinD(2 * Lp) + 0.21 * sinD(2 * omega)) / 3600
        let dEps = (9.20 * cosD(omega) + 0.57 * cosD(2 * L) + 0.10 * cosD(2 * Lp) - 0.09 * cosD(2 * omega)) / 3600
        let eps0 = 23.4392911 - (46.8150 * T + 0.00059 * T * T - 0.001813 * T * T * T) / 3600
        return (dPsi, dEps, eps0)
    }

    // MARK: Libration (chapter 53)

    /// Inclination of the Moon's mean equator to the ecliptic.
    static let inclination = 1.54242

    /// The selenographic longitude and latitude of the point whose zenith is in the direction of ecliptic longitude
    /// `lambda` (of date, apparent) and latitude `beta`, and the angle A used by the position angle, all including
    /// physical libration.
    static func libration(_ T: Double, lambda: Double, beta: Double) -> (l: Double, b: Double, A: Double, rho: Double, sigma: Double) {
        let a = Arguments(T)
        let omega = a.node, dPsi = nutation(T).dPsi
        let W = lambda - dPsi - omega
        let I = inclination
        let A = deg(atan2(sinD(W) * cosD(beta) * cosD(I) - sinD(beta) * sinD(I), cosD(W) * cosD(beta)))
        let l1 = norm180(A - a.F)
        let b1 = deg(asin(-sinD(W) * cosD(beta) * sinD(I) - sinD(beta) * cosD(I)))
        let K1 = 119.75 + 131.849 * T, K2 = 72.56 + 20.186 * T
        let (D, M, Mp, F, E) = (a.D, a.M, a.Mp, a.F, a.E)
        let rho = -0.02752 * cosD(Mp) - 0.02245 * sinD(F) + 0.00684 * cosD(Mp - 2 * F) - 0.00293 * cosD(2 * F)
            - 0.00085 * cosD(2 * F - 2 * D) - 0.00054 * cosD(Mp - 2 * D) - 0.00020 * sinD(Mp + F) - 0.00020 * cosD(Mp + 2 * F)
            - 0.00020 * cosD(Mp - F) + 0.00014 * cosD(Mp + 2 * F - 2 * D)
        let sigma = -0.02816 * sinD(Mp) + 0.02244 * cosD(F) - 0.00682 * sinD(Mp - 2 * F) - 0.00279 * sinD(2 * F)
            - 0.00083 * sinD(2 * F - 2 * D) + 0.00069 * sinD(Mp - 2 * D) + 0.00040 * cosD(Mp + F) - 0.00025 * sinD(2 * Mp)
            - 0.00023 * sinD(Mp + 2 * F) + 0.00020 * cosD(Mp - F) + 0.00019 * sinD(Mp - F) + 0.00013 * sinD(Mp + 2 * F - 2 * D)
            - 0.00010 * cosD(Mp - 3 * F)
        let tau = 0.02520 * E * sinD(M) + 0.00473 * sinD(2 * Mp - 2 * F) - 0.00467 * sinD(Mp) + 0.00396 * sinD(K1)
            + 0.00276 * sinD(2 * Mp - 2 * D) + 0.00196 * sinD(omega) - 0.00183 * cosD(Mp - F) + 0.00115 * sinD(Mp - 2 * D)
            - 0.00096 * sinD(Mp - D) + 0.00046 * sinD(2 * F - 2 * D) - 0.00039 * sinD(Mp - F) - 0.00032 * sinD(Mp - M - D)
            + 0.00027 * sinD(2 * Mp - M - 2 * D) + 0.00023 * sinD(K2) - 0.00014 * sinD(2 * D) + 0.00014 * cosD(2 * Mp - 2 * F)
            - 0.00012 * sinD(Mp - 2 * F) - 0.00012 * sinD(2 * Mp) + 0.00011 * sinD(2 * Mp - 2 * M - 2 * D)
        let l2 = -tau + (rho * cosD(A) + sigma * sinD(A)) * tanD(b1)
        let b2 = sigma * cosD(A) - rho * sinD(A)
        return (norm180(l1 + l2), b1 + b2, A, rho, sigma)
    }

    /// Position angle of the Moon's axis (equation 53.3) for the Moon at apparent right ascension `ra` and total
    /// libration in latitude `b`.
    static func axis(_ T: Double, ra: Double, b: Double, rho: Double, sigma: Double) -> Double {
        let n = nutation(T), I = inclination
        let eps = n.eps0 + n.dEps
        let V = Arguments(T).node + n.dPsi + sigma / sinD(I)
        let X = sinD(I + rho) * sinD(V)
        let Y = sinD(I + rho) * cosD(V) * cosD(eps) - cosD(I + rho) * sinD(eps)
        let omega = deg(atan2(X, Y))
        return norm180(deg(asin(min(max(hypot(X, Y) * cosD(ra - omega) / cosD(b), -1), 1))))
    }

    // MARK: The whole picture

    /// The Moon at `date` for an observer at `latitude`, `longitude` (degrees, east positive), or from Earth's centre.
    static func view(_ date: Date, latitude: Double? = nil, longitude: Double? = nil) -> View {
        let T = centuries(date)
        let m = moon(T), s = sun(T), n = nutation(T)
        let eps = n.eps0 + n.dEps
        // Apparent geocentric positions.
        let lambda = m.longitude + n.dPsi, beta = m.latitude
        let sunLambda = s.longitude + n.dPsi
        let auKm = 149_597_870.7
        let moonEq = equatorial(lambda, beta, eps)
        let sunEq = equatorial(sunLambda, 0, eps)

        // Libration, axis and the selenographic Sun (geocentric; the topocentric shift follows below).
        let lib = libration(T, lambda: lambda, beta: beta)
        let P = axis(T, ra: moonEq.ra, b: lib.b, rho: lib.rho, sigma: lib.sigma)
        let ratio = m.distance / (s.distance * auKm)
        let lambdaH = sunLambda + 180 + ratio * 57.296 * cosD(beta) * sinD(sunLambda - lambda)
        let sol = libration(T, lambda: lambdaH, beta: ratio * beta)

        // The Moon's orientation: body-fixed axes for the geocentric line of sight, sky west and celestial north.
        let mHat = unit(moonEq.ra, moonEq.dec)
        let bodyFromEq = frame(sightLine: mHat, toObserver: unit(lib.l, lib.b), axis: P)

        // The observer's line of sight, displaced by parallax.
        var sight = mHat * m.distance
        if let latitude, let longitude {
            let u = atan(0.99664719 * tanD(latitude))
            let lst = rad(Astro.gmst(date) + longitude)
            let r = 6378.14 * SIMD3(cos(u) * cos(lst), cos(u) * sin(lst), 0.99664719 * sin(u))
            sight -= r
        }
        let distance = simd_length(sight)
        let look = sight / distance
        let east = simd_normalize(simd_cross(SIMD3(0, 0, 1), look)), north = simd_cross(look, east)
        let toObserver = simd_normalize(bodyFromEq * -look)
        let west = simd_normalize(bodyFromEq * -east), up = simd_normalize(bodyFromEq * north)

        // Phase (chapter 48) and the bright limb.
        let psi = acos(cosD(beta) * cosD(lambda - sunLambda))
        let R = s.distance * auKm
        let i = deg(atan2(R * sin(psi), m.distance - R * cos(psi)))
        let chi = deg(atan2(cosD(sunEq.dec) * sinD(sunEq.ra - moonEq.ra),
                            sinD(sunEq.dec) * cosD(moonEq.dec) - cosD(sunEq.dec) * sinD(moonEq.dec) * cosD(sunEq.ra - moonEq.ra)))
        let elongation = norm(lambda - sunLambda)

        let topo = precessToJ2000(look, T)
        let toSun = unit(sol.l, sol.b)
        return View(date: date, ra: norm(deg(atan2(topo.y, topo.x))), dec: deg(asin(topo.z)), distance: distance,
                    illuminated: (1 + cosD(i)) / 2, phaseAngle: i, waxing: elongation < 180, brightLimb: norm(chi),
                    libration: SIMD2(deg(atan2(toObserver.y, toObserver.x)), deg(asin(toObserver.z))), axis: norm(P),
                    subsolar: SIMD2(norm180(sol.l), sol.b), age: age(date), toObserver: toObserver, west: west, north: up, toSun: toSun)
    }

    /// The rotation from equatorial vectors to the Moon's body frame, from the geocentric view: the body vector
    /// `toObserver` points back along `sightLine`, and the Moon's pole stands at position angle `axis` on the sky.
    static func frame(sightLine m: SIMD3<Double>, toObserver e: SIMD3<Double>, axis P: Double) -> simd_double3x3 {
        let eastEq = simd_normalize(simd_cross(SIMD3(0, 0, 1), m)), northEq = simd_cross(m, eastEq)
        // Body frame: projected pole a, and b = e × a. Screen right (sky west) and up (north) follow from P.
        let a = simd_normalize(SIMD3(0, 0, 1) - e * e.z), b = simd_cross(e, a)
        let right = -sinD(P) * a - cosD(P) * b, upward = cosD(P) * a - sinD(P) * b
        let body = simd_double3x3(columns: (e, right, upward))
        let eq = simd_double3x3(columns: (-m, -eastEq, northEq))
        return body * eq.transpose
    }

    /// Days since the most recent new Moon, found by stepping back along the Moon's elongation from the Sun.
    static func age(_ date: Date) -> Double {
        func elongation(_ d: Date) -> Double {
            let T = centuries(d)
            return norm(moon(T).longitude - sun(T).longitude)
        }
        var t = date.addingTimeInterval(-elongation(date) / 360 * synodicMonth * 86400)
        for _ in 0..<4 {
            var e = elongation(t)
            if e > 180 { e -= 360 }
            t = t.addingTimeInterval(-e / 12.19 * 86400)
        }
        return date.timeIntervalSince(t) / 86400
    }

    /// Equatorial coordinates of an ecliptic position.
    static func equatorial(_ lambda: Double, _ beta: Double, _ eps: Double) -> (ra: Double, dec: Double) {
        let ra = deg(atan2(sinD(lambda) * cosD(eps) - tanD(beta) * sinD(eps), cosD(lambda)))
        let dec = deg(asin(sinD(beta) * cosD(eps) + cosD(beta) * sinD(eps) * sinD(lambda)))
        return (norm(ra), dec)
    }

    /// Rotates a direction referred to the equinox of date back to J2000 (Meeus 21.2–21.4).
    static func precessToJ2000(_ v: SIMD3<Double>, _ T: Double) -> SIMD3<Double> {
        let zeta = rad((2306.2181 * T + 0.30188 * T * T + 0.017998 * T * T * T) / 3600)
        let z = rad((2306.2181 * T + 1.09468 * T * T + 0.018203 * T * T * T) / 3600)
        let theta = rad((2004.3109 * T - 0.42665 * T * T - 0.041833 * T * T * T) / 3600)
        func rz(_ a: Double) -> simd_double3x3 { simd_double3x3(rows: [[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]]) }
        func ry(_ a: Double) -> simd_double3x3 { simd_double3x3(rows: [[cos(a), 0, -sin(a)], [0, 1, 0], [sin(a), 0, cos(a)]]) }
        let toDate = rz(z) * ry(theta) * rz(zeta)
        return toDate.transpose * v
    }

    static func unit(_ lon: Double, _ lat: Double) -> SIMD3<Double> { SIMD3(cosD(lat) * cosD(lon), cosD(lat) * sinD(lon), sinD(lat)) }

    // MARK: Helpers

    static func norm(_ x: Double) -> Double { let r = x.truncatingRemainder(dividingBy: 360); return r < 0 ? r + 360 : r }
    static func norm180(_ x: Double) -> Double { let r = norm(x); return r > 180 ? r - 360 : r }
    static func rad(_ x: Double) -> Double { x * .pi / 180 }
    static func deg(_ x: Double) -> Double { x * 180 / .pi }
    private static func sinD(_ x: Double) -> Double { sin(rad(x)) }
    private static func cosD(_ x: Double) -> Double { cos(rad(x)) }
    private static func tanD(_ x: Double) -> Double { tan(rad(x)) }
}
