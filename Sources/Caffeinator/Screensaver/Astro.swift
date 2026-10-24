import Foundation

/// Positions of the Sun, Moon and planets, after Paul Schlyter's "How to compute planetary positions".
/// Good to a few arcminutes for centuries either side of 2000, which is plenty for pictures of the sky.
/// Angles are degrees unless a name says radians; distances are AU (the Moon's are Earth radii).
enum Astro {
    struct Equatorial: Equatable { var ra: Double; var dec: Double; var distance: Double }

    enum Body: String, CaseIterable {
        case mercury, venus, mars, jupiter, saturn, uranus, neptune
        var name: String { rawValue.capitalized }
    }

    static func julianDay(_ date: Date) -> Double { date.timeIntervalSince1970 / 86400 + 2_440_587.5 }

    /// Schlyter's day number: days since 1999 Dec 31, 0h UT.
    static func dayNumber(_ date: Date) -> Double { julianDay(date) - 2_451_543.5 }

    static func obliquity(_ d: Double) -> Double { 23.4393 - 3.563e-7 * d }

    /// Greenwich mean sidereal time, degrees.
    static func gmst(_ date: Date) -> Double {
        let d = julianDay(date) - 2_451_545.0
        return norm(280.46061837 + 360.98564736629 * d)
    }

    /// Local sidereal time at an east longitude, degrees.
    static func lst(_ date: Date, longitude: Double) -> Double { norm(gmst(date) + longitude) }

    // MARK: Orbits

    struct Elements { var N, i, w, a, e, M: Double }

    static func elements(_ body: Body, _ d: Double) -> Elements {
        switch body {
        case .mercury: Elements(N: 48.3313 + 3.24587e-5 * d, i: 7.0047 + 5.00e-8 * d, w: 29.1241 + 1.01444e-5 * d,
                                a: 0.387098, e: 0.205635 + 5.59e-10 * d, M: 168.6562 + 4.0923344368 * d)
        case .venus: Elements(N: 76.6799 + 2.46590e-5 * d, i: 3.3946 + 2.75e-8 * d, w: 54.8910 + 1.38374e-5 * d,
                              a: 0.723330, e: 0.006773 - 1.302e-9 * d, M: 48.0052 + 1.6021302244 * d)
        case .mars: Elements(N: 49.5574 + 2.11081e-5 * d, i: 1.8497 - 1.78e-8 * d, w: 286.5016 + 2.92961e-5 * d,
                             a: 1.523688, e: 0.093405 + 2.516e-9 * d, M: 18.6021 + 0.5240207766 * d)
        case .jupiter: Elements(N: 100.4542 + 2.76854e-5 * d, i: 1.3030 - 1.557e-7 * d, w: 273.8777 + 1.64505e-5 * d,
                                a: 5.20256, e: 0.048498 + 4.469e-9 * d, M: 19.8950 + 0.0830853001 * d)
        case .saturn: Elements(N: 113.6634 + 2.38980e-5 * d, i: 2.4886 - 1.081e-7 * d, w: 339.3939 + 2.97661e-5 * d,
                               a: 9.55475, e: 0.055546 - 9.499e-9 * d, M: 316.9670 + 0.0334442282 * d)
        case .uranus: Elements(N: 74.0005 + 1.3978e-5 * d, i: 0.7733 + 1.9e-8 * d, w: 96.6612 + 3.0565e-5 * d,
                               a: 19.18171 - 1.55e-8 * d, e: 0.047318 + 7.45e-9 * d, M: 142.5905 + 0.011725806 * d)
        case .neptune: Elements(N: 131.7806 + 3.0173e-5 * d, i: 1.7700 - 2.55e-7 * d, w: 272.8461 - 6.027e-6 * d,
                                a: 30.05826 + 3.313e-8 * d, e: 0.008606 + 2.15e-9 * d, M: 260.2471 + 0.005995147 * d)
        }
    }

    /// Solves Kepler's equation and returns the position in the ecliptic frame (x toward the vernal equinox).
    static func position(_ el: Elements) -> SIMD3<Double> {
        let M = rad(norm(el.M))
        var E = M + el.e * sin(M) * (1 + el.e * cos(M))
        for _ in 0..<8 { E -= (E - el.e * sin(E) - M) / (1 - el.e * cos(E)) }
        let xv = el.a * (cos(E) - el.e), yv = el.a * sqrt(1 - el.e * el.e) * sin(E)
        let v = atan2(yv, xv), r = hypot(xv, yv)
        let N = rad(el.N), i = rad(el.i), vw = v + rad(el.w)
        return SIMD3(r * (cos(N) * cos(vw) - sin(N) * sin(vw) * cos(i)),
                     r * (sin(N) * cos(vw) + cos(N) * sin(vw) * cos(i)),
                     r * sin(vw) * sin(i))
    }

    /// The Sun as seen from Earth, in ecliptic coordinates (AU). Earth's heliocentric position is the negative.
    static func sunEcliptic(_ date: Date) -> SIMD3<Double> {
        let d = dayNumber(date)
        let w = 282.9404 + 4.70935e-5 * d, e = 0.016709 - 1.151e-9 * d, M = rad(norm(356.0470 + 0.9856002585 * d))
        let E = M + e * sin(M) * (1 + e * cos(M))
        let xv = cos(E) - e, yv = sqrt(1 - e * e) * sin(E)
        let lon = atan2(yv, xv) + rad(w), r = hypot(xv, yv)
        return SIMD3(r * cos(lon), r * sin(lon), 0)
    }

    /// A planet's heliocentric ecliptic position in AU.
    static func heliocentric(_ body: Body, _ date: Date) -> SIMD3<Double> { position(elements(body, dayNumber(date))) }

    static func sun(_ date: Date) -> Equatorial { equatorial(sunEcliptic(date), date) }

    static func planet(_ body: Body, _ date: Date) -> Equatorial {
        equatorial(heliocentric(body, date) + sunEcliptic(date), date)
    }

    /// The Moon, geocentric, with the largest perturbations in longitude and latitude. Distance in Earth radii.
    static func moon(_ date: Date) -> Equatorial {
        let d = dayNumber(date)
        let el = Elements(N: 125.1228 - 0.0529538083 * d, i: 5.1454, w: 318.0634 + 0.1643573223 * d, a: 60.2666, e: 0.054900,
                          M: 115.3654 + 13.0649929509 * d)
        let p = position(el)
        var lon = deg(atan2(p.y, p.x)), lat = deg(atan2(p.z, hypot(p.x, p.y)))
        var r = length(p)
        let Ms = norm(356.0470 + 0.9856002585 * d), Mm = norm(el.M)
        let Ls = norm(Ms + 282.9404 + 4.70935e-5 * d), Lm = norm(el.N + el.w + el.M)
        let D = Lm - Ls, F = Lm - el.N
        lon += -1.274 * sinD(Mm - 2 * D) + 0.658 * sinD(2 * D) - 0.186 * sinD(Ms) - 0.059 * sinD(2 * Mm - 2 * D)
            - 0.057 * sinD(Mm - 2 * D + Ms) + 0.053 * sinD(Mm + 2 * D) + 0.046 * sinD(2 * D - Ms) + 0.041 * sinD(Mm - Ms)
            - 0.035 * sinD(D) - 0.031 * sinD(Mm + Ms) - 0.015 * sinD(2 * F - 2 * D) + 0.011 * sinD(Mm - 4 * D)
        lat += -0.173 * sinD(F - 2 * D) - 0.055 * sinD(Mm - F - 2 * D) - 0.046 * sinD(Mm + F - 2 * D)
            + 0.033 * sinD(F + 2 * D) + 0.017 * sinD(2 * Mm + F)
        r += -0.58 * cosD(Mm - 2 * D) - 0.46 * cosD(2 * D)
        let q = SIMD3(r * cosD(lat) * cosD(lon), r * cosD(lat) * sinD(lon), r * sinD(lat))
        return equatorial(q, date)
    }

    /// Fraction of the Moon's disc that's lit, 0 (new) to 1 (full), and whether it's waxing.
    static func moonPhase(_ date: Date) -> (lit: Double, waxing: Bool) {
        let m = ecliptic(moon(date), date), s = ecliptic(sun(date), date)
        let elongation = norm(m - s)
        return ((1 - cos(rad(elongation))) / 2, elongation < 180)
    }

    /// Ecliptic longitude of an equatorial position, degrees.
    static func ecliptic(_ q: Equatorial, _ date: Date) -> Double {
        let e = rad(obliquity(dayNumber(date))), ra = rad(q.ra), dec = rad(q.dec)
        return norm(deg(atan2(sin(ra) * cos(e) + tan(dec) * sin(e), cos(ra))))
    }

    static func equatorial(_ ecl: SIMD3<Double>, _ date: Date) -> Equatorial {
        let e = rad(obliquity(dayNumber(date)))
        let x = ecl.x, y = ecl.y * cos(e) - ecl.z * sin(e), z = ecl.y * sin(e) + ecl.z * cos(e)
        return Equatorial(ra: norm(deg(atan2(y, x))), dec: deg(atan2(z, hypot(x, y))), distance: length(ecl))
    }

    /// Altitude above the horizon and azimuth (from north through east) for an observer.
    static func horizontal(_ q: Equatorial, _ date: Date, latitude: Double, longitude: Double) -> (alt: Double, az: Double) {
        let ha = rad(lst(date, longitude: longitude) - q.ra), dec = rad(q.dec), lat = rad(latitude)
        let alt = asin(sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(ha))
        let az = atan2(-sin(ha) * cos(dec), cos(lat) * sin(dec) - sin(lat) * cos(dec) * cos(ha))
        return (deg(alt), norm(deg(az)))
    }

    /// The point on Earth with the Sun (or another body) directly overhead: latitude and east longitude.
    static func subpoint(_ q: Equatorial, _ date: Date) -> (lat: Double, lon: Double) {
        var lon = norm(q.ra - gmst(date))
        if lon > 180 { lon -= 360 }
        return (q.dec, lon)
    }

    // MARK: Helpers

    static func norm(_ x: Double) -> Double { let r = x.truncatingRemainder(dividingBy: 360); return r < 0 ? r + 360 : r }
    static func rad(_ x: Double) -> Double { x * .pi / 180 }
    static func deg(_ x: Double) -> Double { x * 180 / .pi }
    private static func sinD(_ x: Double) -> Double { sin(rad(x)) }
    private static func cosD(_ x: Double) -> Double { cos(rad(x)) }
    private static func length(_ v: SIMD3<Double>) -> Double { (v * v).sum().squareRoot() }
}
