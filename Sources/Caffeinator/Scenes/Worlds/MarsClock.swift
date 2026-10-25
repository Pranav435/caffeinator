import Foundation

/// Where the Sun stands over a place on Mars, after Allison and McEwen (2000), "A post-Pathfinder evaluation of
/// areocentric solar coordinates with improved timing recipes for Mars seasonal/diurnal climate studies" (NASA's Mars24).
enum MarsSun {
    struct Position {
        /// Areocentric solar longitude, degrees: 0 at northern spring, 90 at northern summer.
        var ls: Double
        /// Equation of time, degrees (apparent minus mean solar time).
        var equationOfTime: Double
        /// Solar declination, degrees.
        var declination: Double
        /// Local true solar time at the place, hours 0..<24.
        var localTime: Double
        /// Sun's elevation above the horizon and azimuth clockwise from north, degrees.
        var elevation: Double
        var azimuth: Double
    }

    private static func rad(_ x: Double) -> Double { x * .pi / 180 }
    private static func deg(_ x: Double) -> Double { x * 180 / .pi }
    private static func norm(_ x: Double) -> Double { let r = x.truncatingRemainder(dividingBy: 360); return r < 0 ? r + 360 : r }

    /// Areocentric solar longitude and equation of time (degrees) for a date.
    static func seasonal(_ date: Date) -> (ls: Double, eot: Double, declination: Double) {
        let jd = date.timeIntervalSince1970 / 86400 + 2_440_587.5 + (37 + 32.184) / 86400
        let dt = jd - 2_451_545.0
        let M = 19.3871 + 0.52402073 * dt
        let alphaFMS = 270.3871 + 0.524038496 * dt
        let A = [0.0071, 0.0057, 0.0039, 0.0037, 0.0021, 0.0020, 0.0018]
        let tau = [2.2353, 2.7543, 1.1177, 15.7866, 2.1354, 2.4694, 32.8493]
        let phi = [49.409, 168.173, 191.837, 21.736, 15.704, 95.528, 49.095]
        var pbs = 0.0
        for i in 0..<7 { pbs += A[i] * cos(rad(0.985626 * dt / tau[i] + phi[i])) }
        let nuMinusM = (10.691 + 3.0e-7 * dt) * sin(rad(M)) + 0.623 * sin(rad(2 * M)) + 0.050 * sin(rad(3 * M))
            + 0.005 * sin(rad(4 * M)) + 0.0005 * sin(rad(5 * M)) + pbs
        let ls = norm(alphaFMS + nuMinusM)
        let eot = 2.861 * sin(rad(2 * ls)) - 0.071 * sin(rad(4 * ls)) + 0.002 * sin(rad(6 * ls)) - nuMinusM
        let delta = deg(asin(0.42565 * sin(rad(ls)))) + 0.25 * sin(rad(ls))
        return (ls, eot, delta)
    }

    /// The Sun over latitude `lat` and east longitude `lon` (degrees) at `date`.
    static func position(_ date: Date, lat: Double, lon: Double) -> Position {
        let s = seasonal(date)
        let mtc = Chronology.marsCoordinatedTime(date)
        var ltst = (mtc + lon / 15 + s.eot / 15).truncatingRemainder(dividingBy: 24)
        if ltst < 0 { ltst += 24 }
        let h = rad((ltst - 12) * 15)
        let phi = rad(lat), delta = rad(s.declination)
        let sinEl = sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(h)
        let az = atan2(sin(h), cos(h) * sin(phi) - tan(delta) * cos(phi)) + .pi
        return Position(ls: s.ls, equationOfTime: s.eot, declination: s.declination, localTime: ltst,
                        elevation: deg(asin(max(-1, min(1, sinEl)))), azimuth: norm(deg(az)))
    }

    /// "14:32" for a time in hours.
    static func clock(_ hours: Double) -> String {
        let minutes = Int((hours * 60).rounded()) % 1440
        return String(format: "%02d:%02d", minutes / 60, minutes % 60)
    }
}
