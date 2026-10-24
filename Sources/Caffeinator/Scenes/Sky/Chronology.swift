import Foundation
import simd

/// Clocks and calendars beyond the one on the menu bar: Julian days, sidereal time, atomic and GPS time, Mars time
/// after NASA's Mars24 algorithm (Allison and McEwen 2000), and how far away Voyager 1 is.
enum Chronology {
    /// TAI − UTC since 1 January 2017; there has been no leap second since.
    static let leapSeconds = 37.0
    /// GPS time runs 19 seconds behind TAI, so 18 ahead of UTC.
    static let gpsOffset = leapSeconds - 19

    static func unix(_ date: Date) -> Double { date.timeIntervalSince1970 }
    static func julianDay(_ date: Date) -> Double { date.timeIntervalSince1970 / 86400 + 2_440_587.5 }
    static func modifiedJulianDate(_ date: Date) -> Double { julianDay(date) - 2_400_000.5 }

    /// Terrestrial Time as a Julian day: TAI + 32.184 s.
    static func julianDayTT(_ date: Date) -> Double { julianDay(date) + (leapSeconds + 32.184) / 86400 }

    // MARK: Mars

    /// Mars Sol Date: sols since 29 December 1873, the Martian counterpart of the Julian day (Mars24, eq. C-2).
    static func marsSolDate(_ date: Date) -> Double {
        (julianDayTT(date) - 2_451_545.0 - 4.5) / 1.0274912517 + 44_796.0 - 0.0009626
    }

    /// Coordinated Mars Time, the mean solar time at Mars's prime meridian, in hours.
    static func marsCoordinatedTime(_ date: Date) -> Double { (24 * marsSolDate(date)).truncatingRemainder(dividingBy: 24) }

    /// Local mean solar time on Mars at an east longitude, in hours.
    static func marsLocalTime(_ date: Date, eastLongitude: Double) -> Double {
        let h = (marsCoordinatedTime(date) + eastLongitude / 15).truncatingRemainder(dividingBy: 24)
        return h < 0 ? h + 24 : h
    }

    /// Perseverance at Jezero crater.
    static let jezero = (latitude: 18.4447, longitude: 77.4508)
    /// Touchdown on 18 February 2021 at 20:43:49 UTC (spacecraft time), which began mission sol 0.
    static let perseveranceLanding = Date(timeIntervalSince1970: 1_613_681_029)

    /// The mission sol at Jezero: whole local sols since the landing sol.
    static func perseveranceSol(_ date: Date) -> Int {
        func local(_ d: Date) -> Double { marsSolDate(d) + jezero.longitude / 360 }
        return Int(floor(local(date))) - Int(floor(local(perseveranceLanding)))
    }

    // MARK: Calendar

    static func dayOfYear(_ date: Date, in zone: TimeZone = .current) -> Int {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = zone
        return c.ordinality(of: .day, in: .year, for: date) ?? 0
    }

    /// The ISO 8601 week date, such as "2026-W43-6".
    static func isoWeek(_ date: Date, in zone: TimeZone = .current) -> (year: Int, week: Int, day: Int) {
        var c = Calendar(identifier: .iso8601)
        c.timeZone = zone
        let p = c.dateComponents([.yearForWeekOfYear, .weekOfYear, .weekday], from: date)
        let weekday = ((p.weekday ?? 2) + 5) % 7 + 1
        return (p.yearForWeekOfYear ?? 0, p.weekOfYear ?? 0, weekday)
    }

    // MARK: Voyager 1

    /// Voyager 1's heliocentric position and velocity (ecliptic J2000, AU and AU per day) from JPL Horizons for
    /// 1 January 2026, 0h TDB. It coasts in a nearly straight line now, so a linear model stays good to well under
    /// a light-second over several years.
    static let voyagerEpoch = Date(timeIntervalSince1970: 1_767_225_600 - 69.184)
    static let voyagerPosition = SIMD3<Double>(-31.83641396068854, -134.6784849020786, 97.44480620314309)
    static let voyagerVelocity = SIMD3<Double>(-1.197151941601862e-3, -7.861077920510877e-3, 5.680005847953139e-3)
    static let kilometresPerAU = 149_597_870.7
    static let secondsPerAU = 499.004784

    /// Distance in AU from Earth now to where Voyager 1 was when the light arriving now left it, as JPL quotes it.
    static func voyagerDistance(_ date: Date) -> Double {
        // Earth from Astro's Sun, which is referred to the equinox of date: turn it back to J2000.
        var earth = -Astro.sunEcliptic(date)
        let precession = -1.396971 * (julianDay(date) - 2_451_545.0) / 36525 * .pi / 180
        earth = SIMD3(earth.x * cos(precession) - earth.y * sin(precession), earth.x * sin(precession) + earth.y * cos(precession), earth.z)
        var distance = 0.0
        for _ in 0..<3 {
            let days = (date.timeIntervalSince(voyagerEpoch) - distance * secondsPerAU) / 86400
            distance = simd_length(voyagerPosition + voyagerVelocity * days - earth)
        }
        return distance
    }

    /// One-way light time to Voyager 1, in seconds.
    static func voyagerLightTime(_ date: Date) -> Double { voyagerDistance(date) * secondsPerAU }

    /// When Voyager 1's signals first take a full day to reach Earth: around 19 November 2026.
    static let oneLightDay: Date = {
        var t = Date(timeIntervalSince1970: 1_790_000_000)
        while voyagerLightTime(t) < 86400, t.timeIntervalSince1970 < 1_830_000_000 { t += 3600 }
        return t
    }()

    // MARK: Formatting

    /// Hours as "14:03:20".
    static func clock(_ hours: Double) -> String {
        var s = Int(floor(hours * 3600)) % 86400
        if s < 0 { s += 86400 }
        return String(format: "%02d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
    }

    /// A date's time of day in a zone as "14:03:20".
    static func clock(_ date: Date, in zone: TimeZone) -> String {
        clock((date.timeIntervalSince1970 + Double(zone.secondsFromGMT(for: date))) / 3600)
    }

    /// "2,461,337.98734".
    static func grouped(_ x: Double, decimals: Int) -> String {
        let whole = floor(x)
        let fraction = String(format: "%.\(decimals)f", x - whole)
        if fraction.hasPrefix("1") { return grouped(whole + 1, decimals: decimals) }
        return SkyKit.grouped(Int(whole)) + (decimals > 0 ? String(fraction.dropFirst()) : "")
    }

    /// "23 hours 56 minutes 34 seconds", or the short form "23 h 56 min 34 s".
    static func duration(_ seconds: Double, short: Bool = false) -> String {
        let s = Int(seconds.rounded())
        let (h, m, r) = (s / 3600, s / 60 % 60, s % 60)
        if short { return "\(h) h \(m) min \(r) s" }
        return "\(h) hours \(m) minutes \(r) seconds"
    }
}
