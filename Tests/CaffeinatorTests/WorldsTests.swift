import Foundation
import simd
import Testing
@testable import Caffeinator

struct WorldsTests {
    func utc(_ s: String) -> Date { ISO8601DateFormatter().date(from: s)! }

    /// 1992 April 12, 0h TD, the date of Meeus's worked examples 47.a, 48.a and 53.a.
    let meeusT = -0.077221081451
    var meeusDate: Date { Date(timeIntervalSince1970: (2_448_724.5 - 2_440_587.5) * 86400 - Lunar.ttMinusUTC) }

    // MARK: The Moon

    @Test func moonPosition() {
        let m = Lunar.moon(meeusT)
        #expect(abs(m.longitude - 133.162655) < 0.0005)
        #expect(abs(m.latitude - -3.229126) < 0.0005)
        #expect(abs(m.distance - 368409.7) < 1)
    }

    @Test func sunPosition() {
        // Example 25.a: 1992 October 13, 0h TD. True longitude 199.90988°, less 20.5″ of aberration; R = 0.99766 AU.
        let s = Lunar.sun(-0.072183436)
        #expect(abs(s.longitude - (199.90988 - 0.00569)) < 0.001)
        #expect(abs(s.distance - 0.99766) < 0.00002)
    }

    @Test func libration() {
        // Example 53.a: l = −1.23°, b = +4.20°, P = 15.08°.
        let lib = Lunar.libration(meeusT, lambda: 133.167269, beta: -3.229127)
        #expect(abs(lib.l - -1.23) < 0.01)
        #expect(abs(lib.b - 4.20) < 0.01)
        let P = Lunar.axis(meeusT, ra: 134.688470, b: lib.b, rho: lib.rho, sigma: lib.sigma)
        #expect(abs(P - 15.08) < 0.01)
    }

    @Test func phaseAndBrightLimb() {
        // Example 48.a: k = 0.6786, χ = 285.0°; and the geocentric libration matches 53.a.
        let v = Lunar.view(meeusDate)
        #expect(abs(v.illuminated - 0.6786) < 0.0005)
        #expect(abs(v.brightLimb - 285.0) < 0.1)
        #expect(abs(v.libration.x - -1.23) < 0.02 && abs(v.libration.y - 4.20) < 0.02)
        #expect(abs(v.axis - 15.08) < 0.02)
        #expect(v.waxing)
    }

    @Test(arguments: ["2026-10-25T21:00:00Z", "2026-11-02T03:00:00Z", "2027-01-09T12:00:00Z", "2026-07-21T00:00:00Z"])
    func renderedLimbMatchesAlmanac(_ when: String) {
        // The picture is built from the body-frame vectors; the Sun's direction projected on the sky must land on the
        // bright limb Meeus computes from right ascensions, and the pole on the axis angle.
        let v = Lunar.view(utc(when))
        let pa = Lunar.norm(Lunar.deg(atan2(-simd_dot(v.toSun, v.west), simd_dot(v.toSun, v.north))))
        #expect(abs(Lunar.norm180(pa - v.brightLimb)) < 0.3)
        let pole = Lunar.norm(Lunar.deg(atan2(-v.west.z, v.north.z)))
        #expect(abs(Lunar.norm180(pole - v.axis)) < 0.05)
        // The lit fraction agrees with the angle between the Sun and the observer seen from the Moon.
        let k = (1 + simd_dot(v.toSun, v.toObserver)) / 2
        #expect(abs(k - v.illuminated) < 0.003)
    }

    @Test func libruationStaysInRange() {
        var maxL = 0.0, maxB = 0.0
        for day in stride(from: 0.0, to: 400, by: 0.5) {
            let v = Lunar.view(utc("2026-01-01T00:00:00Z").addingTimeInterval(day * 86400))
            maxL = max(maxL, abs(v.libration.x))
            maxB = max(maxB, abs(v.libration.y))
        }
        // Optical libration reaches about 7.9° in longitude and 6.9° in latitude.
        #expect(maxL > 6.5 && maxL < 8.2)
        #expect(maxB > 6.4 && maxB < 7.0)
    }

    @Test func topocentricParallax() {
        // From the equator with the Moon on the horizon, parallax moves it by nearly a degree.
        let date = utc("2026-10-25T21:00:00Z")
        let geo = Lunar.view(date)
        var largest = 0.0
        for lon in stride(from: -180.0, to: 180, by: 15) {
            let topo = Lunar.view(date, latitude: 0, longitude: lon)
            let shift = Lunar.deg(acos(min(simd_dot(Lunar.unit(geo.ra, geo.dec), Lunar.unit(topo.ra, topo.dec)), 1)))
            largest = max(largest, shift)
            #expect(abs(topo.distance - geo.distance) < 6400)
        }
        #expect(largest > 0.85 && largest < 1.0)
    }

    @Test func moonAge() {
        // New Moon on 2024 January 11 at 11:57 UTC; full Moon on January 25 at 17:54 UTC.
        #expect(abs(Lunar.age(utc("2024-01-12T11:57:00Z")) - 1.0) < 0.02)
        let full = Lunar.view(utc("2024-01-25T17:54:00Z"))
        #expect(full.illuminated > 0.998)
        #expect(abs(full.age - 14.25) < 0.05)
    }

    @Test func precession() {
        // Example 21.b: θ Persei at J2000 (already moved by its proper motion) 41.054063°, 49.227750° precesses to
        // 41.547214°, 49.348483° by 2028 November 13.19 TD.
        let T = (2_462_088.69 - 2_451_545.0) / 36525
        let back = Lunar.precessToJ2000(Lunar.unit(41.547214, 49.348483), T)
        #expect(Lunar.deg(acos(min(simd_dot(back, Lunar.unit(41.054063, 49.227750)), 1))) < 0.0002)
    }

    // MARK: Mars

    @Test func marsSeasons() {
        // Mars Year 36 began (Ls 0) on 2021 February 7; Perseverance landed eleven days later at Ls about 5.5.
        let start = MarsSun.seasonal(utc("2021-02-07T12:00:00Z"))
        #expect(abs(start.ls) < 1.5 || abs(start.ls - 360) < 1.5)
        let landing = MarsSun.seasonal(utc("2021-02-18T20:55:00Z"))
        #expect(abs(landing.ls - 5.5) < 1.5)
        // The seasons advance by about 360 degrees in a Martian year of 686.98 days.
        let a = MarsSun.seasonal(utc("2021-02-07T12:00:00Z")).ls
        let b = MarsSun.seasonal(utc("2021-02-07T12:00:00Z").addingTimeInterval(686.98 * 86400)).ls
        #expect(abs(Lunar.norm180(b - a)) < 1.0)
    }

    @Test func marsEquationOfTimeAndDeclination() {
        var maxEOT = 0.0, maxDec = 0.0
        for day in stride(from: 0.0, to: 687, by: 3) {
            let s = MarsSun.seasonal(utc("2021-02-07T00:00:00Z").addingTimeInterval(day * 86400))
            maxEOT = max(maxEOT, abs(s.eot))
            maxDec = max(maxDec, abs(s.declination))
        }
        // The equation of time reaches about 50 minutes (12.5 degrees); the Sun's declination, the axial tilt of 25.2 degrees.
        #expect(maxEOT > 10 && maxEOT < 14.5)
        #expect(maxDec > 24.5 && maxDec < 25.6)
    }

    @Test func marsLocalSolarTime() {
        // Local true solar time follows mean time at the place by the equation of time.
        let date = utc("2026-10-25T12:00:00Z")
        let p = MarsSun.position(date, lat: 18.4447, lon: 77.4508)
        let mean = Chronology.marsLocalTime(date, eastLongitude: 77.4508)
        var diff = p.localTime - mean
        if diff > 12 { diff -= 24 } else if diff < -12 { diff += 24 }
        #expect(abs(diff * 15 - p.equationOfTime) < 0.001)
        #expect(p.localTime >= 0 && p.localTime < 24)
    }

    @Test func marsSunIsHighAtNoonOnTheEquator() {
        // At local noon on the equator near an equinox the Sun is almost overhead and due north or south.
        var date = utc("2021-02-07T00:00:00Z")
        var best = (elevation: -90.0, localTime: 0.0)
        for _ in 0..<110 {
            date.addTimeInterval(60 * 60)
            let p = MarsSun.position(date, lat: 0, lon: 0)
            if p.elevation > best.elevation { best = (p.elevation, p.localTime) }
        }
        #expect(best.elevation > 85)
        #expect(abs(best.localTime - 12) < 1)
    }

    @Test func marsSunriseAndSetDirections() {
        // In the morning the Sun is in the east half of the sky, in the afternoon in the west half.
        let date = utc("2026-10-25T12:00:00Z")
        var morning = 0, afternoon = 0
        for hour in 0..<48 {
            let p = MarsSun.position(date.addingTimeInterval(Double(hour) * 3600), lat: -13, lon: 295)
            guard p.elevation > 5 else { continue }
            if p.localTime < 12 { if p.azimuth < 180 { morning += 1 } } else if p.azimuth > 180 { afternoon += 1 }
        }
        #expect(morning > 3 && afternoon > 3)
    }
}
