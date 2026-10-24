import Foundation
import Testing
@testable import Caffeinator

struct OrbitTests {
    static let iss = ("ISS (ZARYA)", "1 25544U 98067A   26296.69304404  .00008515  00000+0  16237-3 0  9995",
                      "2 25544  51.6336  24.1765 0006871 300.7285  59.3026 15.49021515591674")
    static let gps = ("NAVSTAR 81 (USA 319)", "1 48859U 21054A   26296.18788634 -.00000044  00000+0  00000+0 0  9993",
                      "2 48859  55.1044 327.3061 0029438 232.8938 159.0071  2.00561558 39320")

    func distance(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double { ((a - b) * (a - b)).sum().squareRoot() }

    /// Reference TEME positions (km) from the python-sgp4 package, 0, 0.5, 1 and 2 days after epoch. Leaving out
    /// SGP4's short-period and deep-space terms costs about 12 km in low orbit and 24 km for GPS.
    @Test func matchesSGP4() throws {
        let cases: [((String, String, String), [SIMD3<Double>], Double)] = [
            (Self.iss, [[6199.99, 2783.33, -0.0], [1569.61, -3918.88, -5328.84], [-6415.35, -2257.69, -24.16], [6577.24, 1707.81, 41.74]], 20),
            (Self.gps, [[23372.62, -5406.9, 11561.77], [23280.99, -5086.23, 11888.44], [23182.22, -4763.84, 12211.42],
                        [22963.37, -4114.31, 12845.98]], 35),
        ]
        for (tle, positions, tolerance) in cases {
            let orbit = try #require(Orbit(name: tle.0, tle.1, tle.2))
            for (days, expected) in zip([0, 0.5, 1, 2], positions) {
                let p = orbit.teme(at: orbit.epoch.addingTimeInterval(days * 86400))
                #expect(distance(p, expected) < tolerance, "\(tle.0) after \(days) days is \(distance(p, expected)) km off")
            }
        }
    }

    @Test func classifies() throws {
        let iss = try #require(Orbit(name: Self.iss.0, Self.iss.1, Self.iss.2))
        let gps = try #require(Orbit(name: Self.gps.0, Self.gps.1, Self.gps.2))
        #expect(iss.kind == .low)
        #expect(gps.kind == .medium)
        #expect(abs(iss.period - 92.9) < 0.3)
        #expect(Orbit.parse("\(Self.iss.0)\n\(Self.iss.1)\n\(Self.iss.2)\nJUNK\n\(Self.gps.0)\n\(Self.gps.1)\n\(Self.gps.2)\n").count == 2)
    }
}

struct EarthTextTests {
    @Test func spokenPlaces() {
        #expect(Earthquakes.spoken("169 km NW of Port-Olry, Vanuatu") == "169 kilometres northwest of Port-Olry, Vanuatu")
        #expect(Earthquakes.spoken("9 km NNE of Indio, CA") == "9 kilometres north-northeast of Indio, California")
        #expect(Earthquakes.spoken("Fiji region") == "near Fiji region")
    }

    @Test func distances() {
        #expect(EarthText.kilometres(418.4) == "418 kilometres")
        #expect(EarthText.kilometres(25.13e9) == "25.1 billion kilometres")
        #expect(EarthText.duration(46 * 3600) == "46 hours")
    }
}
