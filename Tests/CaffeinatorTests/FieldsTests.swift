import Foundation
import Testing
@testable import Caffeinator

@Suite struct FieldsTests {
    @Test func pageThorneFluxVanishesAtTheInnermostStableOrbit() {
        #expect(BlackHole.flux(3) == 0)
        #expect(BlackHole.flux(2) == 0)
        #expect(BlackHole.flux(3.2) > 0)
        // The flux peaks at 9.55 M, which is 4.78 Schwarzschild radii, and falls off like r^-3 far out.
        let radii = stride(from: Float(3.01), through: 20, by: 0.005)
        let peak = radii.max { BlackHole.flux($0) < BlackHole.flux($1) }!
        #expect(abs(peak - 4.775) < 0.02)
        #expect(abs(BlackHole.flux(400) / BlackHole.flux(200) - 0.133) < 0.002)
    }

    @Test func innerEdgeOrbitsAtHalfTheSpeedOfLight() {
        #expect(abs(BlackHole.orbitalSpeed(3) - 0.5) < 1e-6)
        #expect(BlackHole.orbitalSpeed(6) < BlackHole.orbitalSpeed(4))
    }

    @Test func parsesTheCurrentKpFeed() throws {
        let json = """
        [{"time_tag":"2026-10-24T03:00:00","Kp":1.33,"a_running":5,"station_count":8},
         {"time_tag":"2026-10-24T06:00:00","Kp":0.67,"a_running":3,"station_count":8}]
        """
        let reading = try #require(AuroraCurtains.parse(try JSONSerialization.jsonObject(with: Data(json.utf8))))
        #expect(abs(reading.kp - 0.67) < 1e-9)
        #expect(reading.time == Date(timeIntervalSince1970: 1_792_821_600))
    }

    @Test func parsesTheOlderRowFormat() throws {
        let json = """
        [["time_tag","Kp","a_running","station_count"],["2024-05-10 21:00:00.000","8.67","207","8"],["2024-05-11 00:00:00.000","9.00","300","8"]]
        """
        let reading = try #require(AuroraCurtains.parse(try JSONSerialization.jsonObject(with: Data(json.utf8))))
        #expect(reading.kp == 9)
    }

    @Test func skipsMissingAndOddValues() throws {
        let json = """
        [{"time_tag":"2026-10-24T03:00:00","Kp":2.0},{"time_tag":"2026-10-24T06:00:00","Kp":null},{"Kp":3},{"time_tag":"x","Kp":12}]
        """
        let reading = try #require(AuroraCurtains.parse(try JSONSerialization.jsonObject(with: Data(json.utf8))))
        #expect(reading.kp == 2)
        #expect(AuroraCurtains.parse(["not", "rows"]) == nil)
        #expect(AuroraCurtains.parse("nonsense") == nil)
    }

    @Test func describesGeomagneticActivity() {
        #expect(AuroraCurtains.describe(0.67) == "a quiet night")
        #expect(AuroraCurtains.describe(4.33) == "an active night")
        #expect(AuroraCurtains.describe(5.0).hasSuffix("G1"))
        #expect(AuroraCurtains.describe(9).hasSuffix("G5"))
        #expect(AuroraCurtains.activity(0) > 0.1)
        #expect(AuroraCurtains.activity(2) < AuroraCurtains.activity(5))
        #expect(AuroraCurtains.activity(9) == 1)
    }

    @Test func tokyoMapPutsTheBayAtSea() {
        let w = 384, h = 240
        let sea = SlimeTokyo.sea(w, h)
        func at(_ lat: Double, _ lon: Double) -> UInt8 {
            let p = SlimeTokyo.position(lat: lat, lon: lon, Double(w), Double(h))
            return sea[Int(p.y) * w + Int(p.x)]
        }
        #expect(at(35.50, 139.85) > 200)   // the middle of Tokyo Bay
        #expect(at(35.10, 139.30) > 200)   // Sagami Bay
        #expect(at(35.90, 139.62) < 50)    // Saitama
        #expect(at(35.60, 140.25) < 50)    // inland Chiba
        let tokyo = SlimeTokyo.position(SlimeTokyo.cities[0], Double(w), Double(h))
        #expect(abs(tokyo.x - Double(w) / 2) < 3)
        #expect(SlimeTokyo.cities.count == 36)
    }
}
