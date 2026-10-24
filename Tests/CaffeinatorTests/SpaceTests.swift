import Foundation
import simd
import Testing
@testable import Caffeinator

struct SkyCatalogTests {
    @Test func starsDecodeBrightestFirst() {
        let stars = SkyCatalog.stars
        #expect(stars.count == 5044)
        let sirius = stars[0]
        #expect(abs(sirius.mag + 1.44) < 0.05)
        #expect(abs(sirius.ra - 101.29) < 0.05)
        #expect(abs(sirius.dec + 16.72) < 0.05)
        #expect(stars.allSatisfy { $0.dec >= -90 && $0.dec <= 90 && $0.ra >= 0 && $0.ra < 360 })
    }

    @Test func constellationsAndNames() {
        #expect(SkyCatalog.constellations.count == 89)  // Serpens comes in two parts
        #expect(SkyCatalog.constellations.contains { $0.name == "Orion" && $0.rank == 1 })
        #expect(SkyCatalog.lines.count > 80)
        #expect(SkyCatalog.starNames.first?.name == "Sirius")
        // Alnitak sits next to the brighter Alnilam, so only one of the belt stars is labelled.
        #expect(!(SkyCatalog.starNames.contains { $0.name == "Alnilam" } && SkyCatalog.starNames.contains { $0.name == "Alnitak" }))
    }

    @Test func milkyWayIsBrightestTowardTheCentre() {
        let mw = SkyCatalog.milkyWay
        func level(l: Double, b: Double) -> UInt8 { mw[Int((b + 90) * 2) * 720 + Int(l * 2)] }
        #expect(mw.count == 720 * 360)
        #expect(level(l: 0, b: -5) == 5)     // the great Sagittarius star cloud, just below the dusty centre
        #expect(level(l: 180, b: 0) <= 1)    // the faint anticentre
        #expect(level(l: 0, b: 89) == 0)
    }

    @Test func colourTemperatures() {
        #expect(abs(SkyCatalog.temperature(bv: 0.65) - 5800) < 150)   // the Sun
        let blue = SkyCatalog.blackbody(20000), red = SkyCatalog.blackbody(3000)
        #expect(blue.z > blue.x)
        #expect(red.x > red.z)
    }
}

struct ShowerTests {
    func noon(_ y: Int, _ m: Int, _ d: Int) -> Date {
        Calendar.current.date(from: DateComponents(year: y, month: m, day: d, hour: 12))!
    }

    @Test func orionidsThreeDaysAfterPeak() {
        let active = Shower.active(on: noon(2026, 10, 24))
        #expect(active?.0.name == "Orionids")
        #expect((active?.1 ?? 0) > 5 && (active?.1 ?? 0) < 20)
        let radiant = active!.0.radiant(on: noon(2026, 10, 24))
        #expect(abs(radiant.ra - 97.1) < 0.5)
    }

    @Test func peaksAndQuietDays() {
        #expect(abs((Shower.active(on: noon(2026, 8, 12))?.1 ?? 0) - 100) < 1)
        #expect(Shower.active(on: noon(2026, 3, 1)) == nil)
        let quadrantids = Shower.all[0]
        #expect(quadrantids.rate(on: noon(2026, 12, 30)) != nil)   // the window wraps the year
        #expect(quadrantids.rate(on: noon(2026, 12, 20)) == nil)
        #expect(Shower.active(on: noon(2027, 1, 3))?.0.name == "Quadrantids")
    }
}

@MainActor struct OrreryTests {
    @Test func keplerTones() {
        // Earth's year raised 32 octaves is the 136.1 Hz often quoted as Earth's tone.
        #expect(abs(Orrery.tone(period: 365.256, octaves: 32) - 136.1) < 0.1)
        for p in Orrery.planets {
            let f = Orrery.tone(period: p.period, octaves: p.octaves)
            #expect(f >= 110 && f <= 880, "\(p.name) at \(f) Hz")
        }
    }

    @Test func compressedScaleKeepsOrder() {
        let radii = Orrery.planets.map { Orrery.compress(simd_length(Astro.position($0.elements(0)))) }
        #expect(radii == radii.sorted())
        #expect(radii.last! < 1.02)
        #expect(radii.first! > 0.1)
    }
}

@MainActor struct GravitationalWaveTests {
    @Test func gw150914Timing() {
        let i = GravitationalWaves.Inspiral(m1: 35.6, m2: 30.6)
        #expect(abs(i.iscoFrequency - 4397 / 66.2) < 1)
        #expect(abs(i.chirpMass - 28.6) < 0.3)
        // From 35 Hz, where the signal rose out of the noise, to coalescence: about a fifth of a second at leading order.
        let tau = 5.0 / 256 * pow(i.chirpMass * GravitationalWaves.Inspiral.solarTime, -5.0 / 3) * pow(.pi * 35, -8.0 / 3)
        #expect(tau > 0.15 && tau < 0.25)
    }

    @Test func inspiralRunsInwardThenRingsDown() {
        let i = GravitationalWaves.Inspiral(m1: 1.46, m2: 1.27)
        var last = i.state(0)
        #expect(abs(last.separation - pow(i.tauStart / i.tauISCO, 0.25)) < 1e-9)
        for s in stride(from: 0.5, through: GravitationalWaves.Inspiral.inspiralTime, by: 0.5) {
            let now = i.state(s)
            #expect(now.phase > last.phase)
            #expect(now.separation < last.separation)
            #expect(now.amplitude > last.amplitude)
            last = now
        }
        #expect(abs(i.state(GravitationalWaves.Inspiral.inspiralTime).separation - 1) < 1e-6)
        #expect(i.state(i.merger + 5).amplitude < 0.01)
    }

    @Test func catalogueKeepsOneVersionAndShortNames() {
        let json: [String: Any] = ["events": [
            "GW150914-v3": ["commonName": "GW150914", "version": 3, "GPS": 1126259462.4, "mass_1_source": 35.6, "mass_2_source": 30.6,
                            "final_mass_source": 63.1, "luminosity_distance": 440],
            "GW150914-v1": ["commonName": "GW150914", "version": 1, "GPS": 1126259462.4, "mass_1_source": NSNull(), "mass_2_source": NSNull()],
            "GW250114_082203-v2": ["commonName": "GW250114_082203", "version": 2, "GPS": 1420878141.2, "mass_1_source": 33.76,
                                   "mass_2_source": 32.26, "final_mass_source": 62.9, "luminosity_distance": 405],
        ]]
        let events = GravitationalWaves.parse(json)
        #expect(events.map(\.name) == ["GW150914", "GW250114"])
        #expect(events[0].final == 63.1)
    }
}

@MainActor struct PulsarTests {
    @Test func plaqueBinary() {
        for p in Pulsars.all {
            let bits = Pulsars.bits(p.period)
            let n = bits.reduce(0.0) { $0 * 2 + ($1 ? 1 : 0) }
            #expect(abs(n * Pulsars.hydrogen - p.period) < 1e-9)
            #expect(bits.first == true)
        }
        #expect(Pulsars.bits(0.0333924).count == 26)  // the Crab
    }

    @Test func distances() {
        #expect(Pulsars.lightYears(2.0) == 6500)
        #expect(Pulsars.lightYears(0.28) == 910)
        #expect(Pulsars.all.count == 14)
    }
}

@MainActor struct ExoplanetTests {
    @Test func csvFields() {
        #expect(Exoplanets.fields("\"55 Cnc b\",\"55 Cnc\",0.118,,\"Radial, Velocity\"") == ["55 Cnc b", "55 Cnc", "0.118", "", "Radial, Velocity"])
    }

    @Test func parsesAndFillsInMissingOrbits() {
        let csv = """
        pl_name,hostname,pl_orbsmax,pl_orbper,pl_orbeccen,pl_eqt,pl_rade,st_teff,disc_year,discoverymethod
        "51 Peg b","51 Peg",0.052,4.2307966,0.0063,1328.93,14.1,5761.00,1995,"Radial Velocity"
        "Far b","Far",,365.25,,,,5772,2020,"Imaging"
        "Nothing b","Nothing",,,,,,,2020,"Imaging"
        """
        let planets = Exoplanets.parse(Data(csv.utf8))
        #expect(planets.count == 2)
        #expect(abs(planets[1].a - 1) < 1e-6)
        #expect(abs((planets[1].temperature ?? 0) - 278) < 5)
        #expect(planets[0].letter == "b")
    }

    @Test func scalesAndNotes() {
        #expect(Exoplanets.ring(0.01) < Exoplanets.ring(1))
        #expect(Exoplanets.ring(1) < Exoplanets.ring(100))
        #expect(Exoplanets.note(1) > Exoplanets.note(1000))
        #expect(Exoplanets.pace(1) > Exoplanets.pace(365))
    }

    @Test func flybys() {
        let json: [String: Any] = ["fields": ["des", "orbit_id", "jd", "cd", "dist", "dist_min", "dist_max", "v_rel", "v_inf", "t_sigma_f", "h"],
                                   "data": [["2099 UE", "3", "0", "2099-Oct-25 05:51", "0.0134307616673871", "0", "0", "15.9", "15.9", "00:02", "25.775"]]]
        let list = Exoplanets.parseFlybys(json)
        #expect(list.count == 1)
        #expect(abs(list[0].lunar - 5.23) < 0.01)
        #expect(list[0].size > 15 && list[0].size < 40)
    }
}
