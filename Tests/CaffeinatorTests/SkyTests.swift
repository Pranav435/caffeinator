import Foundation
import Testing
@testable import Caffeinator

@MainActor struct SkyTests {
    func utc(_ s: String) -> Date { ISO8601DateFormatter().date(from: s)! }

    // MARK: The Sun

    @Test(arguments: [(4.37e-7, "B4.4"), (1.0e-6, "C1.0"), (9.96e-7, "C1.0"), (2.3e-4, "X2.3"), (1.2e-3, "X12.0"), (5e-9, "A0.5")])
    func flareClasses(flux: Double, expected: String) {
        #expect(SunToday.flareClass(flux) == expected)
    }

    @Test func readsXrayFeed() {
        let row = #"{"time_tag": "2026-10-24T12:13:00Z", "flux": 4.3e-07, "energy": "0.1-0.8nm"}"#
        #expect(SunToday.xrayRows(Data("[\(row)]".utf8))?.count == 1)
        #expect(SunToday.xrayRows(Data("[\(row), \(row)]1, \"time\"}]".utf8))?.count == 2)
        #expect(SunToday.xrayRows(Data("not json".utf8)) == nil)
    }

    @Test func readsSDOStamps() {
        let hourAgo = Date().addingTimeInterval(-3600)
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        let p = c.dateComponents([.year, .month, .day, .hour, .minute, .second], from: hourAgo)
        let aia = String(format: "SDO/AIA 171 | %04d-%02d-%02d %02d:%02d:%02d UT", p.year!, p.month!, p.day!, p.hour!, p.minute!, p.second!)
        let hmi = String(format: "SDO/HMI Quick-Look Continuum: %04d%02d%02d_%02d%02d%02d", p.year!, p.month!, p.day!, p.hour!, p.minute!, p.second!)
        #expect(abs(SunToday.parseStamp(aia)!.timeIntervalSince(hourAgo)) < 1)
        #expect(abs(SunToday.parseStamp(hmi)!.timeIntervalSince(hourAgo)) < 1)
        #expect(SunToday.parseStamp("SDO/AIA 171 | 1999-01-01 00:00:00 UT") == nil)
        #expect(SunToday.parseStamp("no time here") == nil)
    }

    // MARK: Live Earth

    @Test(arguments: [(88.4, "meteosat-9"), (-100.0, "goes-19"), (170.0, "himawari"), (-170.0, "goes-18"), (10.0, "meteosat-0deg"), (-122.4, "goes-18")])
    func nearestSatellite(longitude: Double, slug: String) {
        #expect(LiveEarth.nearest(longitude).slug == slug)
    }

    @Test func picksHalfHourFrames() {
        let stamps: [Int64] = [20261024102021, 20261024101021, 20261024100021, 20261024095021, 20261024094021, 20261024093021,
                               20261024092021, 20261024091021, 20261024090021]
        #expect(LiveEarth.pick(stamps, count: 8) == [20261024100021, 20261024093021, 20261024090021])
        #expect(LiveEarth.pick([20261024100000, 20261024094500, 20261024093000], count: 1) == [20261024100000])
    }

    @Test func tileAddresses() {
        let urls = LiveEarth.tiles(20261024093000, LiveEarth.satellites[4])
        #expect(urls.count == 16)
        #expect(urls[5].absoluteString
            == "https://slider.cira.colostate.edu/data/imagery/2026/10/24/meteosat-9---full_disk/geocolor/20261024093000/02/001_001.png")
        #expect(LiveEarth.date(20261024093000) == utc("2026-10-24T09:30:00Z"))
    }

    @Test func geostationaryProjection() {
        let centre = LiveEarth.geo([0, 0], satellite: 140.7)!
        #expect(abs(centre.lat) < 1e-6 && abs(centre.lon - 140.7) < 1e-6)
        let north = LiveEarth.geo([0, 0.5], satellite: 140.7)!
        #expect(north.lat > 25 && north.lat < 35)
        let east = LiveEarth.geo([0.5, 0], satellite: -75.2)!
        #expect(east.lon > -50 && east.lon < -40)
        #expect(LiveEarth.geo([0.9, 0.9], satellite: 0) == nil)
    }

    @Test func namesPlaces() {
        #expect(SkyPlaces.nearest(14, 88)?.name == "the Bay of Bengal")
        #expect(SkyPlaces.nearest(-25, 134, big: true)?.name == "Australia")
        #expect(SkyPlaces.all.count > 150)
    }

    // MARK: The sky

    @Test func dolbearsLaw() {
        #expect(SkyWindow.dolbear(celsius: 20) == 112)
        #expect(SkyWindow.dolbear(celsius: 0) == 0)
    }

    @Test func sunlightThroughTheAir() {
        let noon = SkyWindow.transmittance(height: 0.05, elevation: 90)
        #expect(noon.x > 0.85 && noon.z > 0.6 && noon.x > noon.z)
        let sunset = SkyWindow.transmittance(height: 0.05, elevation: 1)
        #expect(sunset.x > sunset.y && sunset.y > sunset.z)
        #expect(SkyWindow.transmittance(height: 0.05, elevation: -5) == .zero)
        // High cloud still catches the Sun after it has set for the ground.
        #expect(SkyWindow.transmittance(height: 9, elevation: -2).x > 0.01)
    }

    @Test func readsOpenMeteo() {
        let json: [String: Any] = ["current": ["temperature_2m": 14.2, "cloud_cover": 75, "cloud_cover_low": 60, "weather_code": 61,
                                               "rain": 0.4, "wind_speed_10m": 18.5, "wind_direction_10m": 250]]
        let w = SkyWindow.Weather(json: json)!
        #expect(w.temperature == 14.2 && w.low == 0.6 && w.code == 61 && w.summary == "light rain")
        #expect(w.rainIntensity >= 0.4 && w.snowIntensity == 0)
        #expect(SkyWindow.Weather(json: ["nothing": 1]) == nil)
    }

    @Test func starCatalogue() {
        #expect(SkyStars.all.count > 130)
        let sirius = SkyStars.all.min { $0.mag < $1.mag }!
        #expect(abs(sirius.ra - 6.7525) < 0.01 && abs(sirius.dec + 16.716) < 0.01)
    }

    // MARK: Time

    @Test func voyagerAgreesWithHorizons() {
        // JPL Horizons, one-way light time from Earth in minutes.
        #expect(abs(Chronology.voyagerLightTime(utc("2026-10-26T00:00:00Z")) / 60 - 1436.56994) < 0.02)
        #expect(abs(Chronology.voyagerLightTime(utc("2026-11-20T00:00:00Z")) / 60 - 1440.10844) < 0.02)
        #expect(abs(Chronology.voyagerDistance(utc("2026-10-26T00:00:00Z")) - 172.73220) < 0.002)
        #expect(Chronology.oneLightDay > utc("2026-11-18T00:00:00Z") && Chronology.oneLightDay < utc("2026-11-20T00:00:00Z"))
    }

    @Test func marsTime() {
        // Mars24's worked example: 6 January 2000, 0h UTC, MSD 44,795.9998 and MTC 23:59:39 (with TT − UTC = 64.184 s then).
        #expect(abs(Chronology.marsSolDate(utc("2000-01-06T00:00:00Z")) - 44_795.9998) < 0.0002)
        // Perseverance touched down at about 3:53 pm local mean solar time, on sol 0.
        let landing = Chronology.marsLocalTime(Chronology.perseveranceLanding, eastLongitude: Chronology.jezero.longitude)
        #expect(abs(landing - (15 + 53.0 / 60)) < 0.05)
        #expect(Chronology.perseveranceSol(Chronology.perseveranceLanding) == 0)
        #expect(Chronology.perseveranceSol(Chronology.perseveranceLanding + 1000 * 88_775.244) == 1000)
    }

    @Test func calendars() {
        let d = utc("2026-10-24T12:00:00Z")
        let utcZone = TimeZone(identifier: "UTC")!
        #expect(Chronology.dayOfYear(d, in: utcZone) == 297)
        #expect(Chronology.isoWeek(d, in: utcZone) == (2026, 43, 6))
        #expect(Chronology.isoWeek(utc("2027-01-01T12:00:00Z"), in: utcZone) == (2026, 53, 5))
        #expect(abs(Chronology.julianDay(d) - 2_461_338.0) < 1e-9)
        #expect(Chronology.grouped(2_461_337.98734, decimals: 5) == "2,461,337.98734")
        #expect(Chronology.grouped(61_337.9999999, decimals: 5) == "61,338.00000")
        #expect(Chronology.clock(14.5) == "14:30:00")
        #expect(Chronology.duration(86_179) == "23 hours 56 minutes 19 seconds")
    }
}
