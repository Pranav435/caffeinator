import Foundation

/// A real place on Mars with public elevation and image data from the USGS Astrogeology archive.
struct MarsSite {
    let id: String
    let name: String
    let region: String
    /// Planetocentric latitude and east longitude of the middle of the data, degrees.
    let lat: Double
    let lon: Double
    let dtm: URL
    let ortho: URL
    /// Ground sample distance of both rasters, metres.
    let gsd: Double
    let instrument: String
    let credit: String
    /// How high the camera flies above the terrain, in metres, and how far it ranges across the data (0...1).
    let altitude: Double
    /// Camera pitch below the horizon, radians.
    let pitch: Double
    /// How many cells wide the slope is measured, to smooth stereo noise in the elevation model.
    var smooth: Double = 1.2
    let spoken: String

    private static let ard = "https://astrogeo-ard.s3-us-west-2.amazonaws.com/mars/mro"
    private static func ctx(_ id: String, _ suffix: (String, String)) -> (URL, URL) {
        (URL(string: "\(ard)/ctx/controlled/usgs/\(id)/\(id)\(suffix.0)")!, URL(string: "\(ard)/ctx/controlled/usgs/\(id)/\(id)\(suffix.1)")!)
    }
    private static func hirise(_ dir: String, _ dtm: String, _ ortho: String) -> (URL, URL) {
        (URL(string: "\(ard)/hirise/controlled/dtm/\(dir)/\(dtm).tif")!, URL(string: "\(ard)/hirise/controlled/dtm/\(dir)/\(ortho).tif")!)
    }

    static let all: [MarsSite] = {
        let jezero = ctx("K23_062530_1986_XN_18N282W__K23_062319_1986_XN_18N282W", ("_dtm.tif", "_orthoimage.tif"))
        let gale = ctx("J09_048418_1758_XN_04S222W__G04_019777_1747_XN_05S222W", ("-DEM.tif", "-DRG.tif"))
        let coprates = ctx("D16_033485_1669_XN_13S064W__D18_034197_1669_XN_13S064W", ("_dtm.tif", "_orthoimage.tif"))
        let victoria = hirise("PSP_001414_1780_PSP_001612_1780", "DTEEC_001414_1780_001612_1780_U01", "PSP_001414_1780_RED_C_01_ORTHO")
        let melas = ctx("N09_065819_1733_XN_06S069W__N09_065885_1717_XN_08S068W", ("_dtm.tif", "_orthoimage.tif"))
        let ctxCredit = "NASA/JPL/MSSS/USGS Astrogeology Science Center (CTX digital terrain model, CC0)"
        let hiCredit = "NASA/JPL/University of Arizona/USGS (HiRISE digital terrain model)"
        return [
            MarsSite(id: "jezero", name: "Jezero Crater", region: "Isidis Planitia", lat: 18.477, lon: 77.292, dtm: jezero.0, ortho: jezero.1,
                     gsd: 20, instrument: "MRO CTX  ·  20 m per pixel", credit: ctxCredit, altitude: 4200, pitch: -0.4,
                     spoken: "the river delta of Jezero crater, where a lake once poured in, and where Perseverance landed in 2021"),
            MarsSite(id: "gale", name: "Mount Sharp, Gale Crater", region: "Aeolis Palus", lat: -5.22, lon: 137.87, dtm: gale.0, ortho: gale.1,
                     gsd: 20, instrument: "MRO CTX  ·  20 m per pixel", credit: ctxCredit, altitude: 4200, pitch: -0.4,
                     spoken: "the foot of Mount Sharp, a five kilometre mound of layered sediment rising from the floor of Gale crater, where Curiosity has been climbing since 2014"),
            MarsSite(id: "coprates", name: "Coprates Chasma", region: "Valles Marineris", lat: -13.04, lon: 295.2, dtm: coprates.0, ortho: coprates.1,
                     gsd: 20, instrument: "MRO CTX  ·  20 m per pixel", credit: ctxCredit, altitude: 5200, pitch: -0.4,
                     spoken: "a wall of Coprates Chasma in Valles Marineris, the canyon system that would stretch from New York to Los Angeles and more"),
            MarsSite(id: "victoria", name: "Victoria Crater", region: "Meridiani Planum", lat: -2.051, lon: 354.504, dtm: victoria.0, ortho: victoria.1,
                     gsd: 1, instrument: "MRO HiRISE  ·  1 m per pixel", credit: hiCredit, altitude: 230, pitch: -0.5,
                     spoken: "Victoria crater, 730 metres across, its scalloped walls and dune-filled floor explored by the rover Opportunity in 2006 and 2007"),
            MarsSite(id: "melas", name: "Melas Chasma", region: "Valles Marineris", lat: -6.67, lon: 291.07, dtm: melas.0, ortho: melas.1,
                     gsd: 20, instrument: "MRO CTX  ·  20 m per pixel", credit: ctxCredit, altitude: 4800, pitch: -0.4,
                     spoken: "the floor and slopes of Melas Chasma, the widest part of Valles Marineris, where layered deposits and landslides pile against the canyon walls"),
        ]
    }()
}
