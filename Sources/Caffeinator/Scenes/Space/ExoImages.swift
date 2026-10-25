import Foundation

/// The two spotlit systems whose planets have been photographed directly, with what's needed to show each picture as
/// a scientific figure: where the star and planets are in it, its scale, and the annotations baked into the published
/// file, which are blanked so thinner labels can go in their place.
struct ExoImage {
    let url: URL
    let credit: String
    /// Instrument, filters and date, for a thin caption under the picture.
    let caption: String
    /// Spoken while the picture is on screen.
    let spoken: String
    /// Pixel size of the file, the star's position in it (behind the coronagraph) and its scale.
    let size: SIMD2<Float>
    let star: SIMD2<Float>
    let pixelsPerAU: Float
    /// The scale bar's length.
    let bar: Int
    let planets: [(letter: String, at: SIMD2<Float>)]
    /// Rectangles (x0, y0, x1, y1 in file pixels) of baked-in text to blank.
    let masks: [SIMD4<Float>]

    /// Webb's NIRCam coronagraph picture of HR 8799's four giant planets (ESA/Webb weic2504a, CC BY 4.0). Positions
    /// measured from the 718-pixel file: about 162 pixels per arcsecond at 41.2 parsecs.
    static let hr8799 = ExoImage(
        url: URL(string: "https://cdn.esawebb.org/archives/images/large/weic2504a.jpg")!,
        credit: "HR 8799: NASA, ESA, CSA, STScI, W. Balmer (JHU), L. Pueyo (STScI), M. Perrin (STScI)",
        caption: "JWST NIRCAM CORONAGRAPH  ·  4.1, 4.3, 4.6 µm  ·  2022–2023",
        spoken: "On screen now, Webb's infrared picture of all four giant planets, the star's glare blocked by a coronagraph",
        size: [718, 718], star: [358, 360], pixelsPerAU: 3.93, bar: 50,
        planets: [("b", [88.9, 280.4]), ("c", [399.4, 216.9]), ("e", [396.6, 303.4]), ("d", [447.2, 417.6])],
        masks: [[106, 264, 166, 294], [423, 206, 481, 230], [421, 291, 478, 314], [329, 331, 526, 388], [471, 400, 529, 430]])

    /// The VLT's SPHERE picture of YSES 1 (TYC 8998-760-1) and its two giant planets (ESO eso2011a, CC BY 4.0): about
    /// 326 pixels per arcsecond in the 1,310-pixel file, at 94.6 parsecs.
    static let yses1 = ExoImage(
        url: URL(string: "https://cdn.eso.org/images/large/eso2011a.jpg")!,
        credit: "YSES 1: ESO/Bohn et al.",
        caption: "VLT SPHERE  ·  NEAR INFRARED  ·  2019",
        spoken: "On screen now, the Very Large Telescope's photograph of the two planets, the star itself hidden behind a coronagraph",
        size: [1310, 1310], star: [191, 199.5], pixelsPerAU: 3.45, bar: 100,
        planets: [("b", [490.1, 672.5]), ("c", [921, 1026.3])],
        masks: [])
}
