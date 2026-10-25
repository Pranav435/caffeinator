import Foundation

/// The telescope images the deep-field scene shows, with what each page on esawebb.org and esahubble.org says about
/// it: object, distance, field of view, orientation, filters and the colour each was given, release date and the
/// required credit line. All are CC BY 4.0 (ESA/Webb and ESA/Hubble), fetched at 4000 pixels on the long side.
extension DeepField {
    /// A filter of the composite, and the colour it was assigned on screen.
    struct Filter {
        let name: String
        /// Central wavelength in microns.
        let microns: Double
        /// What it picks out, if it's a line or band worth naming ("hydrogen"), for VoiceOver.
        let feature: String?
        let color: Swatch
    }

    enum Swatch {
        case purple, blue, cyan, green, yellow, orange, red, magenta

        /// The tint drawn in the legend: the published colour, a little muted.
        var rgb: SIMD3<Double> {
            switch self {
            case .purple: [0.66, 0.52, 0.95]
            case .blue: [0.42, 0.6, 1.0]
            case .cyan: [0.4, 0.86, 0.95]
            case .green: [0.5, 0.9, 0.5]
            case .yellow: [0.98, 0.9, 0.45]
            case .orange: [1.0, 0.66, 0.3]
            case .red: [1.0, 0.42, 0.36]
            case .magenta: [0.95, 0.45, 0.85]
            }
        }
    }

    struct Plate {
        let id: String
        let webb: Bool
        let instrument: String
        let name: String
        /// What it is, in a short line under the name.
        let what: String
        /// What the picture looks like, for VoiceOver.
        let spoken: String
        /// Light-years, or nil for a deep field where every galaxy is at its own distance.
        let distance: Double?
        /// The distance as written in the figure, such as "6,500 light-years" or "4.6 billion light-years (the cluster)".
        let distanceText: String
        /// Light-years per arcminute at the object, when it differs from the plain small-angle figure (cosmological
        /// distances), or nil to work it out from `distance`.
        let lyPerArcmin: Double?
        /// Field of view in arcminutes, width × height.
        let field: SIMD2<Double>
        /// Where north points, in degrees clockwise from straight up; nil when the page doesn't say.
        let north: Double?
        let released: String
        let credit: String
        let filters: [Filter]
        /// Ken Burns: the view's centre (0...1 across and down the image) and zoom over the "cover" framing, at the
        /// start and end of the shot.
        let from: SIMD3<Double>
        let to: SIMD3<Double>

        var url: URL {
            URL(string: webb ? "https://cdn.esawebb.org/archives/images/publicationjpg/\(id).jpg"
                             : (id == "heic0515a" ? "https://cdn.esahubble.org/archives/images/large/\(id).jpg"
                                                  : "https://cdn.esahubble.org/archives/images/publicationjpg/\(id).jpg"))!
        }

        var page: String { webb ? "https://esawebb.org/images/\(id)/" : "https://esahubble.org/images/\(id)/" }
        var telescope: String { webb ? "James Webb Space Telescope" : "Hubble Space Telescope" }

        /// Light-years per arcminute of sky at the object.
        var scale: Double? { lyPerArcmin ?? distance.map { $0 * .pi / 180 / 60 } }
    }

    private static func f(_ name: String, _ microns: Double, _ color: Swatch, _ feature: String? = nil) -> Filter {
        Filter(name: name, microns: microns, feature: feature, color: color)
    }

    static let plates: [Plate] = [
        Plate(id: "heic1501a", webb: false, instrument: "WFC3", name: "Pillars of Creation",
              what: "Columns of cold gas and dust in the Eagle Nebula, M16",
              spoken: "Hubble's visible-light view of the Pillars of Creation in the Eagle Nebula: three dark brown columns of gas and dust, the tallest about four light-years high, rise through a teal haze of glowing oxygen, their edges eaten away by ultraviolet light from hot young stars above the frame.",
              distance: 6500, distanceText: "6,500 light-years", lyPerArcmin: nil, field: [4.53, 4.72], north: -35,
              released: "5 January 2015", credit: "NASA, ESA/Hubble and the Hubble Heritage Team",
              filters: [f("F502N", 0.502, .blue, "oxygen"), f("F657N", 0.657, .green, "hydrogen and nitrogen"), f("F673N", 0.673, .orange, "sulphur")],
              from: [0.5, 0.62, 1.0], to: [0.47, 0.4, 1.12]),
        Plate(id: "weic2216b", webb: true, instrument: "NIRCam", name: "Pillars of Creation",
              what: "Star birth in the Eagle Nebula, M16, in near-infrared light",
              spoken: "Webb's near-infrared view of the same Pillars: translucent orange and brown columns climb from the lower left across a deep blue field crowded with thousands of stars that visible light cannot show, and the wavy red tips are jets from stars only a few hundred thousand years old.",
              distance: 6500, distanceText: "6,500 light-years", lyPerArcmin: nil, field: [4.21, 7.30], north: 2.6,
              released: "19 October 2022", credit: "NASA, ESA, CSA, STScI; J. DePasquale, A. Koekemoer, A. Pagan (STScI).",
              filters: [f("F090W", 0.90, .purple), f("F187N", 1.87, .blue, "Paschen-alpha hydrogen"), f("F200W", 2.0, .cyan),
                        f("F335M", 3.35, .yellow, "carbon-rich dust"), f("F444W", 4.44, .orange), f("F470N", 4.70, .red, "molecular hydrogen")],
              from: [0.5, 0.74, 1.0], to: [0.5, 0.36, 1.04]),
        Plate(id: "weic2205a", webb: true, instrument: "NIRCam", name: "Cosmic Cliffs",
              what: "The edge of the star-forming cavity NGC 3324 in the Carina Nebula",
              spoken: "Webb's Cosmic Cliffs in the Carina Nebula: a craggy orange ridge of gas and dust runs across the lower half like mountains at dusk, under a blue sky of hot, glowing gas and bright stars with eight-pointed diffraction spikes. The tallest peaks are about seven light-years high.",
              distance: 7600, distanceText: "7,600 light-years", lyPerArcmin: nil, field: [7.29, 4.22], north: 103,
              released: "12 July 2022", credit: "NASA, ESA, CSA, and STScI",
              filters: [f("F090W", 0.90, .blue), f("F187N", 1.87, .cyan, "Paschen-alpha hydrogen"), f("F200W", 2.0, .green),
                        f("F335M", 3.35, .orange, "carbon-rich dust"), f("F444W", 4.44, .red), f("F470N", 4.70, .yellow, "molecular hydrogen")],
              from: [0.44, 0.5, 1.0], to: [0.56, 0.46, 1.16]),
        Plate(id: "weic2207b", webb: true, instrument: "NIRCam", name: "Southern Ring Nebula",
              what: "Planetary nebula NGC 3132, shells cast off by a dying star",
              spoken: "Webb's Southern Ring Nebula: a glowing blue oval of hot gas inside a ragged orange ring of molecular hydrogen, the shells a dying star has thrown off over thousands of years. The star itself is hidden beside its brighter companion at the centre.",
              distance: 2500, distanceText: "2,500 light-years", lyPerArcmin: nil, field: [2.42, 2.25], north: 112,
              released: "12 July 2022", credit: "NASA, ESA, CSA, STScI, and the Webb ERO Production Team",
              filters: [f("F090W", 0.90, .blue), f("F187N", 1.87, .cyan, "Paschen-alpha hydrogen"), f("F212N", 2.12, .green, "molecular hydrogen"),
                        f("F356W", 3.56, .yellow), f("F405N", 4.05, .red, "Brackett-alpha hydrogen"), f("F470N", 4.70, .red, "molecular hydrogen")],
              from: [0.5, 0.5, 1.0], to: [0.5, 0.47, 1.14]),
        Plate(id: "weic2208a", webb: true, instrument: "NIRCam and MIRI", name: "Stephan’s Quintet",
              what: "Five galaxies in Pegasus, four of them locked in a slow collision",
              spoken: "Webb's Stephan's Quintet: five galaxies, four of them 290 million light-years away and tearing at each other, with shock waves glowing red and gold between them. The bluish galaxy on the left is a bystander, only 40 million light-years away.",
              distance: 290e6, distanceText: "290 million light-years", lyPerArcmin: nil, field: [6.34, 6.08], north: 62,
              released: "12 July 2022", credit: "NASA, ESA, CSA, and STScI",
              filters: [f("F090W", 0.90, .blue), f("F150W", 1.50, .blue), f("F200W", 2.0, .green), f("F277W", 2.77, .yellow),
                        f("F356W", 3.56, .red), f("F444W", 4.44, .red), f("F770W", 7.7, .yellow), f("F1000W", 10.0, .orange)],
              from: [0.48, 0.46, 1.0], to: [0.52, 0.42, 1.1]),
        Plate(id: "weic2209a", webb: true, instrument: "NIRCam", name: "Webb’s First Deep Field",
              what: "Galaxy cluster SMACS 0723 and the lensed universe behind it",
              spoken: "Webb's First Deep Field: the galaxy cluster SMACS 0723, seen as it was 4.6 billion years ago, bends the light of galaxies far behind it into glowing orange arcs. Thousands of galaxies fill a patch of sky the size of a grain of sand held at arm's length.",
              distance: nil, distanceText: "4.6 billion light-years to the cluster", lyPerArcmin: 1.04e6, field: [2.27, 2.31], north: -35,
              released: "12 July 2022", credit: "NASA, ESA, CSA, and STScI",
              filters: [f("F090W", 0.90, .blue), f("F150W", 1.50, .blue), f("F200W", 2.0, .green), f("F277W", 2.77, .green),
                        f("F356W", 3.56, .orange), f("F444W", 4.44, .red)],
              from: [0.5, 0.5, 1.0], to: [0.53, 0.55, 1.32]),
        Plate(id: "heic0406a", webb: false, instrument: "ACS", name: "Hubble Ultra Deep Field",
              what: "About 10,000 galaxies in one dark speck of Fornax",
              spoken: "Hubble's Ultra Deep Field: a speck of sky in Fornax a tenth of the Moon's width, exposed for about a million seconds. Nearly every dot is a galaxy, spirals, ellipticals and red smudges, the faintest seen as they were less than a billion years after the Big Bang.",
              distance: nil, distanceText: "galaxies out to 13 billion light-years", lyPerArcmin: nil, field: [3.10, 3.10], north: -46.5,
              released: "9 March 2004", credit: "NASA, ESA, and S. Beckwith (STScI) and the HUDF Team",
              filters: [f("F435W", 0.435, .blue), f("F606W", 0.606, .green), f("F775W", 0.775, .red)],
              from: [0.5, 0.5, 1.0], to: [0.45, 0.42, 1.3]),
        Plate(id: "weic2211a", webb: true, instrument: "NIRCam and MIRI", name: "Cartwheel Galaxy",
              what: "A ring galaxy reshaped by a head-on collision",
              spoken: "Webb's Cartwheel Galaxy: a wheel of pink star-forming knots around a bright hub, with bluish spokes between them, still rippling outward after a smaller galaxy plunged through it. Two companion galaxies sit to the left.",
              distance: 500e6, distanceText: "500 million light-years", lyPerArcmin: nil, field: [2.34, 2.16], north: -28,
              released: "2 August 2022", credit: "NASA, ESA, CSA, STScI",
              filters: [f("F090W", 0.90, .blue), f("F150W", 1.50, .blue), f("F200W", 2.0, .green), f("F277W", 2.77, .yellow),
                        f("F356W", 3.56, .red), f("F444W", 4.44, .red), f("F770W", 7.7, .orange), f("F1000W", 10.0, .orange),
                        f("F1280W", 12.8, .orange), f("F1800W", 18.0, .orange)],
              from: [0.5, 0.5, 1.0], to: [0.56, 0.5, 1.1]),
        Plate(id: "weic2212a", webb: true, instrument: "NIRCam", name: "Tarantula Nebula",
              what: "30 Doradus, the biggest stellar nursery in the Local Group",
              spoken: "Webb's Tarantula Nebula in the Large Magellanic Cloud: tangled filaments of pale gas and rust-red dust frame a cavity hollowed out by the hot blue stars of the cluster R136, near the centre, where tens of thousands of young stars shine.",
              distance: 161_000, distanceText: "161,000 light-years", lyPerArcmin: nil, field: [7.26, 4.20], north: nil,
              released: "6 September 2022", credit: "NASA, ESA, CSA, and STScI",
              filters: [f("F090W", 0.90, .blue), f("F200W", 2.0, .green), f("F444W", 4.44, .red), f("F470N", 4.70, .orange, "molecular hydrogen")],
              from: [0.45, 0.5, 1.0], to: [0.56, 0.46, 1.18]),
        Plate(id: "heic0515a", webb: false, instrument: "WFPC2", name: "Crab Nebula",
              what: "The wreck of the star that exploded in the year 1054",
              spoken: "Hubble's Crab Nebula: the expanding debris of a star seen to explode in the year 1054, a ragged cage of orange and green filaments around a blue glow powered by the spinning neutron star at its heart.",
              distance: 6500, distanceText: "6,500 light-years", lyPerArcmin: nil, field: [6.41, 6.41], north: 0,
              released: "1 December 2005", credit: "NASA, ESA and Allison Loll/Jeff Hester (Arizona State University). Acknowledgement: Davide De Martin (ESA/Hubble)",
              filters: [f("F502N", 0.502, .red, "doubly ionised oxygen"), f("F631N", 0.631, .blue, "neutral oxygen"), f("F673N", 0.673, .green, "sulphur")],
              from: [0.5, 0.52, 1.0], to: [0.5, 0.47, 1.14]),
        Plate(id: "weic2326a", webb: true, instrument: "NIRCam and MIRI", name: "Crab Nebula",
              what: "Supernova remnant M1 in infrared light",
              spoken: "Webb's infrared Crab Nebula: the same cage of filaments now gold and orange with warm dust, and milky white smoke of synchrotron light filling the inside, swirling around the pulsar that spins thirty times a second.",
              distance: 6500, distanceText: "6,500 light-years", lyPerArcmin: nil, field: [5.33, 4.62], north: 1.9,
              released: "30 October 2023", credit: "NASA, ESA, CSA, STScI, T. Temim (Princeton University)",
              filters: [f("F162M", 1.62, .blue), f("F480M", 4.80, .cyan, "synchrotron light"), f("F560W", 5.6, .cyan),
                        f("F1130W", 11.3, .green, "carbon-rich dust"), f("F1800W", 18.0, .orange, "silicate dust"), f("F2100W", 21.0, .red)],
              from: [0.5, 0.5, 1.0], to: [0.52, 0.48, 1.15]),
        Plate(id: "heic0601a", webb: false, instrument: "ACS", name: "Orion Nebula",
              what: "M42 and M43, the nearest large stellar nursery",
              spoken: "Hubble's sharpest view of the Orion Nebula: billowing walls of red and pink hydrogen around a bright core lit by the four Trapezium stars, dark lanes of dust, and more than 3,000 stars across the field.",
              distance: 1400, distanceText: "1,400 light-years", lyPerArcmin: nil, field: [30.03, 30.03], north: 0,
              released: "11 January 2006",
              credit: "NASA, ESA, M. Robberto (Space Telescope Science Institute/ESA) and the Hubble Space Telescope Orion Treasury Project Team",
              filters: [f("F435W", 0.435, .blue), f("F555W", 0.555, .green), f("F658N", 0.658, .red, "hydrogen-alpha"),
                        f("F775W", 0.775, .red), f("F850LP", 0.85, .red)],
              from: [0.5, 0.48, 1.0], to: [0.5, 0.34, 1.28]),
        Plate(id: "weic2411a", webb: true, instrument: "NIRCam", name: "Horsehead Nebula",
              what: "The crest of a dark cloud in Orion, lit from above",
              spoken: "Webb's close-up of the Horsehead Nebula's mane: a sculpted blue-white ridge of dust along the bottom, its edge glowing where ultraviolet light from a hot star boils it away, under a red haze with a bright spiked star and distant galaxies scattered above.",
              distance: 1300, distanceText: "1,300 light-years", lyPerArcmin: nil, field: [2.13, 2.14], north: -112.2,
              released: "29 April 2024",
              credit: "ESA/Webb, NASA, CSA, K. Misselt (University of Arizona) and A. Abergel (IAS/University Paris-Saclay, CNRS)",
              filters: [f("F140M", 1.40, .blue), f("F250M", 2.50, .blue, "methane"), f("F300M", 3.00, .cyan, "water ice"),
                        f("F323N", 3.23, .cyan, "molecular hydrogen"), f("F335M", 3.35, .green, "carbon-rich dust"),
                        f("F430M", 4.30, .orange, "carbon dioxide"), f("F405N", 4.05, .red, "Brackett-alpha hydrogen"),
                        f("F470N", 4.70, .red, "molecular hydrogen")],
              from: [0.5, 0.62, 1.04], to: [0.5, 0.42, 1.0]),
    ]
}
