@MainActor extension SceneKind {
    static let worldScenes: [SceneKind] = [
        SceneKind(id: "moon", name: "The Moon Tonight", family: .space,
                  about: "The real Moon, right now, as a telescope sees it: NASA's Lunar Reconnaissance Orbiter map wrapped on a sphere and turned to tonight's exact phase and libration. Grey highlands and dark seas catch low sunlight along the terminator, where crater walls throw long black shadows, while the unlit side glows faintly blue with earthshine. The view drifts slowly in to famous craters and out again, with thin labels, a scale bar and the Moon's phase, distance and age.",
                  make: MoonTonight.init, gain: 1.5),
        SceneKind(id: "mars", name: "Mars from Orbit", family: .space,
                  about: "A slow flight over real Martian ground, built from the Mars Reconnaissance Orbiter's elevation maps and photographs: the river delta of Jezero crater, the layered foot of Mount Sharp, two stretches of the Valles Marineris canyon and Victoria crater. The Sun stands where it really is at that spot right now, long shadows stretch across rust and charcoal ground, and butterscotch dust haze swallows the distance, turning blue around the Sun near sunset. A thin readout gives the place, local solar time and instrument.",
                  make: MarsFlight.init, gain: 1.35),
    ]
}
