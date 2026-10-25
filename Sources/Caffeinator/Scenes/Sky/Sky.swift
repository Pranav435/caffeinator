@MainActor extension SceneKind {
    static let skyScenes: [SceneKind] = [
        SceneKind(id: "live-earth", name: "Live Earth", family: .earth,
                  about: "The whole Earth as the nearest weather satellite sees it now: true colour by day, city lights at night. The last few hours loop as a time-lapse, so clouds stream and sunrise sweeps across the globe, which floats in space inside a thin blue glow.",
                  make: LiveEarth.init),
        SceneKind(id: "sun", name: "The Sun Today", family: .earth,
                  about: "NASA's Solar Dynamics Observatory's newest pictures of the Sun, dissolving from wavelength to wavelength: the orange visible surface with its sunspots, golden coronal loops, red prominences, teal flare plasma and the black-and-white magnetic field, slowly turning.",
                  make: SunToday.init),
        SceneKind(id: "sky-window", name: "Window to the Sky", family: .earth,
                  about: "The sky over you right now, photographed through a painted timber window half a metre away. The sky is computed from physics, so its blue, the reddened Sun, twilight purples and the Earth's shadow are real for where the Sun is. Soft volumetric clouds of the kind and amount the weather reports drift with the real wind; rain beads and runs down the glass, snow and mist fall when they really do. Beyond it are trees, rooftops and distant hills. At night the real Milky Way, stars, planets and the Moon in its true phase stand over dark roofs.",
                  make: SkyWindow.init),
        SceneKind(id: "deep-field", name: "Webb and Hubble", family: .space,
                  about: "Real photographs from the James Webb and Hubble space telescopes, fetched from ESA's archives: the Pillars of Creation in visible and then infrared light, the Cosmic Cliffs of Carina, the Southern Ring, Stephan's Quintet, Webb's First Deep Field, the Hubble Ultra Deep Field, the Cartwheel, the Tarantula, the Crab, Orion and the Horsehead. Each drifts slowly for three minutes, labelled like a figure in a paper with its instrument, its filters and the colours they were given, a scale bar in light-years and the date, then dissolves into the next. A low drone takes its chord from each picture's colours, with a soft bell now and then for its stars.",
                  make: DeepField.init),
        SceneKind(id: "time", name: "Time, All of It", family: .earth,
                  about: "A finely set instrument of clocks: local time in large light figures, with UTC, Unix time, the Julian day, sidereal, atomic and GPS time, Mars time at the Perseverance rover, and how many hours Voyager 1's signals take to reach us, over faint turning dials.",
                  make: TimeScene.init, clock: false, gain: 2),
    ]
}
