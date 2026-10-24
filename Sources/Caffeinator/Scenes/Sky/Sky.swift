@MainActor extension SceneKind {
    static let skyScenes: [SceneKind] = [
        SceneKind(id: "live-earth", name: "Live Earth", family: .earth,
                  about: "The whole Earth as the nearest weather satellite sees it now: true colour by day, city lights at night. The last few hours loop as a time-lapse, so clouds stream and sunrise sweeps across the globe, which floats in space inside a thin blue glow.",
                  make: LiveEarth.init),
        SceneKind(id: "sun", name: "The Sun Today", family: .earth,
                  about: "NASA's Solar Dynamics Observatory's newest pictures of the Sun, dissolving from wavelength to wavelength: the orange visible surface with its sunspots, golden coronal loops, red prominences, teal flare plasma and the black-and-white magnetic field, slowly turning.",
                  make: SunToday.init),
        SceneKind(id: "sky-window", name: "Window to the Sky", family: .earth,
                  about: "The sky over you right now, through a window: its real colour for where the Sun is, today's clouds drifting with the real wind, rain, snow or mist when it's really falling, and at night the real stars, planets and the Moon in its true phase above dark rooftops and trees.",
                  make: SkyWindow.init),
        SceneKind(id: "time", name: "Time, All of It", family: .earth,
                  about: "A finely set instrument of clocks: local time in large light figures, with UTC, Unix time, the Julian day, sidereal, atomic and GPS time, Mars time at the Perseverance rover, and how many hours Voyager 1's signals take to reach us, over faint turning dials.",
                  make: TimeScene.init, clock: false, gain: 2),
    ]
}
