@MainActor extension SceneKind {
    static let spaceScenes: [SceneKind] = [
        SceneKind(id: "night-sky", name: "Night Sky", family: .space,
                  about: "The real sky above you right now: thousands of stars in their true colours, the Milky Way, faint constellation figures with their names, the planets and the Moon in its current phase, and meteors from any active shower, over a dark horizon of hills and trees. The view drifts slowly round the compass while the sky turns.",
                  make: NightSky.init),
        SceneKind(id: "orrery", name: "Orrery", family: .space,
                  about: "The solar system seen from above, with every planet where it really is today, then running forward three days a second: glowing planets on fine orbit lines around a bright Sun, the asteroid belt and Jupiter's Trojans between. Each planet hums its own orbital frequency raised by octaves, as Kepler imagined.",
                  make: Orrery.init, gain: 1.3),
        SceneKind(id: "pulsars", name: "Pulsar Map", family: .space,
                  about: "The pulsar map from the Pioneer plaques and the Voyager Golden Record, rebuilt in 3D over a faint spiral Milky Way: fourteen golden lines run out from the Sun, each as long as its pulsar is far and marked in binary with its spin period, and each pulsar flickers at its true rate. The view turns slowly, coming close to two pulsars at a time, which click at their real rhythms.",
                  make: Pulsars.init, gain: 1.8),
        SceneKind(id: "gravitational-waves", name: "Gravitational Waves", family: .space,
                  about: "Real black hole and neutron star mergers detected by LIGO, Virgo and KAGRA. Two bodies spiral together on a glowing grid of spacetime that ripples outward in a two-armed spiral, then merge in a flash and ring down, while a strain trace like LIGO's runs along the bottom and the chirp plays from their real masses.",
                  make: GravitationalWaves.init),
        SceneKind(id: "exoplanets", name: "Every Known Exoplanet", family: .space,
                  about: "Every confirmed planet around another star, thousands of orbits drawn around one shared star on a logarithmic scale, each planet moving at a pace set by its real year, coloured from icy blue to red-hot by its temperature. Famous systems like TRAPPIST-1 take turns in the spotlight, and now and then the asteroids passing Earth this month fly by.",
                  make: Exoplanets.init, gain: 1.6),
    ]
}
