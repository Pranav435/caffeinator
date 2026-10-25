@MainActor extension SceneKind {
    static let spaceScenes: [SceneKind] = [
        SceneKind(id: "night-sky", name: "Night Sky", family: .space,
                  about: "The real sky above you right now, over a dark horizon of layered mountains, hills and a ragged treeline: ESO's photograph of the Milky Way with its dust lanes and star clouds, thousands of catalogue stars in their true colours, faint constellation figures and names, the planets, the Moon in its current phase with the real surface of the Moon, and meteors from any active shower. The air dims and reddens things near the horizon, green airglow and a faint warm town glow touch the sky, and the view drifts slowly round the compass.",
                  make: NightSky.init, gain: 1.15),
        SceneKind(id: "orrery", name: "Orrery", family: .space,
                  about: "The solar system seen from above against the real Milky Way, every planet where it really is today, then running forward three days a second. The planets are lit globes wearing their real maps, Saturn with shadowed rings, Earth with oceans and cloud, the Moon circling it, and the Sun is NASA's live Solar Dynamics Observatory image with its sunspots. The asteroid belt and Jupiter's Trojans are scattered between. Each planet hums its orbital frequency raised by octaves, as Kepler imagined.",
                  make: Orrery.init, gain: 1.3),
        SceneKind(id: "pulsars", name: "Pulsar Map", family: .space,
                  about: "The pulsar map from the Pioneer plaques and the Voyager Golden Record, hung in front of ESO's photograph of the real Milky Way: fourteen fine golden lines run out from the Sun, each as long as its pulsar is far and marked in binary with its spin period. Each pulsar sends out a pair of pale radio beams like a lighthouse and flickers at its true rate. The view turns slowly, coming close to two pulsars at a time, which click at their real rhythms.",
                  make: Pulsars.init, gain: 2.0),
        SceneKind(id: "gravitational-waves", name: "Gravitational Waves", family: .space,
                  about: "Real black hole mergers detected by LIGO and Virgo. Two black holes spiral together as lenses that bend and smear the real Milky Way behind them, then merge and ring down. In the corner, the actual strain from the Hanford and Livingston detectors around the event, whitened and filtered, with a spectrogram showing the rising chirp, and the sound is that real data sped up into hearing range.",
                  make: GravitationalWaves.init, gain: 1.8),
        SceneKind(id: "exoplanets", name: "Every Known Exoplanet", family: .space,
                  about: "Every confirmed planet around another star, thousands of orbits drawn on a logarithmic scale against the real Milky Way, each planet moving at a pace set by its real year and coloured by its temperature. Famous systems take turns in the spotlight with a realistic host star, limb-darkened and glowing at its true temperature, and some show real telescope photographs such as HR 8799 from Webb. Now and then the asteroids passing Earth this month fly by.",
                  make: Exoplanets.init, gain: 1.9),
    ]
}
