@MainActor extension SceneKind {
    static let mathScenes: [SceneKind] = [
        SceneKind(id: "attractors", name: "Strange Attractors", family: .math,
                  about: "A million glowing particles trace a chaotic attractor, the Lorenz butterfly and its cousins, while the camera slowly circles it.",
                  make: Attractors.init, gain: 0.55),
        SceneKind(id: "mandelbrot", name: "Mandelbrot Deep Zoom", family: .math,
                  about: "A deep zoom into the Mandelbrot set, photographed as if it were a relief etched into a metal plate under a studio softbox: black pockets for the set itself, bevelled ridges along its filaments and spirals, and thin-film colours like anodised titanium, with a shallow depth of field that blurs one corner. The view sinks more than ten trillion times deeper, then fades and starts again somewhere new.",
                  make: Mandelbrot.init),
        SceneKind(id: "pendulums", name: "Double Pendulums", family: .math,
                  about: "Two thousand double pendulums, each a thin glowing arm, are released almost exactly together. For a while they swing as one bright pendulum; then chaos fans them out into a rainbow of arms and trails, and every minute or two they start over.",
                  make: Pendulums.init, gain: 0.7),
        SceneKind(id: "lenia", name: "Lenia", family: .math,
                  about: "Darkfield microscope footage: a round field of view on black, speckled with out-of-focus debris and a drifting fibre, with a scale bar of one hundred micrometres. Translucent creatures about a hundred and twenty micrometres across, Orbium gliders from Lenia, a continuous cousin of the Game of Life, glide through it with bright blue-white membranes, amber edges and glowing halos. They sometimes collide and dissolve, and new ones drift in from the edge.",
                  make: Lenia.init, gain: 1.3),
        SceneKind(id: "reaction", name: "Belousov–Zhabotinsky Reaction", family: .math,
                  about: "A chemical clock in a glass petri dish on a dark lab bench, photographed from above: a thin layer of red-orange liquid where blue rings spread out from pacemaker spots and broken fronts curl into turning spirals, as the iron catalyst switches from red ferroin to blue ferriin and back. A few gas bubbles sit in the liquid, and the glass rim catches the light.",
                  make: Belousov.init, gain: 1.05),
        SceneKind(id: "fireflies", name: "Kuramoto Fireflies", family: .math,
                  about: "A long-exposure photograph of a moonlit summer meadow under a forest canopy, with the real Milky Way, as it stands now over your location, shining through a gap in the leaves. Two thousand fireflies draw short green-gold streaks in the mist. At first each flashes on its own; slowly they fall into step until the whole meadow pulses together in waves, like the synchronous fireflies of the Great Smoky Mountains, and then a breeze scatters them and it begins again.",
                  make: Fireflies.init, gain: 1.35),
    ]
}
