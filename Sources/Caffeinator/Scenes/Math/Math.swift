@MainActor extension SceneKind {
    static let mathScenes: [SceneKind] = [
        SceneKind(id: "attractors", name: "Strange Attractors", family: .math,
                  about: "A million glowing particles trace a chaotic attractor, the Lorenz butterfly and its cousins, while the camera slowly circles it.",
                  make: Attractors.init, gain: 0.65),
        SceneKind(id: "mandelbrot", name: "Mandelbrot Deep Zoom", family: .math,
                  about: "A slow, endless dive into the Mandelbrot set: glowing filaments, spirals and tiny copies of the whole set rush outward as the view sinks more than ten trillion times deeper, then fades and starts again somewhere new.",
                  make: Mandelbrot.init),
        SceneKind(id: "pendulums", name: "Double Pendulums", family: .math,
                  about: "Two thousand double pendulums, each a thin glowing arm, are released almost exactly together. For a while they swing as one bright pendulum; then chaos fans them out into a rainbow of arms and trails, and every minute or two they start over.",
                  make: Pendulums.init, gain: 0.7),
        SceneKind(id: "lenia", name: "Lenia", family: .math,
                  about: "Glowing creatures made of nothing but a simple rule glide through a dark sea: Orbium gliders from Lenia, a continuous cousin of the Game of Life. They swim as a loose school, sometimes collide and dissolve, and new ones drift in from the edges.",
                  make: Lenia.init, gain: 1.3),
        SceneKind(id: "reaction", name: "Reaction–Diffusion", family: .math,
                  about: "A glossy relief of two reacting chemicals, lit like molten copper, deep-sea glass or gold-veined obsidian. Spots grow and divide, corals branch, worms crawl and mazes fold as the reaction's feed and kill rates drift slowly from one named pattern to the next.",
                  make: ReactionDiffusion.init, gain: 0.75),
        SceneKind(id: "fireflies", name: "Kuramoto Fireflies", family: .math,
                  about: "Two thousand fireflies blink over a dark summer meadow below a black tree line. At first each flashes on its own; slowly they fall into step until the whole meadow pulses together in soft yellow-green waves that roll out across the grass, like the synchronous fireflies of the Great Smoky Mountains, and then a breeze scatters them and it begins again.",
                  make: Fireflies.init, gain: 1.35),
    ]
}
