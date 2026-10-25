@MainActor extension SceneKind {
    static let labScenes: [SceneKind] = [
        SceneKind(id: "double-slit", name: "Double Slit", family: .physics,
                  about: "A detector screen, filmed in the dark like Tonomura's 1989 single-electron experiment. A faint teal electron wave, solved from the Schrödinger equation, passes through slits in a thin plate and fans into bands. On a phosphor screen at the right, single electrons land as green flashes, one at a time, and the dots slowly pile up into the striped interference pattern.",
                  make: DoubleSlit.init, gain: 1.2),
        SceneKind(id: "hydrogen", name: "Hydrogen Orbitals", family: .physics,
                  about: "The electron cloud of a hydrogen atom, seen as a detector image like the 2013 photoionisation microscope: individual electron hits glow green on a phosphor plate, with counting noise and a faint channel-plate mesh, and pile up into the exact orbital shape from the Schrödinger wavefunction. It steps from the round 1s to many-lobed 4f and beyond, and through mixtures of two levels whose charge sloshes or circles at a real spectral-line frequency.",
                  make: Hydrogen.init, gain: 0.8),
        SceneKind(id: "cloud-chamber", name: "Cloud Chamber", family: .physics,
                  about: "Real footage of a diffusion cloud chamber: a sheet of alcohol vapour lit from the side over black felt, with mist drifting through it. Particle tracks condense out of the vapour as thin white trails that spread and drift. Long straight lines are cosmic-ray muons, short thick bright ones are alpha particles from radon, curly wisps are electrons, and sometimes a shower sprays out of one point.",
                  make: CloudChamber.init, gain: 1.3),
        SceneKind(id: "chladni", name: "Chladni Plate", family: .physics,
                  about: "A brass or aluminium plate, square or round, on a central post, photographed from above at an angle under a studio lamp, with scratched metal and soft shadow. White sand dances off the vibrating plate and settles on the still lines, drawing sharp curving figures. Then the tone glides and the figure dissolves into a new one.",
                  make: Chladni.init),
    ]
}
