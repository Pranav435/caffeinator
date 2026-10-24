@MainActor extension SceneKind {
    static let labScenes: [SceneKind] = [
        SceneKind(id: "double-slit", name: "Double Slit", family: .physics,
                  about: "An electron's wave, coloured by its phase, ripples through two narrow slits in a dark wall and spreads into fanning bands. On a detector screen at the right, single electrons land one dot at a time and slowly build the striped interference pattern.",
                  make: DoubleSlit.init, gain: 1.2),
        SceneKind(id: "hydrogen", name: "Hydrogen Orbitals", family: .physics,
                  about: "The electron cloud of a hydrogen atom glows like a slowly turning nebula, drawn from the exact wavefunction, with colour showing the wave's phase. It steps through orbitals, from the round 1s to many-lobed 4f and beyond, and through mixtures of two levels whose charge sloshes or circles in time.",
                  make: Hydrogen.init),
        SceneKind(id: "cloud-chamber", name: "Cloud Chamber", family: .physics,
                  about: "Looking down into a cloud chamber: faint mist drifts over a black floor while particle tracks condense out of the air as thin white trails. Long straight lines are cosmic-ray muons, short thick bright ones are alpha particles from radon, and curly wisps are electrons; each trail drifts, spreads and fades.",
                  make: CloudChamber.init, gain: 1.6),
        SceneKind(id: "chladni", name: "Chladni Plate", family: .physics,
                  about: "A dark steel plate seen from above, dusted with a quarter of a million grains of warm sand. The plate hums in one vibration mode; sand dances off the moving parts and settles into sharp lace-like lines where the plate stands still, then the tone glides and the figure dissolves into a new one.",
                  make: Chladni.init),
    ]
}
