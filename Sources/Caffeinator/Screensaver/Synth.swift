import AVFoundation
import os

/// The soundscape engine: a drone of sine partials, filtered noise, and one-shot voices (bells, plucks, clicks,
/// thumps and gravitational-wave chirps) through a hall reverb. Scenes set targets from the main thread;
/// the audio thread picks them up with a try-lock, so it never waits on the main thread.
final class Synth: @unchecked Sendable {
    enum Voice: Int32 { case bell, pluck, click, thump, chirp, tone }

    static let partials = 8
    private static let maxVoices = 64
    private static let maxEvents = 64

    private struct Partial { var freq: Float = 0, amp: Float = 0, pan: Float = 0 }
    private struct Event { var voice = Voice.bell, freq: Float = 0, freq2: Float = 0, amp: Float = 0, pan: Float = 0, length: Float = 1 }
    private struct Playing {
        var event = Event()
        var t: Float = 0
        var phase: Float = 0
        var phase2: Float = 0
        var active = false
    }
    private struct Noise { var level: Float = 0, cutoff: Float = 800, crackle: Float = 0, crackleTone: Float = 3000 }

    private let engine = AVAudioEngine()
    private let reverb = AVAudioUnitReverb()
    private var node: AVAudioSourceNode?

    // Written by the main thread under the lock.
    private let lock: UnsafeMutablePointer<os_unfair_lock>
    private let target: UnsafeMutablePointer<Partial>
    private let events: UnsafeMutablePointer<Event>
    private var eventCount = 0
    private var noiseTarget = Noise()
    private var gainTarget: Float = 1

    // Audio thread only.
    private let goal: UnsafeMutablePointer<Partial>
    private let current: UnsafeMutablePointer<Partial>
    private let phases: UnsafeMutablePointer<Double>
    private let voices: UnsafeMutablePointer<Playing>
    private var noise = Noise()
    private var lowL: Float = 0, lowR: Float = 0, brown: Float = 0, rng: UInt32 = 0x9E37_79B9
    private var gainNow: Float = 1, gainGoal: Float = 1
    private var sampleRate: Float = 48000
    /// Renders into a buffer on request instead of to the speakers, for previews and level checks.
    private let offline: Bool

    init(offline: Bool = false) {
        self.offline = offline
        lock = .allocate(capacity: 1)
        lock.initialize(to: os_unfair_lock())
        target = .allocate(capacity: Self.partials)
        target.initialize(repeating: Partial(), count: Self.partials)
        goal = .allocate(capacity: Self.partials)
        goal.initialize(repeating: Partial(), count: Self.partials)
        current = .allocate(capacity: Self.partials)
        current.initialize(repeating: Partial(), count: Self.partials)
        phases = .allocate(capacity: Self.partials)
        phases.initialize(repeating: 0, count: Self.partials)
        events = .allocate(capacity: Self.maxEvents)
        events.initialize(repeating: Event(), count: Self.maxEvents)
        voices = .allocate(capacity: Self.maxVoices)
        voices.initialize(repeating: Playing(), count: Self.maxVoices)
    }

    deinit {
        engine.stop()
        lock.deallocate()
        target.deallocate()
        goal.deallocate()
        current.deallocate()
        phases.deallocate()
        events.deallocate()
        voices.deallocate()
    }

    var volume: Float {
        get { engine.mainMixerNode.outputVolume }
        set { engine.mainMixerNode.outputVolume = newValue }
    }

    func start() {
        if node == nil {
            let format = engine.outputNode.inputFormat(forBus: 0)
            sampleRate = offline || format.sampleRate <= 0 ? 48000 : Float(format.sampleRate)
            let stereo = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 2)!
            if offline { try? engine.enableManualRenderingMode(.offline, format: stereo, maximumFrameCount: 4096) }
            let node = AVAudioSourceNode(format: stereo) { [unowned self] _, _, frames, list in
                self.render(Int(frames), UnsafeMutableAudioBufferListPointer(list))
                return noErr
            }
            reverb.loadFactoryPreset(.largeHall2)
            reverb.wetDryMix = 38
            engine.attach(node)
            engine.attach(reverb)
            engine.connect(node, to: reverb, format: stereo)
            engine.connect(reverb, to: engine.mainMixerNode, format: stereo)
            self.node = node
        }
        guard !engine.isRunning else { return }
        engine.prepare()
        try? engine.start()
    }

    func stop() { engine.stop() }

    /// Offline only: renders the next `seconds` of sound.
    func render(seconds: Double) -> AVAudioPCMBuffer? {
        guard offline, engine.isRunning else { return nil }
        let frames = AVAudioFrameCount(seconds * Double(sampleRate))
        guard let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: frames) else { return nil }
        while buffer.frameLength < frames {
            guard let chunk = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: min(4096, frames - buffer.frameLength)),
                  (try? engine.renderOffline(chunk.frameCapacity, to: chunk)) == .success else { break }
            for c in 0..<Int(buffer.format.channelCount) {
                (buffer.floatChannelData![c] + Int(buffer.frameLength)).update(from: chunk.floatChannelData![c], count: Int(chunk.frameLength))
            }
            buffer.frameLength += chunk.frameLength
        }
        return buffer
    }

    // MARK: Scene API (main thread)

    /// Sets drone partial `i` (0..<8). The audio thread glides to new values, so this can be called every frame.
    func drone(_ i: Int, _ freq: Float, _ amp: Float, pan: Float = 0) {
        guard i >= 0, i < Self.partials else { return }
        os_unfair_lock_lock(lock)
        target[i] = Partial(freq: freq, amp: amp, pan: pan)
        os_unfair_lock_unlock(lock)
    }

    /// Filtered brown noise (wind, rumble, surf) plus random crackles (rain, static, Geiger counts per second).
    func noise(_ level: Float, cutoff: Float = 800, crackle: Float = 0, crackleTone: Float = 3000) {
        os_unfair_lock_lock(lock)
        noiseTarget = Noise(level: level, cutoff: cutoff, crackle: crackle, crackleTone: crackleTone)
        os_unfair_lock_unlock(lock)
    }

    /// Plays a one-shot. `freq2` is the chirp's end frequency. `length` is the decay time in seconds
    /// (for a chirp, the time to merger).
    func play(_ voice: Voice, _ freq: Float, amp: Float = 0.3, pan: Float = 0, length: Float = 2, to freq2: Float = 0) {
        os_unfair_lock_lock(lock)
        if eventCount < Self.maxEvents {
            events[eventCount] = Event(voice: voice, freq: freq, freq2: freq2, amp: amp, pan: max(-1, min(1, pan)), length: max(length, 0.01))
            eventCount += 1
        }
        os_unfair_lock_unlock(lock)
    }

    /// The current scene's loudness trim. The audio thread glides to it over about a second.
    var gain: Float {
        get { os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }; return gainTarget }
        set { os_unfair_lock_lock(lock); gainTarget = max(newValue, 0); os_unfair_lock_unlock(lock) }
    }

    /// Silences the drone and noise and lets ringing voices fade.
    func reset() {
        os_unfair_lock_lock(lock)
        for i in 0..<Self.partials { target[i].amp = 0 }
        noiseTarget = Noise()
        eventCount = 0
        os_unfair_lock_unlock(lock)
    }

    // MARK: Audio thread

    private func render(_ frames: Int, _ buffers: UnsafeMutableAudioBufferListPointer) {
        if os_unfair_lock_trylock(lock) {
            noise.level = noiseTarget.level
            noise.cutoff = noiseTarget.cutoff
            noise.crackle = noiseTarget.crackle
            noise.crackleTone = noiseTarget.crackleTone
            gainGoal = gainTarget
            for e in 0..<eventCount {
                if let slot = (0..<Self.maxVoices).first(where: { !voices[$0].active }) {
                    voices[slot] = Playing(event: events[e], active: true)
                }
            }
            eventCount = 0
            for i in 0..<Self.partials {
                current[i].pan = target[i].pan
                if current[i].amp < 0.0001 { current[i].freq = target[i].freq }
                goal[i] = target[i]
            }
            os_unfair_lock_unlock(lock)
        }
        guard buffers.count >= 2, let left = buffers[0].mData?.assumingMemoryBound(to: Float.self),
              let right = buffers[1].mData?.assumingMemoryBound(to: Float.self) else { return }

        let sr = sampleRate
        let glide = 1 - exp(-1 / (0.25 * sr)) // quarter-second glides
        let lowCoef = 1 - exp(-2 * Float.pi * max(noise.cutoff, 20) / sr)
        let crackleChance = noise.crackle / sr
        let twoPi = Float.pi * 2

        for n in 0..<frames {
            var l: Float = 0, r: Float = 0

            for i in 0..<Self.partials {
                current[i].amp += (goal[i].amp - current[i].amp) * glide
                current[i].freq += (goal[i].freq - current[i].freq) * glide
                let a = current[i].amp
                guard a > 0.00001 else {
                    if goal[i].amp == 0 { current[i].amp = 0 } // no denormals, which are slow on Intel
                    continue
                }
                phases[i] += Double(current[i].freq / sr)
                if phases[i] > 1 { phases[i] -= 1 }
                let s = sin(Float(phases[i]) * twoPi) * a
                l += s * sqrt(0.5 - current[i].pan * 0.5)
                r += s * sqrt(0.5 + current[i].pan * 0.5)
            }

            if noise.level > 0 {
                brown = (brown + (random() * 2 - 1) * 0.02) * 0.998
                lowL += (brown * 3 - lowL) * lowCoef
                lowR += ((random() * 2 - 1) * 0.15 + brown * 2.6 - lowR) * lowCoef
                l += lowL * noise.level
                r += lowR * noise.level
            }
            if crackleChance > 0, random() < crackleChance, let slot = (0..<Self.maxVoices).first(where: { !voices[$0].active }) {
                voices[slot] = Playing(event: Event(voice: .click, freq: noise.crackleTone * (0.7 + random() * 0.6),
                                                    amp: 0.05 + random() * 0.08, pan: random() * 1.6 - 0.8, length: 0.02), active: true)
            }

            for v in 0..<Self.maxVoices where voices[v].active {
                let s = voice(&voices[v], sr)
                let pan = voices[v].event.pan
                l += s * sqrt(0.5 - pan * 0.5)
                r += s * sqrt(0.5 + pan * 0.5)
            }

            gainNow += (gainGoal - gainNow) * glide * 0.25 // about a second
            left[n] = tanh(l * gainNow)
            right[n] = tanh(r * gainNow)
        }
    }

    private func voice(_ p: inout Playing, _ sr: Float) -> Float {
        let e = p.event
        let t = p.t
        p.t += 1 / sr
        let twoPi = Float.pi * 2
        var out: Float = 0
        switch e.voice {
        case .bell:
            // Two-operator FM with an inharmonic ratio, the classic struck-metal sound.
            let index = 2.4 * exp(-t * 3)
            p.phase2 += e.freq * 1.4 / sr
            p.phase += e.freq / sr
            out = sin(twoPi * p.phase + index * sin(twoPi * p.phase2)) * exp(-t * 4 / e.length)
        case .pluck:
            p.phase += e.freq / sr
            out = (sin(twoPi * p.phase) * exp(-t * 5 / e.length) + 0.35 * sin(twoPi * p.phase * 2) * exp(-t * 11 / e.length))
        case .click:
            p.phase += e.freq / sr
            out = (random() * 2 - 1) * 0.6 * exp(-t * 900) + sin(twoPi * p.phase) * exp(-t / e.length * 5)
        case .thump:
            p.phase += e.freq * (1 + 2 * exp(-t * 25)) / sr
            out = sin(twoPi * p.phase) * exp(-t * 3 / e.length) * min(t * 400, 1)
        case .chirp:
            // Inspiral: frequency rises as (time to merger)^(-3/8), amplitude as frequency^(2/3); then a ringdown.
            // phase2 holds the moment of merger.
            let left = e.length - t
            let f = left > 0 ? min(e.freq * pow(e.length / left, 0.375), e.freq2) : e.freq2
            if f >= e.freq2, p.phase2 == 0 { p.phase2 = max(t, 1e-6) }
            p.phase += f / sr
            let ring: Float = p.phase2 > 0 ? exp(-(t - p.phase2) * 14) : 1
            out = sin(twoPi * p.phase) * pow(f / e.freq2, 0.66) * ring * min(t * 20, 1)
            if p.phase2 > 0, t - p.phase2 > 0.5 { p.active = false }
        case .tone:
            p.phase += e.freq / sr
            out = sin(twoPi * p.phase) * min(t * 4, 1) * min(max(e.length - t, 0) * 4, 1)
        }
        if p.phase > 1 { p.phase -= floor(p.phase) }
        if e.voice != .chirp, p.phase2 > 1 { p.phase2 -= floor(p.phase2) }
        if t > e.length * 1.6 + 0.05, e.voice != .chirp { p.active = false }
        return out * e.amp
    }

    private func random() -> Float {
        rng ^= rng << 13
        rng ^= rng >> 17
        rng ^= rng << 5
        return Float(rng) / 4_294_967_296
    }
}
