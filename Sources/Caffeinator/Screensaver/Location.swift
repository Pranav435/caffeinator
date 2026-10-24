import CoreLocation
import Foundation

/// Where the Mac is, for the scenes that show the sky overhead. Uses Location Services once allowed, and until then
/// the main city of the current time zone, read from the system's zone table.
@MainActor @Observable final class Location: NSObject, CLLocationManagerDelegate {
    static let shared = Location()

    /// Degrees, north and east positive.
    private(set) var latitude: Double
    private(set) var longitude: Double
    /// "San Francisco", or the time zone's city when there's no fix.
    private(set) var place: String
    private(set) var precise: Bool
    @ObservationIgnored var onChange: () -> Void = {}
    /// Set while the screensaver is up: macOS's permission prompt would fight it for the keyboard, so a scene's
    /// request waits for `askIfWaiting()` once you've ended it.
    @ObservationIgnored var holdPrompt = false

    @ObservationIgnored private var manager: CLLocationManager?
    @ObservationIgnored private var waiting = false

    private override init() {
        let d = UserDefaults.standard
        if let lat = d.object(forKey: Key.latitude) as? Double, let lon = d.object(forKey: Key.longitude) as? Double {
            latitude = lat
            longitude = lon
            place = d.string(forKey: Key.place) ?? ""
            precise = true
        } else {
            let guess = Self.timeZoneGuess()
            latitude = guess.lat
            longitude = guess.lon
            place = guess.place
            precise = false
        }
        super.init()
    }

    var status: CLAuthorizationStatus { CLLocationManager().authorizationStatus }

    /// Asks for permission if needed, then for one fix. Safe to call often.
    func request() {
        if manager == nil {
            let m = CLLocationManager()
            m.delegate = self
            m.desiredAccuracy = kCLLocationAccuracyKilometer
            manager = m
        }
        guard let manager else { return }
        switch manager.authorizationStatus {
        case .notDetermined:
            waiting = holdPrompt
            if !holdPrompt { manager.requestWhenInUseAuthorization() }
        case .authorizedAlways: manager.requestLocation() // what macOS grants for when-in-use
        default: break
        }
    }

    /// Shows the permission prompt a scene asked for while it was held.
    func askIfWaiting() {
        guard waiting, !holdPrompt else { return }
        request()
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        MainActor.assumeIsolated {
            if let manager = self.manager, manager.authorizationStatus == .authorizedAlways { manager.requestLocation() }
            onChange()
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let fix = locations.last else { return }
        MainActor.assumeIsolated { update(fix) }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}

    private func update(_ fix: CLLocation) {
        latitude = fix.coordinate.latitude
        longitude = fix.coordinate.longitude
        precise = true
        let d = UserDefaults.standard
        d.set(latitude, forKey: Key.latitude)
        d.set(longitude, forKey: Key.longitude)
        onChange()
        CLGeocoder().reverseGeocodeLocation(fix) { marks, _ in
            let name = marks?.first.flatMap { $0.locality ?? $0.administrativeArea ?? $0.country }
            Task { @MainActor in
                guard let name else { return }
                self.place = name
                UserDefaults.standard.set(name, forKey: Key.place)
                self.onChange()
            }
        }
    }

    /// The coordinates of the current time zone's city from zone1970.tab, such as +3403-11814 for Los Angeles.
    nonisolated static func timeZoneGuess(_ zone: String = TimeZone.current.identifier) -> (lat: Double, lon: Double, place: String) {
        let place = zone.split(separator: "/").last.map { $0.replacingOccurrences(of: "_", with: " ") } ?? zone
        for path in ["/usr/share/zoneinfo/zone1970.tab", "/usr/share/zoneinfo/zone.tab"] {
            guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { continue }
            for line in text.split(separator: "\n") where !line.hasPrefix("#") {
                let fields = line.split(separator: "\t")
                guard fields.count >= 3, fields[2] == zone, let c = parse(String(fields[1])) else { continue }
                return (c.lat, c.lon, place)
            }
        }
        return (51.48, 0, "Greenwich")
    }

    /// Parses ISO 6709 "+DDMM+DDDMM" or "+DDMMSS+DDDMMSS".
    nonisolated static func parse(_ s: String) -> (lat: Double, lon: Double)? {
        guard let split = s.dropFirst().firstIndex(where: { $0 == "+" || $0 == "-" }) else { return nil }
        func value(_ part: Substring, degreeDigits: Int) -> Double? {
            let sign: Double = part.first == "-" ? -1 : 1
            let digits = part.dropFirst()
            guard digits.count >= degreeDigits + 2, let deg = Double(digits.prefix(degreeDigits)),
                  let min = Double(digits.dropFirst(degreeDigits).prefix(2)) else { return nil }
            let sec = Double(digits.dropFirst(degreeDigits + 2).prefix(2)) ?? 0
            return sign * (deg + min / 60 + sec / 3600)
        }
        guard let lat = value(s[..<split], degreeDigits: 2), let lon = value(s[split...], degreeDigits: 3) else { return nil }
        return (lat, lon)
    }
}
