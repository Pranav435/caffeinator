// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Caffeinator",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "Caffeinator"),
        .testTarget(name: "CaffeinatorTests", dependencies: ["Caffeinator"]),
    ]
)
