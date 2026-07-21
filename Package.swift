// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "SwairmClient",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.16"),
    ],
    targets: [
        .target(
            name: "SwairmCore",
            dependencies: [.product(name: "ZIPFoundation", package: "ZIPFoundation")]),
        .executableTarget(name: "swairm-client", dependencies: ["SwairmCore"]),
        .testTarget(name: "SwairmCoreTests", dependencies: ["SwairmCore"]),
    ]
)
