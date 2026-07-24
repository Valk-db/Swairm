// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Swairm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(name: "SwairmCore", targets: ["SwairmCore"]),
        .executable(name: "swairm-client", targets: ["swairm-client"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.19")
    ],
    targets: [
        .target(
            name: "SwairmCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                "ZIPFoundation"
            ]
        ),
        .executableTarget(
            name: "swairm-client",
            dependencies: ["SwairmCore"]
        ),
        .testTarget(
            name: "SwairmCoreTests",
            dependencies: ["SwairmCore"]
        )
    ]
)