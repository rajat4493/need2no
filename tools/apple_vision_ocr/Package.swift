// swift-tools-version:5.8
import PackageDescription

let package = Package(
    name: "AppleVisionOCR",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "AppleVisionOCR", targets: ["AppleVisionOCR"])
    ],
    targets: [
        .executableTarget(
            name: "AppleVisionOCR",
            path: "Sources"
        )
    ]
)
