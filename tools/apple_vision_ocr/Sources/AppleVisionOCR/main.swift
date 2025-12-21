import AppKit
import Foundation
import Vision

struct Arguments {
    let imagePath: String
    let roi: CGRect
    let language: String
    let digitsOnly: Bool
}

func parseArguments() -> Arguments {
    var imagePath = ""
    var roi = CGRect(x: 0, y: 0, width: 0, height: 0)
    var language = "en"
    var digitsOnly = false
    var iterator = CommandLine.arguments.makeIterator()
    _ = iterator.next() // skip executable name
    while let arg = iterator.next() {
        switch arg {
        case "--image":
            imagePath = iterator.next() ?? ""
        case "--roi":
            if let raw = iterator.next() {
                let parts = raw.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
                if parts.count == 4 {
                    roi = CGRect(x: parts[0], y: parts[1], width: parts[2] - parts[0], height: parts[3] - parts[1])
                }
            }
        case "--lang":
            language = iterator.next() ?? "en"
        case "--digits-only":
            digitsOnly = (iterator.next() == "1")
        default:
            continue
        }
    }
    return Arguments(imagePath: imagePath, roi: roi, language: language, digitsOnly: digitsOnly)
}

func loadImage(path: String) -> CGImage? {
    guard let nsImage = NSImage(contentsOfFile: path) else { return nil }
    var proposedRect = CGRect.zero
    return nsImage.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil)
}

func cropImage(_ image: CGImage, to rect: CGRect) -> CGImage {
    guard rect.width > 0, rect.height > 0 else { return image }
    let bounded = rect.intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return image.cropping(to: bounded) ?? image
}

func main() {
    let args = parseArguments()
    guard !args.imagePath.isEmpty, let image = loadImage(path: args.imagePath) else {
        fputs("Invalid image path\n", stderr)
        exit(1)
    }
    let cropped = cropImage(image, to: args.roi)
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = [args.language]
    if args.digitsOnly {
        request.customWords = (0...9).map { String($0) }
    }
    let handler = VNImageRequestHandler(cgImage: cropped, options: [:])
    do {
        try handler.perform([request])
    } catch {
        fputs("Vision request failed: \(error)\n", stderr)
        exit(2)
    }
    var words: [[String: Any]] = []
    var textSegments: [String] = []
    var confidences: [Double] = []
    request.results?.forEach { observation in
        guard let candidate = observation.topCandidates(1).first else { return }
        textSegments.append(candidate.string)
        confidences.append(Double(observation.confidence))
        let bbox = observation.boundingBox
        words.append([
            "text": candidate.string,
            "confidence": Double(observation.confidence),
            "bbox": [
                bbox.origin.x,
                bbox.origin.y,
                bbox.origin.x + bbox.size.width,
                bbox.origin.y + bbox.size.height,
            ],
        ])
    }
    let avgConf = confidences.isEmpty ? 0.0 : confidences.reduce(0, +) / Double(confidences.count)
    let payload: [String: Any] = [
        "text": textSegments.joined(separator: " "),
        "avg_conf": avgConf,
        "words": words,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: payload, options: []) {
        FileHandle.standardOutput.write(data)
    } else {
        fputs("Failed to encode JSON\n", stderr)
        exit(3)
    }
}

main()
