// Drives one MLXDeviceLoop against a LAN Anchor from the UI.
// Mirrors DeviceLoopController API but uses real MLX DoRA training.

import SwiftUI
import UIKit
import SwairmCore

@MainActor
@Observable
final class MLXDeviceLoopController {
    // ------------------------------------------------------------ config
    var anchorURLText = "http://192.168.1.100:8000"
    var deviceIndex = 0
    /// Seconds to wait between rounds (mirrors the CLI --interval flag).
    var intervalSeconds = 25.0

    // MLX-specific config
    var modelPath = "models/Qwen2-0.5B-Instruct-4bit"
    var curriculumDirectory = "curriculum"
    var maxStepsPerRound = 60
    var batchSize = 2
    var sequenceLength = 128
    var learningRate: Float = 1e-4

    // ------------------------------------------------------------ state
    private(set) var isRunning = false
    private(set) var anchorVersion: Int?
    private(set) var lastLoss: Float?
    private(set) var log: [LogEntry] = []

    private var runTask: Task<Void, Never>?

    struct LogEntry: Identifiable {
        let id = UUID()
        let date = Date()
        let text: String
        let isError: Bool
    }

    var deviceID: String { "phone\(deviceIndex)" }

    // ------------------------------------------------------------ control
    func start() {
        guard !isRunning else { return }
        guard let url = URL(string: anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(anchorURLText)", isError: true)
            return
        }

        UIDevice.current.isBatteryMonitoringEnabled = true
        let anchor = AnchorClient(base: url)

        do {
            let loop = try MLXDeviceLoop(
                anchor: anchor,
                deviceID: deviceID,
                deviceIndex: deviceIndex,
                config: MLXLoopConfig(
                    modelPath: modelPath,
                    targetModules: ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                    rankMap: ["attn": 4, "mlp": 6],
                    alphaMap: ["attn": 16.0, "mlp": 16.0],
                    learningRate: learningRate,
                    weightDecay: 0.01,
                    maxGradNorm: 1.0,
                    warmupSteps: 10,
                    maxStepsPerRound: maxStepsPerRound,
                    batchSize: batchSize,
                    sequenceLength: sequenceLength,
                    curriculumDirectory: curriculumDirectory,
                    seed: 42 + UInt64(deviceIndex)
                )
            )

            isRunning = true
            append("Started \(deviceID) (MLX) against \(url.absoluteString)")

            let interval = intervalSeconds
            runTask = Task { [weak self] in
                let budget = ResourceBudget(
                    maxSteps: maxStepsPerRound,
                    maxWallClock: 300,
                    minBatteryFraction: 0.2
                )
                while !Task.isCancelled {
                    do {
                        let result = try await loop.runRound(budget: budget)
                        self?.record(result)
                    } catch is CancellationError {
                        break
                    } catch {
                        self?.append("Round failed: \(error)", isError: true)
                    }
                    try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                }
                self?.finish()
            }
        } catch {
            append("Failed to create MLX loop: \(error)", isError: true)
        }
    }

    func stop() {
        runTask?.cancel()
        runTask = nil
    }

    // ------------------------------------------------------------ private
    private func record(_ result: MLXRoundResult) {
        anchorVersion = result.status.version
        lastLoss = result.trainingReport.finalLoss
        append(String(
            format: "round %d | anchor v%d | loss %.4f | %d steps, %.1fs | %@",
            result.round, result.status.version, result.trainingReport.finalLoss ?? -1,
            result.trainingReport.stepsCompleted,
            result.trainingReport.wallClock,
            String(describing: result.trainingReport.termination)
        ))
    }

    private func finish() {
        isRunning = false
        append("Stopped")
    }

    private func append(_ text: String, isError: Bool = false) {
        log.append(LogEntry(text: text, isError: isError))
        if log.count > 200 { log.removeFirst(log.count - 200) }
    }
}