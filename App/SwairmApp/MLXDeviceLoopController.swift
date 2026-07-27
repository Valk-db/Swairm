// Drives one MLXDeviceLoop against a LAN Anchor from the UI.
// Mirrors DeviceLoopController API but uses real MLX DoRA training.
//
// Model & curriculum download: uses ModelDownloader to fetch shards and
// adapter from the Anchor before starting training.

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
    // Matches CI's local_dir naming (mlx-e2e job) -- copy the same
    // mlx-community/Qwen3-0.6B-bf16 files into Documents/mlx-model via
    // Files app to reuse the exact model this default expects.
    var modelPath = "mlx-model"
    var curriculumDirectory = "curriculum"
    var maxStepsPerRound = 60
    var batchSize = 2
    var sequenceLength = 128
    var learningRate: Float = 1e-4

    // Curriculum epoch to download
    var curriculumEpoch = 0

    // ------------------------------------------------------------ state
    private(set) var isRunning = false
    private(set) var isDownloading = false
    private(set) var anchorVersion: Int?
    private(set) var lastLoss: Float?
    private(set) var log: [LogEntry] = []

    private var runTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?

    typealias LogEntry = DeviceLoopController.LogEntry

    var deviceID: String { "phone\(deviceIndex)" }

    // Model downloader for fetching curriculum and adapters
    private let modelDownloader = ModelDownloader()

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
                    modelPath: resolvedInDocuments(modelPath),
                    targetModules: ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                    rankMap: ["": 6],           // uniform rank 6 for all target modules
                    alphaMap: ["": 16.0],       // uniform alpha 16 -> scale = 16/6
                    learningRate: learningRate,
                    weightDecay: 0.01,
                    maxGradNorm: 1.0,
                    warmupSteps: 10,
                    maxStepsPerRound: maxStepsPerRound,
                    batchSize: batchSize,
                    sequenceLength: sequenceLength,
                    curriculumDirectory: resolvedInDocuments(curriculumDirectory),
                    seed: 42 + UInt64(deviceIndex)
                )
            )

            isRunning = true
            append("Started \(deviceID) (MLX) against \(url.absoluteString)")

            let interval = intervalSeconds
            let maxSteps = maxStepsPerRound
            runTask = Task { [weak self] in
                let budget = ResourceBudget(
                    maxSteps: maxSteps,
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

    // ------------------------------------------------------------ Downloads
    /// Download curriculum shards for the configured epoch.
    func downloadCurriculum() {
        guard !isDownloading else { return }
        guard let url = URL(string: anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(anchorURLText)", isError: true)
            return
        }

        let anchor = AnchorClient(base: url)
        let epoch = curriculumEpoch

        isDownloading = true
        append("Downloading curriculum epoch \(epoch)...")

        downloadTask = Task { [weak self] in
            defer { self?.isDownloading = false }

            do {
                let epochDir = try await self?.modelDownloader.downloadCurriculum(
                    epoch: epoch,
                    downloader: anchor
                ) { progress in
                    Task { @MainActor in
                        self?.append(progress.message)
                    }
                }
                await MainActor.run {
                    if let dir = epochDir {
                        self?.append("Curriculum epoch \(epoch) ready at \(dir.lastPathComponent)")
                        self?.curriculumDirectory = "curriculum/epoch_\(epoch)"
                    }
                }
            } catch {
                await MainActor.run {
                    self?.append("Curriculum download failed: \(error)", isError: true)
                }
            }
        }
    }

    /// Download the latest model adapter from the Anchor.
    func downloadModel() {
        guard !isDownloading else { return }
        guard let url = URL(string: anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(anchorURLText)", isError: true)
            return
        }

        let anchor = AnchorClient(base: url)

        isDownloading = true
        append("Downloading latest model adapter...")

        downloadTask = Task { [weak self] in
            defer { self?.isDownloading = false }

            do {
                _ = try await self?.modelDownloader.downloadLatestModel(
                    anchor: anchor
                ) { progress in
                    Task { @MainActor in
                        self?.append(progress.message)
                    }
                }
                await MainActor.run {
                    self?.append("Model adapter downloaded successfully")
                }
            } catch {
                await MainActor.run {
                    self?.append("Model download failed: \(error)", isError: true)
                }
            }
        }
    }

    /// Cancel any in-progress download.
    func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        isDownloading = false
        append("Download cancelled")
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