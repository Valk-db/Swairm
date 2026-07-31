// Drives one MLXDeviceLoop against a LAN Anchor from the UI.
// Mirrors DeviceLoopController API but uses real MLX DoRA training.
//
// Model & curriculum download: uses ModelDownloader to fetch shards and
// adapter from the Anchor before starting training.

import SwiftUI
import UIKit
import SwairmCore

// Clamping helper for numeric bounds
extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}

// Settings persisted via UserDefaults — plain class with computed properties.
// NOT @Observable because the Observation macro requires stored properties.
// MLXDeviceLoopController is @Observable and owns an MLXSettings instance;
// SwiftUI keypath observation through $mlxController.settings.prop works fine.
@MainActor
final class MLXSettings {
    var anchorURLText: String {
        get { UserDefaults.standard.string(forKey: "mlx.anchorURLText") ?? "http://172.20.10.5:8000" }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.anchorURLText") }
    }
    var deviceIndex: Int {
        get {
            let v = UserDefaults.standard.integer(forKey: "mlx.deviceIndex")
            return max(0, min(v, 63))
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.deviceIndex") }
    }
    var intervalSeconds: Double {
        get {
            let v = UserDefaults.standard.double(forKey: "mlx.intervalSeconds")
            return (v == 0 ? 25.0 : v).clamped(to: 5...300)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.intervalSeconds") }
    }
    var modelPath: String {
        get { UserDefaults.standard.string(forKey: "mlx.modelPath") ?? "mlx-model" }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.modelPath") }
    }
    var curriculumDirectory: String {
        get { UserDefaults.standard.string(forKey: "mlx.curriculumDirectory") ?? "curriculum" }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.curriculumDirectory") }
    }
    var maxStepsPerRound: Int {
        get {
            let v = UserDefaults.standard.integer(forKey: "mlx.maxStepsPerRound")
            return (v == 0 ? 1 : v).clamped(to: 1...500)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.maxStepsPerRound") }
    }
    var batchSize: Int {
        get {
            let v = UserDefaults.standard.integer(forKey: "mlx.batchSize")
            return (v == 0 ? 1 : v).clamped(to: 1...4)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.batchSize") }
    }
    var sequenceLength: Int {
        get {
            let v = UserDefaults.standard.integer(forKey: "mlx.sequenceLength")
            return (v == 0 ? 64 : v).clamped(to: 32...256)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.sequenceLength") }
    }
    var learningRate: Float {
        get {
            let v = UserDefaults.standard.float(forKey: "mlx.learningRate")
            return (v == 0 ? 1e-4 : v).clamped(to: 1e-5...1e-3)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.learningRate") }
    }
    var weightDecay: Float {
        get {
            let v = UserDefaults.standard.float(forKey: "mlx.weightDecay")
            return (v == 0 ? 0.01 : v).clamped(to: 0...0.1)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.weightDecay") }
    }
    var maxGradNorm: Float {
        get {
            let v = UserDefaults.standard.float(forKey: "mlx.maxGradNorm")
            return (v == 0 ? 1.0 : v).clamped(to: 0.1...5.0)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.maxGradNorm") }
    }
    var warmupSteps: Int {
        get {
            let v = UserDefaults.standard.integer(forKey: "mlx.warmupSteps")
            return (v == 0 ? 0 : v).clamped(to: 0...50)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.warmupSteps") }
    }
    var targetModules: String {
        get { UserDefaults.standard.string(forKey: "mlx.targetModules") ?? "q_proj,v_proj,gate_proj,up_proj,down_proj" }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.targetModules") }
    }
    var loraRank: Int {
        get {
            let v = UserDefaults.standard.integer(forKey: "mlx.loraRank")
            return (v == 0 ? 6 : v).clamped(to: 1...16)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.loraRank") }
    }
    var loraAlpha: Float {
        get {
            let v = UserDefaults.standard.float(forKey: "mlx.loraAlpha")
            return (v == 0 ? 16.0 : v).clamped(to: 1...64)
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.loraAlpha") }
    }
    var seed: UInt32 {
        get {
            let obj = UserDefaults.standard.object(forKey: "mlx.seed")
            if let u = obj as? UInt32 { return u }
            if let n = obj as? NSNumber { return n.uint32Value }
            return 42
        }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.seed") }
    }
    var baseModelName: String {
        get { UserDefaults.standard.string(forKey: "mlx.baseModelName") ?? "Qwen3-0.6B-8bit" }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.baseModelName") }
    }
    var curriculumEpoch: Int {
        get { UserDefaults.standard.integer(forKey: "mlx.curriculumEpoch").clamped(to: 0...999) }
        set { UserDefaults.standard.set(newValue, forKey: "mlx.curriculumEpoch") }
    }
}

@MainActor
@Observable
final class MLXDeviceLoopController {
    // ------------------------------------------------------------ config (persisted via MLXSettings)
    var settings = MLXSettings()

    // ------------------------------------------------------------ state
    private(set) var isRunning = false
    private(set) var isDownloading = false
    private(set) var anchorVersion: Int?
    private(set) var lastLoss: Float?
    private(set) var log: [LogEntry] = []

    private var runTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?

    typealias LogEntry = DeviceLoopController.LogEntry

    var deviceID: String { "phone\(settings.deviceIndex)" }

    // Model downloader for fetching curriculum and adapters
    private let modelDownloader = ModelDownloader()

    // ------------------------------------------------------------ control
    func start() {
        guard !isRunning else { return }
        guard let url = URL(string: settings.anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(settings.anchorURLText)", isError: true)
            return
        }

        UIDevice.current.isBatteryMonitoringEnabled = true
        let anchor = AnchorClient(base: url)

        // Resolve curriculum directory to the epoch-specific path that downloadCurriculum() writes to
        let resolvedCurriculumDir = "curriculum/epoch_\(settings.curriculumEpoch)"

        do {
            let loop = try MLXDeviceLoop(
                anchor: anchor,
                deviceID: deviceID,
                deviceIndex: settings.deviceIndex,
                config: MLXLoopConfig(
                    modelPath: resolvedInDocuments(settings.modelPath),
                    targetModules: settings.targetModules.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) },
                    rankMap: ["": settings.loraRank],
                    alphaMap: ["": settings.loraAlpha],
                    learningRate: settings.learningRate,
                    weightDecay: settings.weightDecay,
                    maxGradNorm: settings.maxGradNorm,
                    warmupSteps: settings.warmupSteps,
                    maxStepsPerRound: settings.maxStepsPerRound,
                    batchSize: settings.batchSize,
                    sequenceLength: settings.sequenceLength,
                    curriculumDirectory: resolvedInDocuments(resolvedCurriculumDir),
                    seed: UInt64(settings.seed) + UInt64(settings.deviceIndex)
                )
            )

            isRunning = true
            UIApplication.shared.isIdleTimerDisabled = true
            append("Started \(deviceID) (MLX) against \(url.absoluteString)")

            let interval = settings.intervalSeconds
            let maxSteps = settings.maxStepsPerRound
            runTask = Task { [weak self] in
                let budget = ResourceBudget(
                    maxSteps: maxSteps,
                    maxWallClock: 300,
                    stopOnSeriousThermalState: true,
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
        guard let url = URL(string: settings.anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(settings.anchorURLText)", isError: true)
            return
        }

        let anchor = AnchorClient(base: url)
        let epoch = settings.curriculumEpoch

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
                        self?.settings.curriculumDirectory = "curriculum/epoch_\(epoch)"
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
        guard let url = URL(string: settings.anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(settings.anchorURLText)", isError: true)
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

    /// Download a base MLX model from the Anchor.
    func downloadBaseModel() {
        guard !isDownloading else { return }
        guard let url = URL(string: settings.anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(settings.anchorURLText)", isError: true)
            return
        }

        let anchor = AnchorClient(base: url)
        let modelName = settings.baseModelName

        isDownloading = true
        append("Downloading base model \(modelName)...")

        downloadTask = Task { [weak self] in
            defer { self?.isDownloading = false }

            do {
                _ = try await self?.modelDownloader.downloadBaseModel(
                    modelName: modelName,
                    anchor: anchor
                ) { progress in
                    Task { @MainActor in
                        self?.append(progress.message)
                    }
                }
                await MainActor.run {
                    self?.append("Base model \(modelName) downloaded successfully")
                }
            } catch {
                await MainActor.run {
                    self?.append("Base model download failed: \(error)", isError: true)
                }
            }
        }
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
        UIApplication.shared.isIdleTimerDisabled = false
        append("Stopped")
    }

    private func append(_ text: String, isError: Bool = false) {
        log.append(LogEntry(text: text, isError: isError))
        if log.count > 200 { log.removeFirst(log.count - 200) }
    }
}