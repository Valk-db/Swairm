// MLX device round loop: fetch -> prepare -> train -> export -> upload
// using the real MLX DoRA trainer. Mirrors ProxyDeviceLoop structure
// so the iOS app can swap implementations with minimal changes.
//
// Curriculum data streams from .npz shards (token_ids, labels) via
// CurriculumLoader, never fully resident in memory.

import Foundation
import os.log

private let loopLog = OSLog(subsystem: "com.swairm.app", category: "MLXDeviceLoop")

/// Resolve a relative path to the Documents directory on iOS/macOS.
/// Leaves absolute paths (starting with "/") unchanged.
private func resolveInDocuments(_ path: String) -> String {
    guard !path.isEmpty, !path.hasPrefix("/") else { return path }
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    return docs.appendingPathComponent(path).path
}

/// Log to both os_log and Swift print for debugging
private func log(_ message: String, level: OSLogType = .default) {
    os_log("%{public}@", log: loopLog, type: level, message)
    print("[MLXDeviceLoop] \(message)")
}

// ============================================================================
// MARK: - Config & Results
// ============================================================================

public struct MLXLoopConfig: Sendable {
    public let modelPath: String
    public let targetModules: [String]
    public let rankMap: [String: Int]
    public let alphaMap: [String: Float]
    public let learningRate: Float
    public let weightDecay: Float
    public let maxGradNorm: Float
    public let warmupSteps: Int
    public let maxStepsPerRound: Int
    public let batchSize: Int
    public let sequenceLength: Int
    public let curriculumDirectory: String
    public let seed: UInt64

    public init(
        modelPath: String = "models/Qwen2-0.5B-Instruct-4bit",
        targetModules: [String] = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        rankMap: [String: Int] = ["": 6],           // uniform rank 6 for all target modules (max of D6's attn=4, mlp=6); per-module ranks enforced Anchor-side via SVD truncation
        alphaMap: [String: Float] = ["": 16.0],     // uniform alpha 16 -> scale = 16/6
        learningRate: Float = 1e-4,
        weightDecay: Float = 0.01,
        maxGradNorm: Float = 1.0,
        warmupSteps: Int = 10,
        maxStepsPerRound: Int = 60,
        batchSize: Int = 2,
        sequenceLength: Int = 128,
        curriculumDirectory: String = "curriculum",
        seed: UInt64 = 42
    ) {
        self.modelPath = modelPath
        self.targetModules = targetModules
        self.rankMap = rankMap
        self.alphaMap = alphaMap
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.maxGradNorm = maxGradNorm
        self.warmupSteps = warmupSteps
        self.maxStepsPerRound = maxStepsPerRound
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.curriculumDirectory = curriculumDirectory
        self.seed = seed
    }
}

public struct MLXRoundResult: Sendable {
    public let round: Int
    public let status: AnchorStatus
    public let fetchedVersion: Int
    public let trainingReport: TrainingReport
    public let receipt: UploadReceipt

    public init(round: Int, status: AnchorStatus, fetchedVersion: Int,
                trainingReport: TrainingReport, receipt: UploadReceipt) {
        self.round = round
        self.status = status
        self.fetchedVersion = fetchedVersion
        self.trainingReport = trainingReport
        self.receipt = receipt
    }
}

// ============================================================================
// MARK: - Loop
// ============================================================================

/// MLX training loop: one device running real DoRA fine-tuning against an Anchor.
/// Actor-isolated for thread safety; integrates with ProxyDeviceLoop patterns.
public actor MLXDeviceLoop {
    public let deviceID: String
    public let deviceIndex: Int
    public let config: MLXLoopConfig

    private let anchor: AnchorConnecting
    private let trainer: MLXTrainer
    private var roundsRun = 0

    // Checkpoint file URL for this device
    private let checkpointURL: URL

    public init(anchor: AnchorConnecting, deviceID: String, deviceIndex: Int,
                config: MLXLoopConfig = MLXLoopConfig()) throws {
        self.anchor = anchor
        self.deviceID = deviceID
        self.deviceIndex = deviceIndex
        self.config = config

        let trainerConfig = MLXTrainerConfig(
            modelPath: config.modelPath,
            targetModules: config.targetModules,
            rankMap: config.rankMap,
            alphaMap: config.alphaMap,
            learningRate: config.learningRate,
            weightDecay: config.weightDecay,
            maxGradNorm: config.maxGradNorm,
            warmupSteps: config.warmupSteps,
            maxStepsPerRound: config.maxStepsPerRound,
            batchSize: config.batchSize,
            sequenceLength: config.sequenceLength,
            curriculumDirectory: config.curriculumDirectory,
            seed: config.seed + UInt64(deviceIndex)
        )
        self.trainer = MLXTrainer(config: trainerConfig)

        // Checkpoint file in Documents directory
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        self.checkpointURL = docs.appendingPathComponent("checkpoint_\(deviceID).npz")
    }

    // MARK: - Checkpointing

    /// Save trainer checkpoint to disk.
    private func saveCheckpoint() async throws {
        try await trainer.saveCheckpoint(to: checkpointURL)
    }

    /// Load trainer checkpoint from disk if it exists.
    private func loadCheckpoint() async throws {
        guard FileManager.default.fileExists(atPath: checkpointURL.path) else {
            return // First run, no checkpoint
        }
        try await trainer.loadCheckpoint(from: checkpointURL)
    }

    /// Delete checkpoint (e.g., after successful round completion if desired).
    private func deleteCheckpoint() {
        try? FileManager.default.removeItem(at: checkpointURL)
    }

    /// One full round: status -> fetch -> prepare -> train -> export -> upload.
    public func runRound(budget: ResourceBudget = ResourceBudget(
        maxSteps: 60, maxWallClock: 300
    )) async throws -> MLXRoundResult {
        let round = roundsRun
        roundsRun += 1

        log("runRound \(round) starting")

        // Load checkpoint if resuming from interruption
        log("Loading checkpoint...")
        try await loadCheckpoint()
        log("Checkpoint loaded")

        log("Fetching anchor status...")
        let status = try await anchor.status()
        log("Anchor status: version=\(status.version)")

        log("Fetching latest adapter...")
        let globalAdapter = try await anchor.latestAdapter()
        let fetchedVersion = globalAdapter?.version ?? 0
        log("Fetched adapter: version=\(fetchedVersion)")

        // Prepare trainer with global adapter (or fresh if nil)
        // If we loaded a checkpoint, prepare() will re-apply the global adapter on top
        log("Preparing trainer with global adapter...")
        try await trainer.prepare(globalAdapter: globalAdapter)
        log("Trainer prepared")

        // Get curriculum batch stream - resolve path to Documents on iOS
        let resolvedCurriculumPath = resolveInDocuments(config.curriculumDirectory)
        log("Curriculum path: \(resolvedCurriculumPath)")
        guard !config.curriculumDirectory.isEmpty else {
            throw MLXLoopError.noCurriculumDirectory
        }
        log("Creating CurriculumLoader...")
        let curriculumLoader = try CurriculumLoader(
            directory: URL(fileURLWithPath: resolvedCurriculumPath),
            batchSize: config.batchSize,
            sequenceLength: config.sequenceLength
        )
        log("CurriculumLoader created, shards: \(curriculumLoader.shardFiles.count)")

        let batchStream = curriculumLoader.batches()
        log("Starting training...")

        // Train on curriculum batches
        let report = try await trainer.train(batches: batchStream, budget: budget)
        log("Training done: steps=\(report.stepsCompleted) loss=\(report.finalLoss ?? -1) term=\(report.termination)")

        // Export full adapter state (D7: replace semantics)
        log("Exporting adapter...")
        let modules = try await trainer.exportAdapter()
        log("Adapter exported: \(modules.count) modules")

        // Upload to Anchor
        log("Uploading to anchor...")
        let payload = AdapterUploadPayload(
            deviceID: deviceID,
            fetchVersion: fetchedVersion,
            curriculumEpoch: status.curriculum_epoch,
            modules: modules
        )
        let receipt = try await anchor.upload(payload)
        log("Upload done: receipt=\(receipt.accepted)")

        // Save checkpoint after successful round
        log("Saving checkpoint...")
        try await saveCheckpoint()
        log("Checkpoint saved")

        return MLXRoundResult(
            round: round,
            status: status,
            fetchedVersion: fetchedVersion,
            trainingReport: report,
            receipt: receipt
        )
    }
}

enum MLXLoopError: Error {
    case noCurriculumDirectory
    case curriculumLoadFailed(String)
}