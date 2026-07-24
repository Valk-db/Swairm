// MLX device round loop: fetch -> prepare -> train -> export -> upload
// using the real MLX DoRA trainer. Mirrors ProxyDeviceLoop structure
// so the iOS app can swap implementations with minimal changes.
//
// Curriculum data streams from .npz shards (token_ids, labels) via
// CurriculumLoader, never fully resident in memory.

import Foundation

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
        rankMap: [String: Int] = ["attn": 4, "mlp": 6],
        alphaMap: [String: Float] = ["attn": 16.0, "mlp": 16.0],
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

    public init(anchor: AnchorConnecting, deviceID: String, deviceIndex: Int,
                config: MLXLoopConfig = MLXLoopConfig()) throws {
        self.anchor = anchor
        self.deviceID = deviceID
        self.deviceIndex = deviceIndex
        self.config = config

        var trainerConfig = MLXTrainerConfig(
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
    }

    /// One full round: status -> fetch -> prepare -> train -> export -> upload.
    public func runRound(budget: ResourceBudget = ResourceBudget(
        maxSteps: 60, maxWallClock: 300
    )) async throws -> MLXRoundResult {
        let round = roundsRun
        roundsRun += 1

        let status = try await anchor.status()
        let globalAdapter = try await anchor.latestAdapter()
        let fetchedVersion = globalAdapter?.version ?? 0

        // Prepare trainer with global adapter (or fresh if nil)
        try await trainer.prepare(globalAdapter: globalAdapter)

        // Get curriculum batch stream
        guard let curriculumLoader = config.curriculumDirectory.isEmpty ? nil :
              try CurriculumLoader(
                  directory: URL(fileURLWithPath: config.curriculumDirectory),
                  batchSize: config.batchSize,
                  sequenceLength: config.sequenceLength
              ) else {
            throw MLXLoopError.noCurriculumDirectory
        }

        let batchStream = curriculumLoader.batches()

        // Train on curriculum batches
        let report = try await trainer.train(batches: batchStream, budget: budget)

        // Export full adapter state (D7: replace semantics)
        let modules = try await trainer.exportAdapter()

        // Upload to Anchor
        let payload = AdapterUploadPayload(
            deviceID: deviceID,
            fetchVersion: fetchedVersion,
            curriculumEpoch: status.curriculum_epoch,
            modules: modules
        )
        let receipt = try await anchor.upload(payload)

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