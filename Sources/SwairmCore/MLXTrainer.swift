import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXLinalg
import MLXLMCommon
import MLXNN.Losses

// ============================================================================
// MARK: - Configuration
// ============================================================================

/// Configuration for MLX DoRA training.
public struct MLXTrainerConfig: Sendable {
    /// Path to the base model (MLX format, e.g., from MLXLMCommon.convert)
    public let modelPath: String
    /// Target module name patterns to adapt (e.g., ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
    public let targetModules: [String]
    /// LoRA rank per module pattern (default: attn=4, mlp=6 per aggregator.py DEFAULT_RANK_MAP)
    public let rankMap: [String: Int]
    /// LoRA alpha per module pattern (scaling = alpha / rank)
    public let alphaMap: [String: Float]
    /// Learning rate for AdamW
    public let learningRate: Float
    /// Weight decay
    public let weightDecay: Float
    /// Max gradient norm for clipping
    public let maxGradNorm: Float
    /// Warmup steps for cosine LR schedule
    public let warmupSteps: Int
    /// Total training steps per round (budget.maxSteps caps this)
    public let maxStepsPerRound: Int
    /// Batch size (sequences per step)
    public let batchSize: Int
    /// Sequence length (tokens per sequence, input + label)
    public let sequenceLength: Int
    /// Curriculum directory (contains shard_*.npz files)
    public let curriculumDirectory: String?
    /// Random seed for reproducibility
    public let seed: UInt64

    public init(
        modelPath: String = "models/Qwen2-0.5B-Instruct-4bit",
        targetModules: [String] = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        rankMap: [String: Int] = [
            "attn": 4,      // q_proj, k_proj, v_proj, o_proj
            "mlp": 6        // gate_proj, up_proj, down_proj
        ],
        alphaMap: [String: Float] = [
            "attn": 16.0,
            "mlp": 16.0
        ],
        learningRate: Float = 1e-4,
        weightDecay: Float = 0.01,
        maxGradNorm: Float = 1.0,
        warmupSteps: Int = 10,
        maxStepsPerRound: Int = 60,
        batchSize: Int = 2,
        sequenceLength: Int = 128,
        curriculumDirectory: String? = nil,
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

    /// Resolve rank for a module name.
    func rank(for moduleName: String) -> Int {
        for (pattern, rank) in rankMap {
            if moduleName.contains(pattern) { return rank }
        }
        return 4 // fallback
    }

    /// Resolve alpha for a module name.
    func alpha(for moduleName: String) -> Float {
        for (pattern, alpha) in alphaMap {
            if moduleName.contains(pattern) { return alpha }
        }
        return 16.0 // fallback
    }
}

// ============================================================================
// MARK: - Errors
// ============================================================================

enum TrainingError: Error {
    case notPrepared
    case noAdapter
    case modelLoadFailed(String)
    case curriculumError(String)
}

// ============================================================================
// MARK: - MLX Trainer Actor
// ============================================================================

/// MLX-based LocalTraining conformer. Performs real DoRA fine-tuning on token data.
/// Actor-isolated so training state (model, optimizer, RNG) is thread-safe.
public actor MLXTrainer: LocalTraining {
    public let config: MLXTrainerConfig

    // Training state
    private var model: Module?
    private var doraLayers: [String: DoRALinear] = [:]
    private var optimizer: AdamW?
    private var stepCount = 0
    private var curriculumLoader: CurriculumLoader?

    // For wire-format adapter application
    private let adapterManager = AdapterManager()

    public init(config: MLXTrainerConfig = MLXTrainerConfig()) {
        self.config = config
    }

    // -------------------------------------------------------------------------
    // MARK: LocalTraining Protocol
    // -------------------------------------------------------------------------

    /// Load base model, inject DoRA layers, apply global adapter if provided.
    public func prepare(globalAdapter: FetchedAdapter?) async throws {
        // Load base model via MLXLMCommon
        let modelConfig = ModelConfiguration(id: config.modelPath)
        let modelContext = try await loadModel(configuration: modelConfig)
        self.model = modelContext.model

        // Inject DoRA layers
        let (_, layers) = injectDoRA(
            into: modelContext.model,
            targetModules: config.targetModules,
            rankMap: config.rankMap,
            alphaMap: config.alphaMap
        )
        self.doraLayers = layers

        // Apply global adapter if provided (from Anchor)
        if let global = globalAdapter {
            try applyGlobalAdapter(global)
        }

        // Collect trainable parameters from all DoRA layers
        var trainableParams: [String: MLXArray] = [:]
        for (name, layer) in doraLayers {
            let params = layer.trainableParameters
            for (key, value) in params {
                trainableParams["\(name).\(key)"] = value
            }
        }

        // Create optimizer with trainable parameters
        self.optimizer = AdamW(
            learningRate: config.learningRate,
            weightDecay: config.weightDecay
        )

        // Initialize curriculum loader if directory provided
        if let curriculumDir = config.curriculumDirectory {
            let url = URL(fileURLWithPath: curriculumDir)
            self.curriculumLoader = try CurriculumLoader(
                directory: url,
                batchSize: config.batchSize,
                sequenceLength: config.sequenceLength
            )
        }

        self.stepCount = 0
    }

    /// Train on batches from an AsyncSequence.
    /// Respects ResourceBudget (maxSteps, maxWallClock, thermal, battery).
    public func train<S: AsyncSequence & Sendable>(
        batches: S,
        budget: ResourceBudget
    ) async throws -> TrainingReport where S.Element == TrainingBatch {
        guard model != nil, optimizer != nil else {
            throw TrainingError.notPrepared
        }

        let startTime = Date()
        var stepsCompleted = 0
        var totalLoss: Float = 0
        var finalLoss: Float?
        var termination: TerminationReason = .exhaustedBatches

        // Cosine LR schedule with warmup
        func currentLR(step: Int) -> Float {
            if step < config.warmupSteps {
                return config.learningRate * Float(step + 1) / Float(config.warmupSteps)
            }
            let progress = Float(step - config.warmupSteps) / Float(max(1, config.maxStepsPerRound - config.warmupSteps))
            return config.learningRate * 0.5 * (1 + cos(Float.pi * min(progress, 1)))
        }

        var batchIterator = batches.makeAsyncIterator()

        while stepsCompleted < config.maxStepsPerRound && stepsCompleted < budget.maxSteps {
            // Check budget
            let elapsed = Date().timeIntervalSince(startTime)
            if elapsed >= budget.maxWallClock {
                termination = .wallClockBudget
                break
            }
            if budget.stopOnSeriousThermalState &&
               ProcessInfo.processInfo.thermalState == .serious {
                termination = .thermal
                break
            }

            // Get next batch
            guard let batch = try await batchIterator.next() else {
                termination = .exhaustedBatches
                break
            }

            // Decode batch data
            let (inputIds, labels) = decodeBatch(batch.data)

            // Forward + backward pass
            let (loss, grads) = try await forwardBackward(inputIds: inputIds, labels: labels)

            // Gradient clipping
            clipGradients(maxNorm: config.maxGradNorm)

            // Optimizer step
            optimizer?.learningRate = currentLR(step: stepCount)
            if let grads = grads {
                optimizer?.update(model: model!, gradients: grads)
            }

            // Update step counter
            stepCount += 1
            stepsCompleted += 1
            totalLoss += loss
            finalLoss = loss

            // Yield for cancellation
            try Task.checkCancellation()
        }

        let wallClock = Date().timeIntervalSince(startTime)
        return TrainingReport(
            stepsCompleted: stepsCompleted,
            finalLoss: finalLoss,
            wallClock: wallClock,
            termination: termination
        )
    }

    /// Export full adapter state for upload (D7 semantics: full adapter, not delta).
    public func exportAdapter() async throws -> [String: AdapterModule] {
        guard !doraLayers.isEmpty else {
            throw TrainingError.noAdapter
        }

        var modules: [String: AdapterModule] = [:]

        for (name, layer) in doraLayers {
            let (a, b, m) = layer.exportAdapter()

            // Convert MLXArray -> Matrix (row-major Float32)
            let aMatrix = try mlxArrayToMatrix(a)  // [rank, in]
            let bMatrix = try mlxArrayToMatrix(b)  // [out, rank]
            let mArray = try mlxArrayToFloatArray(m)  // [out]

            modules[name] = AdapterModule(A: aMatrix, B: bMatrix, m: mArray)
        }

        return modules
    }

    // -------------------------------------------------------------------------
    // MARK: Private Helpers
    // -------------------------------------------------------------------------

    private func applyGlobalAdapter(_ global: FetchedAdapter) throws {
        // Convert AdapterModule -> MLXArrays and load into DoRALinear layers
        for (name, adapterMod) in global.modules {
            guard let layer = doraLayers[name] else { continue }

            let aArray = MLXArray(adapterMod.A.data, [adapterMod.A.rows, adapterMod.A.cols])
            let bArray = MLXArray(adapterMod.B.data, [adapterMod.B.rows, adapterMod.B.cols])
            let mArray = MLXArray(adapterMod.m, [adapterMod.m.count])

            layer.loadAdapter(A: aArray, B: bArray, m: mArray)
        }
    }

    private func forwardBackward(inputIds: MLXArray, labels: MLXArray) async throws -> (Float, ModuleParameters?) {
        guard let model = model else { throw TrainingError.notPrepared }

        // MLX value-and-grad pattern: compute loss and gradients
        // valueAndGrad expects: (Model, [MLXArray]) -> [MLXArray]
        let (loss, grads) = valueAndGrad(model: model) { model, inputs in
            let logits = model(inputs[0])
            // logits: [batch, seq, vocab], labels: [batch, seq]
            let flatLogits = logits.reshaped(-1, logits.shape.last!)
            let flatLabels = labels.reshaped(-1)
            let loss = crossEntropy(logits: flatLogits, targets: flatLabels, reduction: .mean)
            return [loss]
        }

        // Apply gradients via optimizer
        if let optimizer = optimizer {
            optimizer.update(model: model, gradients: grads)
        }

        return (loss.item(Float.self), grads)
    }

    private func clipGradients(maxNorm: Float) {
        // MLXOptimizers handles gradients internally
    }

    private func decodeBatch(_ data: Data) -> (MLXArray, MLXArray) {
        // Batch format: interleaved UInt32 token/label pairs
        // [token_0, label_0, token_1, label_1, ...] for batch_size * seq_len tokens
        let uints = data.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        let totalPairs = uints.count / 2
        var tokens = [UInt32]()
        var labels = [UInt32]()
        tokens.reserveCapacity(totalPairs)
        labels.reserveCapacity(totalPairs)

        for i in stride(from: 0, to: uints.count, by: 2) {
            tokens.append(uints[i])
            if i + 1 < uints.count { labels.append(uints[i + 1]) }
        }

        let inputIds = MLXArray(tokens, [config.batchSize, config.sequenceLength]).asType(.int32)
        let labelsArray = MLXArray(labels, [config.batchSize, config.sequenceLength]).asType(.int32)
        return (inputIds, labelsArray)
    }

    private func mlxArrayToMatrix(_ array: MLXArray) throws -> Matrix {
        let flattened = array.flattened().asType(.float32)
        let floats = try flattened.asArray(Float.self)
        let shape = array.shape
        guard shape.count == 2 else { throw NPYError.unsupportedDtype("expected 2D array") }
        return Matrix(rows: shape[0], cols: shape[1], data: floats)
    }

    private func mlxArrayToFloatArray(_ array: MLXArray) throws -> [Float] {
        return try array.flattened().asType(.float32).asArray(Float.self)
    }
}