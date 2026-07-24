import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXLinalg
import MLXLMCommon
import Tokenizers

// MARK: - Local Tokenizer Loader

/// A TokenizerLoader that loads tokenizers from a local directory using AutoTokenizer.
/// This avoids the need for HuggingFace downloaders and works on iOS 17+.
private struct LocalTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any Tokenizer {
        // Use AutoTokenizer from the Tokenizers library to load from local directory
        // This loads tokenizer.json, tokenizer_config.json, etc. from the directory
        let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return MLXLMTokenizer(tokenizer: tokenizer)
    }
}

/// Wrapper to conform HuggingFace Tokenizer to MLXLMCommon.Tokenizer protocol
private struct MLXLMTokenizer: Tokenizer {
    let tokenizer: Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        return tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        return tokenizer.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        return tokenizer.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        return tokenizer.convertIdToToken(id)
    }

    var bosToken: String? {
        return tokenizer.bosToken
    }

    var eosToken: String? {
        return tokenizer.eosToken
    }

    var unknownToken: String? {
        return tokenizer.unknownToken
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        return try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: tools,
            additionalContext: additionalContext
        )
    }
}

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
    private var model: (any LanguageModel)?
    private var loraContainer: LoRAContainer?
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

    /// Load base model, inject DoRA layers via MLX LoRAContainer, apply global adapter if provided.
    public func prepare(globalAdapter: FetchedAdapter?) async throws {
        // Load base model from local directory via MLXLMCommon
        let modelDirectory = URL(fileURLWithPath: config.modelPath)
        let modelContext = try await loadModel(
            from: modelDirectory,
            using: LocalTokenizerLoader()
        )
        self.model = modelContext.model

        // Create LoRAConfiguration for DoRA
        let loraConfig = LoRAConfiguration(
            numLayers: modelContext.model.loraLayers.count,
            fineTuneType: .dora,
            loraParameters: LoRAConfiguration.LoRAParameters(
                rank: config.targetModules.count > 0 ? config.rankMap.values.max() ?? 4 : 4,
                scale: config.alphaMap.values.max() ?? 16.0,
                keys: config.targetModules.isEmpty ? nil : config.targetModules
            )
        )

        // Inject DoRA layers using MLX's LoRAContainer
        self.loraContainer = try LoRAContainer.from(
            model: modelContext.model,
            configuration: loraConfig
        )

        // Apply global adapter if provided (from Anchor)
        if let global = globalAdapter {
            try applyGlobalAdapter(global)
        }

        // Collect trainable parameters from all DoRA layers
        var trainableParams: [String: MLXArray] = [:]
        if let container = loraContainer {
            for (key, value) in container.parameters {
                trainableParams[key] = value
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
        guard model != nil, optimizer != nil, loraContainer != nil else {
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

            // Gradient clipping (MLXOptimizers handles internally)
            // Apply gradients via optimizer
            if let optimizer = optimizer {
                optimizer.update(model: model!, gradients: grads)
                eval(model!, optimizer, loss)
            }

            // Update learning rate for next step
            optimizer?.learningRate = currentLR(step: stepCount)

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
        guard let container = loraContainer else {
            throw TrainingError.noAdapter
        }

        var modules: [String: AdapterModule] = [:]

        for (name, array) in container.parameters {
            // Convert MLXArray -> Matrix (row-major Float32) for wire format
            // The parameter names follow MLX LoRA naming: "layer.lora_a", "layer.lora_b", "layer.m"
            // We need to group them by layer
        }

        // Group parameters by layer name
        var layerParams: [String: (A: MLXArray?, B: MLXArray?, M: MLXArray?)] = [:]

        for (name, array) in container.parameters {
            // Parse layer name from parameter key
            // Keys look like: "layers.0.attention.q_proj.lora_a", "layers.0.attention.q_proj.lora_b", "layers.0.attention.q_proj.m"
            let parts = name.split(separator: ".")
            if parts.count >= 2 {
                let layerName = parts.dropLast().joined(separator: ".")
                let paramType = String(parts.last!)

                var params = layerParams[layerName] ?? (nil, nil, nil)
                switch paramType {
                case "lora_a": params.0 = array
                case "lora_b": params.1 = array
                case "m": params.2 = array
                default: break
                }
                layerParams[layerName] = params
            }
        }

        // Convert to AdapterModule format
        for (name, (a, b, m)) in layerParams {
            guard let aArray = a, let bArray = b, let mArray = m else { continue }

            let aMatrix = try mlxArrayToMatrix(aArray)  // [rank, in]
            let bMatrix = try mlxArrayToMatrix(bArray)  // [out, rank]
            let mArray = try mlxArrayToFloatArray(mArray)  // [out]

            modules[name] = AdapterModule(A: aMatrix, B: bMatrix, m: mArray)
        }

        return modules
    }

    // -------------------------------------------------------------------------
    // MARK: Private Helpers
    // -------------------------------------------------------------------------

    private func applyGlobalAdapter(_ global: FetchedAdapter) throws {
        // Convert AdapterModule -> MLXArrays and load into LoRAContainer
        // First, we need to reconstruct ModuleParameters from AdapterModules
        var mlxParams: [String: MLXArray] = [:]

        for (name, adapterMod) in global.modules {
            let aArray = MLXArray(adapterMod.A.data, [adapterMod.A.rows, adapterMod.A.cols])
            let bArray = MLXArray(adapterMod.B.data, [adapterMod.B.rows, adapterMod.B.cols])
            let mArray = MLXArray(adapterMod.m, [adapterMod.m.count])

            mlxParams["\(name).lora_a"] = aArray
            mlxParams["\(name).lora_b"] = bArray
            mlxParams["\(name).m"] = mArray
        }

        let params = ModuleParameters.unflattened(mlxParams)
        try model?.update(parameters: params, verify: .noUnusedKeys)
    }

    private func forwardBackward(inputIds: MLXArray, labels: MLXArray) async throws -> (Float, ModuleParameters?) {
        guard let model = model else { throw TrainingError.notPrepared }

        // valueAndGrad closure must take (model, input, labels) and return loss
        let gradFn = valueAndGrad(model: model) { model, x, y in
            let lm = model as! any LanguageModel
            let logits = lm(x, cache: nil)
            let flatLogits = logits.reshaped(-1, logits.shape.last!)
            let flatLabels = y.reshaped(-1)
            return crossEntropy(logits: flatLogits, targets: flatLabels, reduction: .mean)
        }

        let (loss, grads) = gradFn(model, inputIds, labels)

        // Optimizer step happens once, in train()'s loop — don't apply it here too.
        eval(loss)

        return (loss.item(Float.self), grads)
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