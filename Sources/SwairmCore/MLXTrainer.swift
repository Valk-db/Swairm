import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXLinalg
@preconcurrency import MLXLMCommon
import MLXLLM
import Tokenizers
import os.log

// MLXTrainer logger
private let trainerLog = OSLog(subsystem: "com.swairm.app", category: "MLXTrainer")

private func logTrainer(_ message: String, level: OSLogType = .default) {
    os_log("%{public}@", log: trainerLog, type: level, message)
    print("[MLXTrainer] \(message)")
}

// MARK: - Local Tokenizer Loader

/// A TokenizerLoader that loads tokenizers from a local directory using AutoTokenizer.
/// This avoids the need for HuggingFace downloaders and works on iOS 17+.
fileprivate struct LocalTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        // Use AutoTokenizer from the Tokenizers library to load from local directory
        // This loads tokenizer.json, tokenizer_config.json, etc. from the directory
        let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return MLXLMTokenizer(tokenizer: tokenizer)
    }
}

/// Wrapper to conform HuggingFace Tokenizer to MLXLMCommon.Tokenizer protocol
private struct MLXLMTokenizer: MLXLMCommon.Tokenizer {
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
    /// LoRA rank per module pattern (default: uniform rank 6 for all target modules; per-module ranks attn=4, mlp=6 enforced Anchor-side via SVD truncation in aggregator.py DEFAULT_RANK_MAP)
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
        rankMap: [String: Int] = ["": 6],   // uniform rank 6 (D6 max rank for all target modules)
        alphaMap: [String: Float] = ["": 16.0],  // uniform alpha 16 -> scale = 16/6
        learningRate: Float = 1e-4,
        weightDecay: Float = 0.01,
        maxGradNorm: Float = 1.0,
        warmupSteps: Int = 0,
        maxStepsPerRound: Int = 1,
        batchSize: Int = 1,
        sequenceLength: Int = 64,
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
        // With uniform rankMap ["": rank], all modules resolve to the same rank
        return rankMap[""] ?? 4
    }

    /// Resolve alpha for a module name.
    func alpha(for moduleName: String) -> Float {
        // With uniform alphaMap ["": alpha], all modules resolve to the same alpha
        return alphaMap[""] ?? 16.0
    }

    /// Single rank/scale for the whole LoRAContainer.
    ///
    /// mlx-swift-lm 3.31.3's LoRAContainer applies ONE rank and ONE scale to
    /// every matched key (see createReplacementLayer: each target layer gets
    /// `loraParameters.rank` / `.scale`). It cannot express per-module ranks,
    /// so D6's kv=2/attn=4/mlp=6 split is enforced Anchor-side at aggregation
    /// (aggregate_module truncates each module to its D6 rank via SVD). The
    /// client's job is to emit a consistent, explicit rank -- not to silently
    /// collapse a heterogeneous map to its max. Throws when the resolved
    /// ranks/scales disagree across target modules.
    /// - Parameter keys: the resolved, qualified module keys actually being
    ///   adapted -- not the bare targetModules patterns, which never contain
    ///   "attn"/"mlp" and would silently fall through to the fallback rank.
    func resolvedRankScale(forKeys keys: [String]) throws -> (rank: Int, scale: Float) {
        let ranks = Set(keys.map { rank(for: $0) })
        let scales = Set(keys.map { alpha(for: $0) })
        guard ranks.count <= 1, scales.count <= 1 else {
            throw TrainingError.ambiguousAdapterConfig(
                "LoRAContainer takes one rank/scale for all target modules, but "
                + "keys \(keys) resolve to ranks \(ranks) / "
                + "scales \(scales). Per-module ranks (D6) are applied "
                + "Anchor-side at aggregation, not on-device.")
        }
        return (ranks.first ?? 6, scales.first ?? 16.0)
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
    /// rankMap/alphaMap disagree across the target modules, but LoRAContainer
    /// applies one rank/scale to all of them -- refusing to pick silently.
    case ambiguousAdapterConfig(String)
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

    // For wire-format adapter application
    private let adapterManager = AdapterManager()

    // Logging
    private var logBuffer: [String] = []
    private let maxLogEntries = 200

    public init(config: MLXTrainerConfig = MLXTrainerConfig()) {
        self.config = config
    }

    // Simple internal logging
    private func appendLog(_ message: String) {
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        logBuffer.append("[\(timestamp)] \(message)")
        if logBuffer.count > maxLogEntries { logBuffer.removeFirst(logBuffer.count - maxLogEntries) }
    }

    // -------------------------------------------------------------------------
    // MARK: LocalTraining Protocol
    // -------------------------------------------------------------------------

    /// Load base model, inject DoRA layers via MLX LoRAContainer, apply global adapter if provided.
    ///
    /// The base checkpoint on disk never changes between rounds, so the
    /// disk load + LoRA structure injection below only run once per device
    /// (guarded by `model == nil`), not once per federated round. At 4
    /// devices x 5 rounds this was 20 full model reloads where 4 (one per
    /// device) are actually needed -- confirmed via mlx-e2e on 2026-07-25:
    /// real rounds were completing (anchor.log showed successful
    /// aggregation), just slowly, because every round paid this cost again.
    /// `applyGlobalAdapter` below still runs every round regardless -- that
    /// overwrites parameter VALUES in place, which is all D7's "full
    /// replace semantics" requires; it never needed the container rebuilt.
    public func prepare(globalAdapter: FetchedAdapter?) async throws {
        logTrainer("prepare: start")
        if model == nil {
            logTrainer("prepare: loading base model from \(config.modelPath)")
            // Load base model from local directory via MLXLMCommon
            let modelDirectory = URL(fileURLWithPath: config.modelPath)
            // Use LLMModelFactory directly rather than the free-function
            // MLXLMCommon loadModel: the latter resolves a factory through the
            // ModelFactoryRegistry trampoline (NSClassFromString lookup), which
            // throws noModelFactoryAvailable when the MLXLLM ObjC class isn't
            // realized yet. Calling the concrete factory's own load(from:using:)
            // bypasses the registry entirely.
            //
            // Load in a nonisolated context to avoid Swift 6 Sendable warning:
            // ModelContext is non-sendable but we only need the model (also
            // non-sendable, protected by actor isolation). The helper runs
            // outside actor isolation and returns the model wrapped to
            // silence the cross-actor-boundary warning.
            let loadedModel = try await MLXTrainer.loadModel(
                from: modelDirectory,
                using: LocalTokenizerLoader()
            ).value
            self.model = loadedModel
            // Guard: model was just set above, should never be nil here
            guard let loadedModel = self.model else {
                throw TrainingError.ambiguousAdapterConfig("Model failed to load")
            }
            logTrainer("prepare: base model loaded")

            // LoRAContainer.from matches `keys` by EXACT equality against
            // Module.namedModules() paths, not by suffix/substring -- despite
            // config.targetModules being bare names ("q_proj"). Real submodules
            // are nested ("self_attn.q_proj", "mlp.gate_proj"), so the bare
            // names never matched: replaceLayers() adapted zero layers,
            // model.freeze() (called inside LoRAContainer.from) left the whole
            // model frozen, and the `grad` transform in forwardBackward() then
            // saw zero trainable parameters -- "[grad] Must specify at least
            // one argument." Resolve the bare names to their qualified keys
            // here by suffix so the intended target set (D6: q/v attn + all
            // mlp) is what actually gets adapted, and fail fast with a clear
            // error if a future base model's module names don't match any of
            // targetModules, instead of the opaque MLX-side trap.
            let availableKeys = (loadedModel as? LoRAModel)?.loraDefaultKeys ?? []
            logTrainer("prepare: availableKeys=\(availableKeys)")
            var resolvedKeys: [String]? = nil
            if !config.targetModules.isEmpty {
                let matched = availableKeys.filter { key in
                    config.targetModules.contains { key == $0 || key.hasSuffix("." + $0) }
                }
                guard !matched.isEmpty else {
                    throw TrainingError.ambiguousAdapterConfig(
                        "targetModules \(config.targetModules) matched none of "
                        + "the model's module keys \(availableKeys) -- adapter "
                        + "would train zero parameters.")
                }
                resolvedKeys = matched
            }
            logTrainer("prepare: resolvedKeys=\(resolvedKeys ?? [])")

            // Single rank/scale for the container (see resolvedRankScale): throws
            // rather than silently collapsing a heterogeneous rankMap to its max.
            // Resolved against the qualified keys above (not the bare
            // targetModules patterns), so the "attn"/"mlp" substring matching in
            // rank(for:)/alpha(for:) has real qualified names to match against.
            let (rank, scale) = try config.resolvedRankScale(forKeys: resolvedKeys ?? config.targetModules)
            logTrainer("prepare: rank=\(rank) scale=\(scale)")

            // Adapt every transformer block the model exposes, not a hardcoded
            // count. LoRAContainer.from uses loraLayers.suffix(numLayers), so the
            // true layer count adapts all blocks. Falls back to all layers when
            // the model doesn't report a LoRAModel layer list.
            let numLayers = (loadedModel as? LoRAModel)?.loraLayers.count ?? 0
            logTrainer("prepare: numLayers=\(numLayers)")

            // Create LoRAConfiguration for DoRA
            let loraConfig = LoRAConfiguration(
                numLayers: numLayers,
                fineTuneType: .dora,
                loraParameters: LoRAConfiguration.LoRAParameters(
                    rank: rank,
                    scale: scale,
                    keys: resolvedKeys
                )
            )

            // Inject DoRA layers using MLX's LoRAContainer
            logTrainer("prepare: injecting LoRA...")
            self.loraContainer = try LoRAContainer.from(
                model: loadedModel,
                configuration: loraConfig
            )
            logTrainer("prepare: LoRA injected")

            // Explicitly load the container into the model to ensure trainable
            // parameters (LoRA adapters) are tracked by the model. LoRAContainer.from
            // mutates the model in place, but the documented pattern calls load(into:)
            // to make parameter tracking reliable after any subsequent updates.
            try self.loraContainer?.load(into: loadedModel)
        }

        // Apply global adapter if provided (from Anchor). Runs every round,
        // first or not: on round 0 there's usually no global adapter yet
        // (nil -> the freshly-initialized LoRA weights from injection above
        // are used as-is); on later rounds this resets whatever this device
        // trained locally last round back to the aggregated consensus,
        // which is the actual per-round state D7 cares about.
        if let global = globalAdapter {
            try applyGlobalAdapter(global)
        }

        // Create optimizer with trainable parameters (fresh each round - D7: full replace semantics)
        self.optimizer = AdamW(
            learningRate: config.learningRate,
            weightDecay: config.weightDecay
        )

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

        // Cosine LR schedule with warmup — uses actual step count (may be less than maxStepsPerRound due to budget)
        func currentLR(actualStep: Int) -> Float {
            if actualStep < config.warmupSteps {
                return config.learningRate * Float(actualStep + 1) / Float(config.warmupSteps)
            }
            let totalDecaySteps = max(1, config.maxStepsPerRound - config.warmupSteps)
            let progress = Float(actualStep - config.warmupSteps) / Float(totalDecaySteps)
            return config.learningRate * 0.5 * (1 + cos(Float.pi * min(progress, 1)))
        }

        var batchIterator = batches.makeAsyncIterator()

        appendLog("train: starting loop, maxSteps=\(config.maxStepsPerRound)")
        while stepsCompleted < config.maxStepsPerRound && stepsCompleted < budget.maxSteps {
            // Check budget
            let elapsed = Date().timeIntervalSince(startTime)
            if elapsed >= budget.maxWallClock {
                termination = .wallClockBudget
                break
            }
            if budget.stopOnSeriousThermalState {
                let state = ProcessInfo.processInfo.thermalState
                if state == .serious || state == .critical {
                    // Thermal pacing: yield for a few seconds instead of aborting
                    appendLog("Thermal state \(state) — pacing 2s")
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                    continue
                }
            }

            // Get next batch
            guard let batch = try await batchIterator.next() else {
                termination = .exhaustedBatches
                break
            }

            // Decode batch data
            let (inputIds, labels): (MLXArray, MLXArray)
            do {
                (inputIds, labels) = try decodeBatch(batch.data)
            } catch {
                // Log and skip malformed batch; don't crash the entire round
                appendLog("Batch decode failed, skipping: \(error)")
                continue
            }

            appendLog("train: step \(stepsCompleted) input shape=\(inputIds.shape)")
            // Forward + backward pass
            let (loss, grads) = try await forwardBackward(inputIds: inputIds, labels: labels)
            appendLog("train: step \(stepsCompleted) loss=\(loss)")

            // Gradient clipping by global norm (MLXOptimizers.AdamW does NOT clip internally)
            var clippedGrads = grads
            if let grads = grads, config.maxGradNorm > 0 {
                var totalNormSq: Float = 0
                for (_, grad) in grads.flattened() {
                    totalNormSq += grad.square().sum().item(Float.self)
                }
                let totalNorm = sqrt(totalNormSq)
                let clipCoef = config.maxGradNorm / (totalNorm + 1e-6)
                if clipCoef < 1 {
                    clippedGrads = ModuleParameters.unflattened(
                        grads.flattened().map { ($0.0, $0.1 * clipCoef) }
                    )
                }
            }

            // Apply gradients via optimizer
            if let optimizer = optimizer, let validGrads = clippedGrads {
                appendLog("train: step \(stepsCompleted) optimizer.update")
                optimizer.update(model: model!, gradients: validGrads)
            }

            // Update learning rate for next step
            optimizer?.learningRate = currentLR(actualStep: stepCount)

            // Update step counter
            stepCount += 1
            stepsCompleted += 1
            totalLoss += loss
            finalLoss = loss

            // Yield for cancellation
            try Task.checkCancellation()
        }

        // Loop exits here once we've taken the full requested step count;
        // nothing inside the loop above sets termination for that case.
        if stepsCompleted >= config.maxStepsPerRound || stepsCompleted >= budget.maxSteps {
            termination = .stepBudget
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

        appendLog("exportAdapter: container.parameters=\(container.parameters.flattened().count)")
        var modules: [String: AdapterModule] = [:]

        // Group parameters by layer name
        var layerParams: [String: (A: MLXArray?, B: MLXArray?, M: MLXArray?)] = [:]

        for (name, tensor) in container.parameters.flattened() {
            // Parse layer name from parameter key
            // Keys look like: "layers.0.attention.q_proj.lora_a", "layers.0.attention.q_proj.lora_b", "layers.0.attention.q_proj.m"
            let parts = name.split(separator: ".")
            if parts.count >= 2 {
                let layerName = parts.dropLast().joined(separator: ".")
                let paramType = String(parts.last!)

                var params = layerParams[layerName] ?? (nil, nil, nil)
                switch paramType {
                case "lora_a": params.0 = tensor
                case "lora_b": params.1 = tensor
                case "m": params.2 = tensor
                default: break
                }
                layerParams[layerName] = params
            }
        }

        // Convert to AdapterModule format.
        //
        // Orientation: MLX DoRA stores lora_a as (in, rank) and lora_b as
        // (rank, out) (DoRA+Layers.swift init: loraA is [inputDimensions,
        // rank], loraB is [rank, outputDimensions]). The Anchor wire format is
        // A = (rank, in), B = (out, rank) so that dW = B @ A (aggregator.py,
        // factorToRank). Hence both factors are transposed on the way out.
        for (name, (a, b, m)) in layerParams {
            guard let aArray = a, let bArray = b, let mArray = m else { continue }

            let aMatrix = try mlxArrayToMatrix(aArray.transposed())  // wire A: [rank, in]
            let bMatrix = try mlxArrayToMatrix(bArray.transposed())  // wire B: [out, rank]
            let mFloatArray = try mlxArrayToFloatArray(mArray)       // [out]

            modules[name] = AdapterModule(A: aMatrix, B: bMatrix, m: mFloatArray)
        }

        appendLog("exportAdapter: returning \(modules.count) modules")
        return modules
    }

    // -------------------------------------------------------------------------
    // MARK: Private Helpers
    // -------------------------------------------------------------------------

    private func applyGlobalAdapter(_ global: FetchedAdapter) throws {
        // Convert AdapterModule -> MLXArrays and load into LoRAContainer.
        // Wire A = (rank, in), B = (out, rank); MLX stores lora_a = (in, rank),
        // lora_b = (rank, out). Transpose each factor wire -> MLX (inverse of
        // exportAdapter). `m` is (out,) on both sides -- no transpose.
        var mlxParams: [String: MLXArray] = [:]

        for (name, adapterMod) in global.modules {
            let aT = adapterMod.A.transposed()   // (in, rank) for lora_a
            let bT = adapterMod.B.transposed()   // (rank, out) for lora_b
            mlxParams["\(name).lora_a"] = MLXArray(aT.data, [aT.rows, aT.cols])
            mlxParams["\(name).lora_b"] = MLXArray(bT.data, [bT.rows, bT.cols])
            mlxParams["\(name).m"] = MLXArray(adapterMod.m, [adapterMod.m.count])
        }

        appendLog("applyGlobalAdapter: applying \(global.modules.count) modules")
        let params = ModuleParameters.unflattened(mlxParams)
        try model?.update(parameters: params, verify: .noUnusedKeys)
        appendLog("applyGlobalAdapter: done")
    }

    private func forwardBackward(inputIds: MLXArray, labels: MLXArray) async throws -> (Float, ModuleParameters?) {
        guard let model = model else { throw TrainingError.notPrepared }

        appendLog("forwardBackward: inputIds=\(inputIds.shape) labels=\(labels.shape)")
        // valueAndGrad closure must take (model, input, labels) and return loss
        // Pass model as explicit argument to avoid capturing the `var model` reference
        func inner(parameters: ModuleParameters, arrays: [MLXArray]) -> [MLXArray] {
            model.update(parameters: parameters)
            let logits = model(arrays[0], cache: nil as [any MLXLMCommon.KVCache]?)
            let flatLogits = logits.reshaped(-1, logits.shape.last!)
            let flatLabels = arrays[1].reshaped(-1)
            return [crossEntropy(logits: flatLogits, targets: flatLabels, reduction: .mean)]
        }
        let vg = valueAndGrad(inner)
        let (values, grads) = vg(model.trainableParameters(), [inputIds, labels])
        let loss = values[0]

        // Optimizer step happens once, in train()'s loop — don't apply it here too.

        return (loss.item(Float.self), grads)
    }

    private func decodeBatch(_ data: Data) throws -> (MLXArray, MLXArray) {
        // Batch format: interleaved UInt32 token/label pairs
        // [token_0, label_0, token_1, label_1, ...] for batch_size * seq_len tokens
        let uints = data.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        let totalPairs = uints.count / 2
        let expectedPairs = config.batchSize * config.sequenceLength
        guard totalPairs == expectedPairs else {
            throw TrainingError.curriculumError(
                "Batch decode: expected \(expectedPairs) token/label pairs, got \(totalPairs) (data bytes: \(data.count))"
            )
        }
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
        let flattened = array.flattened().asType(MLX.DType.float32)
        let floats = flattened.asArray(Float.self)
        let shape = array.shape
        guard shape.count == 2 else { throw NPYError.unsupportedDtype("expected 2D array") }
        return Matrix(rows: shape[0], cols: shape[1], data: floats)
    }

    private func mlxArrayToFloatArray(_ array: MLXArray) throws -> [Float] {
        return array.flattened().asType(MLX.DType.float32).asArray(Float.self)
    }

    // Nonisolated helper to load model outside actor isolation.
    // ModelContext is non-sendable but we only extract the model (also
    // non-sendable, protected by actor isolation once stored). This avoids
    // the Swift 6 "non-sendable result type cannot be sent from nonisolated
    // context" warning on the LLMModelFactory.load call inside the actor.
    //
    // The wrapper type silences the call-site warning about sending the
    // non-Sendable LanguageModel across actor boundaries, since the actor
    // isolation protects the stored model after assignment.
    fileprivate nonisolated static func loadModel(
        from directory: URL,
        using loader: LocalTokenizerLoader
    ) async throws -> _UncheckedSendable<any LanguageModel> {
        let context = try await LLMModelFactory.shared.load(
            from: directory,
            using: loader
        )
        return _UncheckedSendable(value: context.model)
    }

    /// Wrapper to silence Swift 6 "non-sendable result cannot cross actor
    /// boundary" warning at the call site. The MLXTrainer actor protects
    /// the model after assignment, so this is safe.
    fileprivate struct _UncheckedSendable<Wrapped>: @unchecked Sendable {
        let value: Wrapped
    }
}

/// MLXTrainer is an actor, but LanguageModel doesn't conform to Sendable.
/// We mark it @unchecked Sendable because the actor isolation protects
/// all mutable state (model, loraContainer, optimizer, stepCount, logBuffer).
extension MLXTrainer: @unchecked Sendable {}

// ============================================================================
// MARK: - Checkpointing
// ============================================================================

extension MLXTrainer {

    /// Save training checkpoint to an NPZ file.
    /// Contains: step count, LoRA parameters ONLY (not full model).
    /// Base model is static on disk - no need to checkpoint it.
    public func saveCheckpoint(to url: URL) async throws {
        guard loraContainer != nil, model != nil else {
            throw TrainingError.notPrepared
        }

        var arrays: [(String, NPYArray)] = []

        // 1. Step count as NPYArray (uint64 in little-endian)
        var stepCountVal = UInt64(stepCount)
        let stepCountData = withUnsafeBytes(of: &stepCountVal) { Data($0) }
        let stepCountArray = NPYArray(descr: "|u8", shape: [1], raw: stepCountData)
        arrays.append(("step_count", stepCountArray))

        // 2. LoRA parameters only (small - just adapter weights)
        let loraParams = loraContainer!.parameters.flattened()
        for (name, tensor) in loraParams {
            let flat = tensor.flattened().asType(MLX.DType.float32)
            let floats = flat.asArray(Float.self)
            let shape = tensor.shape
            let array = MLXArray(floats, shape).asType(MLX.DType.float32)
            let npyArray = try NPYArray.fromMLXArray(array)
            arrays.append(("lora_\(name)", npyArray))
        }

        let data = try NPZ.write(arrays)
        try data.write(to: url, options: .atomic)
        logTrainer("Saved checkpoint to \(url.lastPathComponent) (step \(stepCount))")
    }

    /// Load training checkpoint from an NPZ file.
    /// Restores: step count, LoRA parameters only.
    /// Base model is reloaded from disk in prepare() - not from checkpoint.
    public func loadCheckpoint(from url: URL) async throws {
        guard loraContainer != nil, model != nil else {
            throw TrainingError.notPrepared
        }

        let data = try Data(contentsOf: url)
        let dict = try NPZ.read(data)

        // 1. Restore step count
        if let stepCountArray = dict["step_count"],
           stepCountArray.shape.count == 1,
           stepCountArray.shape[0] == 1,
           stepCountArray.descr == "|u8" {
            let value = stepCountArray.raw.withUnsafeBytes { $0.load(as: UInt64.self) }
            self.stepCount = Int(value)
        }

        // 2. Restore LoRA parameters only (no model params)
        var loraParams: [String: MLXArray] = [:]
        for (key, npyArray) in dict where key.hasPrefix("lora_") {
            let paramName = String(key.dropFirst(5)) // remove "lora_"
            let floats = try npyArray.floats()
            let array = MLXArray(floats, npyArray.shape).asType(MLX.DType.float32)
            loraParams[paramName] = array
        }
        if !loraParams.isEmpty {
            let params = ModuleParameters.unflattened(loraParams)
            try model!.update(parameters: params, verify: .noUnusedKeys)
        }

        logTrainer("Loaded checkpoint from \(url.lastPathComponent) (step \(stepCount))")
    }
}