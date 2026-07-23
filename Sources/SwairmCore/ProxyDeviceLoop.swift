// The shared device round loop for the linear-proxy path: one
// fetch -> prepare -> train -> export -> upload cycle against an Anchor.
//
// This is the exact loop the CLI fleet simulator, the CI integration job,
// and the iOS app all run, so a round that works in CI is byte-identical
// to a round on a sideloaded phone. Pure Foundation: no MLX, no UIKit.
//
// Target synthesis reproduces swairm-client's fleet fabric exactly
// (shared seed 42, per-device seeds 1000+i / 5000+i), so a device with
// index k behaves identically whether it lives in the CLI fleet or on a
// real phone.

import Foundation

// ==========================================================================
// MARK: - Config & results
// ==========================================================================

public struct ProxyLoopConfig: Sendable {
    public let moduleName: String
    public let rows: Int
    public let cols: Int
    public let rank: Int
    public let learningRate: Float
    public let noiseScale: Float
    /// Seed for the fleet-shared component of every device's target.
    public let sharedSeed: UInt64

    public init(moduleName: String = "layers.0.attn.q_proj",
                rows: Int = 128, cols: Int = 256, rank: Int = 4,
                learningRate: Float = 0.5, noiseScale: Float = 0.05,
                sharedSeed: UInt64 = 42) {
        self.moduleName = moduleName
        self.rows = rows
        self.cols = cols
        self.rank = rank
        self.learningRate = learningRate
        self.noiseScale = noiseScale
        self.sharedSeed = sharedSeed
    }

    /// The dimensions the Anchor's simulator fleet uses.
    public static let standard = ProxyLoopConfig()
}

/// A device's synthetic training target (what its local "data" pulls the
/// adapter toward). Deterministic in (sharedSeed, deviceIndex).
public struct ProxyDeviceTarget: Sendable {
    public let dense: Matrix
    public let magnitude: [Float]

    public init(dense: Matrix, magnitude: [Float]) {
        self.dense = dense
        self.magnitude = magnitude
    }
}

/// Everything one round produced, enough for CLI printing and app UI.
public struct ProxyRoundResult: Sendable {
    /// Caller-supplied round index (for logging).
    public let round: Int
    /// Anchor /status snapshot taken at the start of the round.
    public let status: AnchorStatus
    /// Version of the global adapter that was fetched (0 = none yet).
    public let fetchedVersion: Int
    /// Relative Frobenius error of the fetched global direction vs. this
    /// device's own target — shrinks across rounds as the fleet converges.
    public let dirErrorVsTarget: Float
    public let trainingReport: TrainingReport
    public let receipt: UploadReceipt

    public init(round: Int, status: AnchorStatus, fetchedVersion: Int,
                dirErrorVsTarget: Float, trainingReport: TrainingReport,
                receipt: UploadReceipt) {
        self.round = round
        self.status = status
        self.fetchedVersion = fetchedVersion
        self.dirErrorVsTarget = dirErrorVsTarget
        self.trainingReport = trainingReport
        self.receipt = receipt
    }
}

// ==========================================================================
// MARK: - Loop
// ==========================================================================

/// One device's proxy training loop. Owns a persistent LinearProxyTrainer
/// so exploration-noise RNG advances across rounds, matching the original
/// inline CLI loop and the Python simulator's per-device rng.
public actor ProxyDeviceLoop {
    public let deviceID: String
    public let deviceIndex: Int
    public let config: ProxyLoopConfig
    /// Exposed so fleet drivers (the CLI) can compute fleet-average targets.
    public let target: ProxyDeviceTarget

    private let anchor: AnchorConnecting
    private let trainer: LinearProxyTrainer
    private var roundsRun = 0

    public init(anchor: AnchorConnecting, deviceID: String, deviceIndex: Int,
                config: ProxyLoopConfig = .standard,
                batteryFraction: (@Sendable () -> Float?)? = nil) {
        self.anchor = anchor
        self.deviceID = deviceID
        self.deviceIndex = deviceIndex
        self.config = config
        self.target = ProxyDeviceLoop.synthesizeTarget(index: deviceIndex,
                                                       config: config)
        self.trainer = LinearProxyTrainer(
            config: .init(moduleName: config.moduleName,
                          rows: config.rows, cols: config.cols,
                          rank: config.rank,
                          learningRate: config.learningRate,
                          noiseScale: config.noiseScale,
                          seed: UInt64(5000 + deviceIndex)),
            batteryFraction: batteryFraction)
    }

    /// One full round: status -> fetch -> prepare -> train -> export ->
    /// upload. Throws on any transport or trainer failure.
    public func runRound(budget: ResourceBudget = ResourceBudget(
        maxSteps: 1, maxWallClock: 60)) async throws -> ProxyRoundResult {
        let round = roundsRun
        let status = try await anchor.status()
        let globalAdapter = try await anchor.latestAdapter()
        let fetchedVersion = globalAdapter?.version ?? 0

        // Direction error of the fetched global vs. this device's target.
        var gDir = Matrix(rows: config.rows, cols: config.cols)
        if let mod = globalAdapter?.modules[config.moduleName] {
            gDir = mod.B * mod.A
        }
        let targetNorm = target.dense.frobeniusNorm
        let dirError = targetNorm > 0
            ? (gDir - target.dense).frobeniusNorm / targetNorm
            : (gDir - target.dense).frobeniusNorm

        try await trainer.prepare(globalAdapter: globalAdapter)
        let batch = TrainingBatch(index: round, data: LinearProxyBatchCodec
            .encode(dense: target.dense, magnitude: target.magnitude))
        let report = try await trainer.train(batches: BatchStream([batch]),
                                             budget: budget)
        let modules = try await trainer.exportAdapter()

        let receipt = try await anchor.upload(AdapterUploadPayload(
            deviceID: deviceID, fetchVersion: fetchedVersion,
            curriculumEpoch: status.curriculum_epoch,
            modules: modules))

        roundsRun += 1
        return ProxyRoundResult(round: round, status: status,
                                fetchedVersion: fetchedVersion,
                                dirErrorVsTarget: dirError,
                                trainingReport: report,
                                receipt: receipt)
    }

    // ------------------------------------------------------------ targets

    /// Deterministic target synthesis, identical to the original CLI fleet:
    /// a fleet-shared low-rank direction (sharedSeed) plus a per-device
    /// perturbation (seed 1000+index), magnitudes clamped to [0.1, 3.0].
    public static func synthesizeTarget(index: Int,
                                        config: ProxyLoopConfig = .standard)
        -> ProxyDeviceTarget {
        let rankScale = 1.0 / Float(Double(config.rank).squareRoot())
        let dimScale = 1.0 / Float(Double(config.cols).squareRoot())

        var sharedRng = GaussianRNG(seed: config.sharedSeed)
        let dShared = randomNormalMatrix(rows: config.rows, cols: config.rank,
                                         scale: rankScale, rng: &sharedRng)
            * randomNormalMatrix(rows: config.rank, cols: config.cols,
                                 scale: dimScale, rng: &sharedRng)
        var mShared = [Float]()
        for _ in 0..<config.rows { mShared.append(sharedRng.uniform(in: 0.5, 2.5)) }

        var rng = GaussianRNG(seed: UInt64(1000 + index))
        let perturb = randomNormalMatrix(rows: config.rows, cols: config.rank,
                                         scale: rankScale, rng: &rng)
            * randomNormalMatrix(rows: config.rank, cols: config.cols,
                                 scale: dimScale, rng: &rng)
        let dense = dShared + perturb.scaled(by: 0.3)
        var magnitude = [Float]()
        for j in 0..<config.rows {
            magnitude.append(min(3.0, max(0.1, mShared[j] + 0.2 * rng.normal())))
        }
        return ProxyDeviceTarget(dense: dense, magnitude: magnitude)
    }
}
