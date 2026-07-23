// First LocalTraining conformer: the CPU linear-proxy trainer used by the
// simulator and CI. It replaces the loop that previously lived inline in
// swairm-client/main.swift, proving the protocol seam works end to end
// before the MLX conformer exists.
//
// "Training" here is one interpolation step toward a per-device dense
// target per batch, plus Gaussian exploration noise — the same proxy for
// local fine-tuning that swarm_client.py uses. Batches carry the encoded
// target (see LinearProxyBatchCodec), so the trainer itself holds no
// knowledge of how the fleet fabricates its data.

import Foundation

// ==========================================================================
// MARK: - Batch payload codec
// ==========================================================================

/// Wire format for a linear-proxy TrainingBatch payload (little-endian):
///   UInt32 rows | UInt32 cols | UInt32 mCount
///   rows*cols Float32 (dense target, row-major)
///   mCount    Float32 (magnitude target)
public enum LinearProxyBatchCodec {
    public enum CodecError: Error { case truncated, dimensionMismatch }

    public static func encode(dense: Matrix, magnitude: [Float]) -> Data {
        var out = Data(capacity: 12 + 4 * (dense.data.count + magnitude.count))
        for v in [UInt32(dense.rows), UInt32(dense.cols), UInt32(magnitude.count)] {
            withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) }
        }
        for f in dense.data {
            withUnsafeBytes(of: f.bitPattern.littleEndian) { out.append(contentsOf: $0) }
        }
        for f in magnitude {
            withUnsafeBytes(of: f.bitPattern.littleEndian) { out.append(contentsOf: $0) }
        }
        return out
    }

    public static func decode(_ data: Data) throws -> (dense: Matrix, magnitude: [Float]) {
        guard data.count >= 12 else { throw CodecError.truncated }
        let bytes = [UInt8](data)
        func u32(_ offset: Int) -> Int {
            Int(UInt32(bytes[offset])
                | UInt32(bytes[offset + 1]) << 8
                | UInt32(bytes[offset + 2]) << 16
                | UInt32(bytes[offset + 3]) << 24)
        }
        let rows = u32(0), cols = u32(4), mCount = u32(8)
        let floatCount = rows * cols + mCount
        guard data.count == 12 + 4 * floatCount else { throw CodecError.truncated }

        var floats = [Float](repeating: 0, count: floatCount)
        for i in 0..<floatCount {
            let o = 12 + 4 * i
            let bits = UInt32(bytes[o])
                | UInt32(bytes[o + 1]) << 8
                | UInt32(bytes[o + 2]) << 16
                | UInt32(bytes[o + 3]) << 24
            floats[i] = Float(bitPattern: bits)
        }
        let dense = Matrix(rows: rows, cols: cols,
                           data: Array(floats[0..<(rows * cols)]))
        return (dense, Array(floats[(rows * cols)...]))
    }
}

// ==========================================================================
// MARK: - Batch stream helper
// ==========================================================================

/// Minimal Sendable AsyncSequence over an in-memory batch array, for the
/// simulator and tests. Real curriculum data will stream from disk instead.
public struct BatchStream: AsyncSequence, Sendable {
    public typealias Element = TrainingBatch
    private let batches: [TrainingBatch]

    public init(_ batches: [TrainingBatch]) { self.batches = batches }

    public struct AsyncIterator: AsyncIteratorProtocol {
        var remaining: ArraySlice<TrainingBatch>
        public mutating func next() async -> TrainingBatch? {
            remaining.popFirst()
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(remaining: batches[...])
    }
}

// ==========================================================================
// MARK: - Trainer
// ==========================================================================

public enum LinearProxyTrainerError: Error {
    case notPrepared
    case shapeMismatch(expected: String, got: String)
}

public actor LinearProxyTrainer: LocalTraining {
    public struct Config: Sendable {
        public let moduleName: String
        public let rows: Int
        public let cols: Int
        public let rank: Int
        public let learningRate: Float
        public let noiseScale: Float
        public let seed: UInt64

        public init(moduleName: String, rows: Int, cols: Int, rank: Int,
                    learningRate: Float = 0.5, noiseScale: Float = 0.05,
                    seed: UInt64 = 0) {
            self.moduleName = moduleName
            self.rows = rows
            self.cols = cols
            self.rank = rank
            self.learningRate = learningRate
            self.noiseScale = noiseScale
            self.seed = seed
        }
    }

    private let config: Config
    /// Battery level provider, injectable because ProcessInfo has no battery
    /// API and UIDevice only exists on iOS. nil provider == battery unknown.
    private let batteryFraction: (@Sendable () -> Float?)?

    private var direction: Matrix
    private var magnitude: [Float]
    private var rng: GaussianRNG
    private var prepared = false

    public init(config: Config,
                batteryFraction: (@Sendable () -> Float?)? = nil) {
        self.config = config
        self.batteryFraction = batteryFraction
        self.direction = Matrix(rows: config.rows, cols: config.cols)
        self.magnitude = [Float](repeating: 1, count: config.rows)
        self.rng = GaussianRNG(seed: config.seed)
    }

    // ------------------------------------------------------------ prepare

    public func prepare(globalAdapter: FetchedAdapter?) async throws {
        if let mod = globalAdapter?.modules[config.moduleName] {
            let dir = mod.B * mod.A
            guard dir.rows == config.rows, dir.cols == config.cols,
                  mod.m.count == config.rows else {
                throw LinearProxyTrainerError.shapeMismatch(
                    expected: "\(config.rows)x\(config.cols) m[\(config.rows)]",
                    got: "\(dir.rows)x\(dir.cols) m[\(mod.m.count)]")
            }
            direction = dir
            magnitude = mod.m
        } else {
            // Anchor has no global adapter yet: fresh start.
            direction = Matrix(rows: config.rows, cols: config.cols)
            magnitude = [Float](repeating: 1, count: config.rows)
        }
        prepared = true
    }

    // ------------------------------------------------------------ train

    public func train<S: AsyncSequence & Sendable>(
        batches: S, budget: ResourceBudget
    ) async throws -> TrainingReport where S.Element == TrainingBatch {
        guard prepared else { throw LinearProxyTrainerError.notPrepared }

        let start = Date()
        var steps = 0
        var lastLoss: Float?
        var termination = TerminationReason.exhaustedBatches
        let dimScale = 1.0 / Float(Double(config.cols).squareRoot())

        for try await batch in batches {
            if let reason = budgetTrip(steps: steps, start: start, budget: budget) {
                termination = reason
                break
            }

            let (target, targetM) = try LinearProxyBatchCodec.decode(batch.data)
            guard target.rows == config.rows, target.cols == config.cols,
                  targetM.count == config.rows else {
                throw LinearProxyTrainerError.shapeMismatch(
                    expected: "\(config.rows)x\(config.cols) m[\(config.rows)]",
                    got: "\(target.rows)x\(target.cols) m[\(targetM.count)]")
            }

            // One interpolation step toward the target plus exploration noise
            // (the linear proxy for a local fine-tuning step).
            let noise = randomNormalMatrix(rows: config.rows, cols: config.cols,
                                           scale: config.noiseScale * dimScale,
                                           rng: &rng)
            direction = direction
                + (target - direction).scaled(by: config.learningRate)
                + noise
            for j in 0..<config.rows {
                magnitude[j] += config.learningRate * (targetM[j] - magnitude[j])
            }

            steps += 1
            let targetNorm = target.frobeniusNorm
            lastLoss = targetNorm > 0
                ? (direction - target).frobeniusNorm / targetNorm
                : (direction - target).frobeniusNorm
        }

        // A budget can also trip exactly when the stream ends; prefer the
        // budget reason only if we broke out early above.
        return TrainingReport(stepsCompleted: steps,
                              finalLoss: lastLoss,
                              wallClock: Date().timeIntervalSince(start),
                              termination: termination)
    }

    private func budgetTrip(steps: Int, start: Date,
                            budget: ResourceBudget) -> TerminationReason? {
        if Task.isCancelled { return .cancelled }
        if steps >= budget.maxSteps { return .stepBudget }
        if Date().timeIntervalSince(start) >= budget.maxWallClock {
            return .wallClockBudget
        }
        #if !os(Linux) && !os(Windows)
        if budget.stopOnSeriousThermalState {
            let state = ProcessInfo.processInfo.thermalState
            if state == .serious || state == .critical { return .thermal }
        }
        #endif
        if let minBattery = budget.minBatteryFraction,
           let level = batteryFraction?(), level < minBattery {
            return .battery
        }
        return nil
    }

    // ------------------------------------------------------------ export

    public func exportAdapter() async throws -> [String: AdapterModule] {
        guard prepared else { throw LinearProxyTrainerError.notPrepared }
        let (a, b) = factorToRank(direction, rank: config.rank)
        return [config.moduleName: AdapterModule(A: a, B: b, m: magnitude)]
    }
}
