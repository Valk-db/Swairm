// Base abstraction seams for the swarm client (DECISIONS.md D7 semantics).
//
//   AnchorConnecting  — pure transport to the FCS Anchor. No ML types beyond
//                       the wire structs (AdapterModule). Conformers must be
//                       Sendable and fully async: implementations must never
//                       block a thread (no semaphores) so they are safe to
//                       call from iOS background task runners and UI code.
//
//   LocalTraining     — the hardware abstraction. Consumes and produces wire
//                       types ([String: AdapterModule]) only, never backend
//                       tensor types, so MLX / CoreML / CPU-proxy backends
//                       are interchangeable behind the same protocol.
//
// Design invariants:
//   * Full-adapter replace semantics (D7): exportAdapter() returns the whole
//     post-training adapter state, not a delta.
//   * DoRA magnitude `m` is first-class (D4) and travels inside AdapterModule.
//   * Curriculum payloads stream to disk / through AsyncSequence; they are
//     never required to be fully resident in memory.

import Foundation

// ==========================================================================
// MARK: - Anchor transport
// ==========================================================================

/// A fetched global adapter: version is carried alongside the modules so the
/// fetch-version needed for staleness tracking (D7) cannot be forgotten.
public struct FetchedAdapter: Sendable {
    public let version: Int
    public let modules: [String: AdapterModule]

    public init(version: Int, modules: [String: AdapterModule]) {
        self.version = version
        self.modules = modules
    }
}

/// Everything the Anchor needs to ingest one device's contribution.
/// Packing to npz wire bytes is the transport's job, not the caller's.
public struct AdapterUploadPayload: Sendable {
    public let deviceID: String
    public let fetchVersion: Int
    public let curriculumEpoch: Int
    public let modules: [String: AdapterModule]

    public init(deviceID: String, fetchVersion: Int,
                curriculumEpoch: Int, modules: [String: AdapterModule]) {
        self.deviceID = deviceID
        self.fetchVersion = fetchVersion
        self.curriculumEpoch = curriculumEpoch
        self.modules = modules
    }
}

public struct UploadReceipt: Sendable {
    /// Server-assigned queue identifier ("" when the Anchor omits it).
    public let queuedID: String

    public init(queuedID: String) {
        self.queuedID = queuedID
    }
}

/// Metadata for a curriculum shard that was streamed to local storage.
public struct CurriculumManifest: Sendable {
    public let epoch: Int
    public let localURL: URL
    public let byteCount: Int64

    public init(epoch: Int, localURL: URL, byteCount: Int64) {
        self.epoch = epoch
        self.localURL = localURL
        self.byteCount = byteCount
    }
}

/// Pure transport to the Anchor. Implementations must be non-blocking.
public protocol AnchorConnecting: Sendable {
    func status() async throws -> AnchorStatus

    /// nil == the Anchor has no global adapter yet (HTTP 404).
    func latestAdapter() async throws -> FetchedAdapter?

    @discardableResult
    func upload(_ payload: AdapterUploadPayload) async throws -> UploadReceipt

    /// Streams the curriculum shard for `epoch` to `destination` on disk,
    /// never buffering the full payload in memory. Throws
    /// `AnchorClientError.unsupported` until the Anchor grows the endpoint.
    @discardableResult
    func downloadCurriculum(epoch: Int, to destination: URL) async throws -> CurriculumManifest
}

// ==========================================================================
// MARK: - Local training
// ==========================================================================

/// One unit of training data, delivered as opaque bytes. The backend owns
/// decoding (tokenized batches for MLX, dense targets for the CPU proxy, …)
/// so the protocol stays backend-agnostic.
public struct TrainingBatch: Sendable {
    public let index: Int
    public let data: Data

    public init(index: Int, data: Data) {
        self.index = index
        self.data = data
    }
}

/// Explicit iOS resource constraints. The trainer checks these between steps
/// and stops early instead of letting the OS jetsam or throttle the process.
public struct ResourceBudget: Sendable {
    /// Hard cap on optimizer steps for this session.
    public let maxSteps: Int
    /// Hard cap on wall-clock seconds (BGProcessingTask budgets are finite).
    public let maxWallClock: TimeInterval
    /// Stop when ProcessInfo.thermalState reaches .serious or worse.
    public let stopOnSeriousThermalState: Bool
    /// Stop when battery fraction drops below this (nil = ignore battery).
    public let minBatteryFraction: Float?

    public init(maxSteps: Int, maxWallClock: TimeInterval,
                stopOnSeriousThermalState: Bool = true,
                minBatteryFraction: Float? = nil) {
        self.maxSteps = maxSteps
        self.maxWallClock = maxWallClock
        self.stopOnSeriousThermalState = stopOnSeriousThermalState
        self.minBatteryFraction = minBatteryFraction
    }
}

public enum TerminationReason: Sendable, Equatable {
    /// All provided batches were consumed.
    case exhaustedBatches
    case stepBudget
    case wallClockBudget
    case thermal
    case battery
    case cancelled
}

public struct TrainingReport: Sendable {
    public let stepsCompleted: Int
    public let finalLoss: Float?
    public let wallClock: TimeInterval
    public let termination: TerminationReason

    public init(stepsCompleted: Int, finalLoss: Float?,
                wallClock: TimeInterval, termination: TerminationReason) {
        self.stepsCompleted = stepsCompleted
        self.finalLoss = finalLoss
        self.wallClock = wallClock
        self.termination = termination
    }
}

/// The hardware abstraction seam. Actor conformance makes model / optimizer
/// state race-free by construction. Conformers: an MLX trainer on Apple
/// silicon, and a CPU linear-proxy trainer for simulation and CI.
public protocol LocalTraining: Actor {
    /// Load base weights and apply the global adapter, or initialize a fresh
    /// adapter when the Anchor has none yet (nil).
    func prepare(globalAdapter: FetchedAdapter?) async throws

    /// Runs steps until the batch stream ends or the budget trips.
    /// Batches arrive as an AsyncSequence so curriculum data is streamed,
    /// never fully resident.
    func train<S: AsyncSequence & Sendable>(
        batches: S, budget: ResourceBudget
    ) async throws -> TrainingReport where S.Element == TrainingBatch

    /// Full post-training adapter state (D7: replace semantics, not deltas).
    func exportAdapter() async throws -> [String: AdapterModule]
}
