// ModelDownloader: downloads curriculum shards and model adapters from the Anchor.
//
// The FCS Anchor exposes:
//   GET /curriculum/{epoch}/manifest -> CurriculumManifest (JSON)
//   GET /curriculum/{epoch}/{shard}  -> NPZ bytes + X-Shard-SHA256 header
//   GET /adapter/latest              -> NPZ adapter bytes + X-Adapter-Version header
//
// This class uses AnchorClient (which implements CurriculumDownloading) to fetch
// and save files to the app's Documents directory where MLXDeviceLoop expects them.
//
// Curriculum shards -> Documents/curriculum/epoch_{epoch}/shard_XXXXX.npz
// Model adapter     -> Documents/mlx-model/ (NPZ format for MLXTrainer.prepare)
//
// Progress is reported via an AsyncStream for UI updates.

import Foundation
import CommonCrypto

/// Progress updates for download operations.
public struct DownloadProgress: Sendable {
    public let phase: Phase
    public let current: Int
    public let total: Int
    public let message: String

    public enum Phase: Sendable {
        case fetchingManifest
        case downloadingShards
        case downloadingModel
        case verifying
        case complete
        case error
    }

    public init(phase: Phase, current: Int, total: Int, message: String) {
        self.phase = phase
        self.current = current
        self.total = total
        self.message = message
    }
}

/// Errors specific to model/curriculum download.
public enum ModelDownloadError: Error, Sendable {
    case curriculumError(CurriculumError)
    case modelDirectoryCreationFailed
    case modelFetchFailed(underlying: Error)
    case invalidAnchorResponse(String)
    case writeFailed(underlying: Error)
    case notImplemented
}

/// Downloads curriculum shards and model adapters from the Anchor.
/// Uses AnchorClient which conforms to CurriculumDownloading.
public actor ModelDownloader {
    private let documentsURL: URL

    public init() {
        self.documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Download all curriculum shards for an epoch to Documents/curriculum/epoch_{epoch}/
    ///
    /// - Parameters:
    ///   - epoch: Curriculum epoch to download
    ///   - downloader: Transport that implements CurriculumDownloading (e.g., AnchorClient)
    ///   - progress: Optional callback for progress updates
    /// - Returns: URL of the epoch directory containing shards
    public func downloadCurriculum(
        epoch: Int,
        downloader: CurriculumDownloading,
        progress: ((DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        // Phase 1: Fetch manifest
        progress?(DownloadProgress(phase: .fetchingManifest, current: 0, total: 1,
                                   message: "Fetching curriculum manifest..."))
        let manifest: CurriculumManifest
        do {
            manifest = try await downloader.fetchManifest(epoch: epoch)
        } catch let error as CurriculumError {
            progress?(DownloadProgress(phase: .error, current: 0, total: 1,
                                       message: "Failed to fetch manifest: \(error)"))
            throw ModelDownloadError.curriculumError(error)
        } catch {
            progress?(DownloadProgress(phase: .error, current: 0, total: 1,
                                       message: "Failed to fetch manifest: \(error)"))
            throw ModelDownloadError.manifestFetchFailed(underlying: error)
        }

        guard !manifest.shards.isEmpty else {
            progress?(DownloadProgress(phase: .error, current: 0, total: 1,
                                       message: "Manifest contains no shards"))
            throw ModelDownloadError.curriculumError(.shardNotFound(epoch, "no shards in manifest"))
        }

        // Create epoch directory
        let epochDir = documentsURL.appendingPathComponent("curriculum/epoch_\(epoch)")
        do {
            try FileManager.default.createDirectory(at: epochDir, withIntermediateDirectories: true)
        } catch {
            progress?(DownloadProgress(phase: .error, current: 0, total: 1,
                                       message: "Failed to create curriculum directory: \(error)"))
            throw ModelDownloadError.modelDirectoryCreationFailed
        }

        // Phase 2: Download shards
        progress?(DownloadProgress(phase: .downloadingShards, current: 0, total: manifest.shards.count,
                                   message: "Downloading \(manifest.shards.count) shard(s)..."))

        for (index, shardInfo) in manifest.shards.enumerated() {
            let destURL = epochDir.appendingPathComponent(shardInfo.name)

            // Skip if already exists and matches SHA256
            if FileManager.default.fileExists(atPath: destURL.path) {
                do {
                    let existingData = try Data(contentsOf: destURL)
                    let existingSHA = computeSHA256(existingData)
                    if existingSHA == shardInfo.sha256 {
                        progress?(DownloadProgress(phase: .downloadingShards,
                                                   current: index + 1, total: manifest.shards.count,
                                                   message: "Shard \(shardInfo.name) already present (verified)"))
                        continue
                    }
                } catch {
                    // If we can't read/verify, re-download
                }
            }

            do {
                progress?(DownloadProgress(phase: .downloadingShards,
                                           current: index, total: manifest.shards.count,
                                           message: "Downloading shard \(shardInfo.name) (\(index + 1)/\(manifest.shards.count))..."))
                _ = try await downloader.downloadShard(epoch: epoch, shardName: shardInfo.name, to: destURL)
                progress?(DownloadProgress(phase: .downloadingShards,
                                           current: index + 1, total: manifest.shards.count,
                                           message: "Downloaded shard \(shardInfo.name) (\(index + 1)/\(manifest.shards.count))"))
            } catch {
                progress?(DownloadProgress(phase: .error,
                                           current: index + 1, total: manifest.shards.count,
                                           message: "Failed to download shard \(shardInfo.name): \(error)"))
                throw ModelDownloadError.shardDownloadFailed(shard: shardInfo.name, underlying: error)
            }
        }

        progress?(DownloadProgress(phase: .complete, current: manifest.shards.count,
                                   total: manifest.shards.count,
                                   message: "Curriculum epoch \(epoch) ready (\(manifest.totalShards) shards, \(manifest.totalSequences) sequences)"))
        return epochDir
    }

    /// Download the latest global adapter from Anchor to Documents/mlx-model/
    ///
    /// The adapter is saved as a single NPZ file containing all modules (A, B, m per module),
    /// which is the format MLXTrainer.prepare() expects when loading from disk.
    ///
    /// - Parameters:
    ///   - anchor: AnchorClient instance (for fetching latest adapter)
    ///   - progress: Optional callback for progress updates
    /// - Returns: URL of the downloaded model directory
    public func downloadLatestModel(
        anchor: AnchorClient,
        progress: ((DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        progress?(DownloadProgress(phase: .downloadingModel, current: 0, total: 1,
                                   message: "Fetching latest adapter from Anchor..."))

        let fetched: FetchedAdapter
        do {
            guard let adapter = try await anchor.latestAdapter() else {
                throw ModelDownloadError.invalidAnchorResponse("No global adapter available (version 0)")
            }
            fetched = adapter
        } catch {
            progress?(DownloadProgress(phase: .error, current: 0, total: 1,
                                       message: "Failed to fetch adapter: \(error)"))
            throw ModelDownloadError.modelFetchFailed(underlying: error)
        }

        // Create model directory
        let modelDir = documentsURL.appendingPathComponent("mlx-model")
        do {
            try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        } catch {
            progress?(DownloadProgress(phase: .error, current: 0, total: 1,
                                       message: "Failed to create model directory: \(error)"))
            throw ModelDownloadError.modelDirectoryCreationFailed
        }

        // Save adapter as NPZ file
        let modelURL = modelDir.appendingPathComponent("adapter_v\(fetched.version).npz")
        do {
            progress?(DownloadProgress(phase: .verifying, current: 0, total: 1,
                                       message: "Saving adapter (version \(fetched.version))..."))
            let npzData = try AdapterCodec.packUpload(
                deviceID: "download",
                fetchVersion: fetched.version,
                curriculumEpoch: 0,  // not used for model download
                modules: fetched.modules
            )
            try npzData.write(to: modelURL, options: .atomic)
        } catch {
            progress?(DownloadProgress(phase: .error, current: 0, total: 1,
                                       message: "Failed to save adapter: \(error)"))
            throw ModelDownloadError.writeFailed(underlying: error)
        }

        progress?(DownloadProgress(phase: .complete, current: 1, total: 1,
                                   message: "Model adapter v\(fetched.version) downloaded to \(modelURL.lastPathComponent)"))
        return modelDir
    }

    /// Download a specific adapter version by version number.
    /// Anchor endpoint: GET /models/v_{version:05d}.npz (if implemented)
    public func downloadModelVersion(
        version: Int,
        anchor: AnchorClient,
        progress: ((DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        // This would require a new Anchor endpoint. For now, we only support
        // downloading the latest via /adapter/latest.
        // TODO: Add specific version endpoint to AnchorClient if needed.
        throw ModelDownloadError.invalidAnchorResponse("Specific version download not yet implemented")
    }

    /// Check if curriculum for an epoch already exists locally.
    public func hasCurriculum(epoch: Int) -> Bool {
        let epochDir = documentsURL.appendingPathComponent("curriculum/epoch_\(epoch)")
        guard FileManager.default.fileExists(atPath: epochDir.path) else { return false }
        let shards = try? FileManager.default.contentsOfDirectory(
            at: epochDir, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "npz" }
        return !(shards?.isEmpty ?? true)
    }

    /// Check if model adapter exists locally.
    public func hasModel() -> Bool {
        let modelDir = documentsURL.appendingPathComponent("mlx-model")
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return false }
        let files = try? FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "npz" }
        return !(files?.isEmpty ?? true)
    }

    /// Delete all downloaded curriculum for an epoch.
    public func deleteCurriculum(epoch: Int) throws {
        let epochDir = documentsURL.appendingPathComponent("curriculum/epoch_\(epoch)")
        if FileManager.default.fileExists(atPath: epochDir.path) {
            try FileManager.default.removeItem(at: epochDir)
        }
    }

    /// Delete all downloaded model adapters.
    public func deleteModel() throws {
        let modelDir = documentsURL.appendingPathComponent("mlx-model")
        if FileManager.default.fileExists(atPath: modelDir.path) {
            try FileManager.default.removeItem(at: modelDir)
        }
    }

    // MARK: - Helpers

    private func computeSHA256(_ data: Data) -> String {
        var hash = [UInt8](repeating: 0, count: 32)
        data.withUnsafeBytes { _ = CC_SHA256($0.baseAddress, CC_LONG(data.count), &hash) }
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}