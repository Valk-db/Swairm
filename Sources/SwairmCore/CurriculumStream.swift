// Curriculum streaming: AsyncSequence over tokenized shards for MLX training.
// Never fully resident in memory; streams .npz shards from disk.
//
// Shard format (.npz):
//   token_ids: [num_sequences, seq_len]  (UInt32)
//   labels:    [num_sequences, seq_len]  (UInt32)
//
// Batches are yielded as TrainingBatch with encoded binary data matching
// MLXTrainer.decodeBatch expectations (interleaved token/label UInt32).

import Foundation

/// Loads and iterates curriculum shards from a directory.
public struct CurriculumLoader: Sendable {
    public let directory: URL
    public let batchSize: Int
    public let sequenceLength: Int
    private let shardFiles: [URL]

    public init(directory: URL, batchSize: Int, sequenceLength: Int) throws {
        self.directory = directory
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength

        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "npz" }.sorted()
        guard !files.isEmpty else {
            throw CurriculumError.noShardsFound(directory.path)
        }
        self.shardFiles = files
    }

    /// Stream all shards as an AsyncSequence of TrainingBatch.
    public func batches() -> CurriculumBatchSequence {
        CurriculumBatchSequence(
            shardFiles: shardFiles,
            batchSize: batchSize,
            sequenceLength: sequenceLength
        )
    }
}

/// AsyncSequence yielding TrainingBatch from curriculum shards.
public struct CurriculumBatchSequence: AsyncSequence, Sendable {
    public typealias Element = TrainingBatch
    public typealias AsyncIterator = CurriculumBatchIterator

    let shardFiles: [URL]
    let batchSize: Int
    let sequenceLength: Int

    public func makeAsyncIterator() -> CurriculumBatchIterator {
        CurriculumBatchIterator(
            shardFiles: shardFiles,
            batchSize: batchSize,
            sequenceLength: sequenceLength
        )
    }
}

/// Iterator over curriculum shards, yielding batches.
public struct CurriculumBatchIterator: AsyncIteratorProtocol, Sendable {
    let shardFiles: [URL]
    let batchSize: Int
    let sequenceLength: Int

    private var shardIndex = 0
    private var currentShardTokens: [UInt32] = []
    private var currentShardLabels: [UInt32] = []
    private var positionInShard = 0
    private var batchCounter = 0

    public mutating func next() async throws -> TrainingBatch? {
        while true {
            // Load next shard if needed
            if currentShardTokens.isEmpty || positionInShard >= currentShardTokens.count {
                if shardIndex >= shardFiles.count {
                    return nil // exhausted
                }
                try await loadShard(shardFiles[shardIndex])
                shardIndex += 1
                positionInShard = 0
            }

            // Check if we have enough for a batch
            let remaining = currentShardTokens.count - positionInShard
            let needed = batchSize * sequenceLength

            if remaining < needed {
                // Not enough for a full batch; load next shard and continue
                if shardIndex >= shardFiles.count {
                    return nil
                }
                continue
            }

            // Extract batch
            let tokenSlice = Array(currentShardTokens[positionInShard..<positionInShard + needed])
            let labelSlice = Array(currentShardLabels[positionInShard..<positionInShard + needed])
            positionInShard += needed

            // Encode as interleaved UInt32 (token, label, token, label...)
            var data = Data(capacity: needed * 8)
            for i in 0..<needed {
                var token = tokenSlice[i].littleEndian
                var label = labelSlice[i].littleEndian
                data.append(contentsOf: withUnsafeBytes(of: &token) { Array($0) })
                data.append(contentsOf: withUnsafeBytes(of: &label) { Array($0) })
            }

            let batch = TrainingBatch(index: batchCounter, data: data)
            batchCounter += 1
            return batch
        }
    }

    private mutating func loadShard(_ url: URL) async throws {
        let data = try Data(contentsOf: url)
        let arrays = try NPZ.read(data)

        guard let tokenArray = arrays["token_ids"],
              let labelArray = arrays["labels"] else {
            throw CurriculumError.missingArrays(url.path)
        }

        let tokens = try tokenArray.asUInt32()
        let labels = try labelArray.asUInt32()

        // Flatten [num_seq, seq_len] -> [num_seq * seq_len]
        currentShardTokens = tokens
        currentShardLabels = labels
        positionInShard = 0
    }
}

// ============================================================================
// MARK: - NPZ Array Extensions
// ============================================================================

extension NPYArray {
    /// Convert NPY array to [UInt32] (handles various dtypes).
    func asUInt32() throws -> [UInt32] {
        switch descr {
        case "|u4", "<u4", ">u4": // uint32
            return raw.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        case "|i4", "<i4", ">i4": // int32
            return raw.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)).map { UInt32($0) } }
        case "|u2", "<u2", ">u2": // uint16
            return raw.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)).map { UInt32($0) } }
        case "|u1", "|b1": // uint8
            return raw.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)).map { UInt32($0) } }
        default:
            throw CurriculumError.unsupportedDtype(descr, path: "")
        }
    }
}

// ============================================================================
// MARK: - Errors
// ============================================================================

enum CurriculumError: Error {
    case noShardsFound(String)
    case missingArrays(String)
    case unsupportedDtype(String, path: String)
    case decodeError(String)
}