import XCTest
@testable import SwairmCore

final class SwairmCoreTests: XCTestCase {

    // MARK: float16

    func testFloat16RoundTrip() {
        // Values exactly representable in half precision round-trip exactly.
        let exact: [Float] = [0, 1, -1, 0.5, 0.25, -2.5, 1024, 65504,
                              6.103515625e-05,        // 2^-14 min normal
                              5.960464477539063e-08]  // 2^-24 min subnormal
        for v in exact {
            XCTAssertEqual(Float16Codec.decode(Float16Codec.encode(v)), v,
                           "exact value \(v) did not round-trip")
        }
        // Arbitrary values round-trip within half-precision tolerance.
        let approx: [Float] = [0.1, -3.14159, 7.77, 123.456, -0.001]
        for v in approx {
            let back = Float16Codec.decode(Float16Codec.encode(v))
            XCTAssertEqual(back, v, accuracy: max(abs(v) * 0.001, 1e-6),
                           "value \(v) drifted: \(back)")
        }
        XCTAssertEqual(Float16Codec.decode(Float16Codec.encode(.infinity)), .infinity)
        XCTAssertEqual(Float16Codec.decode(Float16Codec.encode(-.infinity)), -.infinity)
        XCTAssertTrue(Float16Codec.decode(Float16Codec.encode(.nan)).isNaN)
    }

    // MARK: npy

    func testNPYRoundTripFloat16() throws {
        let values: [Float] = [1, 2, 3, 0.5, -4, 0]
        let arr = NPYArray(descr: "<f2", shape: [2, 3],
                           raw: Float16Codec.data(from: values))
        let parsed = try NPY.parse(NPY.serialize(arr))
        XCTAssertEqual(parsed.descr, "<f2")
        XCTAssertEqual(parsed.shape, [2, 3])
        XCTAssertEqual(try parsed.floats(), values)
    }

    func testNPYRoundTripUInt8OneD() throws {
        let payload = Data("{\"hello\": 1}".utf8)
        let arr = NPYArray(descr: "|u1", shape: [payload.count], raw: payload)
        let parsed = try NPY.parse(NPY.serialize(arr))
        XCTAssertEqual(parsed.descr, "|u1")
        XCTAssertEqual(parsed.shape, [payload.count])
        XCTAssertEqual(parsed.raw, payload)
    }

    func testNPYHeaderIsAligned() {
        let arr = NPYArray(descr: "<f2", shape: [128, 256],
                           raw: Data(repeating: 0, count: 4))
        let blob = NPY.serialize(arr)
        // Total header block (magic..newline) must be a multiple of 64.
        let headerLen = Int(blob[8]) | (Int(blob[9]) << 8)
        XCTAssertEqual((10 + headerLen) % 64, 0)
    }

    // MARK: npz / adapter codec

    func testNPZPackUnpack() throws {
        // Multiples of 1/8 and 1/16 are exact in float16 at these magnitudes.
        let a = Matrix(rows: 4, cols: 8, data: (0..<32).map { Float($0) * 0.125 })
        let b = Matrix(rows: 16, cols: 4, data: (0..<64).map { Float($0) * -0.0625 })
        let m = (0..<16).map { Float($0) * 0.125 + 0.5 }

        let raw = try AdapterCodec.packUpload(
            deviceID: "dev0", fetchVersion: 7, curriculumEpoch: 3,
            modules: ["layers.0.attn.q_proj": AdapterModule(A: a, B: b, m: m)])

        let meta = try AdapterCodec.unpackMeta(raw)
        XCTAssertEqual(meta.device_id, "dev0")
        XCTAssertEqual(meta.fetch_version, 7)
        XCTAssertEqual(meta.curriculum_epoch, 3)

        let modules = try AdapterCodec.unpackModules(raw)
        let mod = try XCTUnwrap(modules["layers.0.attn.q_proj"])
        XCTAssertEqual(mod.A.rows, 4)
        XCTAssertEqual(mod.A.cols, 8)
        XCTAssertEqual(mod.B.rows, 16)
        XCTAssertEqual(mod.B.cols, 4)
        XCTAssertEqual(mod.A.data, a.data)
        XCTAssertEqual(mod.B.data, b.data)
        XCTAssertEqual(mod.m, m)
    }

    // MARK: math

    func testMatmul() {
        let a = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let b = Matrix(rows: 3, cols: 2, data: [7, 8, 9, 10, 11, 12])
        XCTAssertEqual((a * b).data, [58, 64, 139, 154])
    }

    func testTruncatedSVDReconstructsLowRankMatrix() {
        var rng = GaussianRNG(seed: 7)
        let bTrue = randomNormalMatrix(rows: 64, cols: 4, scale: 0.5, rng: &rng)
        let aTrue = randomNormalMatrix(rows: 4, cols: 96, scale: 0.5, rng: &rng)
        let dense = bTrue * aTrue                       // exactly rank 4

        let (a, b) = factorToRank(dense, rank: 4)
        XCTAssertEqual(a.rows, 4)
        XCTAssertEqual(a.cols, 96)
        XCTAssertEqual(b.rows, 64)
        XCTAssertEqual(b.cols, 4)

        let relErr = ((b * a) - dense).frobeniusNorm / dense.frobeniusNorm
        XCTAssertLessThan(relErr, 1e-3,
                          "rank-4 matrix should reconstruct to float32 noise")
    }

    func testSVDSingularValuesAreSortedDescending() {
        var rng = GaussianRNG(seed: 11)
        let dense = randomNormalMatrix(rows: 32, cols: 48, scale: 1, rng: &rng)
        let svd = truncatedSVD(dense, rank: 6)
        for i in 1..<svd.S.count {
            XCTAssertGreaterThanOrEqual(svd.S[i - 1], svd.S[i])
        }
    }

    // MARK: linear-proxy trainer

    func testLinearProxyBatchCodecRoundTrip() throws {
        let dense = Matrix(rows: 3, cols: 5,
                           data: (0..<15).map { Float($0) * 0.25 - 1 })
        let magnitude: [Float] = [0.5, 1.5, 2.5]
        let data = LinearProxyBatchCodec.encode(dense: dense, magnitude: magnitude)
        let (d2, m2) = try LinearProxyBatchCodec.decode(data)
        XCTAssertEqual(d2, dense)
        XCTAssertEqual(m2, magnitude)
    }

    func testLinearProxyTrainerConvergesTowardTarget() async throws {
        let rows = 16, cols = 24, rank = 4
        var rng = GaussianRNG(seed: 3)
        // Exactly rank-4 target so factorToRank loses nothing.
        let target = randomNormalMatrix(rows: rows, cols: rank, scale: 0.5, rng: &rng)
            * randomNormalMatrix(rows: rank, cols: cols, scale: 0.5, rng: &rng)
        let targetM = (0..<rows).map { _ in rng.uniform(in: 0.5, 2.5) }

        // noiseScale 0 => pure interpolation, loss must shrink by (1-lr)^k.
        let trainer = LinearProxyTrainer(config: .init(
            moduleName: "layers.0.attn.q_proj", rows: rows, cols: cols,
            rank: rank, learningRate: 0.5, noiseScale: 0, seed: 9))
        try await trainer.prepare(globalAdapter: nil)

        let payload = LinearProxyBatchCodec.encode(dense: target, magnitude: targetM)
        let batches = (0..<10).map { TrainingBatch(index: $0, data: payload) }
        let budget = ResourceBudget(maxSteps: 100, maxWallClock: 30,
                                    stopOnSeriousThermalState: false)
        let report = try await trainer.train(batches: BatchStream(batches),
                                             budget: budget)

        XCTAssertEqual(report.stepsCompleted, 10)
        XCTAssertEqual(report.termination, .exhaustedBatches)
        let loss = try XCTUnwrap(report.finalLoss)
        XCTAssertLessThan(loss, 2e-3, "10 halving steps should reach ~1e-3")

        // Export reconstructs the direction near the target (D7 semantics).
        let modules = try await trainer.exportAdapter()
        let mod = try XCTUnwrap(modules["layers.0.attn.q_proj"])
        let relErr = ((mod.B * mod.A) - target).frobeniusNorm
            / target.frobeniusNorm
        XCTAssertLessThan(relErr, 5e-3)
        for j in 0..<rows {
            XCTAssertEqual(mod.m[j], targetM[j], accuracy: 0.01)
        }
    }

    func testLinearProxyTrainerHonorsStepBudget() async throws {
        let trainer = LinearProxyTrainer(config: .init(
            moduleName: "m", rows: 4, cols: 4, rank: 2,
            learningRate: 0.5, noiseScale: 0, seed: 1))
        try await trainer.prepare(globalAdapter: nil)

        let payload = LinearProxyBatchCodec.encode(
            dense: Matrix.identity(4), magnitude: [1, 1, 1, 1])
        let batches = (0..<10).map { TrainingBatch(index: $0, data: payload) }
        let report = try await trainer.train(
            batches: BatchStream(batches),
            budget: ResourceBudget(maxSteps: 3, maxWallClock: 30,
                                   stopOnSeriousThermalState: false))
        XCTAssertEqual(report.stepsCompleted, 3)
        XCTAssertEqual(report.termination, .stepBudget)
    }

    func testLinearProxyTrainerHonorsBatteryFloor() async throws {
        let trainer = LinearProxyTrainer(
            config: .init(moduleName: "m", rows: 4, cols: 4, rank: 2,
                          learningRate: 0.5, noiseScale: 0, seed: 1),
            batteryFraction: { 0.10 })
        try await trainer.prepare(globalAdapter: nil)

        let payload = LinearProxyBatchCodec.encode(
            dense: Matrix.identity(4), magnitude: [1, 1, 1, 1])
        let report = try await trainer.train(
            batches: BatchStream([TrainingBatch(index: 0, data: payload)]),
            budget: ResourceBudget(maxSteps: 10, maxWallClock: 30,
                                   stopOnSeriousThermalState: false,
                                   minBatteryFraction: 0.2))
        XCTAssertEqual(report.stepsCompleted, 0)
        XCTAssertEqual(report.termination, .battery)
    }

    func testLinearProxyTrainerRequiresPrepare() async {
        let trainer = LinearProxyTrainer(config: .init(
            moduleName: "m", rows: 4, cols: 4, rank: 2))
        do {
            _ = try await trainer.exportAdapter()
            XCTFail("exportAdapter before prepare must throw")
        } catch {
            // expected
        }
    }
}
