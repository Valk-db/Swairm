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
}
