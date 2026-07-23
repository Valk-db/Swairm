// ProxyDeviceLoop against a mock Anchor: proves the shared device round
// loop (used by the CLI fleet, CI, and the iOS app) drives the
// fetch -> prepare -> train -> export -> upload cycle correctly without
// any network or Python involved.

import XCTest
@testable import SwairmCore

// ==========================================================================
// MARK: - Mock anchor
// ==========================================================================

/// In-memory AnchorConnecting: serves a configurable global adapter and
/// records every upload. Actor so the test can safely inspect state.
private actor MockAnchor: AnchorConnecting {
    var version = 0
    var curriculumEpoch = 0
    var global: [String: AdapterModule]? = nil
    private(set) var uploads: [AdapterUploadPayload] = []

    func setGlobal(version: Int, modules: [String: AdapterModule]) {
        self.version = version
        self.global = modules
    }

    func status() async throws -> AnchorStatus {
        AnchorStatus(version: version, curriculum_epoch: curriculumEpoch,
                     rounds: version, skew_detected: false,
                     pending: uploads.count)
    }

    func latestAdapter() async throws -> FetchedAdapter? {
        guard let global else { return nil }
        return FetchedAdapter(version: version, modules: global)
    }

    @discardableResult
    func upload(_ payload: AdapterUploadPayload) async throws -> UploadReceipt {
        uploads.append(payload)
        return UploadReceipt(queuedID: "q\(uploads.count)")
    }

    @discardableResult
    func downloadCurriculum(epoch: Int, to destination: URL) async throws
        -> CurriculumManifest {
        throw AnchorClientError.unsupported("mock has no curriculum")
    }
}

// ==========================================================================
// MARK: - Tests
// ==========================================================================

final class ProxyDeviceLoopTests: XCTestCase {
    /// Small dims keep the truncated SVD in exportAdapter fast.
    private let config = ProxyLoopConfig(rows: 16, cols: 24, rank: 4)

    func testTargetSynthesisIsDeterministic() {
        let a = ProxyDeviceLoop.synthesizeTarget(index: 3, config: config)
        let b = ProxyDeviceLoop.synthesizeTarget(index: 3, config: config)
        XCTAssertEqual(a.dense, b.dense)
        XCTAssertEqual(a.magnitude, b.magnitude)

        let other = ProxyDeviceLoop.synthesizeTarget(index: 4, config: config)
        XCTAssertNotEqual(a.dense, other.dense,
                          "different device indices must get different targets")
        for m in a.magnitude {
            XCTAssertGreaterThanOrEqual(m, 0.1)
            XCTAssertLessThanOrEqual(m, 3.0)
        }
    }

    func testRoundAgainstEmptyAnchorUploadsFullAdapter() async throws {
        let anchor = MockAnchor()
        let loop = ProxyDeviceLoop(anchor: anchor, deviceID: "phone0",
                                   deviceIndex: 0, config: config)

        let result = try await loop.runRound()

        XCTAssertEqual(result.round, 0)
        XCTAssertEqual(result.fetchedVersion, 0, "empty anchor => version 0")
        XCTAssertEqual(result.trainingReport.stepsCompleted, 1)
        XCTAssertEqual(result.receipt.queuedID, "q1")
        // Fresh adapter is all zeros, so error vs. target is exactly 1.
        XCTAssertEqual(result.dirErrorVsTarget, 1.0, accuracy: 1e-5)

        let uploads = await anchor.uploads
        XCTAssertEqual(uploads.count, 1)
        let payload = try XCTUnwrap(uploads.first)
        XCTAssertEqual(payload.deviceID, "phone0")
        XCTAssertEqual(payload.fetchVersion, 0)
        let mod = try XCTUnwrap(payload.modules[config.moduleName])
        XCTAssertEqual(mod.B.rows, config.rows)
        XCTAssertEqual(mod.A.cols, config.cols)
        XCTAssertEqual(mod.m.count, config.rows)
    }

    func testRoundFetchesGlobalAndTracksVersion() async throws {
        let anchor = MockAnchor()
        let loop = ProxyDeviceLoop(anchor: anchor, deviceID: "phone1",
                                   deviceIndex: 1, config: config)

        // Seed the anchor with a global adapter built from the device's own
        // target: the fetched direction then matches the target closely, so
        // dirErrorVsTarget must be far below 1.
        let target = ProxyDeviceLoop.synthesizeTarget(index: 1, config: config)
        let (a, b) = factorToRank(target.dense, rank: config.rank)
        await anchor.setGlobal(version: 7, modules: [
            config.moduleName: AdapterModule(A: a, B: b, m: target.magnitude)
        ])

        let result = try await loop.runRound()

        XCTAssertEqual(result.fetchedVersion, 7)
        XCTAssertLessThan(result.dirErrorVsTarget, 0.9,
                          "fetched global built from the target should be close")
        let uploads = await anchor.uploads
        let payload = try XCTUnwrap(uploads.first)
        XCTAssertEqual(payload.fetchVersion, 7,
                       "upload must carry the fetched version for staleness tracking")
    }

    func testConsecutiveRoundsConverge() async throws {
        let anchor = MockAnchor()
        let loop = ProxyDeviceLoop(anchor: anchor, deviceID: "phone2",
                                   deviceIndex: 2, config: config)

        // Simulate the Anchor aggregating this single device's uploads:
        // after each round, promote the upload to the new global.
        var lastError: Float = .infinity
        for round in 0..<3 {
            let result = try await loop.runRound()
            XCTAssertEqual(result.round, round)
            if round > 0 {
                XCTAssertLessThan(result.dirErrorVsTarget, lastError,
                                  "error vs. own target should shrink")
            }
            lastError = result.dirErrorVsTarget

            let uploads = await anchor.uploads
            let payload = try XCTUnwrap(uploads.last)
            await anchor.setGlobal(version: round + 1, modules: payload.modules)
        }
        XCTAssertLessThan(lastError, 0.6)
    }
}
