// Async HTTP client for the FCS Anchor. Endpoints pinned by main.py:
//   GET  /status          -> JSON status
//   GET  /adapter/latest  -> npz bytes + X-Adapter-Version header (404 = none yet)
//   POST /upload          -> raw npz body, no parsing server-side
//
// Fully non-blocking (no semaphores): safe from UI code and iOS background
// task runners. Conforms to AnchorConnecting so orchestration and tests can
// substitute mock transports.

import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct AnchorStatus: Codable, Sendable {
    public let version: Int
    public let curriculum_epoch: Int
    public let rounds: Int
    public let skew_detected: Bool
    public let pending: Int
}

public enum AnchorClientError: Error {
    case invalidURL(String)
    case transport(Error)
    case noResponse
    case httpStatus(Int)
    /// The Anchor does not expose this endpoint yet (e.g. curriculum download).
    case unsupported(String)
}

public final class AnchorClient: AnchorConnecting {
    public let base: URL
    private let session: URLSession

    public init(base: URL) {
        self.base = base
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 15
        self.session = URLSession(configuration: cfg)
    }

    public func status() async throws -> AnchorStatus {
        let (data, http) = try await request(path: "/status", method: "GET", body: nil)
        guard http.statusCode == 200 else {
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        return try JSONDecoder().decode(AnchorStatus.self, from: data)
    }

    /// Returns nil when the Anchor has no global adapter yet (HTTP 404).
    public func latestAdapter() async throws -> FetchedAdapter? {
        let (data, http) = try await request(path: "/adapter/latest", method: "GET", body: nil)
        if http.statusCode == 404 { return nil }
        guard http.statusCode == 200 else {
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        let version = Int(headerValue("X-Adapter-Version", in: http) ?? "0") ?? 0
        return FetchedAdapter(version: version,
                              modules: try AdapterCodec.unpackModules(data))
    }

    @discardableResult
    public func upload(_ payload: AdapterUploadPayload) async throws -> UploadReceipt {
        let raw = try AdapterCodec.packUpload(
            deviceID: payload.deviceID,
            fetchVersion: payload.fetchVersion,
            curriculumEpoch: payload.curriculumEpoch,
            modules: payload.modules)
        return try await uploadRaw(raw)
    }

    /// Escape hatch for callers that already hold packed npz wire bytes.
    @discardableResult
    public func uploadRaw(_ raw: Data) async throws -> UploadReceipt {
        let (data, http) = try await request(path: "/upload", method: "POST", body: raw)
        guard http.statusCode == 200 else {
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let queued = obj["queued"] as? String {
            return UploadReceipt(queuedID: queued)
        }
        return UploadReceipt(queuedID: "")
    }

    /// The Anchor (main.py) does not serve curriculum data yet; the protocol
    /// reserves the slot so orchestration code can be written against it now.
    @discardableResult
    public func downloadCurriculum(epoch: Int, to destination: URL) async throws -> CurriculumManifest {
        throw AnchorClientError.unsupported(
            "the Anchor exposes no curriculum endpoint yet (epoch \(epoch))")
    }

    // ------------------------------------------------------------- internals

    private func makeURL(_ path: String) -> URL? {
        var baseText = base.absoluteString
        if baseText.hasSuffix("/") { baseText = String(baseText.dropLast()) }
        return URL(string: baseText + path)
    }

    private func request(path: String, method: String,
                         body: Data?) async throws -> (Data, HTTPURLResponse) {
        guard let url = makeURL(path) else {
            throw AnchorClientError.invalidURL(base.absoluteString + path)
        }
        var req = URLRequest(url: url)
        req.httpMethod = method
        if let body = body {
            req.httpBody = body
            req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        }
        return try await perform(req)
    }

    /// Continuation-based bridge over dataTask: non-blocking and portable
    /// across Darwin Foundation and swift-corelibs FoundationNetworking.
    private func perform(_ req: URLRequest) async throws -> (Data, HTTPURLResponse) {
        try await withCheckedThrowingContinuation { continuation in
            let task = session.dataTask(with: req) { data, resp, err in
                if let err = err {
                    continuation.resume(throwing: AnchorClientError.transport(err))
                    return
                }
                guard let http = resp as? HTTPURLResponse else {
                    continuation.resume(throwing: AnchorClientError.noResponse)
                    return
                }
                continuation.resume(returning: (data ?? Data(), http))
            }
            task.resume()
        }
    }

    private func headerValue(_ name: String, in resp: HTTPURLResponse) -> String? {
        for (key, value) in resp.allHeaderFields {
            if String(describing: key).lowercased() == name.lowercased() {
                return String(describing: value)
            }
        }
        return nil
    }
}

extension AnchorClient: @unchecked Sendable {}
// @unchecked justification: all stored properties (base, session) are `let`
// and URLSession is itself thread-safe; the class holds no mutable state.
