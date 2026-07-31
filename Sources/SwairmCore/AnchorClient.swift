// Async HTTP client for the FCS Anchor. Endpoints pinned by main.py:
//   GET  /status          -> JSON status
//   GET  /adapter/latest  -> npz bytes + X-Adapter-Version header (404 = none yet)
//   POST /upload          -> raw npz body, no parsing server-side
//   GET  /curriculum/{epoch}/manifest -> JSON manifest
//   GET  /curriculum/{epoch}/{shard}  -> npz bytes + X-Shard-SHA256 header
//   GET  /models/base/{model_name}/manifest -> JSON manifest
//   GET  /models/base/{model_name}/{file}  -> file bytes
//
// Fully non-blocking (no semaphores): safe from UI code and iOS background
// task runners. Conforms to AnchorConnecting so orchestration and tests can
// substitute mock transports.
//
// HMAC-SHA256 signing (when secret provided): X-HMAC-Signature: sha256=<hex>
// Signature = HMAC-SHA256(secret, METHOD\nPATH\nBODY)

import Foundation
import CommonCrypto
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
    /// HMAC secret not configured but required by server.
    case hmacRequired
    /// HMAC signature verification failed.
    case hmacInvalid
    /// Base model file verification failed.
    case modelVerificationFailed(String)
}

public final class AnchorClient: AnchorConnecting, CurriculumDownloading, BaseModelDownloading {
    public let base: URL
    private let session: URLSession
    private let hmacSecret: Data?

    /// Initialize with optional HMAC secret for request signing.
    /// When nil (default), no HMAC signature is sent (dev mode).
    public init(base: URL, hmacSecret: Data? = nil) {
        self.base = base
        self.hmacSecret = hmacSecret
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 30
        cfg.timeoutIntervalForResource = 300  // 5 min for large model downloads
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

    // ------------------------------------------------------------- curriculum download
    /// Fetch manifest for a curriculum epoch.
    public func fetchManifest(epoch: Int) async throws -> CurriculumManifest {
        let (data, http) = try await request(path: "/curriculum/\(epoch)/manifest", method: "GET", body: nil)
        guard http.statusCode == 200 else {
            if http.statusCode == 404 {
                throw CurriculumError.manifestNotFound(epoch)
            }
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        return try JSONDecoder().decode(CurriculumManifest.self, from: data)
    }

    /// Stream a single shard to disk, validating SHA256 after download.
    public func downloadShard(epoch: Int, shardName: String, to destination: URL) async throws -> URL {
        // Validate shard name to prevent path traversal
        if shardName.contains("..") || shardName.contains("/") || shardName.contains("\\") {
            throw CurriculumError.invalidShardName(shardName)
        }
        let (data, http) = try await request(path: "/curriculum/\(epoch)/\(shardName)", method: "GET", body: nil)
        guard http.statusCode == 200 else {
            if http.statusCode == 404 {
                throw CurriculumError.shardNotFound(epoch, shardName)
            }
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        // Validate SHA256 if provided in header
        if let expectedSHA = headerValue("X-Shard-SHA256", in: http) {
            let actualSHA = computeSHA256(data)
            if actualSHA != expectedSHA {
                throw CurriculumError.integrityCheckFailed(expected: expectedSHA, actual: actualSHA)
            }
        }
        try data.write(to: destination, options: .atomic)
        return destination
    }

    // ------------------------------------------------------------- base model download

    /// Fetch manifest for a base model (list of files with SHA256).
    public func fetchBaseModelManifest(modelName: String) async throws -> BaseModelManifest {
        let (data, http) = try await request(path: "/models/base/\(modelName)/manifest", method: "GET", body: nil)
        guard http.statusCode == 200 else {
            if http.statusCode == 404 {
                throw BaseModelDownloadError.manifestNotFound(modelName)
            }
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        // Decode server manifest and map to our BaseModelManifest
        struct ServerManifest: Codable {
            let model_name: String
            let files: [ServerFileInfo]
        }
        struct ServerFileInfo: Codable {
            let name: String
            let sha256: String
            let size: Int
        }
        let serverManifest = try JSONDecoder().decode(ServerManifest.self, from: data)
        let files = serverManifest.files.map { BaseModelFile(name: $0.name, sha256: $0.sha256, size: Int64($0.size)) }
        return BaseModelManifest(modelName: serverManifest.model_name, files: files)
    }

    /// Stream a single base model file to disk, validating SHA256 after download.
    public func downloadBaseModelFile(modelName: String, fileName: String, to destination: URL, expectedSHA: String) async throws -> URL {
        // Validate file name to prevent path traversal
        if fileName.contains("..") || fileName.contains("/") || fileName.contains("\\") {
            throw BaseModelDownloadError.invalidFileName(fileName)
        }
        let (data, http) = try await request(path: "/models/base/\(modelName)/\(fileName)", method: "GET", body: nil)
        guard http.statusCode == 200 else {
            if http.statusCode == 404 {
                throw BaseModelDownloadError.fileNotFound(modelName, fileName)
            }
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        // Validate SHA256
        let actualSHA = computeSHA256(data)
        if actualSHA != expectedSHA {
            throw BaseModelDownloadError.integrityCheckFailed(expected: expectedSHA, actual: actualSHA)
        }
        try data.write(to: destination, options: .atomic)
        return destination
    }

    // ------------------------------------------------------------- internals

    private func makeURL(_ path: String) -> URL? {
        var baseText = base.absoluteString
        if baseText.hasSuffix("/") { baseText = String(baseText.dropLast()) }
        return URL(string: baseText + path)
    }

    private func headerValue(_ name: String, in http: HTTPURLResponse) -> String? {
        // HTTP header field names are case-insensitive; try exact match first,
        // then case-insensitive fallback (Foundation sometimes normalizes keys)
        if let exact = http.allHeaderFields[name] as? String {
            return exact
        }
        let lower = name.lowercased()
        for (key, value) in http.allHeaderFields {
            if (key as? String)?.lowercased() == lower,
               let str = value as? String {
                return str
            }
        }
        return nil
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
        // Add HMAC signature if secret is configured
        if let secret = hmacSecret {
            let bodyForSig = body ?? Data()
            let canonical = "\(method)\n\(path)\n".data(using: .utf8)! + bodyForSig
            let signature = hmacSHA256(secret, canonical)
            req.setValue("sha256=\(signature)", forHTTPHeaderField: "X-HMAC-Signature")
        }
        return try await perform(req)
    }

    /// Compute HMAC-SHA256 signature
    private func hmacSHA256(_ key: Data, _ data: Data) -> String {
        var hmac = [UInt8](repeating: 0, count: 32)
        key.withUnsafeBytes { keyPtr in
            data.withUnsafeBytes { dataPtr in
                CCHmac(CCHmacAlgorithm(kCCHmacAlgSHA256),
                       keyPtr.baseAddress, key.count,
                       dataPtr.baseAddress, data.count,
                       &hmac)
            }
        }
        return hmac.map { String(format: "%02x", $0) }.joined()
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

    /// Compute SHA256 hash of data as hex string
    private func computeSHA256(_ data: Data) -> String {
        var hash = [UInt8](repeating: 0, count: 32)
        data.withUnsafeBytes { _ = CC_SHA256($0.baseAddress, CC_LONG(data.count), &hash) }
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}

extension AnchorClient: @unchecked Sendable {}
// @unchecked justification: all stored properties (base, session) are `let`
// and URLSession is itself thread-safe; the class holds no mutable state.