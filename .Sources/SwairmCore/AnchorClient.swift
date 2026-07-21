// Synchronous HTTP client for the FCS Anchor. Endpoints pinned by main.py:
//   GET  /status          -> JSON status
//   GET  /adapter/latest  -> npz bytes + X-Adapter-Version header (404 = none yet)
//   POST /upload          -> raw npz body, no parsing server-side

import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct AnchorStatus: Codable {
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
}

public final class AnchorClient {
    public let base: URL
    private let session: URLSession

    public init(base: URL) {
        self.base = base
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 15
        self.session = URLSession(configuration: cfg)
    }

    public func status() throws -> AnchorStatus {
        let (data, http) = try request(path: "/status", method: "GET", body: nil)
        guard http.statusCode == 200 else {
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        return try JSONDecoder().decode(AnchorStatus.self, from: data)
    }

    /// Returns nil when the Anchor has no global adapter yet (HTTP 404).
    public func latestAdapter() throws -> (version: Int, modules: [String: AdapterModule])? {
        let (data, http) = try request(path: "/adapter/latest", method: "GET", body: nil)
        if http.statusCode == 404 { return nil }
        guard http.statusCode == 200 else {
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        let version = Int(headerValue("X-Adapter-Version", in: http) ?? "0") ?? 0
        return (version, try AdapterCodec.unpackModules(data))
    }

    @discardableResult
    public func upload(_ raw: Data) throws -> String {
        let (data, http) = try request(path: "/upload", method: "POST", body: raw)
        guard http.statusCode == 200 else {
            throw AnchorClientError.httpStatus(http.statusCode)
        }
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let queued = obj["queued"] as? String {
            return queued
        }
        return ""
    }

    // ------------------------------------------------------------- internals

    private func makeURL(_ path: String) -> URL? {
        var baseText = base.absoluteString
        if baseText.hasSuffix("/") { baseText = String(baseText.dropLast()) }
        return URL(string: baseText + path)
    }

    private func request(path: String, method: String,
                         body: Data?) throws -> (Data, HTTPURLResponse) {
        guard let url = makeURL(path) else {
            throw AnchorClientError.invalidURL(base.absoluteString + path)
        }
        var req = URLRequest(url: url)
        req.httpMethod = method
        if let body = body {
            req.httpBody = body
            req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        }
        return try perform(req)
    }

    private func perform(_ req: URLRequest) throws -> (Data, HTTPURLResponse) {
        var out: (Data?, URLResponse?, Error?) = (nil, nil, nil)
        let sem = DispatchSemaphore(value: 0)
        let task = session.dataTask(with: req) { data, resp, err in
            out = (data, resp, err)
            sem.signal()
        }
        task.resume()
        sem.wait()
        if let err = out.2 { throw AnchorClientError.transport(err) }
        guard let http = out.1 as? HTTPURLResponse else {
            throw AnchorClientError.noResponse
        }
        return (out.0 ?? Data(), http)
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
