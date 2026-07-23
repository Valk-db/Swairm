// NPZ = zip of .npy blobs (numpy savez/savez_compressed).
// AdapterCodec mirrors pack_upload()/unpack_upload() from the Python side.

import Foundation
import ZIPFoundation

public enum NPZError: Error {
    case notAnArchive
    case missingMeta
    case badModule(String)
}

public enum NPZ {
    public static func read(_ data: Data) throws -> [String: NPYArray] {
        let archive: Archive
        do {
            archive = try Archive(data: data, accessMode: .read, pathEncoding: nil)
        } catch {
            throw NPZError.notAnArchive
        }
        var out: [String: NPYArray] = [:]
        for entry in archive {
            guard entry.type == .file else { continue }
            var blob = Data()
            _ = try archive.extract(entry) { blob.append($0) }
            var key = entry.path
            if key.hasSuffix(".npy") { key = String(key.dropLast(4)) }
            out[key] = try NPY.parse(blob)
        }
        return out
    }

    public static func write(_ arrays: [(String, NPYArray)]) throws -> Data {
        let archive = try Archive(data: Data(), accessMode: .create)
        for (key, arr) in arrays {
            let blob = NPY.serialize(arr)
            try archive.addEntry(with: key + ".npy", type: .file,
                                 uncompressedSize: Int64(blob.count),
                                 compressionMethod: .deflate) { position, size in
                blob.subdata(in: Int(position)..<Int(position) + size)
            }
        }
        guard let out = archive.data else { throw NPZError.notAnArchive }
        return out
    }
}

public struct AdapterModule: Sendable {
    public var A: Matrix
    public var B: Matrix
    public var m: [Float]

    public init(A: Matrix, B: Matrix, m: [Float]) {
        self.A = A
        self.B = B
        self.m = m
    }
}

public struct UploadMeta: Codable {
    public var device_id: String
    public var fetch_version: Int
    public var curriculum_epoch: Int
}

public enum AdapterCodec {
    public static func packUpload(deviceID: String, fetchVersion: Int,
                                  curriculumEpoch: Int,
                                  modules: [String: AdapterModule]) throws -> Data {
        let meta = UploadMeta(device_id: deviceID, fetch_version: fetchVersion,
                              curriculum_epoch: curriculumEpoch)
        let metaData = try JSONEncoder().encode(meta)
        var arrays: [(String, NPYArray)] = [
            ("__meta__", NPYArray(descr: "|u1", shape: [metaData.count], raw: metaData))
        ]
        for name in modules.keys.sorted() {
            let mod = modules[name]!
            arrays.append(("\(name)::A", NPYArray(descr: "<f2",
                shape: [mod.A.rows, mod.A.cols],
                raw: Float16Codec.data(from: mod.A.data))))
            arrays.append(("\(name)::B", NPYArray(descr: "<f2",
                shape: [mod.B.rows, mod.B.cols],
                raw: Float16Codec.data(from: mod.B.data))))
            arrays.append(("\(name)::m", NPYArray(descr: "<f2",
                shape: [mod.m.count],
                raw: Float16Codec.data(from: mod.m))))
        }
        return try NPZ.write(arrays)
    }

    public static func unpackMeta(_ npzData: Data) throws -> UploadMeta {
        let arrays = try NPZ.read(npzData)
        guard let meta = arrays["__meta__"] else { throw NPZError.missingMeta }
        return try JSONDecoder().decode(UploadMeta.self, from: meta.raw)
    }

    public static func unpackModules(_ npzData: Data) throws -> [String: AdapterModule] {
        let arrays = try NPZ.read(npzData)
        var grouped: [String: [String: NPYArray]] = [:]
        for (key, arr) in arrays where key != "__meta__" {
            guard let r = key.range(of: "::", options: .backwards) else { continue }
            grouped[String(key[..<r.lowerBound]), default: [:]][String(key[r.upperBound...])] = arr
        }
        var out: [String: AdapterModule] = [:]
        for (name, parts) in grouped {
            guard let a = parts["A"], let b = parts["B"], let m = parts["m"],
                  a.shape.count == 2, b.shape.count == 2, m.shape.count == 1 else {
                throw NPZError.badModule(name)
            }
            out[name] = AdapterModule(
                A: Matrix(rows: a.shape[0], cols: a.shape[1], data: try a.floats()),
                B: Matrix(rows: b.shape[0], cols: b.shape[1], data: try b.floats()),
                m: try m.floats())
        }
        return out
    }
}
