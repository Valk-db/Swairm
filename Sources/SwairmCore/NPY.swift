// NPY format + float16 codec. Wire format pinned by main.py/swarm_client.py:
// uploads are float16 ('<f2'), Anchor snapshots are float32 ('<f4'),
// __meta__ is uint8 ('|u1') JSON bytes.

import Foundation

public enum Float16Codec {
    public static func encode(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let fexp = Int((bits >> 23) & 0xFF)
        var mant = bits & 0x7F_FFFF
        if fexp == 0xFF {                                  // inf / nan
            return sign | 0x7C00 | (mant != 0 ? 0x0200 : 0)
        }
        if (bits & 0x7FFF_FFFF) == 0 { return sign }       // +/- zero
        let hexp = fexp - 127 + 15
        if hexp >= 0x1F { return sign | 0x7C00 }           // overflow -> inf
        if hexp <= 0 {                                     // subnormal half
            if hexp < -10 { return sign }                  // underflow -> 0
            mant |= 0x80_0000
            let shift = UInt32(14 - hexp)
            var half = UInt16(mant >> shift)
            if (mant >> (shift - 1)) & 1 != 0 { half &+= 1 }
            return sign | half
        }
        var half = UInt16(hexp << 10) | UInt16(mant >> 13)
        if (mant & 0x1000) != 0 { half &+= 1 }             // round to nearest
        return sign | half
    }

    public static func decode(_ half: UInt16) -> Float {
        let sign = UInt32(half & 0x8000) << 16
        let hexp = UInt32((half >> 10) & 0x1F)
        let mant = UInt32(half & 0x3FF)
        let bits: UInt32
        if hexp == 0 {
            if mant == 0 {
                bits = sign
            } else {                                       // subnormal half
                var m = mant
                var shifts: UInt32 = 0
                while m & 0x400 == 0 { m <<= 1; shifts += 1 }
                m &= 0x3FF
                bits = sign | ((113 - shifts) << 23) | (m << 13)
            }
        } else if hexp == 0x1F {                           // inf / nan
            bits = sign | 0x7F80_0000 | (mant << 13)
        } else {
            bits = sign | ((hexp + 112) << 23) | (mant << 13)
        }
        return Float(bitPattern: bits)
    }

    public static func data(from floats: [Float]) -> Data {
        var out = Data(capacity: floats.count * 2)
        for f in floats {
            let h = encode(f)
            out.append(UInt8(h & 0xFF))
            out.append(UInt8(h >> 8))
        }
        return out
    }

    public static func floats(from data: Data) -> [Float] {
        let bytes = [UInt8](data)
        var out = [Float]()
        out.reserveCapacity(bytes.count / 2)
        var i = 0
        while i + 1 < bytes.count {
            out.append(decode(UInt16(bytes[i]) | (UInt16(bytes[i + 1]) << 8)))
            i += 2
        }
        return out
    }
}

public enum NPYError: Error {
    case badMagic
    case badHeader
    case unsupportedDtype(String)
    case fortranOrderUnsupported
}

public struct NPYArray {
    public var descr: String
    public var shape: [Int]
    public var raw: Data

    public init(descr: String, shape: [Int], raw: Data) {
        self.descr = descr
        self.shape = shape
        self.raw = raw
    }

    public var count: Int { shape.reduce(1, *) }

    public func floats() throws -> [Float] {
        switch descr {
        case "<f2", "=f2":
            return Float16Codec.floats(from: raw)
        case "<f4", "=f4":
            let bytes = [UInt8](raw)
            var out = [Float]()
            out.reserveCapacity(bytes.count / 4)
            var i = 0
            while i + 3 < bytes.count {
                let u = UInt32(bytes[i])
                    | (UInt32(bytes[i + 1]) << 8)
                    | (UInt32(bytes[i + 2]) << 16)
                    | (UInt32(bytes[i + 3]) << 24)
                out.append(Float(bitPattern: u))
                i += 4
            }
            return out
        default:
            throw NPYError.unsupportedDtype(descr)
        }
    }
}

public enum NPY {
    public static func parse(_ data: Data) throws -> NPYArray {
        let bytes = [UInt8](data)
        guard bytes.count > 10, bytes[0] == 0x93,
              String(bytes: bytes[1...5], encoding: .ascii) == "NUMPY" else {
            throw NPYError.badMagic
        }
        let major = bytes[6]
        let headerLen: Int
        let headerStart: Int
        if major >= 2 {
            guard bytes.count >= 12 else { throw NPYError.badHeader }
            headerLen = Int(bytes[8]) | (Int(bytes[9]) << 8)
                | (Int(bytes[10]) << 16) | (Int(bytes[11]) << 24)
            headerStart = 12
        } else {
            headerLen = Int(bytes[8]) | (Int(bytes[9]) << 8)
            headerStart = 10
        }
        guard bytes.count >= headerStart + headerLen,
              let header = String(bytes: bytes[headerStart..<headerStart + headerLen],
                                  encoding: .utf8) else {
            throw NPYError.badHeader
        }
        if header.contains("'fortran_order': True") {
            throw NPYError.fortranOrderUnsupported
        }
        guard let descr = quotedValue(after: "descr", in: header) else {
            throw NPYError.badHeader
        }
        guard let shapeRange = header.range(of: "'shape':"),
              let open = header[shapeRange.upperBound...].firstIndex(of: "("),
              let close = header[open...].firstIndex(of: ")") else {
            throw NPYError.badHeader
        }
        let inner = header[header.index(after: open)..<close]
        let shape = inner.split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        let payload = Data(bytes[(headerStart + headerLen)...])
        return NPYArray(descr: descr, shape: shape, raw: payload)
    }

    public static func serialize(_ array: NPYArray) -> Data {
        var header = "{'descr': '\(array.descr)', 'fortran_order': False, "
            + "'shape': \(shapeText(array.shape)), }"
        let unpadded = 10 + header.count + 1
        let pad = (64 - unpadded % 64) % 64
        header += String(repeating: " ", count: pad) + "\n"
        var out = Data([0x93])
        out.append("NUMPY".data(using: .ascii)!)
        out.append(contentsOf: [1, 0])
        let hl = UInt16(header.count)
        out.append(UInt8(hl & 0xFF))
        out.append(UInt8(hl >> 8))
        out.append(header.data(using: .ascii)!)
        out.append(array.raw)
        return out
    }

    private static func shapeText(_ shape: [Int]) -> String {
        switch shape.count {
        case 0: return "()"
        case 1: return "(\(shape[0]),)"
        default: return "(" + shape.map(String.init).joined(separator: ", ") + ")"
        }
    }

    private static func quotedValue(after key: String, in header: String) -> String? {
        guard let keyRange = header.range(of: "'\(key)':") else { return nil }
        let rest = header[keyRange.upperBound...]
        guard let q1 = rest.firstIndex(of: "'") else { return nil }
        let afterQ1 = rest.index(after: q1)
        guard let q2 = rest[afterQ1...].firstIndex(of: "'") else { return nil }
        return String(rest[afterQ1..<q2])
    }
}
