// Minimal dense math for the client: row-major Matrix, seeded Gaussian RNG,
// and a randomized truncated SVD (power iteration + Gram-Schmidt QR +
// Jacobi eigendecomposition of the small Gram matrix). Mirrors what
// sklearn's randomized_svd provides for train_step's rank refactoring.

import Foundation

public struct Matrix: Equatable, Sendable {
    public let rows: Int
    public let cols: Int
    public var data: [Float]

    public init(rows: Int, cols: Int, repeating value: Float = 0) {
        self.rows = rows
        self.cols = cols
        self.data = [Float](repeating: value, count: rows * cols)
    }

    public init(rows: Int, cols: Int, data: [Float]) {
        precondition(data.count == rows * cols, "data count != rows*cols")
        self.rows = rows
        self.cols = cols
        self.data = data
    }

    public subscript(_ i: Int, _ j: Int) -> Float {
        get { data[i * cols + j] }
        set { data[i * cols + j] = newValue }
    }

    public static func identity(_ n: Int) -> Matrix {
        var m = Matrix(rows: n, cols: n)
        for i in 0..<n { m[i, i] = 1 }
        return m
    }

    public func transposed() -> Matrix {
        var out = Matrix(rows: cols, cols: rows)
        for i in 0..<rows {
            for j in 0..<cols {
                out.data[j * rows + i] = data[i * cols + j]
            }
        }
        return out
    }

    public static func * (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.cols == rhs.rows, "matmul dimension mismatch")
        var out = Matrix(rows: lhs.rows, cols: rhs.cols)
        for i in 0..<lhs.rows {
            let lhsBase = i * lhs.cols
            let outBase = i * rhs.cols
            for k in 0..<lhs.cols {
                let v = lhs.data[lhsBase + k]
                if v == 0 { continue }
                let rhsBase = k * rhs.cols
                for j in 0..<rhs.cols {
                    out.data[outBase + j] += v * rhs.data[rhsBase + j]
                }
            }
        }
        return out
    }

    public static func + (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var out = lhs
        for i in 0..<out.data.count { out.data[i] += rhs.data[i] }
        return out
    }

    public static func - (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var out = lhs
        for i in 0..<out.data.count { out.data[i] -= rhs.data[i] }
        return out
    }

    public func scaled(by s: Float) -> Matrix {
        var out = self
        for i in 0..<out.data.count { out.data[i] *= s }
        return out
    }

    public var frobeniusNorm: Float {
        var sum: Float = 0
        for v in data { sum += v * v }
        return sum.squareRoot()
    }
}

public func vectorNorm(_ v: [Float]) -> Float {
    var sum: Float = 0
    for x in v { sum += x * x }
    return sum.squareRoot()
}

// ------------------------------------------------------------------ RNG

public struct GaussianRNG {
    private var state: UInt64
    private var cache: Float?

    public init(seed: UInt64) {
        state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    private mutating func nextUInt64() -> UInt64 {   // SplitMix64
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func uniform() -> Float {
        Float(Double(nextUInt64() >> 11) * (1.0 / 9007199254740992.0))
    }

    public mutating func uniform(in lo: Float, _ hi: Float) -> Float {
        lo + (hi - lo) * uniform()
    }

    /// Standard normal via Box-Muller (with caching of the second value).
    public mutating func normal() -> Float {
        if let c = cache {
            cache = nil
            return c
        }
        var u1 = Double(uniform())
        if u1 < 1e-12 { u1 = 1e-12 }
        let u2 = Double(uniform())
        let r = (-2.0 * log(u1)).squareRoot()
        let theta = 2.0 * Double.pi * u2
        cache = Float(r * sin(theta))
        return Float(r * cos(theta))
    }
}

public func randomNormalMatrix(rows: Int, cols: Int, scale: Float,
                               rng: inout GaussianRNG) -> Matrix {
    var m = Matrix(rows: rows, cols: cols)
    for i in 0..<m.data.count { m.data[i] = scale * rng.normal() }
    return m
}

// ------------------------------------------------------------------ SVD

public struct SVDResult {
    public let U: Matrix     // rows x rank
    public let S: [Float]    // rank
    public let Vt: Matrix    // rank x cols
}

/// Modified Gram-Schmidt orthonormalization of the columns of `m`, in place.
/// Projections run twice ("twice is enough") to keep Q orthogonal in float32,
/// and columns whose residual collapses relative to their original norm are
/// zeroed instead of normalizing rounding noise into a fake direction.
func orthonormalizeColumns(_ m: inout Matrix) {
    for j in 0..<m.cols {
        var originalNorm: Float = 0
        for r in 0..<m.rows { originalNorm += m[r, j] * m[r, j] }
        originalNorm = originalNorm.squareRoot()

        for _ in 0..<2 {
            for i in 0..<j {
                var dot: Float = 0
                for r in 0..<m.rows { dot += m[r, i] * m[r, j] }
                for r in 0..<m.rows { m[r, j] -= dot * m[r, i] }
            }
        }

        var norm: Float = 0
        for r in 0..<m.rows { norm += m[r, j] * m[r, j] }
        norm = norm.squareRoot()

        if norm > max(1e-6 * originalNorm, 1e-12) {
            for r in 0..<m.rows { m[r, j] /= norm }
        } else {
            for r in 0..<m.rows { m[r, j] = 0 }   // dependent column: drop it
        }
    }
}


/// Jacobi eigendecomposition of a small symmetric matrix.
/// Returns eigenvalues (descending) and eigenvectors as matching columns.
func jacobiEigen(_ input: Matrix, maxSweeps: Int = 50) -> (values: [Float], vectors: Matrix) {
    precondition(input.rows == input.cols)
    let n = input.rows
    var a = input
    var v = Matrix.identity(n)
    for _ in 0..<maxSweeps {
        var off: Float = 0
        for p in 0..<n {
            for q in (p + 1)..<n { off += a[p, q] * a[p, q] }
        }
        if off < 1e-18 { break }
        for p in 0..<n {
            for q in (p + 1)..<n {
                let apq = a[p, q]
                if abs(apq) < 1e-12 { continue }
                let theta = 0.5 * atan2(2.0 * Double(apq),
                                        Double(a[q, q]) - Double(a[p, p]))
                let c = Float(cos(theta))
                let s = Float(sin(theta))
                for k in 0..<n {
                    let akp = a[k, p]
                    let akq = a[k, q]
                    a[k, p] = c * akp - s * akq
                    a[k, q] = s * akp + c * akq
                }
                for k in 0..<n {
                    let apk = a[p, k]
                    let aqk = a[q, k]
                    a[p, k] = c * apk - s * aqk
                    a[q, k] = s * apk + c * aqk
                }
                for k in 0..<n {
                    let vkp = v[k, p]
                    let vkq = v[k, q]
                    v[k, p] = c * vkp - s * vkq
                    v[k, q] = s * vkp + c * vkq
                }
            }
        }
    }
    var order = Array(0..<n)
    order.sort { a[$0, $0] > a[$1, $1] }
    let values = order.map { a[$0, $0] }
    var vectors = Matrix(rows: n, cols: n)
    for (newCol, oldCol) in order.enumerated() {
        for r in 0..<n { vectors[r, newCol] = v[r, oldCol] }
    }
    return (values, vectors)
}

/// Randomized truncated SVD (Halko-style): range finding with power
/// iterations, then exact SVD of the small projected matrix via the
/// eigendecomposition of its Gram matrix.
public func truncatedSVD(_ D: Matrix, rank: Int, oversample: Int = 2,
                         powerIterations: Int = 4, seed: UInt64 = 42) -> SVDResult {
    let l = min(rank + oversample, min(D.rows, D.cols))
    var rng = GaussianRNG(seed: seed)
    let omega = randomNormalMatrix(rows: D.cols, cols: l, scale: 1, rng: &rng)
    var y = D * omega                       // rows x l
    orthonormalizeColumns(&y)
    let dt = D.transposed()
    for _ in 0..<powerIterations {
        var z = dt * y                      // cols x l
        orthonormalizeColumns(&z)
        y = D * z
        orthonormalizeColumns(&y)
    }
    let bSmall = y.transposed() * D         // l x cols
    let gram = bSmall * bSmall.transposed() // l x l
    let (vals, vecs) = jacobiEigen(gram)

    let r = min(rank, l)
    var s = [Float](repeating: 0, count: r)
    var u = Matrix(rows: D.rows, cols: r)
    var vt = Matrix(rows: r, cols: D.cols)
    for i in 0..<r {
        let sv = vals[i] > 0 ? vals[i].squareRoot() : 0
        s[i] = sv
        for row in 0..<D.rows {
            var acc: Float = 0
            for k in 0..<l { acc += y[row, k] * vecs[k, i] }
            u[row, i] = acc
        }
        if sv > 0 {
            for col in 0..<D.cols {
                var acc: Float = 0
                for k in 0..<l { acc += vecs[k, i] * bSmall[k, col] }
                vt[i, col] = acc / sv
            }
        }
    }
    return SVDResult(U: u, S: s, Vt: vt)
}

/// Refactor a dense update to rank-r factors, matching the Python client:
/// A = diag(sqrt(S)) @ Vt, B = U @ diag(sqrt(S)).
public func factorToRank(_ dense: Matrix, rank: Int) -> (A: Matrix, B: Matrix) {
    let svd = truncatedSVD(dense, rank: rank)
    var a = svd.Vt
    var b = svd.U
    for i in 0..<svd.S.count {
        let f = max(svd.S[i], 0).squareRoot()
        for j in 0..<a.cols { a[i, j] *= f }
        for r in 0..<b.rows { b[r, i] *= f }
    }
    return (a, b)
}
