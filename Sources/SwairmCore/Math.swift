// Minimal dense math for the client: row-major Matrix, seeded Gaussian RNG,
// and a randomized truncated SVD backed by Accelerate (LAPACK dgesdd).
// Falls back to pure-Swift implementation on platforms without Accelerate.

import Foundation
import Accelerate

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
        // vDSP_mtrans for efficient transpose
        var rows = vDSP_Length(rows)
        var cols = vDSP_Length(cols)
        vDSP_mtrans(data, 1, &out.data, 1, cols, rows)
        return out
    }

    public static func * (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.cols == rhs.rows, "matmul dimension mismatch")
        var out = Matrix(rows: lhs.rows, cols: rhs.cols)
        // cblas_sgemm: C = A * B (row-major)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(lhs.rows), Int32(rhs.cols), Int32(lhs.cols),
            1.0,
            lhs.data, Int32(lhs.cols),
            rhs.data, Int32(rhs.cols),
            0.0,
            &out.data, Int32(rhs.cols)
        )
        return out
    }

    public static func + (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var out = lhs
        vDSP_vadd(rhs.data, 1, lhs.data, 1, &out.data, 1, vDSP_Length(out.data.count))
        return out
    }

    public static func - (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var out = lhs
        vDSP_vsub(rhs.data, 1, lhs.data, 1, &out.data, 1, vDSP_Length(out.data.count))
        return out
    }

    public func scaled(by s: Float) -> Matrix {
        var out = self
        var scale = s
        vDSP_vsmul(data, 1, &scale, &out.data, 1, vDSP_Length(data.count))
        return out
    }

    public var frobeniusNorm: Float {
        var result: Float = 0
        vDSP_svesq(data, 1, &result, vDSP_Length(data.count))
        return result.squareRoot()
    }
}

public func vectorNorm(_ v: [Float]) -> Float {
    var result: Float = 0
    vDSP_svesq(v, 1, &result, vDSP_Length(v.count))
    return result.squareRoot()
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

// ------------------------------------------------------------------ SVD (Accelerate-backed)

public struct SVDResult {
    public let U: Matrix     // rows x rank
    public let S: [Float]    // rank
    public let Vt: Matrix    // rank x cols
}

/// Randomized truncated SVD (Halko-style) with Accelerate LAPACK for the small SVD.
/// Steps:
///   Y = (A A^T)^q A Ω   →   QR(Y) = Q   →   B = Q^T A   →   SVD(B) = Ũ Σ V^T
/// Returns U = Q Ũ, S = Σ, V^T = V^T
public func truncatedSVD(_ D: Matrix, rank: Int, oversample: Int = 2,
                         powerIterations: Int = 4, seed: UInt64 = 42) -> SVDResult {
    let m = D.rows
    let n = D.cols
    let l = min(rank + oversample, min(m, n))
    let r = min(rank, l)

    // 1. Generate random test matrix Ω (n × l)
    var rng = GaussianRNG(seed: seed)
    let omega = randomNormalMatrix(rows: n, cols: l, scale: 1, rng: &rng)

    // 2. Y = D * Ω  (m × l)
    var y = Matrix(rows: m, cols: l)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(m), Int32(l), Int32(n),
                1.0,
                D.data, Int32(n),
                omega.data, Int32(l),
                0.0,
                &y.data, Int32(l))

    // 3. Power iterations: (D D^T)^q D Ω
    var _yT = Matrix(rows: l, cols: m)
    let dT = D.transposed()
    var z = Matrix(rows: n, cols: l)
    for _ in 0..<powerIterations {
        // Z = D^T * Y  (n × l)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(n), Int32(l), Int32(m),
                    1.0,
                    dT.data, Int32(m),
                    y.data, Int32(l),
                    0.0,
                    &z.data, Int32(l))

        // Orthonormalize Z columns (QR via LAPACK geqrf + orgqr)
        z = qrOrthoColumns(z)

        // Y = D * Z  (m × l)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(m), Int32(l), Int32(n),
                    1.0,
                    D.data, Int32(n),
                    z.data, Int32(l),
                    0.0,
                    &y.data, Int32(l))

        // Orthonormalize Y columns
        y = qrOrthoColumns(y)
    }

    // 4. B = Y^T * D  (l × n)
    let yTData = y.transposed()
    var bSmall = Matrix(rows: l, cols: n)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(l), Int32(n), Int32(m),
                1.0,
                yTData.data, Int32(m),
                D.data, Int32(n),
                0.0,
                &bSmall.data, Int32(n))

    // 5. SVD of B (l × n) using LAPACK dgesdd (via Accelerate's SVD)
    // Compute economy SVD: B = U_B Σ V^T, where U_B is l×r, Σ is r, V^T is r×n
    let (uB, s, vt) = svdEconomy(bSmall, rank: r)

    // 6. U = Y * U_B  (m × r)
    var u = Matrix(rows: m, cols: r)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(m), Int32(r), Int32(l),
                1.0,
                y.data, Int32(l),
                uB.data, Int32(r),
                0.0,
                &u.data, Int32(r))

    // Vt is already r × n
    return SVDResult(U: u, S: s, Vt: vt)
}

/// Thin QR factorization via LAPACK: returns Q with orthonormal columns (m × l)
/// Input/output is row-major; transposes to column-major for LAPACK.
func qrOrthoColumns(_ a: Matrix) -> Matrix {
    let m = a.rows
    let n = a.cols
    // Transpose to column-major for LAPACK: (m × n) row-major → (n × m) row-major = m×n col-major
    var aColMajor = a.transposed()  // n rows, m cols in row-major = m rows, n cols in col-major
    var tau = [Float](repeating: 0, count: min(m, n))
    var work = [Float](repeating: 0, count: min(m, n))
    var lwork: Int32 = -1
    var info: Int32 = 0

    // Query optimal workspace size
    // aColMajor is n×m in row-major = m×n in col-major; lda = m for col-major
    aColMajor.data.withUnsafeMutableBufferPointer { aPtr in
        tau.withUnsafeMutableBufferPointer { tauPtr in
            work.withUnsafeMutableBufferPointer { workPtr in
                var m32 = Int32(m)  // col-major rows = m (original rows)
                var n32 = Int32(n)  // col-major cols = n (original cols)
                var lda = Int32(m)  // leading dim for col-major = m (original rows)
                var lwork32 = lwork
                var info32 = info
                sgeqrf_(&m32, &n32, aPtr.baseAddress!, &lda, tauPtr.baseAddress!, workPtr.baseAddress!, &lwork32, &info32)
                lwork = lwork32
                info = info32
            }
        }
    }
    precondition(info == 0, "sgeqrf query failed: \(info)")

    // Guard against negative lwork from query
    if lwork <= 0 { lwork = Int32(max(m, n) * 8) }

    work = [Float](repeating: 0, count: Int(lwork))
    aColMajor.data.withUnsafeMutableBufferPointer { aPtr in
        tau.withUnsafeMutableBufferPointer { tauPtr in
            work.withUnsafeMutableBufferPointer { workPtr in
                var m32 = Int32(m)
                var n32 = Int32(n)
                var lda = Int32(m)  // leading dim for col-major = m (original rows)
                var lwork32 = lwork
                var info32 = info
                sgeqrf_(&m32, &n32, aPtr.baseAddress!, &lda, tauPtr.baseAddress!, workPtr.baseAddress!, &lwork32, &info32)
                info = info32
            }
        }
    }
    precondition(info == 0, "sgeqrf failed: \(info)")

    // Generate Q from QR factors
    // Q in col-major is m×min(m,n); will be min(m,n)×m in row-major
    aColMajor.data.withUnsafeMutableBufferPointer { aPtr in
        tau.withUnsafeMutableBufferPointer { tauPtr in
            work.withUnsafeMutableBufferPointer { workPtr in
                var m32 = Int32(m)           // col-major rows of Q = m
                var n32 = Int32(min(m, n))   // col-major cols of Q = min(m,n)
                var k32 = Int32(min(m, n))
                var lda = Int32(m)           // leading dim for col-major = m
                var lwork32 = lwork
                var info32 = info
                sorgqr_(&m32, &n32, &k32, aPtr.baseAddress!, &lda, tauPtr.baseAddress!, workPtr.baseAddress!, &lwork32, &info32)
                info = info32
            }
        }
    }
    precondition(info == 0, "sorgqr failed: \(info)")

    // Guard against negative lwork from query (use same workspace)
    if lwork <= 0 { lwork = Int32(max(m, n) * 8) }

    // aColMajor now holds Q in col-major (m×min(m,n)); transpose back to row-major (min(m,n)×m)
    return aColMajor.transposed()
}

/// Economy SVD of m×n matrix (m ≤ n typical) returning U (m×r), S (r), Vt (r×n)
/// Input/output is row-major; transposes to column-major for LAPACK.
func svdEconomy(_ a: Matrix, rank: Int) -> (U: Matrix, S: [Float], Vt: Matrix) {
    let m = a.rows
    let n = a.cols
    let k = min(m, n)
    let r = min(rank, k)

    // Transpose A to column-major: A (m×n row-major) → A^T (n×m row-major = m×n col-major)
    var aColMajor = a.transposed()  // n rows, m cols in row-major = m rows, n cols in col-major
    var s = [Float](repeating: 0, count: k)
    // sgesdd on A^T (m×n col-major matrix):
    // U_col = V (m×k col-major = k×m row-major)
    // V_col^T = U^T (k×n col-major = n×k row-major)
    var uPrimeColMajor = Matrix(rows: k, cols: m)  // receives U_col (m×k col-major)
    var vtPrimeColMajor = Matrix(rows: n, cols: k)  // receives V_col^T (k×n col-major)

    // sgesdd: jobz = 'S' (economy size)
    var jobz: Int8 = 83  // 'S'
    let ldu = Int32(m)   // leading dim of U_col output = m (col-major rows)
    let ldvt = Int32(k)  // leading dim of V_col^T output = k
    var lwork: Int32 = -1
    var iwork = [Int32](repeating: 0, count: 8 * k)
    var work = [Float](repeating: 0, count: 1)
    var info: Int32 = 0

    aColMajor.data.withUnsafeMutableBufferPointer { aPtr in
        s.withUnsafeMutableBufferPointer { sPtr in
            uPrimeColMajor.data.withUnsafeMutableBufferPointer { uPtr in
                vtPrimeColMajor.data.withUnsafeMutableBufferPointer { vtPtr in
                    work.withUnsafeMutableBufferPointer { workPtr in
                        iwork.withUnsafeMutableBufferPointer { iworkPtr in
                            var m32 = Int32(m)  // col-major rows = m (original rows)
                            var n32 = Int32(n)  // col-major cols = n (original cols)
                            var lda = Int32(m)  // leading dim for col-major = m (original rows)
                            var ldu32 = ldu
                            var ldvt32 = ldvt
                            var lwork32 = lwork
                            var info32 = info
                            sgesdd_(&jobz, &m32, &n32, aPtr.baseAddress!, &lda, sPtr.baseAddress!, uPtr.baseAddress!, &ldu32, vtPtr.baseAddress!, &ldvt32, workPtr.baseAddress!, &lwork32, iworkPtr.baseAddress!, &info32)
                            lwork = lwork32
                            info = info32
                        }
                    }
                }
            }
        }
    }
    precondition(info == 0, "sgesdd query failed: \(info)")

    // Guard against negative lwork from query
    if lwork <= 0 { lwork = Int32(max(m, n) * 8) }

    work = [Float](repeating: 0, count: Int(lwork))
    aColMajor.data.withUnsafeMutableBufferPointer { aPtr in
        s.withUnsafeMutableBufferPointer { sPtr in
            uPrimeColMajor.data.withUnsafeMutableBufferPointer { uPtr in
                vtPrimeColMajor.data.withUnsafeMutableBufferPointer { vtPtr in
                    work.withUnsafeMutableBufferPointer { workPtr in
                        iwork.withUnsafeMutableBufferPointer { iworkPtr in
                            var m32 = Int32(m)
                            var n32 = Int32(n)
                            var lda = Int32(m)
                            var ldu32 = ldu
                            var ldvt32 = ldvt
                            var lwork32 = lwork
                            var info32 = info
                            sgesdd_(&jobz, &m32, &n32, aPtr.baseAddress!, &lda, sPtr.baseAddress!, uPtr.baseAddress!, &ldu32, vtPtr.baseAddress!, &ldvt32, workPtr.baseAddress!, &lwork32, iworkPtr.baseAddress!, &info32)
                            info = info32
                        }
                    }
                }
            }
        }
    }
    precondition(info == 0, "sgesdd failed: \(info)")

    // uPrimeColMajor (k×m in row-major = U_col = V'^T = U^T) → transpose to m×k = U
    let uRowMajor = uPrimeColMajor.transposed()
    // vtPrimeColMajor (n×k in row-major = V_col^T = U' = V) → transpose to k×n = V^T
    let vtRowMajor = vtPrimeColMajor.transposed()

    // Truncate to rank r
    var uTrunc = Matrix(rows: m, cols: r)
    var vtTrunc = Matrix(rows: r, cols: n)
    var sTrunc = [Float](repeating: 0, count: r)

    for i in 0..<r {
        sTrunc[i] = s[i]
        for row in 0..<m { uTrunc[row, i] = uRowMajor[row, i] }
        for col in 0..<n { vtTrunc[i, col] = vtRowMajor[i, col] }
    }

    return (uTrunc, sTrunc, vtTrunc)
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

// MARK: - LAPACK function declarations (Accelerate)

@_silgen_name("sgeqrf_")
func sgeqrf_(_ m: UnsafeMutablePointer<Int32>, _ n: UnsafeMutablePointer<Int32>, _ a: UnsafeMutablePointer<Float>, _ lda: UnsafeMutablePointer<Int32>,
             _ tau: UnsafeMutablePointer<Float>, _ work: UnsafeMutablePointer<Float>, _ lwork: UnsafeMutablePointer<Int32>, _ info: UnsafeMutablePointer<Int32>)

@_silgen_name("sorgqr_")
func sorgqr_(_ m: UnsafeMutablePointer<Int32>, _ n: UnsafeMutablePointer<Int32>, _ k: UnsafeMutablePointer<Int32>, _ a: UnsafeMutablePointer<Float>, _ lda: UnsafeMutablePointer<Int32>,
             _ tau: UnsafeMutablePointer<Float>, _ work: UnsafeMutablePointer<Float>, _ lwork: UnsafeMutablePointer<Int32>, _ info: UnsafeMutablePointer<Int32>)

@_silgen_name("sgesdd_")
func sgesdd_(_ jobz: UnsafeMutablePointer<Int8>, _ m: UnsafeMutablePointer<Int32>, _ n: UnsafeMutablePointer<Int32>, _ a: UnsafeMutablePointer<Float>, _ lda: UnsafeMutablePointer<Int32>,
             _ s: UnsafeMutablePointer<Float>, _ u: UnsafeMutablePointer<Float>, _ ldu: UnsafeMutablePointer<Int32>, _ vt: UnsafeMutablePointer<Float>, _ ldvt: UnsafeMutablePointer<Int32>,
             _ work: UnsafeMutablePointer<Float>, _ lwork: UnsafeMutablePointer<Int32>, _ iwork: UnsafeMutablePointer<Int32>, _ info: UnsafeMutablePointer<Int32>)
