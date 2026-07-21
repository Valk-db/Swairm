// Swift port of swarm_client.py: fake phone fleet exercising the full
// Anchor loop over HTTP with the linear-proxy trainer. Full-adapter
// FedAvg semantics (DECISIONS.md D7). RNG streams differ from numpy's,
// so per-device targets are not bit-identical to the Python sim; the
// convergence behavior and wire format are the same.

import Foundation
import SwairmCore

let mDim = 128
let nDim = 256
let rank = 4
let moduleName = "layers.0.attn.q_proj"
let clientLR: Float = 0.5
let noiseScale: Float = 0.05

// ------------------------------------------------------------------ args
var anchor = "http://127.0.0.1:8000"
var fleet = 12
var rounds = 10
var interval: Double = 25

let argv = Array(CommandLine.arguments.dropFirst())
var idx = 0
while idx < argv.count {
    let arg = argv[idx]
    let value: String? = idx + 1 < argv.count ? argv[idx + 1] : nil
    switch arg {
    case "--anchor":
        if let v = value { anchor = v; idx += 1 }
    case "--fleet":
        if let v = value, let n = Int(v) { fleet = n; idx += 1 }
    case "--rounds":
        if let v = value, let n = Int(v) { rounds = n; idx += 1 }
    case "--interval":
        if let v = value, let d = Double(v) { interval = d; idx += 1 }
    default:
        print("unknown argument: \(arg)")
        exit(2)
    }
    idx += 1
}

guard let baseURL = URL(string: anchor) else {
    print("invalid --anchor URL: \(anchor)")
    exit(2)
}

// ------------------------------------------------------------------ targets
struct DeviceTarget {
    let dense: Matrix
    let magnitude: [Float]
    var rng: GaussianRNG
}

let rankScale = 1.0 / Float(Double(rank).squareRoot())
let dimScale = 1.0 / Float(Double(nDim).squareRoot())

var sharedRng = GaussianRNG(seed: 42)
let dShared = randomNormalMatrix(rows: mDim, cols: rank, scale: rankScale, rng: &sharedRng)
    * randomNormalMatrix(rows: rank, cols: nDim, scale: dimScale, rng: &sharedRng)
var mShared = [Float]()
for _ in 0..<mDim { mShared.append(sharedRng.uniform(in: 0.5, 2.5)) }

var targets: [String: DeviceTarget] = [:]
for i in 0..<fleet {
    var rng = GaussianRNG(seed: UInt64(1000 + i))
    let perturb = randomNormalMatrix(rows: mDim, cols: rank, scale: rankScale, rng: &rng)
        * randomNormalMatrix(rows: rank, cols: nDim, scale: dimScale, rng: &rng)
    let dK = dShared + perturb.scaled(by: 0.3)
    var mK = [Float]()
    for j in 0..<mDim {
        mK.append(min(3.0, max(0.1, mShared[j] + 0.2 * rng.normal())))
    }
    targets["dev\(i)"] = DeviceTarget(dense: dK, magnitude: mK, rng: rng)
}

var fleetDir = Matrix(rows: mDim, cols: nDim)
var fleetM = [Float](repeating: 0, count: mDim)
for t in targets.values {
    fleetDir = fleetDir + t.dense
    for j in 0..<mDim { fleetM[j] += t.magnitude[j] }
}
fleetDir = fleetDir.scaled(by: 1.0 / Float(fleet))
for j in 0..<mDim { fleetM[j] /= Float(fleet) }
let fleetNorm = fleetDir.frobeniusNorm

// ------------------------------------------------------------------ rounds
let client = AnchorClient(base: baseURL)

do {
    for rnd in 0..<rounds {
        let status = try client.status()

        var version = 0
        var gDir = Matrix(rows: mDim, cols: nDim)
        var gM = [Float](repeating: 1, count: mDim)
        if let adapter = try client.latestAdapter(),
           let mod = adapter.modules[moduleName] {
            version = adapter.version
            gDir = mod.B * mod.A
            gM = mod.m
        }

        let err = (gDir - fleetDir).frobeniusNorm / fleetNorm
        print("[round \(rnd)] anchor v\(status.version) "
            + "(epoch \(status.curriculum_epoch), skew=\(status.skew_detected)) | "
            + "global-vs-fleet-target err=" + String(format: "%.4f", err))

        for deviceID in targets.keys.sorted() {
            var target = targets[deviceID]!
            let noise = randomNormalMatrix(rows: mDim, cols: nDim,
                                           scale: noiseScale * dimScale,
                                           rng: &target.rng)
            let newDir = gDir + (target.dense - gDir).scaled(by: clientLR) + noise
            let (a, b) = factorToRank(newDir, rank: rank)
            var mNew = [Float](repeating: 0, count: mDim)
            for j in 0..<mDim {
                mNew[j] = gM[j] + clientLR * (target.magnitude[j] - gM[j])
            }
            let raw = try AdapterCodec.packUpload(
                deviceID: deviceID, fetchVersion: version,
                curriculumEpoch: status.curriculum_epoch,
                modules: [moduleName: AdapterModule(A: a, B: b, m: mNew)])
            try client.upload(raw)
            targets[deviceID] = target      // persist advanced RNG state
        }
        print("          uploaded \(fleet) adapters; waiting "
            + "\(Int(interval))s for the worker...")
        Thread.sleep(forTimeInterval: interval)
    }

    if let final = try client.latestAdapter(),
       let mod = final.modules[moduleName] {
        let gDir = mod.B * mod.A
        let err = (gDir - fleetDir).frobeniusNorm / fleetNorm
        var mDiff = [Float](repeating: 0, count: mDim)
        for j in 0..<mDim { mDiff[j] = mod.m[j] - fleetM[j] }
        let mErr = vectorNorm(mDiff) / vectorNorm(fleetM)
        print("\nFinal: dir err=" + String(format: "%.4f", err)
            + ", magnitude err=" + String(format: "%.4f", mErr)
            + " (both should shrink toward a noise floor across rounds)")
    }
} catch {
    print("swairm-client failed: \(error)")
    exit(1)
}
