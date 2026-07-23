// Swift port of swarm_client.py: fake phone fleet exercising the full
// Anchor loop over HTTP with the linear-proxy trainer. Full-adapter
// FedAvg semantics (DECISIONS.md D7). RNG streams differ from numpy's,
// so per-device targets are not bit-identical to the Python sim; the
// convergence behavior and wire format are the same.
//
// Thin wrapper around SwairmCore.ProxyDeviceLoop — the CLI fleet, the CI
// integration job, and the iOS app all run the same round loop.

import Foundation
import SwairmCore

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

// ------------------------------------------------------------------ fleet
let config = ProxyLoopConfig.standard
let client = AnchorClient(base: baseURL)

// One persistent loop per device so its trainer's noise RNG advances
// across rounds, like the old inline loop's per-device rng did.
var loops: [ProxyDeviceLoop] = []
for i in 0..<fleet {
    loops.append(ProxyDeviceLoop(anchor: client, deviceID: "dev\(i)",
                                 deviceIndex: i, config: config))
}

// Fleet-average target, for the global-vs-fleet convergence metric.
var fleetDir = Matrix(rows: config.rows, cols: config.cols)
var fleetM = [Float](repeating: 0, count: config.rows)
for loop in loops {
    fleetDir = fleetDir + loop.target.dense
    for j in 0..<config.rows { fleetM[j] += loop.target.magnitude[j] }
}
fleetDir = fleetDir.scaled(by: 1.0 / Float(fleet))
for j in 0..<config.rows { fleetM[j] /= Float(fleet) }
let fleetNorm = fleetDir.frobeniusNorm

// ------------------------------------------------------------------ rounds
do {
    let budget = ResourceBudget(maxSteps: 1, maxWallClock: 60,
                                stopOnSeriousThermalState: false)
    for rnd in 0..<rounds {
        // Round header: global adapter vs. fleet-average target.
        let status = try await client.status()
        let globalAdapter = try await client.latestAdapter()
        var gDir = Matrix(rows: config.rows, cols: config.cols)
        if let mod = globalAdapter?.modules[config.moduleName] {
            gDir = mod.B * mod.A
        }
        let err = (gDir - fleetDir).frobeniusNorm / fleetNorm
        print("[round \(rnd)] anchor v\(status.version) "
            + "(epoch \(status.curriculum_epoch), skew=\(status.skew_detected)) | "
            + "global-vs-fleet-target err=" + String(format: "%.4f", err))

        // Each device: fetch -> prepare -> train -> export -> upload,
        // the same loop a real phone runs behind the LocalTraining seam.
        for loop in loops {
            _ = try await loop.runRound(budget: budget)
        }
        print("          uploaded \(fleet) adapters; waiting "
            + "\(Int(interval))s for the worker...")
        try await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
    }

    if let final = try await client.latestAdapter(),
       let mod = final.modules[config.moduleName] {
        let gDir = mod.B * mod.A
        let err = (gDir - fleetDir).frobeniusNorm / fleetNorm
        var mDiff = [Float](repeating: 0, count: config.rows)
        for j in 0..<config.rows { mDiff[j] = mod.m[j] - fleetM[j] }
        let mErr = vectorNorm(mDiff) / vectorNorm(fleetM)
        print("\nFinal: dir err=" + String(format: "%.4f", err)
            + ", magnitude err=" + String(format: "%.4f", mErr)
            + " (both should shrink toward a noise floor across rounds)")
    }
} catch {
    print("swairm-client failed: \(error)")
    exit(1)
}
