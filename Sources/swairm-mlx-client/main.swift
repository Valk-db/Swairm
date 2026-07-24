// Real on-device MLX DoRA fleet driver: the CI oracle for the active
// frontier. Runs a small fleet of MLXDeviceLoop instances (real fetch ->
// prepare -> train -> export -> upload) against a live Anchor, proving the
// MLX training path -- not just the linear proxy -- round-trips the wire
// format. Unlike swift test, `swift run` provides the Metal backend MLX
// needs (XCTest cannot load the default metallib on the runner).
//
// Requires a local MLX model dir (config.json + *.safetensors + tokenizer)
// and a curriculum dir of token_ids/labels .npz shards -- both prepared by
// the CI job, not committed.

import Foundation
import SwairmCore

// ------------------------------------------------------------------ args
var anchor = "http://127.0.0.1:8000"
var modelPath = "models/mlx-model"
var curriculumDirectory = "curriculum"
var fleet = 2
var rounds = 1
var maxSteps = 4
var batchSize = 1
var sequenceLength = 64

let argv = Array(CommandLine.arguments.dropFirst())
var idx = 0
while idx < argv.count {
    let arg = argv[idx]
    let value: String? = idx + 1 < argv.count ? argv[idx + 1] : nil
    switch arg {
    case "--anchor":
        if let v = value { anchor = v; idx += 1 }
    case "--model":
        if let v = value { modelPath = v; idx += 1 }
    case "--curriculum":
        if let v = value { curriculumDirectory = v; idx += 1 }
    case "--fleet":
        if let v = value, let n = Int(v) { fleet = n; idx += 1 }
    case "--rounds":
        if let v = value, let n = Int(v) { rounds = n; idx += 1 }
    case "--max-steps":
        if let v = value, let n = Int(v) { maxSteps = n; idx += 1 }
    case "--batch-size":
        if let v = value, let n = Int(v) { batchSize = n; idx += 1 }
    case "--seq-len":
        if let v = value, let n = Int(v) { sequenceLength = n; idx += 1 }
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
let client = AnchorClient(base: baseURL)

func makeConfig(deviceIndex: Int) -> MLXLoopConfig {
    MLXLoopConfig(
        modelPath: modelPath,
        // Bare projection names: matched by suffix/substring against the
        // model's qualified module keys inside LoRAContainer. rankMap/alphaMap
        // resolve uniformly (see resolvedRankScale) -- the container applies
        // one rank/scale; D6's per-module split stays Anchor-side.
        targetModules: ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        rankMap: ["attn": 4, "mlp": 6],
        alphaMap: ["attn": 16.0, "mlp": 16.0],
        learningRate: 1e-4,
        maxStepsPerRound: maxSteps,
        batchSize: batchSize,
        sequenceLength: sequenceLength,
        curriculumDirectory: curriculumDirectory,
        seed: 42 + UInt64(deviceIndex)
    )
}

print("swairm-mlx-client: fleet=\(fleet) rounds=\(rounds) "
    + "maxSteps=\(maxSteps) batch=\(batchSize) seq=\(sequenceLength)")
print("  model: \(modelPath)")
print("  curriculum: \(curriculumDirectory)")
print("  anchor: \(anchor)")

// One persistent loop per device (actor owns its trainer state across rounds).
var loops: [MLXDeviceLoop] = []
do {
    for i in 0..<fleet {
        loops.append(try MLXDeviceLoop(
            anchor: client, deviceID: "mlxdev\(i)", deviceIndex: i,
            config: makeConfig(deviceIndex: i)))
    }
} catch {
    print("failed to create MLX loops: \(error)")
    exit(1)
}

// ------------------------------------------------------------------ rounds
do {
    let v0 = try await client.status().version
    print("[start] anchor version = \(v0)")

    let budget = ResourceBudget(
        maxSteps: maxSteps, maxWallClock: 600,
        stopOnSeriousThermalState: false)

    for rnd in 0..<rounds {
        for (i, loop) in loops.enumerated() {
            let result = try await loop.runRound(budget: budget)
            let loss = result.trainingReport.finalLoss ?? -1
            print(String(
                format: "[round %d dev %d] fetched v%d | steps %d | loss %.4f | %@",
                rnd, i, result.fetchedVersion,
                result.trainingReport.stepsCompleted, loss,
                String(describing: result.trainingReport.termination)))
        }
    }

    // The worker aggregates on its own interval; give it a moment, then
    // confirm the version advanced past where we started.
    try await Task.sleep(nanoseconds: 6_000_000_000)
    let vEnd = try await client.status().version
    print("[end] anchor version = \(vEnd) (started at \(v0))")
    if vEnd <= v0 {
        print("ERROR: anchor version did not advance -- round(s) not aggregated")
        exit(1)
    }
    print("OK: real MLX fleet round-trip advanced the Anchor to v\(vEnd)")
} catch {
    print("swairm-mlx-client failed: \(error)")
    exit(1)
}
