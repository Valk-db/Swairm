// DoRA (Weight-Decomposed Low-Rank Adaptation) layer for MLX.
// Wraps a frozen base Linear layer with trainable LoRA A, B and magnitude m.
// Forward: m * (W + scaling * B @ A) / ||W + scaling * B @ A||

import Foundation
import MLX
import MLXNN

/// DoRA-wrapped linear layer. Base weight is frozen; only LoRA params train.
public final class DoRALinear: Module, @unchecked Sendable {
    // Frozen base parameters
    public let baseWeight: MLXArray      // [out_features, in_features]
    public let baseBias: MLXArray?       // [out_features]

    // Trainable DoRA parameters
    public var loraA: MLXArray           // [rank, in_features]
    public var loraB: MLXArray           // [out_features, rank]
    public var magnitude: MLXArray       // [out_features]

    public let rank: Int
    public let scaling: Float            // alpha / rank

    /// Initialize from a frozen base Linear layer.
    public init(base: Linear, rank: Int, alpha: Float) {
        self.baseWeight = base.weight
        self.baseBias = base.bias
        self.rank = rank
        self.scaling = alpha / Float(rank)

        let outFeatures = baseWeight.shape[0]
        let inFeatures = baseWeight.shape[1]

        // LoRA A: Kaiming uniform init
        let bound = sqrt(5.0 / Float(inFeatures))
        self.loraA = MLX.randomUniform(-bound, bound, [rank, inFeatures])

        // LoRA B: zeros
        self.loraB = MLXArray.zeros([outFeatures, rank])

        // Magnitude: L2 norm of base weight rows
        self.magnitude = sqrt(sum(baseWeight * baseWeight, axis: 1))

        super.init()
    }

    /// Forward pass: DoRA reparameterization.
    /// weight = magnitude * (baseWeight + scaling * loraB @ loraA) / ||baseWeight + scaling * loraB @ loraA||
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Delta weight: scaling * B @ A  [out_features, in_features]
        let deltaW = matmul(loraB, loraA) * scaling

        // Combined weight
        let combined = baseWeight + deltaW

        // Row-wise L2 norm
        let norms = sqrt(sum(combined * combined, axis: 1))

        // Direction vectors (normalized rows)
        let direction = combined / norms.reshaped(-1, 1)

        // Scale by learned magnitude
        let weight = direction * magnitude.reshaped(-1, 1)

        // Linear: x @ weight.T + bias
        var out = matmul(x, weight.transposed(0, 1))
        if let bias = baseBias {
            out = out + bias
        }
        return out
    }

    /// Export adapter parameters for wire format (AdapterModule).
    public func exportAdapter() -> (A: MLXArray, B: MLXArray, m: MLXArray) {
        return (loraA, loraB, magnitude)
    }

    /// Load adapter parameters from wire format.
    public func loadAdapter(A: MLXArray, B: MLXArray, m: MLXArray) {
        self.loraA = A
        self.loraB = B
        self.magnitude = m
    }

    /// Trainable parameters for optimizer collection.
    public var trainableParameters: [String: MLXArray] {
        return [
            "loraA": loraA,
            "loraB": loraB,
            "magnitude": magnitude
        ]
    }
}

// ============================================================================
// MARK: - DoRA Injection
// ============================================================================

/// Inject DoRALinear layers into a model, replacing target Linear modules.
/// Returns (model, [moduleName: DoRALinear]) for optimizer collection.
public func injectDoRA(
    into model: Module,
    targetModules: [String],
    rankMap: [String: Int],
    alphaMap: [String: Float]
) -> (Module, [String: DoRALinear]) {
    var doraLayers: [String: DoRALinear] = [:]

    // Collect all matching Linear modules first (avoids nested closure mutation)
    var matches: [(module: Module, name: String, fullPath: String, pattern: String)] = []

    func collect(_ module: Module, path: String = "") {
        let children = module.children
        for (name, child) in children {
            let fullPath = path.isEmpty ? name : "\(path).\(name)"
            if let linear = child as? Linear {
                for pattern in targetModules {
                    if fullPath.contains(pattern) {
                        matches.append((module, name, fullPath, pattern))
                        break
                    }
                }
            } else {
                collect(child, path: fullPath)
            }
        }
    }

    collect(model)

    // Replace collected matches
    for (parent, name, fullPath, pattern) in matches {
        let children = parent.children
        guard let linear = children[name] as? Linear else { continue }
        let rank = rankMap[pattern] ?? 4
        let alpha = alphaMap[pattern] ?? 16.0
        let dora = DoRALinear(base: linear, rank: rank, alpha: alpha)
        doraLayers[fullPath] = dora
        // Update module by replacing the child
        var newChildren = children
        newChildren[name] = dora
        _ = parent.update(modules: newChildren)
    }

    return (model, doraLayers)
}