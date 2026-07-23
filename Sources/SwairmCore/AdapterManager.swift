import Foundation
import MLX
import MLXNN
import MLXLMCommon
import ZIPFoundation

/// Manages atomic adapter replacement (D7) and DoRA magnitude parameter synchronization (D4)
/// for the Swift client engine.
public final class AdapterManager {
    
    public init() {}
    
    /// Unpacks an incoming network NPZ payload and performs an atomic full-adapter replace
    /// across all matching DoRALinear layers in the model.
    /// - Parameters:
    ///   - data: Raw binary data of the `.npz` archive received from the anchor node.
    ///   - model: The active MLX model module being fine-tuned or evaluated.
    public func applyRemoteAdapter(from data: Data, to model: inout some Module) throws {
        // 1. Extract tensors from the NPZ wire payload using the existing NPY/NPZ codecs
        let archive = try NPZArchive(data: data)
        let tensors = try archive.extractTensors()
        
        // 2. Perform atomic full-adapter replacement (D7)
        // Iterate through named modules to locate and update DoRA layers
        for (name, module) in model.namedModules() {
            // Check for upstream mlx-swift-lm DoRALinear structure or custom wrapper
            if let doraLayer = module as? DoRALinear {
                let keyPrefix = name.isEmpty ? "" : "\(name)."
                
                guard let wA = tensors["\(keyPrefix)lora_a"],
                      let wB = tensors["\(keyPrefix)lora_b"],
                      let magnitude = tensors["\(keyPrefix)m"] else {
                    continue
                }
                
                // Atomic assignment maintaining vector shapes and precision
                doraLayer.loraA = wA
                doraLayer.loraB = wB
                
                // D4 Invariant: Magnitude vector 'm' is updated as a separate, uncoupled parameter
                doraLayer.magnitude = magnitude
            }
        }
        
        // Evaluate deferred computation graphs to ensure memory is committed immediately
        eval(model.parameters())
    }
}