import Foundation
import MLX
import MLXNN
import MLXLMCommon
import SwairmCore

public class AdapterManager {
    public init() {}

    public func applyRemoteAdapter(from data: Data, to model: inout some Module) throws {
        // 1. Unpack modules from the NPZ / adapter wire payload using your codec
        let modulesMap = try AdapterCodec.unpackModules(data)
        
        // 2. Build the nested parameter dictionary expected by MLX Module updates
        var parameterUpdates: [String: Any] = [:]
        
        for (moduleName, adapterMod) in modulesMap {
            // Convert Swift Data / Floats into MLXArrays with precise shapes
            let wA = MLXArray(adapterMod.A.data, [adapterMod.A.rows, adapterMod.A.cols])
            let wB = MLXArray(adapterMod.B.data, [adapterMod.B.rows, adapterMod.B.cols])
            let magnitude = MLXArray(adapterMod.m, [adapterMod.m.count])
            
            // Map keys to match the internal @ParameterInfo keys ("lora_a", "lora_b", "m")
            parameterUpdates[moduleName] = [
                "lora_a": wA,
                "lora_b": wB,
                "m": magnitude
            ]
        }
        
        // 3. Apply updates safely through MLX's reflection-safe update engine
        model.update(parameters: parameterUpdates)
    }
}