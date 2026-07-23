import Foundation
import MLX
import MLXNN
import MLXLMCommon

public class AdapterManager {
    public init() {}

    public func applyRemoteAdapter(from data: Data, to model: inout some Module) throws {
        // 1. Unpack modules from the NPZ / adapter wire payload using your codec
        let modulesMap = try AdapterCodec.unpackModules(data)
        
        // 2. Build the explicit NestedDictionary structure required by MLX
        var rootDict: [String: NestedDictionary<String, MLXArray>] = [:]
        
        for (moduleName, adapterMod) in modulesMap {
            // Convert Swift Data / Floats into MLXArrays with precise shapes
            let wA = MLXArray(adapterMod.A.data, [adapterMod.A.rows, adapterMod.A.cols])
            let wB = MLXArray(adapterMod.B.data, [adapterMod.B.rows, adapterMod.B.cols])
            let magnitude = MLXArray(adapterMod.m, [adapterMod.m.count])
            
            // Wrap parameters into leaves within the nested dictionary tree
            let innerDict: [String: NestedDictionary<String, MLXArray>] = [
                "lora_a": .value(wA),
                "lora_b": .value(wB),
                "m": .value(magnitude)
            ]
            
            rootDict[moduleName] = .dictionary(innerDict)
        }
        
        let parameterUpdates: ModuleParameters = .dictionary(rootDict)
        
        // 3. Apply updates safely through MLX's reflection-safe update engine
        model.update(parameters: parameterUpdates)
    }
}