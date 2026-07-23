import Foundation
import MLX
import MLXNN
import MLXLMCommon

public class AdapterManager {
    public init() {}

    public func applyRemoteAdapter(from data: Data, to model: inout some Module) throws {
        // 1. Unpack modules from the NPZ / adapter wire payload using your codec
        let modulesMap = try AdapterCodec.unpackModules(data)
        
        // 2. We use a regular Swift dictionary to hold the top-level branches
        var rootDict: [String: NestedDictionary<String, MLXArray>] = [:]
        
        for (moduleName, adapterMod) in modulesMap {
            // Convert Swift Data / Floats into MLXArrays with precise shapes
            let wA = MLXArray(adapterMod.A.data, [adapterMod.A.rows, adapterMod.A.cols])
            let wB = MLXArray(adapterMod.B.data, [adapterMod.B.rows, adapterMod.B.cols])
            let magnitude = MLXArray(adapterMod.m, [adapterMod.m.count])
            
            // Wrap parameters into an inner NestedDictionary using its standard dictionary initializer
            let innerDict = NestedDictionary<String, MLXArray>([
                "lora_a": wA,
                "lora_b": wB,
                "m": magnitude
            ])
            
            // Assign the branch to the root tree
            rootDict[moduleName] = innerDict
        }
        
        // 3. Initialize the final ModuleParameters object from our nested tree
        let parameterUpdates = ModuleParameters(rootDict)
        
        // 4. Apply updates safely through MLX's reflection-safe update engine
        model.update(parameters: parameterUpdates)
    }
}