import Foundation
import MLX
import MLXNN
import MLXLMCommon

public class AdapterManager {
    public init() {}

    public func applyRemoteAdapter(from data: Data, to model: inout some Module) throws {
        // 1. Unpack modules from the NPZ / adapter wire payload using your codec
        let modulesMap = try AdapterCodec.unpackModules(data)

        // 2. Flatten into "moduleName.paramName" -> MLXArray. This mirrors what
        //    MLXLMCommon's own LoRAContainer.from(directory:) does when it turns
        //    a loaded adapters.safetensors dict into parameters:
        //        let parameters = ModuleParameters.unflattened(weights)
        //    (mlx-swift-lm: Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift:128)
        var flat: [String: MLXArray] = [:]
        for (moduleName, adapterMod) in modulesMap {
            flat["\(moduleName).lora_a"] = MLXArray(adapterMod.A.data, [adapterMod.A.rows, adapterMod.A.cols])
            flat["\(moduleName).lora_b"] = MLXArray(adapterMod.B.data, [adapterMod.B.rows, adapterMod.B.cols])
            flat["\(moduleName).m"] = MLXArray(adapterMod.m, [adapterMod.m.count])
        }

        // 3. Build the nested parameter tree and apply it. `.noUnusedKeys` makes a
        //    moduleName/key that doesn't land anywhere in `model` throw instead of
        //    silently no-op'ing -- worth keeping since a background adapter fetch
        //    has no other feedback path if a path is wrong.
        let parameterUpdates = ModuleParameters.unflattened(flat)
        try model.update(parameters: parameterUpdates, verify: .noUnusedKeys)
    }
}