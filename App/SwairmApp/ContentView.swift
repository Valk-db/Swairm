import SwiftUI
import SwairmCore

struct ContentView: View {
    @State private var useMLXTrainer = false
    @State private var proxyController = DeviceLoopController()
    @State private var mlxController = MLXDeviceLoopController()

    var isRunning: Bool {
        useMLXTrainer ? mlxController.isRunning : proxyController.isRunning
    }

    var anchorVersion: Int? {
        useMLXTrainer ? mlxController.anchorVersion : proxyController.anchorVersion
    }

    var lastLoss: Float? {
        useMLXTrainer ? mlxController.lastLoss : proxyController.lastDirError
    }

    var log: [LogEntry] {
        useMLXTrainer ? mlxController.log : proxyController.log
    }

    var deviceID: String {
        useMLXTrainer ? mlxController.deviceID : proxyController.deviceID
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Mode") {
                    Toggle("Use MLX Trainer (Real DoRA)", isOn: $useMLXTrainer)
                        .onChange(of: useMLXTrainer) { _, _ in
                            if isRunning { stop() }
                        }
                }

                Section("Anchor") {
                    if useMLXTrainer {
                        TextField("http://host:8000", text: $mlxController.settings.anchorURLText)
                            .keyboardType(.URL)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .disabled(isRunning)
                        Stepper("Device index: \(mlxController.settings.deviceIndex)",
                                value: $mlxController.settings.deviceIndex, in: 0...63)
                            .disabled(isRunning)
                        Stepper("Interval: \(Int(mlxController.settings.intervalSeconds))s",
                                value: $mlxController.settings.intervalSeconds,
                                in: 5...300, step: 5)
                            .disabled(isRunning)
                    } else {
                        TextField("http://host:8000", text: $proxyController.settings.anchorURLText)
                            .keyboardType(.URL)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .disabled(isRunning)
                        Stepper("Device index: \(proxyController.settings.deviceIndex)",
                                value: $proxyController.settings.deviceIndex, in: 0...63)
                            .disabled(isRunning)
                        Stepper("Interval: \(Int(proxyController.settings.intervalSeconds))s",
                                value: $proxyController.settings.intervalSeconds,
                                in: 5...300, step: 5)
                            .disabled(isRunning)
                    }
                }

                if useMLXTrainer {
                    Section("MLX Model") {
                        TextField("Model path", text: $mlxController.settings.modelPath)
                            .disabled(isRunning)
                        TextField("Curriculum dir", text: $mlxController.settings.curriculumDirectory)
                            .disabled(isRunning)
                        Stepper("Steps/round: \(mlxController.settings.maxStepsPerRound)",
                                value: $mlxController.settings.maxStepsPerRound, in: 1...500, step: 1)
                            .disabled(isRunning)
                        Stepper("Batch size: \(mlxController.settings.batchSize)",
                                value: $mlxController.settings.batchSize, in: 1...8)
                            .disabled(isRunning)
                        Stepper("Seq length: \(mlxController.settings.sequenceLength)",
                                value: $mlxController.settings.sequenceLength, in: 32...512, step: 32)
                            .disabled(isRunning)
                        Stepper("Learning rate: \(String(format: "%.0e", mlxController.settings.learningRate))",
                                value: $mlxController.settings.learningRate, in: 1e-5...1e-3, step: 1e-5)
                            .disabled(isRunning)
                        Stepper("Weight decay: \(String(format: "%.4f", mlxController.settings.weightDecay))",
                                value: $mlxController.settings.weightDecay, in: 0...0.1, step: 0.01)
                            .disabled(isRunning)
                        Stepper("Max grad norm: \(String(format: "%.2f", mlxController.settings.maxGradNorm))",
                                value: $mlxController.settings.maxGradNorm, in: 0.1...5.0, step: 0.1)
                            .disabled(isRunning)
                        Stepper("Warmup steps: \(mlxController.settings.warmupSteps)",
                                value: $mlxController.settings.warmupSteps, in: 0...100, step: 1)
                            .disabled(isRunning)
                    }

                    Section("MLX Advanced") {
                        TextField("Target modules (comma-separated)", text: $mlxController.settings.targetModules)
                            .disabled(isRunning)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                        Stepper("LoRA rank: \(mlxController.settings.loraRank)",
                                value: $mlxController.settings.loraRank, in: 1...16, step: 1)
                            .disabled(isRunning)
                        Stepper("LoRA alpha: \(String(format: "%.1f", mlxController.settings.loraAlpha))",
                                value: $mlxController.settings.loraAlpha, in: 1...64, step: 1)
                            .disabled(isRunning)
                        Stepper("Seed: \(mlxController.settings.seed)",
                                value: $mlxController.settings.seed, in: 0...UInt32.max, step: 1)
                            .disabled(isRunning)
                    }

                    Section("Base Model") {
                        TextField("Base model name", text: $mlxController.settings.baseModelName)
                            .disabled(isRunning || mlxController.isDownloading)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()

                        Button("Download Base Model") {
                            mlxController.downloadBaseModel()
                        }
                        .disabled(isRunning || mlxController.isDownloading)
                        .fontWeight(.semibold)
                    }

                    Section("Downloads") {
                        Stepper("Curriculum epoch: \(mlxController.settings.curriculumEpoch)",
                                value: $mlxController.settings.curriculumEpoch, in: 0...999)
                            .disabled(isRunning || mlxController.isDownloading)

                        Button("Download Curriculum") {
                            mlxController.downloadCurriculum()
                        }
                        .disabled(isRunning || mlxController.isDownloading)
                        .fontWeight(.semibold)

                        Button("Download Model Adapter") {
                            mlxController.downloadModel()
                        }
                        .disabled(isRunning || mlxController.isDownloading)
                        .fontWeight(.semibold)

                        if mlxController.isDownloading {
                            Button("Cancel Download") {
                                mlxController.cancelDownload()
                            }
                            .foregroundStyle(.red)
                        }
                    }
                }

                Section("Status") {
                    LabeledContent("Device", value: deviceID)
                    LabeledContent("Anchor version",
                                   value: anchorVersion.map(String.init) ?? "—")
                    if useMLXTrainer {
                        LabeledContent("Last loss",
                                       value: lastLoss.map { String(format: "%.4f", $0) } ?? "—")
                    } else {
                        LabeledContent("Dir error vs target",
                                       value: lastLoss.map { String(format: "%.4f", $0) } ?? "—")
                    }
                    Button(isRunning ? "Stop" : "Start") {
                        if isRunning { stop() } else { start() }
                    }
                    .fontWeight(.semibold)
                }

                Section("Rounds") {
                    if log.isEmpty {
                        Text("No rounds yet")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(log.reversed()) { entry in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(entry.text)
                                .font(.caption.monospaced())
                                .foregroundStyle(entry.isError ? .red : .primary)
                            Text(entry.date, style: .time)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Swairm")
        }
    }

    private func start() {
        if useMLXTrainer {
            mlxController.start()
        } else {
            proxyController.start()
        }
    }

    private func stop() {
        if useMLXTrainer {
            mlxController.stop()
        } else {
            proxyController.stop()
        }
    }
}

// Unify LogEntry types
typealias LogEntry = DeviceLoopController.LogEntry

#Preview {
    ContentView()
}