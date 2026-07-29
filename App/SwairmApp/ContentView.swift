import SwiftUI
import SwairmCore

struct ContentView: View {
    @State private var useMLXTrainer = false
    @State private var proxyController = DeviceLoopController()
    @State private var mlxController = MLXDeviceLoopController()
    @State private var bgScheduler = BackgroundTaskScheduler.shared

    // Shared state properties that both controllers expose
    @State private var anchorURLText = "http://172.20.10.10:8000"
    @State private var deviceIndex = 0
    @State private var intervalSeconds = 25.0

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
                            // Stop any running loop when switching modes
                            if isRunning { stop() }
                        }
                }

                Section("Anchor") {
                    TextField("http://host:8000", text: $anchorURLText)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .disabled(isRunning)
                    Stepper("Device index: \(deviceIndex)",
                            value: $deviceIndex, in: 0...63)
                        .disabled(isRunning)
                    Stepper("Interval: \(Int(intervalSeconds))s",
                            value: $intervalSeconds,
                            in: 5...300, step: 5)
                        .disabled(isRunning)
                }

                if useMLXTrainer {
                    Section("MLX Model") {
                        TextField("Model path", text: $mlxController.modelPath)
                            .disabled(isRunning)
                        TextField("Curriculum dir", text: $mlxController.curriculumDirectory)
                            .disabled(isRunning)
                        Stepper("Steps/round: \(mlxController.maxStepsPerRound)",
                                value: $mlxController.maxStepsPerRound, in: 10...500, step: 10)
                            .disabled(isRunning)
                        Stepper("Batch size: \(mlxController.batchSize)",
                                value: $mlxController.batchSize, in: 1...8)
                            .disabled(isRunning)
                        Stepper("Seq length: \(mlxController.sequenceLength)",
                                value: $mlxController.sequenceLength, in: 64...512, step: 32)
                            .disabled(isRunning)
                    }

                    Section("Base Model") {
                        TextField("Base model name", text: $mlxController.baseModelName)
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
                        Stepper("Curriculum epoch: \(mlxController.curriculumEpoch)",
                                value: $mlxController.curriculumEpoch, in: 0...999)
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

                Section("Background Training") {
                    Toggle("Enable Background Rounds", isOn: $bgScheduler.isBackgroundEnabled)
                        .disabled(bgScheduler.isBackgroundTaskRunning)

                    if bgScheduler.lastResult != nil {
                        let result = bgScheduler.lastResult!
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Last Round: \(result.round)")
                                Spacer()
                                Text(result.isMLXMode ? "MLX" : "Proxy")
                                    .font(.caption)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(result.isMLXMode ? Color.blue.opacity(0.2) : Color.green.opacity(0.2))
                                    .cornerRadius(4)
                            }
                            if let loss = result.finalLoss {
                                Text("Loss: \(String(format: "%.4f", loss))")
                                    .font(.caption.monospaced())
                            }
                            Text("Anchor v\(result.anchorVersion) • \(result.stepsCompleted) steps • \(result.termination)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(result.timestamp, style: .relative)
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                            if let error = result.error {
                                Text("Error: \(error)")
                                    .font(.caption)
                                    .foregroundStyle(.red)
                            }
                        }
                    } else {
                        Text("No background rounds completed yet")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Button(bgScheduler.isBackgroundTaskRunning ? "Background task running…" : "Run Background Round Now") {
                        Task {
                            await bgScheduler.runBackgroundRoundNow()
                        }
                    }
                    .disabled(bgScheduler.isBackgroundTaskRunning || bgScheduler.anchorURLText.isEmpty)
                    .font(.caption)
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
        // Sync URL to background scheduler
        bgScheduler.anchorURLText = anchorURLText
        bgScheduler.config.deviceIndex = deviceIndex
        bgScheduler.config.useMLXTrainer = useMLXTrainer
        if useMLXTrainer {
            bgScheduler.config.modelPath = mlxController.modelPath
            bgScheduler.config.curriculumDirectory = mlxController.curriculumDirectory
            bgScheduler.config.maxStepsPerRound = mlxController.maxStepsPerRound
            bgScheduler.config.batchSize = mlxController.batchSize
            bgScheduler.config.sequenceLength = mlxController.sequenceLength
            bgScheduler.config.learningRate = mlxController.learningRate
        }

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