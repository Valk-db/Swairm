// Single-screen UI: Anchor connection settings, run control, and a live
// round log. Supports both Proxy (linear proxy) and MLX (real DoRA) modes.

import SwiftUI

struct ContentView: View {
    @State private var useMLXTrainer = false
    @State private var proxyController = DeviceLoopController()
    @State private var mlxController = MLXDeviceLoopController()

    var controller: any ObservableObject {
        useMLXTrainer ? mlxController : proxyController
    }

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

    var deviceIndex: Int {
        get { useMLXTrainer ? mlxController.deviceIndex : proxyController.deviceIndex }
        set {
            if useMLXTrainer { mlxController.deviceIndex = newValue }
            else { proxyController.deviceIndex = newValue }
        }
    }

    var intervalSeconds: Double {
        get { useMLXTrainer ? mlxController.intervalSeconds : proxyController.intervalSeconds }
        set {
            if useMLXTrainer { mlxController.intervalSeconds = newValue }
            else { proxyController.intervalSeconds = newValue }
        }
    }

    var anchorURLText: String {
        get { useMLXTrainer ? mlxController.anchorURLText : proxyController.anchorURLText }
        set {
            if useMLXTrainer { mlxController.anchorURLText = newValue }
            else { proxyController.anchorURLText = newValue }
        }
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