// Single-screen UI: Anchor connection settings, run control, and a live
// round log. Deliberately spartan — the app's job is to run rounds, not
// to be a dashboard.

import SwiftUI

struct ContentView: View {
    @State private var controller = DeviceLoopController()

    var body: some View {
        NavigationStack {
            Form {
                Section("Anchor") {
                    TextField("http://host:8000", text: $controller.anchorURLText)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .disabled(controller.isRunning)
                    Stepper("Device index: \(controller.deviceIndex)",
                            value: $controller.deviceIndex, in: 0...63)
                        .disabled(controller.isRunning)
                    Stepper("Interval: \(Int(controller.intervalSeconds))s",
                            value: $controller.intervalSeconds,
                            in: 5...300, step: 5)
                        .disabled(controller.isRunning)
                }

                Section("Status") {
                    LabeledContent("Device", value: controller.deviceID)
                    LabeledContent("Anchor version",
                                   value: controller.anchorVersion.map(String.init) ?? "—")
                    LabeledContent("Dir error vs target",
                                   value: controller.lastDirError
                                       .map { String(format: "%.4f", $0) } ?? "—")
                    Button(controller.isRunning ? "Stop" : "Start") {
                        if controller.isRunning {
                            controller.stop()
                        } else {
                            controller.start()
                        }
                    }
                    .fontWeight(.semibold)
                }

                Section("Rounds") {
                    if controller.log.isEmpty {
                        Text("No rounds yet")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(controller.log.reversed()) { entry in
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
}

#Preview {
    ContentView()
}
