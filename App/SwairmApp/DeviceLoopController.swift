// Drives one ProxyDeviceLoop against a LAN Anchor from the UI.
//
// Owns the run task and republishes round results as log entries. All
// mutable state lives on the MainActor; the loop itself is an actor, so
// there is no shared mutable state between the UI and training.

import SwiftUI
import UIKit
import SwairmCore

@MainActor
@Observable
final class DeviceLoopController {
    // ------------------------------------------------------------- config
    var anchorURLText = "http://192.168.1.100:8000"
    var deviceIndex = 0
    /// Seconds to wait between rounds (mirrors the CLI --interval flag).
    var intervalSeconds = 25.0

    // ------------------------------------------------------------- state
    private(set) var isRunning = false
    private(set) var anchorVersion: Int?
    private(set) var lastDirError: Float?
    private(set) var log: [LogEntry] = []

    private var runTask: Task<Void, Never>?

    struct LogEntry: Identifiable {
        let id = UUID()
        let date = Date()
        let text: String
        let isError: Bool
    }

    var deviceID: String { "phone\(deviceIndex)" }

    // ------------------------------------------------------------- control
    func start() {
        guard !isRunning else { return }
        guard let url = URL(string: anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(anchorURLText)", isError: true)
            return
        }

        UIDevice.current.isBatteryMonitoringEnabled = true
        let anchor = AnchorClient(base: url)
        let loop = ProxyDeviceLoop(
            anchor: anchor, deviceID: deviceID, deviceIndex: deviceIndex,
            batteryFraction: {
                let level = UIDevice.current.batteryLevel
                return level >= 0 ? level : nil
            })

        isRunning = true
        append("Started \(deviceID) against \(url.absoluteString)")

        let interval = intervalSeconds
        runTask = Task { [weak self] in
            let budget = ResourceBudget(maxSteps: 1, maxWallClock: 60,
                                        minBatteryFraction: 0.2)
            while !Task.isCancelled {
                do {
                    let result = try await loop.runRound(budget: budget)
                    self?.record(result)
                } catch is CancellationError {
                    break
                } catch {
                    self?.append("Round failed: \(error)", isError: true)
                }
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
            }
            self?.finish()
        }
    }

    func stop() {
        runTask?.cancel()
        runTask = nil
    }

    // ------------------------------------------------------------- private
    private func record(_ result: ProxyRoundResult) {
        anchorVersion = result.status.version
        lastDirError = result.dirErrorVsTarget
        append(String(
            format: "round %d | anchor v%d | dir err %.4f | %d step(s), %.2fs",
            result.round, result.status.version, result.dirErrorVsTarget,
            result.trainingReport.stepsCompleted,
            result.trainingReport.wallClock))
    }

    private func finish() {
        isRunning = false
        append("Stopped")
    }

    private func append(_ text: String, isError: Bool = false) {
        log.append(LogEntry(text: text, isError: isError))
        if log.count > 200 { log.removeFirst(log.count - 200) }
    }
}
