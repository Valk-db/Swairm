// Drives one ProxyDeviceLoop against a LAN Anchor from the UI.
// Owns the run task and republishes round results as log entries. All
// mutable state lives on the MainActor; the loop itself is an actor, so
// there is no shared mutable state between the UI and training.

import SwiftUI
import UIKit
import SwairmCore

// Settings persisted via UserDefaults — kept separate from @Observable
// because the Observation macro doesn't support property wrappers on iOS targets.
@MainActor
@Observable
final class ProxySettings {
    var anchorURLText: String {
        get { UserDefaults.standard.string(forKey: "proxy.anchorURLText") ?? "http://172.20.10.5:8000" }
        set { UserDefaults.standard.set(newValue, forKey: "proxy.anchorURLText") }
    }
    var deviceIndex: Int {
        get { UserDefaults.standard.integer(forKey: "proxy.deviceIndex") }
        set { UserDefaults.standard.set(newValue, forKey: "proxy.deviceIndex") }
    }
    var intervalSeconds: Double {
        get { UserDefaults.standard.double(forKey: "proxy.intervalSeconds") == 0 ? 25.0 : UserDefaults.standard.double(forKey: "proxy.intervalSeconds") }
        set { UserDefaults.standard.set(newValue, forKey: "proxy.intervalSeconds") }
    }
}

@MainActor
@Observable
final class DeviceLoopController {
    // ------------------------------------------------------------- config (persisted via ProxySettings)
    var settings = ProxySettings()

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

    var deviceID: String { "phone\(settings.deviceIndex)" }

    // ------------------------------------------------------------- control
    func start() {
        guard !isRunning else { return }
        guard let url = URL(string: settings.anchorURLText), url.scheme != nil else {
            append("Invalid Anchor URL: \(settings.anchorURLText)", isError: true)
            return
        }

        UIDevice.current.isBatteryMonitoringEnabled = true
        let anchor = AnchorClient(base: url)
        let deviceID = deviceID
        let batteryFractionProvider: @Sendable () -> Float? = { @MainActor in
            let level = UIDevice.current.batteryLevel
            return level >= 0 ? level : nil
        }
        let loop = ProxyDeviceLoop(
            anchor: anchor, deviceID: deviceID, deviceIndex: settings.deviceIndex,
            batteryFraction: batteryFractionProvider)

        isRunning = true
        append("Started \(deviceID) against \(url.absoluteString)")

        let interval = settings.intervalSeconds
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