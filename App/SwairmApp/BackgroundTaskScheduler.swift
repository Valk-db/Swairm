// Background task scheduler for iOS federated learning rounds.
// Registers BGAppRefreshTask and BGProcessingTask to run periodic
// federated learning rounds when the app is backgrounded.
//
// Integrates with DeviceLoopController and MLXDeviceLoopController
// to execute a single training round per background task invocation.

import Foundation
import BackgroundTasks
import UIKit
import SwairmCore
import os.log

private let logger = OSLog(subsystem: "com.swairm.app", category: "BackgroundTaskScheduler")

/// Background task identifiers registered in Info.plist
/// BGAppRefreshTaskIdentifier: "com.swairm.app.refresh" (30s budget)
/// BGProcessingTaskIdentifier: "com.swairm.app.processing" (minutes budget)
enum BackgroundTaskIdentifier {
    static let refresh = "com.swairm.app.refresh"
    static let processing = "com.swairm.app.processing"
}

/// Result of a background training round, stored for UI retrieval on next launch.
struct BackgroundRoundResult: Codable {
    let deviceID: String
    let round: Int
    let anchorVersion: Int
    let finalLoss: Float?
    let stepsCompleted: Int
    let wallClock: TimeInterval
    let termination: String
    let timestamp: Date
    let isMLXMode: Bool
    let error: String?
}

/// Configuration for background training rounds.
struct BackgroundTrainingConfig: Codable, Sendable {
    var anchorURL: String
    var deviceIndex: Int
    var intervalSeconds: TimeInterval
    var useMLXTrainer: Bool
    var modelPath: String?
    var curriculumDirectory: String?
    var maxStepsPerRound: Int
    var batchSize: Int
    var sequenceLength: Int
    var learningRate: Float
    var minBatteryFraction: Float?
    var hmacSecret: Data?

    // No default host baked in on purpose: a hardcoded LAN IP silently
    // breaks (or worse, silently points at the wrong box) the moment the
    // Anchor moves. Empty string fails the existing URL(string:) guard in
    // executeTrainingRound/runBackgroundRoundNow with a clear
    // "Invalid Anchor URL" error until the user sets one via anchorURLText.
    static let `default` = BackgroundTrainingConfig(
        anchorURL: "",
        deviceIndex: 0,
        intervalSeconds: 15 * 60,  // 15 minutes between background refreshes
        useMLXTrainer: false,
        modelPath: nil,
        curriculumDirectory: nil,
        maxStepsPerRound: 1,
        batchSize: 2,
        sequenceLength: 128,
        learningRate: 1e-4,
        minBatteryFraction: 0.2,
        hmacSecret: nil
    )
}

/// Background task scheduler for federated learning rounds.
/// Registers background tasks, executes training rounds, and persists results.
@MainActor
@Observable
final class BackgroundTaskScheduler {
    static let shared = BackgroundTaskScheduler()

    // Configuration (mutable for UI binding)
    var config: BackgroundTrainingConfig = .default

    // State
    private(set) var isBackgroundTrainingEnabled = false
    var isBackgroundEnabled: Bool {
        get { isBackgroundTrainingEnabled }
        set { setEnabled(newValue) }
    }
    private(set) var lastResult: BackgroundRoundResult?
    private(set) var lastError: String?
    private(set) var isBackgroundTaskRunning = false

    // Computed property for UI binding to anchor URL
    var anchorURLText: String {
        get { config.anchorURL }
        set {
            config = BackgroundTrainingConfig(
                anchorURL: newValue,
                deviceIndex: config.deviceIndex,
                intervalSeconds: config.intervalSeconds,
                useMLXTrainer: config.useMLXTrainer,
                modelPath: config.modelPath,
                curriculumDirectory: config.curriculumDirectory,
                maxStepsPerRound: config.maxStepsPerRound,
                batchSize: config.batchSize,
                sequenceLength: config.sequenceLength,
                learningRate: config.learningRate,
                minBatteryFraction: config.minBatteryFraction,
                hmacSecret: config.hmacSecret
            )
        }
    }

    // Internal
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    private var currentRound = 0
    private let resultsStorageURL: URL
    private let configStorageURL: URL
    private let log = OSLog(subsystem: "com.swairm.app", category: "BackgroundTaskScheduler")

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        self.resultsStorageURL = docs.appendingPathComponent("background_results.json")
        self.configStorageURL = docs.appendingPathComponent("background_config.json")
        loadConfig()
        loadLastResult()
    }

    // MARK: - Public Configuration

    /// Update configuration and persist to disk.
    func updateConfig(_ newConfig: BackgroundTrainingConfig) {
        config = newConfig
        saveConfig()
        if isBackgroundTrainingEnabled {
            scheduleBackgroundTasks()
        }
    }

    /// Enable or disable background training.
    func setEnabled(_ enabled: Bool) {
        isBackgroundTrainingEnabled = enabled
        if enabled {
            scheduleBackgroundTasks()
        } else {
            cancelAllBackgroundTasks()
        }
    }

    /// Call when app enters background to schedule next round.
    func applicationDidEnterBackground() {
        guard isBackgroundTrainingEnabled else { return }
        scheduleBackgroundTasks()
        os_log("App backgrounded - scheduled background training", log: log, type: .info)
    }

    /// Call when app enters foreground to cancel pending background tasks.
    func applicationWillEnterForeground() {
        cancelAllBackgroundTasks()
        os_log("App foregrounded - cancelled background training tasks", log: log, type: .info)
    }

    /// Get the last completed background round result (for UI display).
    func getLastResult() -> BackgroundRoundResult? {
        return lastResult
    }

    /// Clear stored results.
    func clearResults() {
        lastResult = nil
        lastError = nil
        try? FileManager.default.removeItem(at: resultsStorageURL)
    }

    /// Run a background round immediately (for testing from UI).
    func runBackgroundRoundNow() async {
        guard !isBackgroundTaskRunning else { return }
        guard URL(string: config.anchorURL) != nil else {
            lastError = "Invalid Anchor URL: \(config.anchorURL)"
            return
        }

        isBackgroundTaskRunning = true
        defer { isBackgroundTaskRunning = false }

        do {
            try await executeTrainingRound(isMLXMode: config.useMLXTrainer)
        } catch {
            lastError = error.localizedDescription
        }
    }

    // MARK: - Background Task Registration

    /// Register background task handlers. Call from AppDelegate or @main entry point.
    static func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: BackgroundTaskIdentifier.refresh, using: nil) { task in
            Task { @MainActor in
                await BackgroundTaskScheduler.shared.handleRefreshTask(task as! BGAppRefreshTask)
            }
        }

        BGTaskScheduler.shared.register(forTaskWithIdentifier: BackgroundTaskIdentifier.processing, using: nil) { task in
            Task { @MainActor in
                await BackgroundTaskScheduler.shared.handleProcessingTask(task as! BGProcessingTask)
            }
        }

        os_log("Registered background task identifiers: %s, %s", log: logger, type: .info,
               BackgroundTaskIdentifier.refresh, BackgroundTaskIdentifier.processing)
    }

    // MARK: - Task Scheduling

    private func scheduleBackgroundTasks() {
        // Schedule BGAppRefreshTask for periodic quick rounds (15 min interval, 30s budget)
        let refreshRequest = BGAppRefreshTaskRequest(identifier: BackgroundTaskIdentifier.refresh)
        refreshRequest.earliestBeginDate = Date(timeIntervalSinceNow: config.intervalSeconds)

        do {
            try BGTaskScheduler.shared.submit(refreshRequest)
            os_log("Scheduled BGAppRefreshTask at %@", log: log, type: .info,
                   refreshRequest.earliestBeginDate! as NSDate)
        } catch {
            os_log("Failed to schedule BGAppRefreshTask: %@", log: log, type: .error, error as CVarArg)
        }

        // Also schedule BGProcessingTask for longer MLX training sessions (if MLX mode)
        if config.useMLXTrainer {
            let processingRequest = BGProcessingTaskRequest(identifier: BackgroundTaskIdentifier.processing)
            processingRequest.earliestBeginDate = Date(timeIntervalSinceNow: config.intervalSeconds)
            processingRequest.requiresNetworkConnectivity = true
            processingRequest.requiresExternalPower = false

            do {
                try BGTaskScheduler.shared.submit(processingRequest)
                os_log("Scheduled BGProcessingTask at %@", log: log, type: .info,
                       processingRequest.earliestBeginDate! as NSDate)
            } catch {
                os_log("Failed to schedule BGProcessingTask: %@", log: log, type: .error, error as CVarArg)
            }
        }
    }

    private func cancelAllBackgroundTasks() {
        BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: BackgroundTaskIdentifier.refresh)
        BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: BackgroundTaskIdentifier.processing)
        os_log("Cancelled all background tasks", log: log, type: .info)
    }

    // MARK: - Task Handlers

    private func handleRefreshTask(_ task: BGAppRefreshTask) async {
        os_log("BGAppRefreshTask started", log: log, type: .info)
        isBackgroundTaskRunning = true

        // Begin background task for extra time if needed
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "FL-Round-Refresh") { [weak self] in
            self?.endBackgroundTask()
        }

        // Set expiration handler
        task.expirationHandler = { [weak self] in
            os_log("BGAppRefreshTask expired", log: self?.log ?? logger, type: .error)
            self?.endBackgroundTask()
            self?.isBackgroundTaskRunning = false
        }

        do {
            try await executeTrainingRound(isMLXMode: config.useMLXTrainer)
            // Schedule next refresh
            scheduleBackgroundTasks()
            task.setTaskCompleted(success: true)
        } catch {
            os_log("BGAppRefreshTask failed: %@", log: log, type: .error, error as CVarArg)
            lastError = error.localizedDescription
            task.setTaskCompleted(success: false)
        }

        endBackgroundTask()
        isBackgroundTaskRunning = false
    }

    private func handleProcessingTask(_ task: BGProcessingTask) async {
        os_log("BGProcessingTask started", log: log, type: .info)
        isBackgroundTaskRunning = true

        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "FL-Round-Processing") { [weak self] in
            self?.endBackgroundTask()
        }

        task.expirationHandler = { [weak self] in
            os_log("BGProcessingTask expired", log: self?.log ?? logger, type: .error)
            self?.endBackgroundTask()
            self?.isBackgroundTaskRunning = false
        }

        do {
            try await executeTrainingRound(isMLXMode: config.useMLXTrainer)
            scheduleBackgroundTasks()
            task.setTaskCompleted(success: true)
        } catch {
            os_log("BGProcessingTask failed: %@", log: log, type: .error, error as CVarArg)
            lastError = error.localizedDescription
            task.setTaskCompleted(success: false)
        }

        endBackgroundTask()
        isBackgroundTaskRunning = false
    }

    private func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }

    // MARK: - Training Round Execution

    private func executeTrainingRound(isMLXMode: Bool) async throws {
        currentRound += 1
        let round = currentRound

        guard let anchorURL = URL(string: config.anchorURL) else {
            throw BackgroundSchedulerError.invalidAnchorURL(config.anchorURL)
        }

        os_log("Starting background round %d (MLX=%@)", log: log, type: .info, round, isMLXMode ? "true" : "false")

        let anchor = AnchorClient(base: anchorURL, hmacSecret: config.hmacSecret)

        // Battery check
        if let minBattery = config.minBatteryFraction {
            UIDevice.current.isBatteryMonitoringEnabled = true
            let level = UIDevice.current.batteryLevel
            if level >= 0 && level < minBattery {
                throw BackgroundSchedulerError.batteryTooLow(level)
            }
        }

        // Thermal check
        if ProcessInfo.processInfo.thermalState == .serious || ProcessInfo.processInfo.thermalState == .critical {
            throw BackgroundSchedulerError.thermalState(ProcessInfo.processInfo.thermalState)
        }

        let deviceID = "bg-phone\(config.deviceIndex)"
        // let startTime = Date()  // not used

        let budget = ResourceBudget(
            maxSteps: config.maxStepsPerRound,
            maxWallClock: isMLXMode ? 300 : 60,  // MLX gets more time
            stopOnSeriousThermalState: true,
            minBatteryFraction: config.minBatteryFraction
        )

        var finalLoss: Float?
        var stepsCompleted = 0
        var wallClock: TimeInterval = 0
        var termination: TerminationReason = .cancelled
        var anchorVersion = 0
        var error: String?

        do {
            if isMLXMode {
                guard let modelPath = config.modelPath, !modelPath.isEmpty else {
                    throw BackgroundSchedulerError.missingModelPath
                }
                // Curriculum directory is now resolved automatically to curriculum/epoch_{epoch}

                let loop = try MLXDeviceLoop(
                    anchor: anchor,
                    deviceID: deviceID,
                    deviceIndex: config.deviceIndex,
                    config: MLXLoopConfig(
                        modelPath: resolvedInDocuments(modelPath),
                        targetModules: ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                        rankMap: ["": 6],
                        alphaMap: ["": 16.0],
                        learningRate: config.learningRate,
                        weightDecay: 0.01,
                        maxGradNorm: 1.0,
                        warmupSteps: 10,
                        maxStepsPerRound: config.maxStepsPerRound,
                        batchSize: config.batchSize,
                        sequenceLength: config.sequenceLength,
                        curriculumDirectory: resolvedInDocuments("curriculum/epoch_\(config.deviceIndex)"),
                        seed: 42 + UInt64(config.deviceIndex)
                    )
                )

                let result = try await loop.runRound(budget: budget)
                finalLoss = result.trainingReport.finalLoss
                stepsCompleted = result.trainingReport.stepsCompleted
                wallClock = result.trainingReport.wallClock
                termination = result.trainingReport.termination
                anchorVersion = result.status.version

            } else {
                // Proxy mode - linear proxy training
                let batteryFractionProvider: @Sendable () -> Float? = { @MainActor in
                    let level = UIDevice.current.batteryLevel
                    return level >= 0 ? level : nil
                }

                let loop = ProxyDeviceLoop(
                    anchor: anchor,
                    deviceID: deviceID,
                    deviceIndex: config.deviceIndex,
                    batteryFraction: batteryFractionProvider
                )

                let result = try await loop.runRound(budget: budget)
                // Proxy returns dirErrorVsTarget as "loss" equivalent
                finalLoss = result.dirErrorVsTarget
                stepsCompleted = result.trainingReport.stepsCompleted
                wallClock = result.trainingReport.wallClock
                termination = result.trainingReport.termination
                anchorVersion = result.status.version
            }

        } catch let caughtError {
            error = caughtError.localizedDescription
            os_log("Background round %d failed: %@", log: log, type: .error, round, caughtError as CVarArg)
            throw caughtError
        }

        let result = BackgroundRoundResult(
            deviceID: deviceID,
            round: round,
            anchorVersion: anchorVersion,
            finalLoss: finalLoss,
            stepsCompleted: stepsCompleted,
            wallClock: wallClock,
            termination: String(describing: termination),
            timestamp: Date(),
            isMLXMode: isMLXMode,
            error: error
        )

        lastResult = result
        lastError = error
        saveResult(result)

        os_log("Background round %d completed: anchor v%d, loss=%@, steps=%d, term=%@",
               log: log, type: .info, round, anchorVersion,
               finalLoss.map { String(format: "%.4f", $0) } ?? "nil",
               stepsCompleted, String(describing: termination))
    }

    // MARK: - Persistence

    private func saveConfig() {
        do {
            let data = try JSONEncoder().encode(config)
            try data.write(to: configStorageURL, options: .atomic)
        } catch {
            os_log("Failed to save background config: %@", log: log, type: .error, error as CVarArg)
        }
    }

    private func loadConfig() {
        guard let data = try? Data(contentsOf: configStorageURL),
              let decoded = try? JSONDecoder().decode(BackgroundTrainingConfig.self, from: data) else {
            return
        }
        config = decoded
    }

    private func saveResult(_ result: BackgroundRoundResult) {
        do {
            let data = try JSONEncoder().encode(result)
            try data.write(to: resultsStorageURL, options: .atomic)
        } catch {
            os_log("Failed to save background result: %@", log: log, type: .error, error as CVarArg)
        }
    }

    private func loadLastResult() {
        guard let data = try? Data(contentsOf: resultsStorageURL),
              let decoded = try? JSONDecoder().decode(BackgroundRoundResult.self, from: data) else {
            return
        }
        lastResult = decoded
    }
}

// MARK: - Errors

enum BackgroundSchedulerError: LocalizedError {
    case invalidAnchorURL(String)
    case batteryTooLow(Float)
    case thermalState(ProcessInfo.ThermalState)
    case missingModelPath
    case missingCurriculumDirectory
    case backgroundTaskExpired

    var errorDescription: String? {
        switch self {
        case .invalidAnchorURL(let url):
            return "Invalid Anchor URL: \(url)"
        case .batteryTooLow(let level):
            return "Battery too low for background training: \(Int(level * 100))%"
        case .thermalState(let state):
            return "Thermal state too high for background training: \(state)"
        case .missingModelPath:
            return "MLX model path not configured"
        case .missingCurriculumDirectory:
            return "Curriculum directory not configured"
        case .backgroundTaskExpired:
            return "Background task expired before completion"
        }
    }
}

// MARK: - AppDelegate Integration

/// Call this from your AppDelegate or @main App to register background tasks.
/// Add to AppDelegate.application(_:didFinishLaunchingWithOptions:):
///     BackgroundTaskScheduler.registerBackgroundTasks()
///
/// And handle lifecycle:
///     func applicationDidEnterBackground(_ application: UIApplication) {
///         BackgroundTaskScheduler.shared.applicationDidEnterBackground()
///     }
///     func applicationWillEnterForeground(_ application: UIApplication) {
///         BackgroundTaskScheduler.shared.applicationWillEnterForeground()
///     }
///
/// Also add these to Info.plist:
/// <key>BGTaskSchedulerPermittedIdentifiers</key>
/// <array>
///     <string>com.swairm.app.refresh</string>
///     <string>com.swairm.app.processing</string>
/// </array>
/// <key>UIBackgroundModes</key>
/// <array>
///     <string>background-processing</string>
///     <string>background-fetch</string>
/// </array>