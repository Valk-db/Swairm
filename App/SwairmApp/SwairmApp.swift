// Sideloadable Swairm iOS app entry point. The app is a thin shell over
// SwairmCore.ProxyDeviceLoop — the same round loop the CLI fleet and the
// CI integration job run, so a phone round is byte-identical to a CI round.
//
// Background task scheduler is registered to enable federated learning
// rounds when the app is backgrounded.

import SwiftUI
import UIKit
import SwairmCore

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        BackgroundTaskScheduler.registerBackgroundTasks()
        return true
    }

    func applicationDidEnterBackground(_ application: UIApplication) {
        BackgroundTaskScheduler.shared.applicationDidEnterBackground()
    }

    func applicationWillEnterForeground(_ application: UIApplication) {
        BackgroundTaskScheduler.shared.applicationWillEnterForeground()
    }
}

@main
struct SwairmApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
