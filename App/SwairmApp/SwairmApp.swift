// Sideloadable Swairm iOS app entry point. The app is a thin shell over
// SwairmCore.ProxyDeviceLoop — the same round loop the CLI fleet and the
// CI integration job run, so a phone round is byte-identical to a CI round.

import SwiftUI

@main
struct SwairmApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
