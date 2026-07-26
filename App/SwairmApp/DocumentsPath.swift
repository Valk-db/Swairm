// URL(fileURLWithPath:) on a relative string resolves against
// FileManager.default.currentDirectoryPath -- meaningful for the macOS CLI
// (swift run resolves relative to the package dir), but on iOS a fresh
// process's current directory is not the app's Documents folder, which is
// where files copied in via Files app (UIFileSharingEnabled, see
// project.yml) actually land. So a relative modelPath/curriculumDirectory
// typed into the UI would resolve to nowhere useful on-device. Anchor
// relative paths to Documents explicitly; leave already-absolute paths
// (a leading "/") untouched.

import Foundation

func resolvedInDocuments(_ path: String) -> String {
    guard !path.isEmpty, !path.hasPrefix("/") else { return path }
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    return docs.appendingPathComponent(path).path
}