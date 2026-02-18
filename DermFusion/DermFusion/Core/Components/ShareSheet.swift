//
//  ShareSheet.swift
//  DermFusion
//
//  Presents the system share sheet (UIActivityViewController) for sharing URLs or other items.
//

import SwiftUI
import UIKit

/// Wraps UIActivityViewController for use in SwiftUI.
struct ShareSheet: UIViewControllerRepresentable {

    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
