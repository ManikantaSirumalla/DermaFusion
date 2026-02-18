//
//  DermaFusionApp.swift
//  DermFusion
//
//  App entry point and dependency root for DermaFusion.
//  Enforces first-launch disclaimer gating and initializes local-only persistence.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

@main
struct DermaFusionApp: App {

    // MARK: - Properties

    @AppStorage(DFMedicalDisclaimer.acceptanceKey) private var hasAcceptedDisclaimer = false
    @StateObject private var appDataStore = AppDataStore()

    // MARK: - Body

    var body: some Scene {
        WindowGroup {
            Group {
                if hasAcceptedDisclaimer {
                    ContentView()
                        .environmentObject(appDataStore)
                } else {
                    DisclaimerView(
                        onAccept: {
                            hasAcceptedDisclaimer = true
                        }
                    )
                }
            }
        }
    }
}
